import os.path as osp
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler
from dassl.data.datasets import Datum
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from clip_w_local import clip
import numpy as np
from tqdm import tqdm
from torchvision import datasets
from utils.detection_util import get_and_print_results
from dassl.data.transforms import build_transform
from torch.utils.data import Dataset as TorchDataset
from utils.training_util import build_data_loader
from .custom_clip import load_clip_to_cpu, TextEncoder, PromptLearner, CustomCLIP, CustomZSCLIP

from PIL import Image, ImageDraw
import cv2
import shutil
import numpy as np
import os
import torch
from PIL import Image
import random
from concurrent.futures import ThreadPoolExecutor  # 引入线程池加速IO
from torchvision import transforms
def entropy_select_topk(p, bs, label, num_of_local_feature, mask):
    if num_of_local_feature == 196:
        patch_size = 16
    else:
        patch_size = 32
    mask_patches = mask.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patch_counts = (mask_patches == 255).sum(dim=(3, 4))
    patch_counts = patch_counts.view(bs, -1)
    patch_counts = patch_counts > 100
    patch_counts[label == p.shape[1]] = False

    patch_counts = patch_counts.view(-1)
    label_repeat = label.repeat_interleave(num_of_local_feature)
    p = F.softmax(p, dim=-1)
    selected_p = p[~patch_counts]

    if selected_p.shape[0] == 0:
        return torch.tensor([0]).cuda()
    return -torch.mean(torch.sum(selected_p * torch.log(selected_p + 1e-5), 1))

def copy_images_with_overwrite(folder_a, folder_b, image_extensions=None):

    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']

    os.makedirs(folder_b, exist_ok=True)

    if not os.path.exists(folder_a):
        print(f"错误：源文件夹 {folder_a} 不存在！")
        return
    
    copied_count = 0

    for filename in os.listdir(folder_a):
        source_file = os.path.join(folder_a, filename)

        if os.path.isfile(source_file):
            file_ext = os.path.splitext(filename)[1].lower()

            if file_ext in image_extensions:
                target_file = os.path.join(folder_b, filename)
                shutil.copy2(source_file, target_file)


                copied_count += 1

    print("\n===== 复制完成 =====")
    print(f"成功复制/覆盖：{copied_count} 个图片文件")


@TRAINER_REGISTRY.register()
class OSPCoOp(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.OSPCOOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        cfg.num_classes = len(classnames)
        self.DATA_ROOT = cfg.DATA_ROOT[0]
        self.saved_classes_count = {} 
        self.id_aug_thre = self.cfg.id_aug_thre
        self.num_aug_versions = self.cfg.num_aug_versions
        self.num_classes = len(classnames)
        self.train_root = self.cfg.train_root

        self.loss1 = cfg.loss1
        self.loss2 = cfg.loss2
        self.augmented_img_paths = set()
        self.glmcm_local_weight = cfg.glmcm_local_weight
        self.T = cfg.T
        self.eval_bg = cfg.eval_bg
        self.ood_aug = cfg.ood_aug
        self.id_aug_type = cfg.id_aug_type
        self.eval_freq = cfg.eval_freq
        self.start_eval_epoch = cfg.start_eval_epoch
        self.inpaint_thre = cfg.inpaint_thre
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.OSPCOOP.PREC == "fp32" or cfg.TRAINER.OSPCOOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        self.zsmodel = CustomZSCLIP(cfg, classnames, clip_model)
        self.zsmodel.eval()
        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.OSPCOOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        self.ood_loader()
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label, mask = self.parse_batch_train(batch)

        output, output_local = self.model(image)

        valid_indices = (label >= 0) & (label <= output.shape[1] - 1)

        filtered_output = output[valid_indices]
        filtered_gt = label[valid_indices]

        loss_id = F.cross_entropy(filtered_output, filtered_gt)

        valid_indices_ood = (label == output.shape[1])
        filtered_output_ood = output[valid_indices_ood]
        filtered_output_ood = F.softmax(filtered_output_ood, dim=-1)

        loss_ood = torch.mean(torch.sum(filtered_output_ood * torch.log(filtered_output_ood + 1e-5), 1))

        batch_size, num_of_local_feature = output_local.shape[0], output_local.shape[1]

        output_local = output_local.view(batch_size * num_of_local_feature, -1)
        loss_en = - entropy_select_topk(output_local, batch_size, label, num_of_local_feature, mask)

        if filtered_output.shape[0] == 0:
            loss = self.loss1 * loss_en + self.loss2 * loss_ood
            self.model_backward_and_update(loss)
            loss_summary = {
                "loss": loss.item(),
                "loss_ood": loss_ood.item(),
                "loss_en": loss_en.item(),
            }
        elif filtered_output_ood.shape[0] == 0:
            loss = loss_id + self.loss1 * loss_en
            self.model_backward_and_update(loss)
            loss_summary = {
                "loss": loss.item(),
                "loss_id": loss_id.item(),
                "loss_en": loss_en.item(),
                "acc": compute_accuracy(filtered_output, filtered_gt)[0].item(),
            }
        else:
            loss = loss_id + self.loss1 * loss_en + self.loss2 * loss_ood
            self.model_backward_and_update(loss)
            loss_summary = {
                "loss": loss.item(),
                'loss_id': loss_id.item(),
                "loss_ood": loss_ood.item(),
                "loss_en": loss_en.item(),
                "acc": compute_accuracy(filtered_output, filtered_gt)[0].item(),

            }
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        mask = batch["mask"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label, mask

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    def train(self):  # , start_epoch, max_epoch):
        """Generic training loops."""
        self.before_train()
        self.gen_idaug(aug_case = self.id_aug_type, num_aug_versions= self.num_aug_versions)
        
        self.ood_loader()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            if (self.epoch + 1) > self.start_eval_epoch:
                if (self.epoch + 1) % self.eval_freq == 0:
                    mcm_aurod, glmcm_auroc = self.eval_ood()
                    self.after_epoch(glmcm_auroc)

                elif (self.epoch + 1) == self.max_epoch:
                    mcm_aurod, glmcm_auroc = self.eval_ood()
                    self.after_epoch(glmcm_auroc)
        self.load_model(directory=self.output_dir)
        _, _ = self.eval_ood()

    def after_epoch(self, auroc):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST

        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        curr_result = auroc
        is_best = curr_result > self.best_result
        if is_best:
            self.best_result = curr_result
            self.save_model(
                self.epoch,
                self.output_dir,
                val_result=curr_result,
                model_name="model-best.pth.tar"
            )

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

    @torch.no_grad()
    def test_ood(self, data_loader, T):
        """Test-time OOD detection pipeline."""
        to_np = lambda x: x.data.cpu().numpy()
        concat = lambda x: np.concatenate(x, axis=0)

        self.set_model_mode("eval")
        self.evaluator.reset()

        glmcm_score = []
        mcm_score = []
        for batch_idx, (images, labels, *id_flag) in enumerate(tqdm(data_loader)):
            images = images.cuda()
            output, output_local = self.model_inference(images)

            output /= 100.0
            output_local /= 100.0
            smax_global = to_np(F.softmax(output / T, dim=-1))
            smax_local = to_np(F.softmax(output_local / T, dim=-1))
            mcm_global_score = -np.max(smax_global, axis=1)
            local_score = -np.max(smax_local, axis=(1, 2))

            mcm_score.append(mcm_global_score)
            glmcm_score.append(local_score)

        return concat(mcm_score)[:len(data_loader.dataset)].copy(), concat(glmcm_score)[
                                                                    :len(data_loader.dataset)].copy()

    @torch.no_grad()
    def eval_ood(self):

        _, preprocess = clip.load("ViT-B/16", device='cpu')

        id_data_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(self.DATA_ROOT['ID'], transform=preprocess),
            batch_size=512, shuffle=False, num_workers = 4, pin_memory = True)

        auroc_list_mcm, aupr_list_mcm, fpr_list_mcm = [], [], []
        auroc_list_gl, aupr_list_gl, fpr_list_gl = [], [], []

        in_score_mcm, in_score_gl = self.test_ood(id_data_loader, T=self.T)

        if self.eval_bg:
            out_datasets = ['iNaturalist', 'SUN', 'places365', 'Texture','ImageNet-Bg']
        else:
            out_datasets = ['iNaturalist', 'SUN', 'places365', 'Texture']
        in_score_gl = in_score_gl * self.glmcm_local_weight + in_score_mcm

        for out_dataset in out_datasets:
            print(f"Evaluting OOD dataset {out_dataset}")
            ood_loader = self.set_ood_loader_ImageNet(out_dataset, preprocess)
            out_score_mcm, out_score_gl = self.test_ood(ood_loader, T=self.T)

            out_score_gl = out_score_gl * self.glmcm_local_weight + out_score_mcm
            print("MCM score")
            get_and_print_results(in_score_mcm, out_score_mcm,
                                  auroc_list_mcm, aupr_list_mcm, fpr_list_mcm)

            print("GL-MCM score")
            get_and_print_results(in_score_gl, out_score_gl,
                                  auroc_list_gl, aupr_list_gl, fpr_list_gl)

        print("MCM avg. FPR:{}, AUROC:{}, AUPR:{}".format(np.mean(fpr_list_mcm), np.mean(auroc_list_mcm),
                                                          np.mean(aupr_list_mcm)))
        print("GL-MCM avg. FPR:{}, AUROC:{}, AUPR:{}".format(np.mean(fpr_list_gl), np.mean(auroc_list_gl),
                                                             np.mean(aupr_list_gl)))

        return np.mean(auroc_list_mcm), np.mean(auroc_list_gl)

    def set_ood_loader_ImageNet(self, out_dataset, preprocess):
        testsetout = datasets.ImageFolder(root=self.DATA_ROOT['OOD'][out_dataset],
                                          transform=preprocess)
        testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=512,
                                                    shuffle=False, num_workers=4)
        return testloaderOut
    
            
    @torch.no_grad()
    def gen_idaug(self, batch_size=16, aug_case='mix', num_aug_versions=64):
        
        self.set_model_mode("eval")
        self.evaluator.reset()
        
        dataset = self.raw_data_source
        
        # Device config (prefer GPU, enable half-precision inference)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            torch.backends.cudnn.benchmark = True  # Enable cuDNN optimization
            dtype = torch.float16  # Half-precision inference
        else:
            dtype = torch.float32
        # Load CLIP model for preprocessing (consistent with reference function)
        clip_model, preprocess = clip.load("ViT-B/16", device=device)
        clip_model.eval()  # Keep CLIP model in eval mode
        
        # Load background image lists (separate bg/texture, support mix mode)
        bg_img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
        # ========== Core change 1: store bg and texture paths/tensors separately ==========
        bg_paths = []       # bg folder paths
        texture_paths = []  # texture folder paths
        save_dir = self.output_dir
        mask_dir = os.path.join('./data/' + self.train_root, 'mask')
        
        # Load bg folder
        bg_dir = os.path.join('./data/' + self.train_root, 'score_inpaint')
        for f in os.listdir(bg_dir):
            if f.lower().endswith(bg_img_extensions):
                score = float(f[:-5].split('_')[-1])
                if score > self.inpaint_thre:
                    bg_paths.append(os.path.join(bg_dir, f))
        
        # Load texture folder
        texture_dir = os.path.join('./data/' + self.train_root, 'texture', 'inpaint', 'top')
        for f in os.listdir(texture_dir):
            if f.lower().endswith(bg_img_extensions):
                texture_paths.append(os.path.join(texture_dir, f))
        
        # ========== Core change 2: configure paths and save dirs based on aug_case ==========
        if aug_case == 'bg':
            bg_img_paths = bg_paths
        elif aug_case == 'texture':
            bg_img_paths = texture_paths
            
        id_aug_dir = os.path.join(save_dir, "id_aug")
        id_aug_vis_dir = os.path.join(save_dir, "id_aug_vis")
        fallback_dir = os.path.join(save_dir, "id_aug_fallback")
        
        # Create all output directories upfront
        for dir_path in [id_aug_dir, id_aug_vis_dir, fallback_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Check for valid background images
        if aug_case != 'mix' and not bg_img_paths:
            print(f"Warning: no images found in {aug_case} folder!")
            return
        
        # Image transforms (unified size, pre-compiled)
        resize_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ])
        pil_transform = transforms.ToPILImage()
        
        # Statistics counters
        total_processed = 0
        total_selected_aug = 0
        total_selected_fallback = 0
        total_skipped = 0
        
        # Core tracking variable
        if not hasattr(self, 'augmented_img_paths'):
            self.augmented_img_paths = set()
        
        # ========== Optimization 1: thread pool for image loading (parallelize IO) ==========
        # Batch image loading helper
        def load_image(img_path):
            try:
                return Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Failed to load image {img_path}: {e}")
                return None
        
        # Initialize thread pool (4-8 workers recommended based on CPU cores)
        executor = ThreadPoolExecutor(max_workers=8)
        
        # ========== Core change 3: preload bg/texture tensors (support mix mode) ==========
        # Preload background images as tensors (load once)
        print("Preloading background images...")
        bg_tensors = []     # bg folder tensors
        texture_tensors = []  # texture folder tensors
        
        # Preload bg tensors
        for bg_path in bg_paths:
            bg_img = load_image(bg_path)
            if bg_img is not None:
                bg_tensor = resize_transform(bg_img).to(device, dtype=dtype)
                bg_tensors.append(bg_tensor)
        
        # Preload texture tensors
        for texture_path in texture_paths:
            texture_img = load_image(texture_path)
            if texture_img is not None:
                texture_tensor = resize_transform(texture_img).to(device, dtype=dtype)
                texture_tensors.append(texture_tensor)
        
        # Validate preload results
        if aug_case == 'bg' and not bg_tensors:
            print("Warning: no valid images in bg folder!")
            return
        elif aug_case == 'texture' and not texture_tensors:
            print("Warning: no valid images in texture folder!")
            return
        elif aug_case == 'mix':
            if not bg_tensors or not texture_tensors:
                print(f"Warning: mix mode requires valid images in both bg and texture! "
                      f"bg: {len(bg_tensors)}, texture: {len(texture_tensors)}")
                return
        
        # Non-mix mode: assign directly
        if aug_case == 'bg':
            all_bg_tensors = bg_tensors
        elif aug_case == 'texture':
            all_bg_tensors = texture_tensors
        
        print(f"Background preload complete - bg: {len(bg_tensors)}, texture: {len(texture_tensors)}")
        
        # Keep outer loop with optimized exit logic
        for repeat in range(1):
            has_new_process = False
            total_samples = len(dataset)
            
            for batch_idx in range(0, total_samples, batch_size):
                batch_samples = dataset[batch_idx:batch_idx + batch_size]
                batch_img_paths = [item.impath for item in batch_samples]
                batch_labels = [item.label for item in batch_samples]
                batch_size_current = len(batch_img_paths)
                
                if batch_size_current == 0:
                    continue
                
                # Log progress every 10 batches (reduce logging overhead)
                if batch_idx % (batch_size * 10) == 0:
                    print(f"Outer loop round {repeat+1} | batch {batch_idx // batch_size + 1}/"
                          f"{(total_samples + batch_size - 1) // batch_size}")
                
                # ========== Optimization 3: batch filter already-processed images ==========
                # Collect unprocessed image paths and labels
                unprocessed_data = []
                for img_path, label in zip(batch_img_paths, batch_labels):
                    if img_path not in self.augmented_img_paths:
                        unprocessed_data.append((img_path, label))
                    else:
                        total_skipped += 1
                if not unprocessed_data:
                    continue
                
                # ========== Optimization 4: batch load originals + batch score inference ==========
                # Load originals in parallel
                img_paths = [d[0] for d in unprocessed_data]
                labels = [d[1] for d in unprocessed_data]
                future_to_img = {executor.submit(load_image, path): (path, label) for path, label in unprocessed_data}
                
                # Collect valid original image data
                valid_original_tensors = []
                valid_original_imgs = []
                valid_img_paths = []
                valid_labels = []
                valid_original_scores = []
                
                for future in future_to_img:
                    img = future.result()
                    img_path, label = future_to_img[future]
                    if img is None:
                        total_skipped += 1
                        continue
                    
                    # Preprocess original for batch inference
                    original_tensor = preprocess(img).to(device, dtype=dtype)
                    valid_original_tensors.append(original_tensor)
                    valid_original_imgs.append(img)
                    valid_img_paths.append(img_path)
                    valid_labels.append(label)
                
                if not valid_original_tensors:
                    continue
                
                # Batch infer original scores (replaces per-image inference)
                original_tensor_batch = torch.stack(valid_original_tensors)
                original_output, _ = self.zsmodel.forward(original_tensor_batch)
                original_labels_tensor = torch.tensor(valid_labels).to(device)
                original_scores = original_output[torch.arange(len(original_output)), original_labels_tensor].cpu().numpy()
                
                # ========== Optimization 5: batch generate augmentations + batch score inference ==========
                # Store all augmentation versions per image
                all_aug_data = []  # [[(aug_tensor, aug_img), ...], ...] per image, num_aug_versions each
                
                for idx, (img_path, original_img) in enumerate(zip(valid_img_paths, valid_original_imgs)):
                    # Load mask (once per image)
                    img_name = os.path.basename(img_path)
                    mask_name = os.path.splitext(img_name)[0] + '.png'
                    mask_path = os.path.join(mask_dir, mask_name)
                    
                    mask_img = load_image(mask_path)
                    if mask_img is None:
                        all_aug_data.append([])
                        total_skipped += 1
                        continue
                    
                    # Convert mask and original to tensors (once per image)
                    mask_img_resized = mask_img.resize((224, 224), Image.Resampling.NEAREST)
                    original_img_resized = original_img.resize((224, 224), Image.Resampling.BILINEAR)
                    
                    mask_tensor = resize_transform(mask_img_resized)[0:1, :, :].to(device, dtype=dtype)
                    original_resized_tensor = resize_transform(original_img_resized).to(device, dtype=dtype)
                    mask_binary = (mask_tensor > 0.5).float()
                    
                    # Generate num_aug_versions augmented images for current image
                    aug_tensors = []
                    aug_imgs = []
                    
                    # ========== Core change 4: equal-probability bg selection in mix mode ==========
                    if aug_case == 'mix':
                        selected_bg_tensors = []
                        # 50% bg, 50% texture per augmentation version
                        for _ in range(num_aug_versions):
                            # Randomly pick folder with equal probability
                            if random.random() < 0.5:
                                # Pick bg
                                if bg_tensors:
                                    selected_bg = random.choice(bg_tensors)
                                else:
                                    selected_bg = random.choice(texture_tensors)
                            else:
                                # Pick texture
                                if texture_tensors:
                                    selected_bg = random.choice(texture_tensors)
                                else:
                                    selected_bg = random.choice(bg_tensors)
                            selected_bg_tensors.append(selected_bg)
                    else:
                        # Non-mix mode: random background from current folder
                        selected_bg_tensors = random.choices(all_bg_tensors, k=num_aug_versions)
                    
                    for bg_tensor in selected_bg_tensors:
                        # Blend foreground and background (vectorized)
                        augmented_tensor = original_resized_tensor * mask_binary + bg_tensor * (1 - mask_binary)
                        aug_img = pil_transform(augmented_tensor.cpu())
                        
                        # Preprocess augmented image for inference
                        aug_infer_tensor = preprocess(aug_img).to(device, dtype=dtype)
                        aug_tensors.append(aug_infer_tensor)
                        aug_imgs.append(aug_img)
                    
                    all_aug_data.append(list(zip(aug_tensors, aug_imgs)))
                    total_processed += 1
                    has_new_process = True
                
                # ========== Optimization 6: batch infer all augmented image scores ==========
                # Flatten all aug tensors for batch inference
                flat_aug_tensors = []
                flat_indices = []  # Map each aug tensor to its source image index
                for img_idx, aug_versions in enumerate(all_aug_data):
                    for aug_tensor, _ in aug_versions:
                        flat_aug_tensors.append(aug_tensor)
                        flat_indices.append(img_idx)
                
                if not flat_aug_tensors:
                    continue
                
                # Batch infer augmented scores
                aug_tensor_batch = torch.stack(flat_aug_tensors)
                aug_output, _ = self.zsmodel.forward(aug_tensor_batch)
                
                # Split scores by source image index
                aug_scores_per_img = [[] for _ in range(len(all_aug_data))]
                for flat_idx, img_idx in enumerate(flat_indices):
                    label = valid_labels[img_idx]
                    aug_score = aug_output[flat_idx, label].cpu().item()
                    aug_scores_per_img[img_idx].append(aug_score)
                
                # ========== Save best result (one version per image) ==========
                threshold = self.id_aug_thre
                for img_idx in range(len(all_aug_data)):
                    if img_idx >= len(valid_img_paths):
                        continue
                    
                    img_path = valid_img_paths[img_idx]
                    original_score = original_scores[img_idx]
                    aug_versions = all_aug_data[img_idx]
                    aug_scores = aug_scores_per_img[img_idx]
                    
                    # No valid augmentations; mark as processed
                    if not aug_versions or not aug_scores:
                        self.augmented_img_paths.add(img_path)
                        total_skipped += 1
                        continue
                    
                    # Select highest-scoring augmentation
                    best_idx = aug_scores.index(max(aug_scores))
                    best_aug_score = aug_scores[best_idx]
                    best_aug_tensor, best_aug_img = aug_versions[best_idx]
                    score_diff = best_aug_score - original_score
                    
                    # Save logic
                    img_name = os.path.basename(img_path)
                    if score_diff > threshold:
                        # Save augmented image
                        save_path = os.path.join(id_aug_dir, img_name)
                        best_aug_img.save(save_path)
                        total_selected_aug += 1
                        
                        print(f"[Saved augmented] {img_name} | original: {original_score:.4f} | "
                              f"best aug: {best_aug_score:.4f} | gain: {score_diff:.4f}")
                        
                        # Concatenate comparison visualization
                        original_img_vis = valid_original_imgs[img_idx].resize((224, 224), Image.Resampling.BILINEAR)
                        vis_concat_img = Image.new('RGB', (224, 448))
                        vis_concat_img.paste(original_img_vis, (0, 0))
                        vis_concat_img.paste(best_aug_img, (0, 224))
                        # Save visualization
                        vis_save_path = os.path.join(id_aug_vis_dir, img_name)
                        vis_concat_img.save(vis_save_path)
                    else:
                        # Save original to fallback path
                        save_path = os.path.join(fallback_dir, img_name)
                        valid_original_imgs[img_idx].save(save_path)
                        total_selected_fallback += 1
                        
                        print(f"[Saved original (below threshold)] {img_name} | original: {original_score:.4f} | "
                              f"best aug: {best_aug_score:.4f} | gain: {score_diff:.4f} (threshold: {threshold:.4f})")
                    
                    # Mark as processed
                    self.augmented_img_paths.add(img_path)
            
            # Outer loop exit condition
            if not has_new_process:
                print(f"Outer loop round {repeat+1}: no new images processed, exiting early")
                break
        
        # Shut down thread pool
        executor.shutdown()
        
        # Final statistics
        total_increment = total_selected_aug + total_selected_fallback
        print("\n" + "="*80)
        print(f"Done! Total generated: {total_increment} (accelerated, aug_case={aug_case})")
        print(f"  1. Originals processed: {total_processed}")
        print(f"  2. Skipped: {total_skipped}")
        print(f"  3. Augmented images saved: {total_selected_aug} -> {id_aug_dir}")
        print(f"  4. Fallback originals saved: {total_selected_fallback} -> {fallback_dir}")
        print(f"  5. Total processed originals: {len(self.augmented_img_paths)}")
        print("="*80 + "\n")
        
    @torch.no_grad()
    def ood_loader(self):
        self.inpainted_dir = './data/' + self.train_root + '/score_inpaint/'
        self.texture_dir = './data/' + self.train_root + '/texture/inpaint/top/'
        trainset = self.dm.train_loader_x.dataset
        
        tmp_data_source = []
        raw_data_source = []
        files = []
        self.id2clsname = {}
        for item in trainset.data_source:
            self.id2clsname[item.impath.split('/')[-1].split('_')[0]] = [item.label, item.classname]
        # load background data 
        for filename in os.listdir(self.inpainted_dir):
            if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.JPEG'):
                impath = os.path.join(self.inpainted_dir, filename)
                scores = float(os.path.splitext(filename)[0].split('_')[2])
                if scores > self.inpaint_thre:
                    classname = 'background'
                    label = self.num_classes
                    tmp_data_source.append(Datum(impath=impath, label=label, classname=classname, domain=2)) # Domain 2: Background
                    files.append(filename.split('_')[0] + '_' + filename.split('_')[1])
        # load OOD Aug data 
        if self.ood_aug:
            image_folder = self.texture_dir
            for filename in os.listdir(image_folder):
                if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.JPEG'):
                    impath = os.path.join(image_folder, filename)
                    classname = 'background'
                    label = self.num_classes
                    tmp_data_source.append(Datum(impath=impath, label=label, classname=classname, domain=2))
        # load few-shot ID data 
        image_folder_raw = './data/' + self.train_root + '/raw/'
        for filename in os.listdir(image_folder_raw):
            if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.JPEG'):
                impath = os.path.join(image_folder_raw, filename)
                imageid = filename.split('_')[0]
                classname = self.id2clsname[imageid][1]
                label = self.id2clsname[imageid][0]
                
                if filename.split('.')[0] in files:
                    tmp_data_source.append(Datum(impath=impath, label=label, classname=classname, domain=1)) # Domain 1: ID data with mask
                    raw_data_source.append(Datum(impath=impath, label=label, classname=classname, domain=1)) 
                else:
                    tmp_data_source.append(Datum(impath=impath, label=label, classname=classname, domain=0)) # Domain 0: ID data without mask

        # load background data 
        for filename in os.listdir(self.inpainted_dir):
            if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.JPEG'):
                impath = os.path.join(self.inpainted_dir, filename)
                scores = float(os.path.splitext(filename)[0].split('_')[2])
                if scores > self.inpaint_thre:
                    classname = 'background'
                    label = self.num_classes
                    tmp_data_source.append(Datum(impath=impath, label=label, classname=classname, domain=2)) # Domain 2: Background
                    files.append(filename.split('_')[0] + '_' + filename.split('_')[1])
        # load OOD Aug data 
        if self.ood_aug:
            image_folder = self.texture_dir
            for filename in os.listdir(image_folder):
                if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.JPEG'):
                    impath = os.path.join(image_folder, filename)
                    classname = 'background'
                    label = self.num_classes
                    tmp_data_source.append(Datum(impath=impath, label=label, classname=classname, domain=2))
                    
                                        
        # load id_aug data
        id_aug_dir = os.path.join(self.output_dir, "id_aug_all")
        copy_images_with_overwrite(image_folder_raw,id_aug_dir)
        copy_images_with_overwrite(os.path.join(self.output_dir, "id_aug"),id_aug_dir)
   
        try:
            image_folder_raw = id_aug_dir
            for filename in os.listdir(image_folder_raw):
                if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.JPEG'):
                    impath = os.path.join(image_folder_raw, filename)
                    imageid = filename.split('_')[0]
                    classname = self.id2clsname[imageid][1]
                    label = self.id2clsname[imageid][0]
                    
                    if filename.split('.')[0] in files:
                        tmp_data_source.append(Datum(impath=impath, label=label, classname=classname, domain=1)) # Domain 1: ID data with mask
                    else:
                        tmp_data_source.append(Datum(impath=impath, label=label, classname=classname, domain=0)) # Domain 0: ID data without mask
        except:
            pass   
            
        self.raw_data_source = raw_data_source
        trainset.data_source = tmp_data_source
        tfm_train = build_transform(self.cfg, is_train=True)
        self.train_loader_x = build_data_loader(
            self.cfg,
            sampler_type=self.cfg.DATALOADER.TRAIN_X.SAMPLER,
            data_source=trainset,
            batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            n_domain=self.cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=self.cfg.DATALOADER.TRAIN_X.N_INS,
            tfm=tfm_train,
            is_train=True,
        )

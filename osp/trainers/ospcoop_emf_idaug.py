import os
import os.path as osp
from datetime import datetime
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from collections import OrderedDict
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
import json
from clip_w_local import clip
from clip_w_local.simple_tokenizer import SimpleTokenizer as _Tokenizer
import numpy as np
from tqdm import tqdm
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count
from PIL import Image
from torchvision import datasets
from utils.detection_util import get_and_print_results
_tokenizer = _Tokenizer()
from dassl.data.datasets import build_dataset
from dassl.data.samplers import build_sampler
from dassl.data.transforms import build_transform
from dassl.data.data_manager import build_data_loader
softmax = nn.Softmax(dim=1).cuda()
import random
from scipy.spatial.distance import cdist
import shutil
from .custom_clip import load_clip_to_cpu, TextEncoder, PromptLearner, CustomCLIP, CustomZSCLIP
from PIL import Image, ImageDraw
import cv2
import shutil
import numpy as np
import os
import torch
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import random
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor  # 引入线程池加速IO
import warnings
warnings.filterwarnings('ignore')

def copy_images_with_overwrite(folder_a, folder_b, image_extensions=None):
    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']

    os.makedirs(folder_b, exist_ok=True)
    
    if not os.path.exists(folder_a):
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
    pass
class CustomRandomResizedCrop:
    def __init__(self, size=(224,224), scale=(0.08, 1.0)):
        self.size = size
        self.scale = scale

    def __call__(self, img,mask):
        area = img.size[0] * img.size[1]
        target_area = torch.randint(int(self.scale[0] * area), int(self.scale[1] * area) + 1, (1,)).item()

        aspect_ratio = 1
        w = int(round((target_area * aspect_ratio) ** 0.5))
        h = int(round((target_area / aspect_ratio) ** 0.5))

        return transforms.functional.resized_crop(img, 0, 0, h, w, self.size),transforms.functional.resized_crop(mask, 0, 0, h, w, self.size)


class CustomRandomFlip:
    def __init__(self, horizontal=True, vertical=False, p=0.5):
        self.horizontal = horizontal
        self.vertical = vertical
        self.p = p

    def __call__(self, img,mask):
        flip_h = False
        flip_v = False

        if self.horizontal and random.random() < self.p:
            img = transforms.functional.hflip(img)
            mask = transforms.functional.hflip(mask)
            flip_h = True

        if self.vertical and random.random() < self.p:
            img = transforms.functional.vflip(img)
            flip_v = True
            mask = transforms.functional.hflip(mask)

        return img,mask



class DataManager:

    def __init__(
        self,
        cfg,
        custom_tfm_train=None,
        custom_tfm_test=None,
        dataset_wrapper=None
    ):
        # Load dataset
        dataset = build_dataset(cfg)
        # Build transform
        if custom_tfm_train is None:
            tfm_train = build_transform(cfg, is_train=True)
        else:
            print("* Using custom transform for training")
            tfm_train = custom_tfm_train

        if custom_tfm_test is None:
            tfm_test = build_transform(cfg, is_train=False)
        else:
            print("* Using custom transform for testing")
            tfm_test = custom_tfm_test

        # Build train_loader_x
        train_loader_x = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
            data_source=dataset.train_x,
            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
            tfm=tfm_train,
            is_train=True,
            dataset_wrapper=dataset_wrapper,
        )

        # Build train_loader_u
        train_loader_u = None
        if dataset.train_u:
            sampler_type_ = cfg.DATALOADER.TRAIN_U.SAMPLER
            batch_size_ = cfg.DATALOADER.TRAIN_U.BATCH_SIZE
            n_domain_ = cfg.DATALOADER.TRAIN_U.N_DOMAIN
            n_ins_ = cfg.DATALOADER.TRAIN_U.N_INS

            if cfg.DATALOADER.TRAIN_U.SAME_AS_X:
                sampler_type_ = cfg.DATALOADER.TRAIN_X.SAMPLER
                batch_size_ = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
                n_domain_ = cfg.DATALOADER.TRAIN_X.N_DOMAIN
                n_ins_ = cfg.DATALOADER.TRAIN_X.N_INS

            train_loader_u = build_data_loader(
                cfg,
                sampler_type=sampler_type_,
                data_source=dataset.train_u,
                batch_size=batch_size_,
                n_domain=n_domain_,
                n_ins=n_ins_,
                tfm=tfm_train,
                is_train=True,
                dataset_wrapper=dataset_wrapper,
            )

        # Build val_loader
        val_loader = None
        if dataset.val:
            val_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=dataset.val,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper,
            )

        # Build test_loader
        test_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=dataset.test,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=dataset_wrapper,
        )

        # Attributes
        self._num_classes = dataset.num_classes
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)
        self._lab2cname = dataset.lab2cname

        # Dataset and data-loaders
        self.dataset = dataset
        self.train_loader_x = train_loader_x
        self.train_loader_u = train_loader_u
        self.val_loader = val_loader
        self.test_loader = test_loader

        if cfg.VERBOSE:
            self.show_dataset_summary(cfg)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_source_domains(self):
        return self._num_source_domains

    @property
    def lab2cname(self):
        return self._lab2cname

    def show_dataset_summary(self, cfg):
        print("***** Dataset statistics *****")

        print("  Dataset: {}".format(cfg.DATASET.NAME))

        if cfg.DATASET.SOURCE_DOMAINS:
            print("  Source domains: {}".format(cfg.DATASET.SOURCE_DOMAINS))
        if cfg.DATASET.TARGET_DOMAINS:
            print("  Target domains: {}".format(cfg.DATASET.TARGET_DOMAINS))

        print("  # classes: {:,}".format(self.num_classes))

        print("  # train_x: {:,}".format(len(self.dataset.train_x)))

        if self.dataset.train_u:
            print("  # train_u: {:,}".format(len(self.dataset.train_u)))

        if self.dataset.val:
            print("  # val: {:,}".format(len(self.dataset.val)))

        print("  # test: {:,}".format(len(self.dataset.test)))
from dassl.data.samplers import build_sampler
def build_data_loader(
    cfg,
    sampler_type="SequentialSampler",
    data_source=None,
    batch_size=64,
    n_domain=0,
    n_ins=2,
    tfm=None,
    is_train=True,
    dataset_wrapper=None,
):
    # Build sampler
    sampler = build_sampler(
        sampler_type,
        cfg=cfg,
        data_source=data_source,
        batch_size=batch_size,
        n_domain=n_domain,
        n_ins=n_ins,
    )

    if dataset_wrapper is None:
        dataset_wrapper = DatasetWrapper

    # Build data loader  num_workers=cfg.DATALOADER.NUM_WORKERS,
    data_loader = torch.utils.data.DataLoader(
        dataset_wrapper(cfg, data_source, transform=tfm, is_train=is_train),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=is_train and len(data_source) >= batch_size,
        pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA),
    )
    assert len(data_loader) > 0

    return data_loader
from torch.utils.data import Dataset as TorchDataset
INTERPOLATION_MODES = {
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "nearest": Image.NEAREST,
}
import torchvision.transforms as T
from dassl.utils import read_image
class DatasetWrapper(TorchDataset):

    def __init__(self, cfg, data_source, transform=None, is_train=False):
        self.cfg = cfg
        self.train_root = self.cfg.train_root
        self.output_dir = self.cfg.OUTPUT_DIR
        self.data_source = data_source
        self.transform = transform  # accept list (tuple) as input
        self.is_train = is_train
        self.topk = cfg.topk
        # Augmenting an image K>1 times is only allowed during training
        self.k_tfm = cfg.DATALOADER.K_TRANSFORMS if is_train else 1
        self.return_img0 = cfg.DATALOADER.RETURN_IMG0

        if self.k_tfm > 1 and transform is None:
            raise ValueError(
                "Cannot augment the image {} times "
                "because transform is None".format(self.k_tfm)
            )

        # Build transform that doesn't apply any data augmentation
        interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
        to_tensor = []
        to_tensor += [T.Resize(cfg.INPUT.SIZE, interpolation=interp_mode)]
        to_tensor += [T.ToTensor()]
        if "normalize" in cfg.INPUT.TRANSFORMS:
            normalize = T.Normalize(
                mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
            )
            to_tensor += [normalize]

        self.RandomResizedCrop = CustomRandomResizedCrop()
        self.RandomFlip = CustomRandomFlip()
        self.transform_img = transforms.Compose([
            transforms.ToTensor(),  # 将PIL图像转换为Tensor
            transforms.Normalize(mean= [0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),  # 标准化
        ])
        self.to_tensor = T.Compose(to_tensor)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        if item['domain'] == 1:
            name = item['impath'].split('/')[-1].split('.')[0]
            mask_root = self.output_dir+ '/gen_mask_'+str(self.topk)+'/'+name.split('_')[0]+'_'+name.split('_')[1]+'.png'
            mask_img = Image.open(mask_root)
            mask_img = np.array(mask_img)
            item['mask'] = mask_img
        if item['domain'] == 0:
            item['mask'] =  np.full((224, 224), 255, dtype=np.uint8)
        if item['domain'] == 2:
            item['mask'] =  np.full((224, 224), 0, dtype=np.uint8)

        img0 = read_image(item['impath'])
        mask = item['mask']

        mask1 = np.stack((mask,) * 3, axis=-1)
        mask1 = Image.fromarray(mask1)


        mask1 = np.array(mask1)[:,:,0]

        mask1 = np.where(mask1 > 50, 255, 0)

        img2 = self.transform_img(img0)
        item['img'] = img2
        item['mask'] = mask1
        

        return item


    def _transform_image(self, tfm, img0):
        img_list = []

        for k in range(self.k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img

def entropy_select_topk(p, top_k, label, num_of_local_feature):
    """
    Extract non-Top-K regions and calculate entropy.
    """
    label_repeat = label.repeat_interleave(num_of_local_feature)
    p = F.softmax(p, dim=-1)
    pred_topk = torch.topk(p, k=top_k, dim=1)[1]
    contains_label = pred_topk.eq(torch.tensor(label_repeat).unsqueeze(1)).any(dim=1)
    selected_p = p[~contains_label]

    if selected_p.shape[0] == 0:
        return torch.tensor([0]).cuda()
    return -torch.mean(torch.sum(selected_p * torch.log(selected_p+1e-5), 1))


@TRAINER_REGISTRY.register()
class OSPCoOp_EMF(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.OSPCOOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        cfg.num_classes = len(classnames)
        self.DATA_ROOT = cfg.DATA_ROOT[0]
        self.num_classes = len(classnames)
        self.train_root = self.cfg.train_root
        self.loss1 = cfg.loss1
        self.loss2 = cfg.loss2
        self.glmcm_local_weight = cfg.glmcm_local_weight
        self.T = cfg.T
        self.eval_bg = cfg.eval_bg
        self.ood_aug = cfg.ood_aug
        self.id_aug_dir = cfg.id_aug_dir
        self.topk = cfg.topk
        self.eval_freq = cfg.eval_freq
        self.start_eval_epoch = cfg.start_eval_epoch

        
        self.crop_num_per_img = cfg.crop_num_per_img
        self.num_aug_versions = cfg.num_aug_versions
        self.id_aug_thre = cfg.id_aug_thre
        
        
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
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        try:
            topk = self.topk
            save_dir = './data/' + self.train_root
            shutil.rmtree(os.path.join(save_dir, "ood_select_"+str(topk)))
            shutil.rmtree(os.path.join(save_dir, "vis_"+str(topk)))
            shutil.rmtree('./data/' + self.train_root+ '/id_aug_'+str(topk)) 
        except:
            pass
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

        loss_en = - entropy_select_topk(output_local, self.topk, label, num_of_local_feature)

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
        self.ood_loader()
        self.gen_mask()
        self.gen_ood(crop_num_per_img= self.crop_num_per_img)
        self.gen_idaug(num_aug_versions = self.num_aug_versions,id_aug_thre = self.id_aug_thre)
        self.ood_loader()

        for self.epoch in range(self.start_epoch, self.max_epoch):
            
            self.before_epoch()
            self.run_epoch()
            if (self.epoch + 1) > self.start_eval_epoch:

                if (self.epoch + 1) % self.eval_freq == 0:
                    mcm_aurod, glmcm_auroc = self.eval_ood()
                    self.after_epoch(glmcm_auroc)
                    pass
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
        glmcm_score_entropy = []
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
    def gen_idaug(self, batch_size=16, num_aug_versions=64, id_aug_thre=5.0):
        """
        Args:
            batch_size: Batch processing size (default 16)
            num_aug_versions: Number of augmented versions per original image (default 64, keep best)
            id_aug_thre: Threshold for augmented score exceeding original score (default 5.0)
        """
        self.set_model_mode("eval")
        self.evaluator.reset()
        dataset = self.raw_data_source
        topk = self.topk  # Keep original topk parameter
        
        # ========== 1. Device setup ==========
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            torch.backends.cudnn.benchmark = True  # Enable cuDNN benchmark optimization
            dtype = torch.float16  # Half-precision inference for speed
        else:
            dtype = torch.float32
        
        # Load CLIP model for preprocessing
        _, preprocess = clip.load("ViT-B/16", device=device)
        
        # ========== 2. Path setup ==========
        save_dir = self.output_dir
        mask_dir = os.path.join(save_dir, "gen_mask_"+str(topk))  # Original mask directory
        bg_dir = os.path.join(save_dir, "ood_select_"+str(topk))  # Original background image directory
        id_aug_dir = os.path.join(save_dir, "id_aug_"+str(topk))  # Augmented image save directory
        id_aug_vis_dir = os.path.join(save_dir, "id_aug_vis_"+str(topk))  # Visualization directory
        fallback_dir = os.path.join(save_dir, "id_aug_fallback_"+str(topk))  # Fallback directory when threshold not met
        
        # Create directories
        for dir_path in [id_aug_dir, id_aug_vis_dir, fallback_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # ========== 3. Load background images ==========
        bg_img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
        bg_img_paths = [
            os.path.join(bg_dir, f) for f in os.listdir(bg_dir)
            if f.lower().endswith(bg_img_extensions)
        ]
        
        if not bg_img_paths:
            print(f"Warning: No images found in background folder {bg_dir}!")
            return
        
        # ========== 4. Thread pool + preload background tensors ==========
        def load_image(img_path):
            try:
                return Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Failed to load image {img_path}: {e}")
                return None
        
        # Initialize thread pool (adjust max_workers to CPU cores)
        executor = ThreadPoolExecutor(max_workers=8)
        
        # Preload background images as tensors
        print("Preloading background images...")
        futures = [executor.submit(load_image, path) for path in bg_img_paths]
        bg_tensors = []
        for future, path in zip(futures, bg_img_paths):
            bg_img = future.result()
            if bg_img is not None:
                resize_transform = transforms.Compose([
                    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.ToTensor()
                ])
                bg_tensor = resize_transform(bg_img).to(device, dtype=dtype)
                bg_tensors.append(bg_tensor)
        
        if not bg_tensors:
            print(f"Warning: No valid images in background folder {bg_dir}!")
            executor.shutdown()
            return
        print(f"Background preload complete - {len(bg_tensors)} images")
        
        # ========== 5. Core tracking variables ==========
        total_processed = 0
        total_selected_aug = 0  # Count of augmented images meeting threshold
        total_selected_fallback = 0  # Count of originals saved due to insufficient threshold
        total_skipped = 0  # Count of skipped/processed-or-failed images
        
        # Track processed image paths only; no class/total count limits
        if not hasattr(self, 'augmented_img_paths'):
            self.augmented_img_paths = set()
        
        # ========== 6. Image transforms ==========
        resize_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ])
        pil_transform = transforms.ToPILImage()
        
        # ========== 7. Outer loop ==========
        for repeat in range(1):
            has_new_process = False
            total_samples = len(dataset)
            
            for batch_idx in range(0, total_samples, batch_size):
                # Extract current batch samples
                batch_samples = dataset[batch_idx:batch_idx + batch_size]
                batch_img_paths = [item.impath for item in batch_samples]
                batch_labels = [item.label for item in batch_samples]
                batch_size_current = len(batch_img_paths)
                
                if batch_size_current == 0:
                    continue
                
                # Print progress every 10 batches only (reduce log overhead)
                if batch_idx % (batch_size * 10) == 0:
                    print(f"Processing batch {batch_idx // batch_size + 1}/{(total_samples + batch_size - 1) // batch_size}")
                
                # ========== 8. Batch filter already-processed images ==========
                unprocessed_data = []
                for img_path, label in zip(batch_img_paths, batch_labels):
                    if img_path not in self.augmented_img_paths:
                        unprocessed_data.append((img_path, label))
                    else:
                        total_skipped += 1
                
                if not unprocessed_data:
                    continue
                has_new_process = True
                
                # ========== 9. Batch load originals + batch inference scores ==========
                img_paths = [d[0] for d in unprocessed_data]
                labels = [d[1] for d in unprocessed_data]
                
                # Load originals in parallel
                future_to_img = {executor.submit(load_image, path): (path, label) for path, label in unprocessed_data}
                valid_original_tensors = []
                valid_original_imgs = []
                valid_img_paths = []
                valid_labels = []
                
                for future in future_to_img:
                    img = future.result()
                    img_path, label = future_to_img[future]
                    if img is None:
                        total_skipped += 1
                        continue
                    
                    # Preprocess original image
                    original_tensor = preprocess(img).to(device, dtype=dtype)
                    valid_original_tensors.append(original_tensor)
                    valid_original_imgs.append(img)
                    valid_img_paths.append(img_path)
                    valid_labels.append(label)
                
                if not valid_original_tensors:
                    continue
                
                # Batch inference on original scores
                original_tensor_batch = torch.stack(valid_original_tensors)
                original_output, _ = self.zsmodel.forward(original_tensor_batch)
                original_labels_tensor = torch.tensor(valid_labels).to(device)
                original_scores = original_output[torch.arange(len(original_output)), original_labels_tensor].cpu().numpy()
                
                # ========== 10. Batch generate multiple augmented versions ==========
                all_aug_data = []  # Store num_aug_versions augmented versions per image
                
                for idx, (img_path, original_img) in enumerate(zip(valid_img_paths, valid_original_imgs)):
                    # Load mask
                    img_name = os.path.basename(img_path)
                    mask_name = os.path.splitext(img_name)[0] + '.png'
                    mask_path = os.path.join(mask_dir, mask_name)
                    
                    mask_img = load_image(mask_path)
                    if mask_img is None:
                        all_aug_data.append([])
                        total_skipped += 1
                        continue
                    
                    # Resize to uniform size and convert to tensor
                    mask_img_resized = mask_img.resize((224, 224), Image.Resampling.NEAREST)
                    original_img_resized = original_img.resize((224, 224), Image.Resampling.BILINEAR)
                    
                    mask_tensor = resize_transform(mask_img_resized)[0:1, :, :].to(device, dtype=dtype)
                    original_resized_tensor = resize_transform(original_img_resized).to(device, dtype=dtype)
                    mask_binary = (mask_tensor > 0.5).float()
                    
                    # Generate num_aug_versions augmented images
                    aug_tensors = []
                    aug_imgs = []
                    # Randomly select num_aug_versions background images
                    selected_bg_tensors = random.choices(bg_tensors, k=num_aug_versions)
                    
                    for bg_tensor in selected_bg_tensors:
                        # Blend foreground and background
                        augmented_tensor = original_resized_tensor * mask_binary + bg_tensor * (1 - mask_binary)
                        aug_img = pil_transform(augmented_tensor.cpu())
                        
                        # Preprocess for inference
                        aug_infer_tensor = preprocess(aug_img).to(device, dtype=dtype)
                        aug_tensors.append(aug_infer_tensor)
                        aug_imgs.append(aug_img)
                    
                    all_aug_data.append(list(zip(aug_tensors, aug_imgs)))
                    total_processed += 1
                
                # ========== 11. Batch inference on all augmented scores ==========
                flat_aug_tensors = []
                flat_indices = []  # Map each augmented tensor to its original image index
                
                for img_idx, aug_versions in enumerate(all_aug_data):
                    for aug_tensor, _ in aug_versions:
                        flat_aug_tensors.append(aug_tensor)
                        flat_indices.append(img_idx)
                
                if not flat_aug_tensors:
                    continue
                
                # Batch inference on all augmented scores
                aug_tensor_batch = torch.stack(flat_aug_tensors)
                aug_output, _ = self.zsmodel.forward(aug_tensor_batch)
                
                # Split scores by original image index
                aug_scores_per_img = [[] for _ in range(len(all_aug_data))]
                for flat_idx, img_idx in enumerate(flat_indices):
                    label = valid_labels[img_idx]
                    aug_score = aug_output[flat_idx, label].cpu().item()
                    aug_scores_per_img[img_idx].append(aug_score)
                
                # ========== 12. Save best candidates ==========
                for img_idx in range(len(all_aug_data)):
                    if img_idx >= len(valid_img_paths):
                        break
                    
                    img_path = valid_img_paths[img_idx]
                    original_score = original_scores[img_idx]
                    aug_versions = all_aug_data[img_idx]
                    aug_scores = aug_scores_per_img[img_idx]
                    
                    # No valid augmented images; mark as processed
                    if not aug_versions or not aug_scores:
                        self.augmented_img_paths.add(img_path)
                        total_skipped += 1
                        continue
                    
                    # Select highest-scoring augmented version
                    best_idx = aug_scores.index(max(aug_scores))
                    best_aug_score = aug_scores[best_idx]
                    best_aug_img = aug_versions[best_idx][1]
                    score_diff = best_aug_score - original_score
                    img_name = os.path.basename(img_path)
                    
                    # ========== Threshold-only selection + single pass per image ==========
                    if score_diff > id_aug_thre:
                        # Save augmented image
                        try:
                            aug_save_path = os.path.join(id_aug_dir, img_name)
                            best_aug_img.save(aug_save_path)
                            
                            # Concatenate visualization image
                            original_img_vis = valid_original_imgs[img_idx].resize((224, 224), Image.Resampling.BILINEAR)
                            concat_img = Image.new('RGB', (224, 448))
                            concat_img.paste(original_img_vis, (0, 0))
                            concat_img.paste(best_aug_img, (0, 224))
                            vis_save_path = os.path.join(id_aug_vis_dir, img_name)
                            concat_img.save(vis_save_path)
                            
                            print(f"[Saved augmented image] {img_name} | original score: {original_score:.4f} | augmented score: {best_aug_score:.4f} | score gain: {score_diff:.4f}")
                            total_selected_aug += 1
                        except Exception as e:
                            print(f"Failed to save augmented image {img_path}: {e}")
                    else:
                        # Threshold not met; save original to fallback path
                        try:
                            fallback_save_path = os.path.join(fallback_dir, img_name)
                            valid_original_imgs[img_idx].save(fallback_save_path)
                            print(f"[Saved original (threshold not met)] {img_name} | original score: {original_score:.4f} | best augmented score: {best_aug_score:.4f} | score gain: {score_diff:.4f} (threshold: {id_aug_thre:.4f})")
                            total_selected_fallback += 1
                        except Exception as e:
                            print(f"Failed to save original image {img_path}: {e}")
                    
                    # Mark as processed to avoid reprocessing
                    self.augmented_img_paths.add(img_path)
            
            # Outer loop exit condition
            if not has_new_process:
                print(f"No new images processed this round, exiting early")
                break
        
        # ========== 13. Shutdown thread pool + final stats ==========
        executor.shutdown()
        
        total_increment = total_selected_aug + total_selected_fallback
        print("\n" + "="*80)
        print(f"Processing complete! [Total generated images: {total_increment}] (threshold-only version)")
        print(f"  1. Total originals processed: {total_processed}")
        print(f"  2. Skipped images: {total_skipped}")
        print(f"  3. Augmented images generated: {total_selected_aug} -> {id_aug_dir}")
        print(f"  4. Original images saved: {total_selected_fallback} -> {fallback_dir}")
        print(f"  5. Total originals processed: {len(self.augmented_img_paths)}")
        print(f"  6. Visualization path: {id_aug_vis_dir}")
        print("="*80 + "\n")
    
    @torch.no_grad()
    def gen_ood(self, batch_size=64, crop_num_per_img=32):

        self.set_model_mode("eval")
        self.evaluator.reset()
        dataset = self.raw_data_source
        topk = self.topk
        # Basic setup
        save_dir = self.output_dir
        ood_select_dir = os.path.join(save_dir, "ood_select_"+str(topk))
        vis_dir = os.path.join(save_dir, "vis_"+str(topk))
        
        # Keep only required folders (crops no longer saved to disk; remove related folder ops)
        try:
            shutil.rmtree('./data/' + self.train_root+ '/id_aug_'+str(topk)) 
        except:
            pass
    
        # Create directories
        for dir_path in [ood_select_dir, vis_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Crop inference batch size (increased to 64 for better GPU utilization)
        crop_infer_batch_size = 64
                    
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Preload CLIP model (comment out if unused to reduce memory)
        _, preprocess = clip.load("ViT-B/16", device=device, jit=False)  # jit=False for faster loading
        
        # Global storage: score info for all selected images
        selected_scores_info = []
        # Thread pool: accelerate image loading (CPU-bound IO)
        executor = ThreadPoolExecutor(max_workers=8)  # Adjust to CPU core count
        
        # Split dataset into batches
        total_samples = len(dataset)
        for batch_idx in range(0, total_samples, batch_size):
            # Extract original image samples for current batch
            batch_samples = dataset[batch_idx:batch_idx + batch_size]
            batch_img_paths = [item.impath for item in batch_samples]
            batch_labels = [item.label for item in batch_samples]
            batch_size_current = len(batch_img_paths)
    
            # -------------------------- 1. Batch load originals + masks (thread pool) --------------------------
            # Define load function
            def load_img_mask(img_path, mask_ori_dir):
                try:
                    # Load original image
                    raw_img = Image.open(img_path).convert("RGB")
                    img_name = os.path.basename(img_path)
                    mask_name = os.path.splitext(img_name)[0] + '.png'
                    mask_ori_path = os.path.join(mask_ori_dir, mask_name)
                    
                    # Load original mask
                    if os.path.exists(mask_ori_path):
                        raw_mask = Image.open(mask_ori_path).convert("L")
                        raw_mask_np = np.array(raw_mask)
                        ori_255_count = np.sum(raw_mask_np == 255)
                        if ori_255_count == 0:
                            ori_255_count = 1
                    else:
                        return None
                    
                    return {
                        "img_path": img_path,
                        "raw_img": raw_img,
                        "raw_mask": raw_mask,
                        "ori_255_count": ori_255_count,
                        "label": batch_labels[batch_img_paths.index(img_path)],
                        "mask_ori_path": mask_ori_path
                    }
                except Exception as e:
                    return None
            
            # Batch load (thread pool)
            mask_ori_dir = os.path.join(save_dir, "gen_mask_"+str(topk))
            futures = [executor.submit(load_img_mask, path, mask_ori_dir) for path in batch_img_paths]
            img_mask_data = [f.result() for f in futures if f.result() is not None]
            if not img_mask_data:
                continue
    
            # -------------------------- 2. Generate crops in memory (no disk write) --------------------------
            # Define crop transform (reuse parameter generator)
            crop_transform = transforms.RandomResizedCrop(
                size=224,
                scale=(0.08, 1.0),
                ratio=(0.7, 1.5),
                interpolation=transforms.InterpolationMode.BILINEAR
            )
            
            # Storage: {original path: [(crop tensor, mask_ratio, crop mask tensor), ...]}
            original_to_crops = {}
            # Preallocate tensor memory (reduce dynamic allocation)
            crop_tensors_list = []
            crop_metas = []  # Metadata: (original path, mask_ratio, label)
            
            for data in img_mask_data:
                img_path = data["img_path"]
                raw_img = data["raw_img"]
                raw_mask = data["raw_mask"]
                ori_255_count = data["ori_255_count"]
                label = data["label"]
                
                # Convert to tensor
                raw_img_tensor = transforms.ToTensor()(raw_img).to(device, non_blocking=True)
                raw_mask_tensor = transforms.ToTensor()(raw_mask).to(device, non_blocking=True)
                
                # Generate specified number of crops (in memory)
                original_to_crops[img_path] = []
                for crop_idx in range(crop_num_per_img):
                    # Generate crop params once and reuse
                    i, j, h, w = transforms.RandomResizedCrop.get_params(
                        raw_img_tensor, scale=(0.08, 1.0), ratio=(0.7, 1.5)
                    )
                    
                    # Crop + resize image (on GPU)
                    crop_img_tensor = transforms.functional.resized_crop(
                        raw_img_tensor, i, j, h, w, (224, 224), 
                        interpolation=transforms.InterpolationMode.BILINEAR
                    )
                    # Preprocess (CLIP)
                    crop_img_clip = preprocess(transforms.ToPILImage()(crop_img_tensor.cpu())).to(device, non_blocking=True)
                    
                    # Crop + resize mask
                    crop_mask_tensor = transforms.functional.resized_crop(
                        raw_mask_tensor, i, j, h, w, (224, 224), 
                        interpolation=transforms.InterpolationMode.NEAREST
                    )
                    # Compute mask ratio (vectorized)
                    crop_mask_np = crop_mask_tensor.cpu().numpy().squeeze()
                    mask_ratio = np.sum(crop_mask_np > 0.5) / ori_255_count
                    
                    # Store
                    crop_tensors_list.append(crop_img_clip)
                    crop_metas.append((img_path, mask_ratio, label))
                    original_to_crops[img_path].append((crop_img_tensor, mask_ratio, crop_mask_tensor, crop_idx))
    
            if not crop_tensors_list:
                continue
            
            # -------------------------- 3. Batch inference (maximize GPU utilization) --------------------------
            # 3.1 Batch inference on crops
            crop_tensors = torch.stack(crop_tensors_list)
            total_crop_imgs = len(crop_tensors)
            crop_scores = torch.zeros(total_crop_imgs, device=device)
            
            # Large-batch inference (fewer kernel launches)
            for start in range(0, total_crop_imgs, crop_infer_batch_size):
                end = min(start + crop_infer_batch_size, total_crop_imgs)
                batch = crop_tensors[start:end]
                labels = torch.tensor([crop_metas[i][2] for i in range(start, end)], device=device)
                
                # Inference
                output, _ = self.zsmodel.forward(batch)
                crop_scores[start:end] = output[torch.arange(end-start), labels]
            
            # Convert to numpy (once)
            crop_scores_np = crop_scores.cpu().numpy()
            
            # 3.2 Batch inference on originals (dedup + vectorized)
            unique_original_paths = list(original_to_crops.keys())
            original_tensors = []
            for path in unique_original_paths:
                data = next(d for d in img_mask_data if d["img_path"] == path)
                img = preprocess(data["raw_img"]).to(device, non_blocking=True)
                original_tensors.append(img)
            
            original_tensors = torch.stack(original_tensors)
            original_labels = torch.tensor([d["label"] for d in img_mask_data if d["img_path"] in unique_original_paths], device=device)
            original_output, _ = self.zsmodel.forward(original_tensors)
            original_scores = original_output[torch.arange(len(original_output)), original_labels].cpu().numpy()
            original_path_to_score = dict(zip(unique_original_paths, original_scores))
    
            # -------------------------- 4. Select lowest-scoring crop per original (vectorized) --------------------------
            # Build group indices
            original_indices = []
            for meta in crop_metas:
                original_indices.append(unique_original_paths.index(meta[0]))
            
            original_indices_np = np.array(original_indices)
            mask_ratios_np = np.array([meta[1] for meta in crop_metas])
            
            # Find lowest score per original image group
            selected_crops = []
            for i, img_path in enumerate(unique_original_paths):
                # Find all crop indices for this original
                crop_indices = np.where(original_indices_np == i)[0]
                if len(crop_indices) == 0:
                    continue
                
                # Find index with lowest score
                crop_scores_group = crop_scores_np[crop_indices]
                min_idx = crop_indices[np.argmin(crop_scores_group)]
                
                # Get metadata
                mask_ratio = mask_ratios_np[min_idx]
                crop_score = crop_scores_np[min_idx]
                original_score = original_path_to_score[img_path]
                score_diff = original_score - crop_score
                
                # Condition check
                cond1 = (mask_ratio < 0.5) & (score_diff > 7)
                cond2 = (score_diff > 12)
                
                if cond1 or cond2:
                    # Get crop tensor and mask tensor
                    crop_idx_in_group = list(crop_indices).index(min_idx)
                    crop_img_tensor, mask_ratio, crop_mask_tensor, crop_idx = original_to_crops[img_path][crop_idx_in_group]
                    
                    selected_crops.append({
                        "img_path": img_path,
                        "crop_img_tensor": crop_img_tensor,
                        "crop_mask_tensor": crop_mask_tensor,
                        "crop_idx": crop_idx,
                        "mask_ratio": mask_ratio,
                        "crop_score": crop_score,
                        "original_score": original_score,
                        "score_diff": score_diff,
                        "condition": "cond1" if cond1 else "cond2",
                        "mask_ori_path": next(d["mask_ori_path"] for d in img_mask_data if d["img_path"] == img_path)
                    })
    
            # -------------------------- 5. Batch save selected images (reduce IO) --------------------------
            for crop in selected_crops:
                # Print score info
                original_name = os.path.basename(crop["img_path"])
                crop_name = f"{os.path.splitext(original_name)[0]}_crop_{crop['crop_idx']}{os.path.splitext(original_name)[1]}"
                print(f"[Selected image] original: {original_name} | crop: {crop_name}")
                print(f"  original score: {crop['original_score']:.4f} | crop score: {crop['crop_score']:.4f} | score diff: {crop['score_diff']:.4f} | mask ratio: {crop['mask_ratio']:.4f}")
                print(f"  condition met: {crop['condition']}")
                print("-"*80)
                
                # Record stats
                selected_scores_info.append({
                    "original_name": original_name,
                    "crop_name": crop_name,
                    "original_score": crop["original_score"],
                    "crop_score": crop["crop_score"],
                    "score_diff": crop["score_diff"],
                    "mask_ratio": crop["mask_ratio"],
                    "condition": crop["condition"]
                })
                
                # Save crop to ood_select
                try:
                    crop_img = transforms.ToPILImage()(crop["crop_img_tensor"].cpu())
                    target_path = os.path.join(ood_select_dir, crop_name)
                    crop_img.save(target_path)
                    
                    # Concatenate images (batch preprocessing)
                    try:
                        # Load original image and mask
                        original_img = Image.open(crop["img_path"]).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
                        original_mask = Image.open(crop["mask_ori_path"]).convert("L").resize((224, 224), Image.Resampling.NEAREST)
                        original_mask_rgb = Image.merge("RGB", (original_mask, original_mask, original_mask))
                        
                        # Crop image and crop mask
                        crop_mask = transforms.ToPILImage()(crop["crop_mask_tensor"].cpu()).resize((224, 224), Image.Resampling.NEAREST)
                        crop_mask_rgb = Image.merge("RGB", (crop_mask, crop_mask, crop_mask))
                        
                        # Concatenate
                        concat_img = Image.new('RGB', (224*4, 224))
                        concat_img.paste(original_img, (0, 0))
                        concat_img.paste(original_mask_rgb, (224, 0))
                        concat_img.paste(crop_img, (448, 0))
                        concat_img.paste(crop_mask_rgb, (672, 0))
                        
                        # Save concatenated image
                        vis_path = os.path.join(vis_dir, crop_name)
                        concat_img.save(vis_path)
                        
                    except Exception as e:
                        print(f"Concatenation failed {crop_name}: {e}")
                        
                except Exception as e:
                    print(f"Save failed {crop_name}: {e}")
    
        # Shutdown thread pool
        executor.shutdown()
        
        # -------------------------- Summary stats --------------------------
        try:
            vis_files = [f for f in os.listdir(vis_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            vis_img_count = len(vis_files)
            
            print("\n" + "="*80)
            print(f"All processing complete! vis folder contains {vis_img_count} concatenated images")
            print(f" vis path: {vis_dir}")
            print("="*80)
            
            if selected_scores_info:
                avg_original_score = np.mean([info["original_score"] for info in selected_scores_info])
                avg_crop_score = np.mean([info["crop_score"] for info in selected_scores_info])
                avg_score_diff = np.mean([info["score_diff"] for info in selected_scores_info])
                cond1_count = len([info for info in selected_scores_info if info["condition"] == "cond1"])
                cond2_count = len([info for info in selected_scores_info if info["condition"] == "cond2"])
                
                print("\n[Selected image score statistics]")
                print(f"  avg original score: {avg_original_score:.4f}")
                print(f"  avg crop score: {avg_crop_score:.4f}")
                print(f"  avg score diff: {avg_score_diff:.4f}")
                print(f"  images meeting cond1: {cond1_count}")
                print(f"  images meeting cond2: {cond2_count}")
                print("="*80 + "\n")
            else:
                print("  No images met the selection criteria")
                
        except Exception as e:
            print(f"Statistics failed: {e}")
     

    @torch.no_grad()
    def gen_mask(self, batch_size=512):
        """
        Args:
            dataset: Dataset with impath and label attributes (item.impath / item.label)
            batch_size: Batch size (adjust to GPU memory, default 32)
        """
        print('geb_mask ing..........')
        print(self.output_dir)
        self.set_model_mode("eval")
        self.zsmodel.cuda()
        self.evaluator.reset()
        save_dir = self.output_dir
        topk = self.topk
        # Save directory setup
        vis_save_dir = os.path.join(save_dir,'vis_mask_'+str(topk))
        save_mask_dir = os.path.join(save_dir,'gen_mask_'+str(topk))
        os.makedirs(vis_save_dir, exist_ok=True)
        os.makedirs(save_mask_dir, exist_ok=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _, preprocess = clip.load("ViT-B/16", device=device)
        
        dataset = self.raw_data_source
                                                  
        # Device setup
        grid_size = 224 // 16  # 14x14 grid
        num_regions = grid_size * grid_size  # 196 regions

        # Split dataset into batches
        total_samples = len(dataset)
        for batch_idx in range(0, total_samples, batch_size):
            # Extract samples for current batch
            batch_samples = dataset[batch_idx:batch_idx + batch_size]
            batch_img_paths = [item.impath for item in batch_samples]
            batch_labels = [item.label for item in batch_samples]
            
            batch_size_current = len(batch_img_paths)

            print(f"Processing batch {batch_idx // batch_size + 1}/{(total_samples + batch_size - 1) // batch_size} "
                  f"(samples: {batch_size_current})")

            # -------------------------- 1. Batch load original images (keep size for drawing) --------------------------
            raw_imgs = []  # Store original PIL images
            img_sizes = []  # Store original image size (w, h) per image
            for img_path in batch_img_paths:
                raw_img = Image.open(img_path).convert("RGB")
                raw_imgs.append(raw_img)
                img_sizes.append(raw_img.size)  # (w, h)

            # -------------------------- 2. Batch preprocess images (for model inference) --------------------------
            batch_images = []
            for img_path in batch_img_paths:
                img = preprocess(Image.open(img_path)).to(device)  # (C, 224, 224)
                batch_images.append(img)
            batch_images = torch.stack(batch_images)  # (batch_size, C, 224, 224)

            # -------------------------- 3. Batch model inference --------------------------
            with torch.no_grad():
                output, output_local = self.zsmodel.forward(batch_images)  # output_local: (batch_size, 196, num_classes)

            # -------------------------- 4. Batch process local predictions --------------------------
            # Convert labels to tensor and expand dims
            batch_labels_tensor = torch.tensor(batch_labels, device=device)  # (batch_size,)
            label_repeat = batch_labels_tensor.repeat_interleave(num_regions)  # (batch_size * 196,)

            # Process output_local: softmax + reshape
            output_local = F.softmax(output_local, dim=-1)  # (batch_size, 196, num_classes)
            output_local = output_local.view(-1, output_local.shape[-1])  # (batch_size * 196, num_classes)

            # Filter regions where top-k predictions contain the label
            pred_topk = torch.topk(output_local, k=topk, dim=1)[1]  # (batch_size * 196, 50)
            contains_label = pred_topk.eq(label_repeat.unsqueeze(1)).any(dim=1)  # (batch_size * 196,)
            contains_label = contains_label.view(batch_size_current, num_regions)  # (batch_size, 196)

            # -------------------------- 5. Batch compute coordinates (vectorized, no loop) --------------------------
            # Generate 196 region coordinates per sample (vectorized)
            indices = np.arange(num_regions)
            rows = indices // grid_size  # (196,)
            cols = indices % grid_size   # (196,)

            # Coordinates on 224x224 image
            x1_224 = cols * 16  # (196,)
            y1_224 = rows * 16  # (196,)
            x2_224 = (cols + 1) * 16  # (196,)
            y2_224 = (rows + 1) * 16  # (196,)

            # -------------------------- 6. Draw and save per sample (avoid batch drawing conflicts) --------------------------
            for idx_in_batch in range(batch_size_current):
                img_path = batch_img_paths[idx_in_batch]
                label = batch_labels[idx_in_batch]
                raw_img = raw_imgs[idx_in_batch]
                img_w, img_h = img_sizes[idx_in_batch]
                contains_label_single = contains_label[idx_in_batch].cpu().numpy()  # (196,)

                # Filter valid region coordinates for current sample
                valid_x1 = (x1_224 / 224 * img_w).astype(int)[contains_label_single]
                valid_y1 = (y1_224 / 224 * img_h).astype(int)[contains_label_single]
                valid_x2 = (x2_224 / 224 * img_w).astype(int)[contains_label_single]
                valid_y2 = (y2_224 / 224 * img_h).astype(int)[contains_label_single]

                # -------------------------- Draw original image + gray grid cells --------------------------
                draw_img = ImageDraw.Draw(raw_img, mode="RGBA")
                for x1, y1, x2, y2 in zip(valid_x1, valid_y1, valid_x2, valid_y2):
                    draw_img.rectangle([x1, y1, x2, y2], fill=(128, 128, 128, 255))

                # -------------------------- Create and draw mask --------------------------
                mask = Image.new("L", (img_w, img_h), color=255)
                draw_mask = ImageDraw.Draw(mask)
                for x1, y1, x2, y2 in zip(valid_x1, valid_y1, valid_x2, valid_y2):
                    draw_mask.rectangle([x1, y1, x2, y2], fill=0)

                # -------------------------- Dilate mask + invert colors --------------------------
                mask_cv = np.array(mask)
                kernel = np.ones((5, 5), np.uint8)
                dilated_mask_cv = cv2.dilate(mask_cv, kernel, iterations=1)
                inverted_dilated_mask_cv = 255 - dilated_mask_cv
                inverted_dilated_mask = Image.fromarray(inverted_dilated_mask_cv)

                # -------------------------- Save images --------------------------
                img_name = img_path.split('/')[-1]
                raw_img_save_path = os.path.join(save_dir, img_name)
                img_name_no_ext = os.path.splitext(img_name)[0]
                inverted_mask_save_path = os.path.join(save_mask_dir, f"{img_name_no_ext}.png")

                #raw_img.save(raw_img_save_path)
                inverted_dilated_mask.save(inverted_mask_save_path, format="PNG")

              

    @torch.no_grad()
    def ood_loader(self):
        topk = self.topk
        self.inpainted_dir =  self.output_dir+ '/ood_select_'+str(topk)
        self.id_aug_dir =   self.output_dir+ '/id_aug_'+str(topk) 
        trainset = self.dm.train_loader_x.dataset
        tmp_data_source = []
        raw_data_source = []
        
        files = []
        self.id2clsname = {}
        for item in trainset.data_source:
            if item.label == 1000:
                continue
            self.id2clsname[item.impath.split('/')[-1].split('_')[0]] = [item.label, item.classname]
        # load background data
        os.makedirs(self.inpainted_dir, exist_ok=True)
        
        image_folder_raw = './data/' + self.train_root + '/raw/'
        for filename in os.listdir(image_folder_raw):
            if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.JPEG'):
                impath = os.path.join(image_folder_raw, filename)
                imageid = filename.split('_')[0]
                classname = self.id2clsname[imageid][1]
                label = self.id2clsname[imageid][0]
                tmp_data_source.append(Datum(impath=impath, label=label, classname=classname, domain=1)) # Domain 1: ID data with mask
                raw_data_source.append(Datum(impath=impath, label=label, classname=classname, domain=1)) 
        for filename in os.listdir(self.inpainted_dir ):
            if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.JPEG'):
                impath = os.path.join(self.inpainted_dir, filename)
                classname = 'background'
                label = self.num_classes
                tmp_data_source.append(Datum(impath=impath, label=label, classname=classname, domain=2)) # Domain 2: Background
        
        # load id_aug data
        
        id_aug_dir = os.path.join(self.output_dir, "id_aug")
        copy_images_with_overwrite(image_folder_raw,id_aug_dir)
        copy_images_with_overwrite(self.id_aug_dir,id_aug_dir)

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
            for filename in os.listdir(self.inpainted_dir ):
                if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.JPEG'):
                    impath = os.path.join(self.inpainted_dir, filename)
                    classname = 'background'
                    label = self.num_classes
                    tmp_data_source.append(Datum(impath=impath, label=label, classname=classname, domain=2)) # Domain 2: Background
                                          
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

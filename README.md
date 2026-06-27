# OSPCoOpv2

The official code repository for **"Bg\-Bench: Benchmarking OOD Background Interference and Overcoming Shortcuts with Model\-Free Context Optimization"**

## Bg\-Bench

### Dataset Download

Full dataset \(Version 1, 2\.5 GB\) is hosted on Kaggle:

[Dataset: Bg-Bench](https://www.kaggle.com/datasets/xiangexiang/bg-bench/data)

### Directory Structure

```plain text
Bg-Bench
├── CIFAR100-Bg       # CIFAR-100 domain background OOD split
│   └── Bg/images     # Pure background OOD images for OOD evaluation
├── ImageNet-Bg       # ImageNet domain background OOD split
│   └── Bg/images     # Pure background OOD images for OOD evaluation
├── OES-Bg            # Remote sensing OES domain background OOD split
│   ├── Bg            # Pure background OOD images
│   └── ID/test       # Corresponding in-distribution test images
└── Hallucination     # Multimodal hallucination evaluation subset
    ├── CIFAR/images_all
    │   └── cifar_dataset.json  # Hallucination ground-truth labels
    ├── ImageNet
    ├── OES
    └── coco-hard
```

### Task Usage Guide

#### 1\. Background OOD Detection

- Target folders: All directories suffixed with `-Bg` \(CIFAR100\-Bg, ImageNet\-Bg, OES\-Bg\)

- Images under`Bg/images` are pure background samples with all foreground objects removed, served as OOD test data\.

- `ID/test` folder provides matched in\-distribution samples.

#### 2\.  Hallucination Evaluation

- Target folder: `Hallucination`

- Contains multi\-domain test images divided into CIFAR, ImageNet, OES and coco\-hard subfolders\.

- `xxxx.json` stores complete ground\-truth labels for evaluating object hallucination of multimodal large language models\.

## Dataset Preparation

- ID Datasets: ImageNet-1K, The ImageNet-1k dataset (ILSVRC-2012) can be downloaded [here](https://image-net.org/challenges/LSVRC/2012/index.php#).
 
- OOD Datasets: iNaturalist, SUN, Places, and Texture. Please follow the instruction from [MOS](https://github.com/deeplearning-wisc/large_scale_ood#out-of-distribution-dataset).

- Background Interference OOD Datasets: Bg-Bench.

After downloading the datasets, update the data root path in `./OSPCoOp/root_config.py` to point to your local dataset location.

Please run the following command to sample few-shot training data for further Pseudo-OOD Generation.
```
CUDA_VISIBLE_DEVICES=0 python train.py  --trainer OSPCoOp --shots 16 --few_shot_sampler True --seed 1
```

## OSPCoOp-EMF
Our experiments are conducted with Python 3.9.18 and Pytorch 2.1.0. Followed [LoCoOp](https://github.com/AtsuMiyai/LoCoOp) and [CoOp](https://github.com/KaiyangZhou/CoOp), the training code is built on top of the awesome toolbox [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch). So, you need to install `dassl` first. 

To train OSPCoOp, please run the command in `./osp/train_ospemf.sh`.

## OSPCoOp


### Pseudo-OOD Generation

For quick start, we have provided our generated Pseudo-OOD data, which can be downloaded here ([Google Drive](https://drive.google.com/drive/folders/1hht3yDUNXsXqZSgaI0MMTYgot4D5F7bl?usp=drive_link) or [Baidu Cloud](https://pan.baidu.com/s/1tH8JCnVta0cxZyA-z8CJsw?pwd=dm5u)).

#### Image Masking
Please follow these steps:

Step1: Please install [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything).

Step2: Copy the ./GroundingSAM/masking.py and ./GroundingSAM/classnames.py file into your Grounded-Segment-Anything project directory.

Step3: Run the following command:
```
python masking.py --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py   --grounded_checkpoint groundingdino_swint_ogc.pth   --sam_checkpoint sam_vit_h_4b8939.pth   --output_dir "output_dir"   --box_threshold 0.3   --text_threshold 0.25    --device "cuda"
```
Note: Set "output_dir" to your image directory, and place the images to be processed in the "./output_dir/raw" directory.
#### Image Inpainting
Please follow these steps:

Step1: Please install [Inpaint-Anything](https://github.com/geekyutao/Inpaint-Anything).

Step2: Copy the ./Inpainting/inpainting.py, ./Inpainting/Texture_inpainting.py and ./GroundingSAM/classnames.py file into your Inpaint-Anything project directory.

Step3: Run the following command to generate the inpainting images:
```
python inpainting.py --output_dir output    --lama_ckpt ./pretrained_models/big-lama  --min=0.0  --max=0.25
```

Step4: Run the following command to compute the score of inpainting images:
```
python get_clip_score.py  --output_dir output
```
Step5: To generate the texture OOD aug data, please run the following command:
```
python Texture_inpainting.py  --output_dir output    --lama_ckpt ./pretrained_models/big-lama  --min=0.0  --max=0.25
```
#### Training
To train OSPCoOp, please run the command in `./osp/train_osp.sh`.
### OOD Detection Evaluation

To evaluate iNaturalist, SUN, Places, and Texture OOD data, please run the following command.
```
CUDA_VISIBLE_DEVICES=0 python train.py  --trainer OSPCoOp --shots 16 --eval_only True --model-dir xxx 
```
To evaluate ImageNet-Bg in Bg-Bench, please run the following command.
```
CUDA_VISIBLE_DEVICES=0 python train.py  --trainer OSPCoOp --shots 16 --eval_only True --model-dir xx  --eval-bg True
```


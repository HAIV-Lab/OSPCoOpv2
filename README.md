# OSPCoOpv2

The official code repository for **"Bg\-Bench: Benchmarking OOD Background Interference and Overcoming Shortcuts with Model\-Free Context Optimization"**

## Bg\-Bench

### Dataset Download

Full dataset \(Version 1, 2\.5 GB\) is hosted on Kaggle:

[https://www\.kaggle\.com/datasets/xiangexiang/bg\-bench/data](https://www.kaggle.com/datasets/xiangexiang/bg-bench/data)

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

- `ID/test` folder provides matched in\-distribution samples for calculating standard OOD metrics \(AUROC, FPR95\)\.

#### 2\. Multimodal Hallucination Evaluation

- Target folder: `Hallucination`

- Contains multi\-domain test images divided into CIFAR, ImageNet, OES and coco\-hard subfolders\.

- `cifar_dataset.json` stores complete ground\-truth labels for evaluating object hallucination of multimodal large language models\.


```

> （注：部分内容可能由 AI 生成）

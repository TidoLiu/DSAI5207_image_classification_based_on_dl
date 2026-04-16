# DSAI5207_image_classification_based_on_dl
Group Members: GUO Ziqi (25085124g); LIU Xiduo (25053421g); GUO Jingyao (25055797g); YANG Liuyue (25049452g)

# Overview

This project implements a hierarchical dog breed classification system:

1. **Coarse Classification**: Binary classification (long hair vs short hair) using VGG16
2. **Fine Classification**: Multi-class breed classification (48 breeds) using feature fusion from multiple CNN models

## Key Results

| Model | Validation Accuracy | Description |
|:---|:---|:---|
| VGG16 (Coarse) | ~69% | Binary classification |
| InceptionV3 (Fine) | ~98.4% | Single model feature extraction |
| Fused Features (Fine) | ~99.1% | 4-model feature fusion (InceptionV3, Xception, NASNetLarge, InceptionResNetV2) |

---

# Dependencies

## Core Libraries
tensorflow>=2.0
keras
numpy
pandas
matplotlib
seaborn
scikit-learn
opencv-python
scikit-image
tqdm
pillow

## Installation

```bash
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn opencv-python scikit-image tqdm pillow
```
---
# Dataset Preparation

## Required Data Files
| File              | Description                                                              |
| :---------------- | :----------------------------------------------------------------------- |
| `labels.csv`      | CSV file with columns `id` and `breed_label` (for coarse classification) |
| `labels-long.csv` | CSV file with columns `id` and `breed` (for fine classification)         |
| `train/`          | Directory containing training images named as `{image_id}.jpg`           |


## Dataset Organization
The code will automatically organize images into:
```
dog_datasets/
├── train/
│   ├── long/
│   │   └── {image_id}.jpg
│   └── short/
│       └── ...
└── val/
    ├── long/
    └── short/
```
## Dataset Statistics

- Coarse Classification: 10,297 images (2 classes: long/short)
- Fine Classification (Long-haired): 4,132 images (48 breeds)
- Image Size: 331x331x3 (for fine classification), 224x224x3 (for coarse)

## The Pytorch Implementation of Deep Photo Enhancer

## Introduction

This project is based on the thesis《Deep Photo Enhancer: Unpaired Learning for Image Enhancement from Photographs with GANs》。

The author's project address is：[nothinglo/Deep-Photo-Enhancer](https://github.com/nothinglo/Deep-Photo-Enhancer)

中文文档说明请看[这里](https://github.com/mtics/deep-photo-enhancer/blob/master/README_zh_cn.md)

## Prerequisites

- Python 3.6
- CUDA 10.0
- The needed packages are listed in `requirements.txt`，please use the following command to install：
  `pip install -r requirements.txt`

## Data Preparation

Expert-C on [MIT-Adobe FiveK dataset](https://data.csail.mit.edu/graphics/fivek/)

## Preparation

1. All hyperparameters are in `libs\constant.py`
2. There are some folders need to be created:
   1. `images_LR`：Used to store datasets
      1. `Expert-C`
      2. `input`
      3. In each of the above two folders, the following three new folders need to be created:
         1. `Testing`
         2. `Training1`
         3. `Training2`
   2. `models`：Used to store all the training generated files：
      1. `gt_images`
      2. `input_images`
      3. `pretrain_checkpoint`
      4. `pretrain_images`
      5. `test_images`
      6. `train_checkpoint`
      7. `train_images`
      8. `train_test_images`
      9. In each of the above folders, the following two new folders need to be created:
         1. `1Way`
         2. `2Way`
   3. `model`: Used to store `log_PreTraining.txt`
   4. The last generated `gan1_pretrain_xxx_xxx.pth` should be placed in the root directory.

## Cost Time

- Pretrain: 3H55M  8H45M 9H25M
- Train: 2H45M  2H49M 3H03M 5H38M(2Way) 4H45M(2Way)

## Problem 

1. There may be a problem in computing the value of PSNR or not. It needs to be  proved.
2. The compute functions in `libs\compute.py` are wrong, which cause the discriminator loss cannot converge and the output is awful.
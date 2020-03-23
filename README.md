# 代码运行文档

## 复现代码

本代码根据论文《Deep Photo Enhancer: Unpaired Learning for Image Enhancement from Photographs with GANs》。

原作者代码库：[nothinglo/Deep-Photo-Enhancer](https://github.com/nothinglo/Deep-Photo-Enhancer)

## 所需环境

- Python 3.6
- CUDA 10.0
- 具体见requirements.txt，使用如下命令安装依赖：
`pip install -r requirements.txt`

## 所需数据集

Expert-C on [MIT-Adobe FiveK dataset](https://data.csail.mit.edu/graphics/fivek/)

## 部分操作

1. PreTrain和Train中的参数可以修改
2. 在SourceCode下要新建多个文件夹，分为：
   1. images_LR：用来存放数据集
      1. Expert-C
      2. input
      3. 上述两文件夹中都需要新建下面三个文件夹
         1. Testing
         2. Training1
         3. Training2
   2. model：用来存放所有训练产生的文件，其下还需新建：
      1. gt_images
         1. 1Way
         2. 2Way
      2. input_images
         1. 1Way
         2. 2Way
      3. pretrain_checkpoint
         1. 1Way
         2. 2Way
      4. pretrain_images
         1. 1Way
         2. 2Way
      5. test_images
         1. 1Way
         2. 2Way
      6. train_checkpoint
         1. 1Way
         2. 2Way
      7. train_images
         1. 1Way
         2. 2Way
      8. train_test_images
         1. 1Way
         2. 2Way
   3. models:用来存放log_PreTraining.txt
   4. 训练后的gan1_pretrain_XXx_xxx.pth需要放在根目录下

## Cost Time

- Pretrain: 3H55M  8H45M 9H25M
- Train: 2H45M  2H49M 3H03M 5H38M(2Way) 4H45M(2Way)

## Problem 

1. There may be a problem in computing the value of PSNR or not. It needs to be  proved.

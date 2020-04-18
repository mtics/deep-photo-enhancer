## The Pytorch Implementation of Deep Photo Enhancer

This project is based on the thesis《Deep Photo Enhancer: Unpaired Learning for Image Enhancement from Photographs with GANs》。

The author's project address is：[nothinglo/Deep-Photo-Enhancer](https://github.com/nothinglo/Deep-Photo-Enhancer)

My code is based on [hyerania\Deep-Learning-Project](https://github.com/hyerania/Deep-Learning-Project)

中文文档说明请看[这里](https://github.com/mtics/deep-photo-enhancer/blob/master/README_zh_cn.md)

## Requirements

- Python 3.6
- CUDA 10.0
- To install requirements：
  `pip install -r requirements.txt`

## Prerequisites

### Data

Expert-C on [MIT-Adobe FiveK dataset](https://data.csail.mit.edu/graphics/fivek/)

### Folders

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
      10. `log`: used to store all logs and the result of data visualization
          1. `pretrain`: store all charts of pretrain
          2. `train`: store all charts of train

## Training

1. If you don't have a pretrain module, the first thing you need to do is running "1WayGAN_PreTrain.py":
   `python 1WayGAN_PreTrain.py`
   
2. The last generated `\models\pretrain_checkpoint\gan1_pretrain_xxx_xxx.pth` should be placed in the root directory, like "gan1_pretrain_100_113.pth" in my repository.

3. Next you need to change the line 15 in “1WayGAN_Train.py” or the same line in “2WayGAN_Train.py”.  
   
4. To train the model, please run this command:

   ```python
   # Only if you want to use 1 way Gan
   python 1WayGAN_Train.py
   # Only if you want to use 2 way Gan
   python 2WayGAN_Train.py
   ```

## Evaluation

For now, the evaluation and training are simultaneous. So there is no need to run anything.

To evaluate my module, I use PSNR in “XWayGAN_Train.py”

## Results

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;}
.tg .tg-18eh{font-weight:bold;border-color:#000000;text-align:center;vertical-align:middle}
.tg .tg-wp8o{border-color:#000000;text-align:center;vertical-align:top}
.tg .tg-xwyw{border-color:#000000;text-align:center;vertical-align:middle}
.tg .tg-mqa1{font-weight:bold;border-color:#000000;text-align:center;vertical-align:top}
</style>
<table class="tg">
  <tr>
    <th class="tg-xwyw"></th>
    <th class="tg-mqa1"></th>
    <th class="tg-mqa1">1Way GAN</th>
    <th class="tg-mqa1">2Way GAN</th>
    <th class="tg-18eh">BATCH_SIZE</th>
    <th class="tg-18eh">NUM_EPOCHS_PRETRAIN</th>
    <th class="tg-18eh">NUM_EPOCHS_TRAIN</th>
    <th class="tg-18eh">Discriminator Loss</th>
    <th class="tg-18eh">Generator Loss</th>
    <th class="tg-18eh">PSNR</th>
    <th class="tg-18eh">Time</th>
  </tr>
  <tr>
    <td class="tg-18eh" rowspan="3">Pretrain</td>
    <td class="tg-wp8o">1</td>
    <td class="tg-wp8o">\</td>
    <td class="tg-wp8o">\</td>
    <td class="tg-xwyw"></td>
    <td class="tg-xwyw"></td>
    <td class="tg-xwyw"></td>
    <td class="tg-xwyw">\</td>
    <td class="tg-xwyw"></td>
    <td class="tg-xwyw">\</td>
    <td class="tg-xwyw">3H55M</td>
  </tr>
  <tr>
    <td class="tg-wp8o">2</td>
    <td class="tg-wp8o">\</td>
    <td class="tg-wp8o">\</td>
    <td class="tg-xwyw"></td>
    <td class="tg-xwyw"></td>
    <td class="tg-xwyw"></td>
    <td class="tg-xwyw">\</td>
    <td class="tg-xwyw"></td>
    <td class="tg-xwyw">\</td>
    <td class="tg-xwyw">8H45M</td>
  </tr>
  <tr>
    <td class="tg-wp8o">3</td>
    <td class="tg-wp8o">\</td>
    <td class="tg-wp8o">\</td>
    <td class="tg-wp8o"></td>
    <td class="tg-wp8o"></td>
    <td class="tg-wp8o"></td>
    <td class="tg-wp8o">\</td>
    <td class="tg-wp8o"></td>
    <td class="tg-wp8o">\</td>
    <td class="tg-wp8o">9H25M</td>
  </tr>
  <tr>
    <td class="tg-18eh" rowspan="3">Train</td>
    <td class="tg-wp8o">1</td>
    <td class="tg-wp8o">√</td>
    <td class="tg-wp8o"></td>
    <td class="tg-xwyw"></td>
    <td class="tg-xwyw"></td>
    <td class="tg-xwyw"></td>
    <td class="tg-xwyw"></td>
    <td class="tg-xwyw"></td>
    <td class="tg-xwyw"></td>
    <td class="tg-xwyw">2H45M</td>
  </tr>
  <tr>
    <td class="tg-wp8o">2</td>
    <td class="tg-wp8o"></td>
    <td class="tg-wp8o">√</td>
    <td class="tg-xwyw"></td>
    <td class="tg-xwyw"></td>
    <td class="tg-xwyw"></td>
    <td class="tg-xwyw"></td>
    <td class="tg-xwyw"></td>
    <td class="tg-xwyw"></td>
    <td class="tg-xwyw">5H38M</td>
  </tr>
  <tr>
    <td class="tg-wp8o">3</td>
    <td class="tg-wp8o"></td>
    <td class="tg-wp8o">√</td>
    <td class="tg-wp8o"></td>
    <td class="tg-wp8o"></td>
    <td class="tg-wp8o"></td>
    <td class="tg-wp8o"></td>
    <td class="tg-wp8o"></td>
    <td class="tg-wp8o"></td>
    <td class="tg-wp8o">4H45M</td>
  </tr>
</table>

## Problem

1. There may be a problem in computing the value of PSNR or not. It needs to be  proved.
2. The compute functions in `libs\compute.py` are wrong, which cause the discriminator loss cannot converge and the output is awful.

## License

This repo is released under  the [MIT License](LICENSE.md)

## Contributor

For now, This repo is maintained by Zhiwei Li.

Welcome to join me to maintain it together.

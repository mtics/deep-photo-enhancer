## 实现基于Pytorch的“深度照片增强器”

本项目根据论文《Deep Photo Enhancer: Unpaired Learning for Image Enhancement from Photographs with GANs》进行实验。

论文作者的项目地址为：[nothinglo/Deep-Photo-Enhancer](https://github.com/nothinglo/Deep-Photo-Enhancer)。

我的代码是基于[hyerania\Deep-Learning-Project](https://github.com/hyerania/Deep-Learning-Project)进行改进的。

## 版本要求

- Python 3.6
- CUDA 10.0
- 运行下列命令来安装依赖：
  `pip install -r requirements.txt`

## 预先准备

### 数据集

使用[MIT-Adobe FiveK dataset](https://data.csail.mit.edu/graphics/fivek/)的Expert-C

### Folders

1. 所有的超参都存放于`libs\constant.py`
2. 有一些文件夹需要用户自己创建:
   1. `images_LR`：用来存放数据集
      1. `Expert-C`
      2. `input`
      3. 上述两个文件夹中, 都需要新建下面三个文件夹:
         1. `Testing`
         2. `Training1`
         3. `Training2`
   2. `models`：用来存放所有训练生成的文件：
      1. `gt_images`
      2. `input_images`
      3. `pretrain_checkpoint`
      4. `pretrain_images`
      5. `test_images`
      6. `train_checkpoint`
      7. `train_images`
      8. `train_test_images`
      9. 在上述的文件夹中, 都需要新建两个文件夹:
         1. `1Way`
         2. `2Way`
      10. `log`: 用来存放所有的日志文本和数据可视化的结果
          1. `pretrain`: 用来存放所有预训练生成的图表
          2. `train`: 用来存放所有训练生成的图表

## Training

1. 若您还没有预训练模型, 首先需要运行"1WayGAN_PreTrain.py":
   `python 1WayGAN_PreTrain.py`
2. 最后一个生成的`\models\pretrain_checkpoint\gan1_pretrain_xxx_xxx.pth`应该放在根目录下, 比如项目中的 "gan1_pretrain_100_113.pth".
3. 接下来您需要将 “1WayGAN_Train.py” 和 “2WayGAN_Train.py”的第15行进行相应的修改。
4. 运行下列命令来进行训练:
   
   ```python
   # 若您想训练单向GAN
   python 1WayGAN_Train.py
   # 若您想训练双向GAN
   python 2WayGAN_Train.py
   ```

## 评估

目前，评估和训练是同步进行的，所以不需要单独运行什么.

为了评估我的模型，我在“XWayGAN_Train.py”中使用了PSNR算法

## 结果

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

## 问题

1. 模型中目前存在严重问题，造成训练后的效果并不理想

## 许可证

该项目根据[MIT许可证](LICENSE.md)进行开源

## 贡献者

本项目目前仅由李志伟进行维护。

欢迎大家加入与我共同维护。

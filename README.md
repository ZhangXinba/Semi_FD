# Semi-supervised Transformable Architecture Search for Feature Distillation


## Requirements
-requirements.txt

## CIFAR
Attention! During training, change right locations of dtaset and pretrained models in right places.

The code expects to find the data in specific directories inside the data-local directory. You can prepare the CIFAR with this command:

```
./data-local/bin/prepare_cifar10.sh
```

create a list of image filenames and their labels:

```
./datalocal/labels/prepare_semisupervised_sets.
```

After generating labels, change the location in argument, and 'data_dir' in datasets.py


## Settings
The structure settings are as follows.

| Setup | Compression type |   Teacher   |   Student   | Teacher size | Student size | Size ratio |
|:-----:|:----------------:|:-----------:|:-----------:|:------------:|:------------:|:----------:|
|  (a)  |       Depth      |   WRN 28-4  |   WRN 16-4  |     5.87M    |     2.77M    |    47.2%   |
|  (b)  |      Channel     |   WRN 28-4  |   WRN 28-2  |     5.87M    |     1.47M    |    25.0%   |
|  (c)  |  Depth & channel |   WRN 28-4  |   WRN 16-2  |     5.87M    |     0.70M    |    11.9%   |
|  (d)  |   Architecture   |   WRN 28-4  |  ResNet 56  |     5.87M    |     0.86M    |    14.7%   |
|  (e)  |   Architecture   | Pyramid-200 |   WRN 28-4  |    26.84M    |     5.87M    |    21.9%   |
|  (f)  |   Architecture   | Pyramid-200 | Pyramid-110 |    26.84M    |     3.91M    |    14.6%   |

## Teacher models
Download following pre-trained teacher network and put them into ./models directory
- [Wide Residual Network 28-4](https://drive.google.com/open?id=1Quxgs5teXVXwD3jBdkk-WeNLNpxbiZXN)
- [PyramidNet-200(240)](https://drive.google.com/open?id=1_QgG81fNM3OvVIbMAxDPykKWuSIyKnmz)

## Training
Run ```CIFAR/train_with_distillation.py``` with setting alphabet (a - f)
```
python train_with_distillation.py --setting a
```

# ImageNet

In 2016, the Google deepmind team extracted a small part (about 3gb in size) from the Imagenet dataset and produced the mini Imagenet dataset, which has 100 categories. Each category has 600 pictures, 60000 in total (all files ending in. JPG), and the size of the images is not fixed.It is not feasible to train your classification network directly with mini imgenet data set, because train CSV and val.csv are not sampled from each category, so we need to build a new train CSV and val.csv files.

Download the data, PAN:https://pan.baidu.com/s/1tD8xdkOO6MiWuThOff07aA Code：ZMMM
And use python ./devide.py to re-build new lists.

## Settings

| Setup | Compression type |   Teacher  |  Student  | Teacher size | Student size | Size ratio |
|:-----:|:----------------:|:----------:|:---------:|:------------:|:------------:|:----------:|
|  (a)  |       Depth      | ResNet 152 | ResNet 50 |    60.19M    |    25.56M    |   42.47%   |
|  (b)  |   Architecture   |  ResNet 50 | MobileNet |    25.56M    |     4.23M    |   16.55%   |

## The teacher model will be automatically downloaded from PyTorch sites.

## Training
python train_with_distillation.py


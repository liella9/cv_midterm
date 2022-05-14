# DATA130051.01 Midterm Project - Guide

本次作业主要实现了：

- 在CIFAR-100数据集上训练CNN
- 在VOC数据集上训练Faster R-CNN和YOLO V3

各个模型的下载链接在report文件夹的报告中。

## 在CIFAR-100数据集上训练CNN

### 训练步骤
1. 安装程序运行对应的包
2. 下载CIFAR-100数据集，在Part 1中新建data文件夹并放入数据集
3. 在Part 1文件夹的train.py文件中设置好训练时的参数
4. 运行train.py，进行训练
5. 最终得到的模型以及loss和accuracy曲线将保存于Part 1文件夹中

### 测试步骤
1. 将下载的模型放入Part 1文件夹
2. 在train.py文件中，设置学习率为0
3. 在train.py文件中，设置模型对应的.pth文件名
4. 运行train.py，进行测试


## 在VOC数据集上训练Faster R-CNN和YOLO V3




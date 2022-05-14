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

### Faster R-CNN

#### 训练步骤
1. 提前准备好数据集
2. 提前下载好对应预训练模型权重(要重命名)backbone: https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
5. 使用```train_resnet50_fpn.py```训练脚本
6. 最终得到的模型以及loss和mAP曲线保存

#### 测试步骤
1. 使用predict.py
2. train_weights设置为训练好的模型,original_img为待测试的图片
3. 得到带得分和类别的box


### YOLO V3

#### 训练步骤
1. 将VOC标注数据转为YOLO标注数据，使用```trans_voc2yolo.py```脚本进行转换
2. 根据摆放好的数据集信息生成一系列相关准备文件
3. 预训练权重下载地址```yolov3-spp-ultralytics-512.pt```: 链接: https://pan.baidu.com/s/1k5yeTZZNv8Xqf0uBXnUK-g  密码: e3k1
4. 直接使用train.py训练脚本

#### 测试步骤
1. predict_test.py，使用训练好的权重进行预测测试
2. weights设置为训练好的模型,img_path为待测试的图片
3. 得到带得分和类别的box

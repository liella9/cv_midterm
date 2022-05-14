import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from torchtoolbox.transform import Cutout
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':
    train_set = torchvision.datasets.VOCDetection(root='./data', year='2007', image_set='trainval', download=True)
    test_set = torchvision.datasets.VOCDetection(root='./data', year='2007', image_set='test', download=True)
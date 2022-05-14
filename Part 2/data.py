import numpy as np
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import cv2

data_path = './data'

def plot_box(img, box, classes, scores, line_thickness=2):
    start_point = (int(box['xmin']), int(box['ymin']))
    end_point = (int(box['xmax']), int(box['ymax']))
    cv2.rectangle(img, start_point, end_point, (255, 0, 0), line_thickness)
    cv2.putText(img, classes, start_point, cv2.FONT_HERSHEY_COMPLEX,
                0.5, (255, 0, 0), 2)


if __name__ == '__main__':
    train_set = torchvision.datasets.VOCDetection(root=data_path, year='2007', image_set='trainval', download=False)
    test_set = torchvision.datasets.VOCDetection(root=data_path, year='2007', image_set='test', download=False)

    for index, data in enumerate(train_set):
        image, label = data[0], data[1]
        objects = label['annotation']['object']
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        for curr_object in objects:
            class_name, bounding_box = curr_object['name'], curr_object['bndbox']
            plot_box(image, bounding_box, class_name, scores=1)

        cv2.imwrite('image_' + str(index) + '.png', image)

        if index == 3:
            break

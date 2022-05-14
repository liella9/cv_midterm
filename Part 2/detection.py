import torch
import torchvision
import argparse
import cv2
import sys
import random

sys.path.append('./')

coco_labels_name = ["unlabeled", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                    "traffic light", "fire hydrant", "street sign", "stop sign", "parking meter", "bench", "bird",
                    "cat", "dog", "horse",
                    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe",
                    "eye glasses", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports_ball", "kite",
                    "baseball bat",
                    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "plate", "wine glass",
                    "cup", "fork", "knife",
                    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot_dog", "pizza",
                    "donut", "cake", "chair", "couch", "potted plant", "bed", "mirror", "dining table", "window",
                    "desk",
                    "toilet", "door", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
                    "toaster", "sink", "refrigerator", "blender", "book", "clock", "vase", "scissors", "teddy bear",
                    "hair drier",
                    "toothbrush", "hair brush"]

voc_names = ['unlabeled', 'aeroplane', 'bicycle', 'bird', 'boat',
             'bottle', 'bus', 'car', 'cat', 'chair',
             'cow', 'diningtable', 'dog', 'horse', 'motorbike',
             'person', 'pottedplant', 'sheep', 'sofa', 'train',
             'tvmonitor']


def get_args():
    parser = argparse.ArgumentParser(description='Pytorch Faster-rcnn Detection')

    parser.add_argument('--image_path', default='example.jpg', type=str, help='image path')
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', help='model')
    parser.add_argument('--dataset', default='coco', help='model')
    parser.add_argument('--score', type=float, default=0.8, help='objectness score threshold')
    args = parser.parse_args()

    return args


def random_color():
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)

    return b, g, r


def main():
    args = get_args()
    input_img = []
    num_classes = 91
    names = voc_names

    # Model creating
    print("Creating model")
    model = torchvision.models.detection.__dict__[args.model](num_classes=num_classes, pretrained=True)
    print(model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    model = model.to(device)

    model.eval()

    src_img = cv2.imread(args.image_path)
    img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float().cuda()
    input_img.append(img_tensor)
    out = model(input_img)
    boxes = out[0]['boxes']
    labels = out[0]['labels']
    scores = out[0]['scores']

    for idx in range(boxes.shape[0]):
        if scores[idx] >= args.score:
            x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]
            name = names[labels[idx].item()]
            cv2.rectangle(src_img, (x1, y1), (x2, y2), random_color(), thickness=2)
            cv2.putText(src_img, text=name, org=(x1, y1 + 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 0, 255))

    cv2.imwrite('example-result.jpg', src_img)


if __name__ == "__main__":
    main()

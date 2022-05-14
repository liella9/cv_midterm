import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision import models
from torchvision.models.resnet import resnet18
from torchvision.transforms.transforms import Resize

from train import train, test, plot_curves

epochs = 200
batch_size = 100
model_path = 'model_transfer.pth'
initial_learning_rate = 1e-2
learning_rate_decay = 0.98
momentum = 0.9
weight_decay = 1e-5


transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
transform_test = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

# read and randomly shuffle and batch data
train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

# check the device we are using
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# transfer learning
net = models.resnet18(pretrained=True)
for param in net.parameters():
    print(param.names)
    param.requires_grad = False

fc_inputs = net.fc.in_features
net.fc = nn.Sequential(
    nn.Linear(fc_inputs, 100),
    nn.LogSoftmax(dim=1)
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
net = net.to(device)

loss_fn = nn.CrossEntropyLoss()

# train
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    learning_rate = initial_learning_rate * (learning_rate_decay ** t)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                                momentum=momentum, weight_decay=weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
    #                              betas=betas, weight_decay=weight_decay)
    train(train_loader, net, loss_fn, optimizer)
    test(test_loader, net, loss_fn)

print("Done!")
torch.save(net.state_dict(), model_path)
print("Saved PyTorch Model State to " + model_path)

# plot and save accuracy and loss curves
plot_curves()
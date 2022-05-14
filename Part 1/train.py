import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from resnet import ResNet18
import torchvision
import torchvision.transforms as transforms
from torchtoolbox.transform import Cutout
from torch.utils.tensorboard import SummaryWriter
from cutmix import cutmix, CutMixCriterion
from mixup import mixup, MixUpCriterion

train_size = 50000
epochs = 200
batch_size = 100
batches_to_check = 20
initial_learning_rate = 1e-2
learning_rate_decay = 0.98
momentum = 0.9
betas = (0.9, 0.999)
weight_decay = 1e-3

train_loss_curve = []
test_loss_curve = []
train_accuracy_curve = []
test_accuracy_curve = []

cut_out = False
cut_mix = False
mix_up = True

if cut_out:
    model_path = 'model_cutout.pth'
elif cut_mix:
    model_path = 'model_cutmix.pth'
elif mix_up:
    model_path = 'model_mixup.pth'
else:
    model_path = 'model.pth'

accuracy_curve_path = 'accuracy6.png'
loss_curve_path = 'loss6.png'

cutmix_criterion = CutMixCriterion(reduction='mean')
mixup_criterion = MixUpCriterion(reduction='mean')

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    # train
    for batch, (X, y) in enumerate(dataloader, start=1):
        X, y = X.to(device), y.to(device)
        model.train()

        # Compute forward and calculate prediction error
        if cut_mix:
            r = np.random.rand(1)
            if r < 0.5:
                X, targets = cutmix(batch=(X, y), alpha=1.0)
                yhat = model(X)
                loss = cutmix_criterion(yhat, targets)
            else:
                yhat = model(X)
                loss = loss_fn(yhat, y)
        elif mix_up:
            X, targets = mixup(batch=(X, y), alpha=0.2)
            yhat = model(X)
            loss = mixup_criterion(yhat, targets)
        else:
            # using baseline or cutout
            yhat = model(X)
            loss = loss_fn(yhat, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # optimize
        optimizer.step()

        # print intermediate results
        if batch % batches_to_check == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    # evaluate loss and accuracy on train set
    num_batches = len(dataloader)
    model.eval()
    train_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            yhat = model(X)

            train_loss += loss_fn(yhat, y).item()
            correct += (yhat.argmax(1) == y).type(torch.float).sum().item()

    train_loss /= num_batches
    correct /= size
    print(f"Train Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {train_loss:>8f} \n")
    train_loss_curve.append(train_loss)
    train_accuracy_curve.append(100 * correct)


def test(dataloader, model, loss_fn):
    # evaluate loss and accuracy on test set
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            yhat = model(X)
            # y_one_hot = F.one_hot(y, num_classes=10)
            test_loss += loss_fn(yhat, y).item()
            # test_loss += loss_fn(yhat, y_one_hot).item()
            correct += (yhat.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    test_loss_curve.append(test_loss)
    test_accuracy_curve.append(100 * correct)

def plot_curves():
    # plot curves and save
    # total_num_of_check = int(epochs * train_size / (batch_size * batches_to_check))
    # x = batches_to_check * (np.arange(total_num_of_check) + 1)
    x = np.arange(epochs) + 1
    y = np.asarray(train_loss_curve)
    z = np.asarray(test_loss_curve)
    r = np.asarray(train_accuracy_curve)
    w = np.asarray(test_accuracy_curve)

    f1 = plt.figure(1)
    plt.xlabel('num of epochs')
    plt.ylabel('loss')
    plt.plot(x, y, label='Loss on train set')
    plt.plot(x, z, label='Loss on test set')
    plt.legend()
    plt.savefig(loss_curve_path)

    f2 = plt.figure(2)
    plt.xlabel('num of epochs')
    plt.ylabel('accuracy')
    plt.plot(x, r, label='Accuracy on train set')
    plt.plot(x, w, label='Accuracy on test set')
    plt.legend()
    plt.savefig(accuracy_curve_path)


if __name__ == '__main__':

    # data preprocessing and augmentation
    if cut_out:
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              Cutout(0.5),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    else:
        # baseline method
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

    # load the model
    model = ResNet18().to(device)
    # if there's a model, then load it
    # model.load_state_dict(torch.load(model_path))

    loss_fn = nn.CrossEntropyLoss()

    # train
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        learning_rate = initial_learning_rate * (learning_rate_decay ** t)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                    momentum=momentum, weight_decay=weight_decay)
        train(train_loader, model, loss_fn, optimizer)
        test(test_loader, model, loss_fn)
    print("Done!")

    # save model
    torch.save(model.state_dict(), model_path)
    print("Saved PyTorch Model State to " + model_path)

    # plot and save accuracy and loss curves
    plot_curves()
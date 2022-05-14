import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torchtoolbox.transform import Cutout
from mixup import mixup, MixUpCollator
from cutmix import cutmix, CutMixCollator
from resnet import ResNet18

model_path = 'model.pth'
model_cutout_path = 'model_cutout.pth'
model_cutmix_path = 'model_cutmix.pth'
model_mixup_path = 'model_mixup.pth'

cutmix_collator = CutMixCollator(alpha=1.0)
mixup_collator = MixUpCollator(alpha=0.2)

def save_sample_img(n, dataloader, model, name, type):

    for batch in dataloader:

        images, labels = batch
        if type == 'cutmix':
            images, targets = cutmix(batch, alpha=1.0)
        elif type == 'mixup':
            images, targets = mixup(batch, alpha=0.2)

        # save input image
        base_directory = name + '.png'

        torchvision.utils.save_image(images, base_directory, nrow=n, padding=2)
        hidden_outs = model.get_hidden_outputs(images)

        # save hidden layers' image (just choose one feature)
        i = 1
        for hidden_out in hidden_outs:
            hidden_out_path = name + 'hidden' + str(i) + '.png'
            example_hidden_out = hidden_out[:, 0:2, :, :]
            torchvision.utils.save_image(example_hidden_out, hidden_out_path, nrow=n, padding=2)
            i += 1
        break

    return


if __name__ == '__main__':
    model = ResNet18()
    model.load_state_dict(torch.load(model_mixup_path))
    model.eval()

    # show network structure
    total_params = sum([param.nelement() for param in model.parameters()])
    print(total_params)

    # cutout
    batch_size = 3
    # transform = transforms.Compose([Cutout(1),
    #                                 transforms.ToTensor()])
    # cutmix
    transform = transforms.ToTensor()
    train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    base_name = 'sample_mixup'
    save_sample_img(n=batch_size, dataloader=train_loader, model=model, name=base_name, type='mixup')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from utils import set_seed, device


def get_dataloaders(dataset_name="CIFAR10",batch_size=128,remove_idx=None):
    if dataset_name == "CIFAR10":
        tr = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomCrop(32,4), transforms.ToTensor()])
        te = transforms.Compose([transforms.ToTensor()])
        train = datasets.CIFAR10('./data',train=True,download=True,transform=tr)
        test = datasets.CIFAR10('./data',train=False,download=True,transform=te)
    elif dataset_name == 'MNIST':
            tr = transforms.Compose([transforms.ToTensor()])
            train = datasets.MNIST('./data', train=True, download=True, transform=tr)
            test = datasets.MNIST('./data', train=False, download=True, transform=tr)
    else:
        raise NotImplementedError

    if remove_idx is not None:
        mask = [i for i in range(len(train)) if i not in set(remove_idx)]
        train = Subset(train, mask)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader


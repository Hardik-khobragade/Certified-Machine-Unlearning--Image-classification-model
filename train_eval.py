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

def build_model(num_classes=10):
    m= models.resnet18(pretrained=False)
    m.fc =nn.Linear(m.fc.in_features , num_classes)
    return m

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x,y in loader:
        x,y= x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        _,pred = out.max(1)
        correct += (pred==y).sum().item()
        total += x.size(0)
    return total_loss/total, correct/total

def evaluation(model, loader, device):
    model.eval()
    total=0
    correct=0
    losses=[]
    criterion = nn.CrossEntropyLoss(reduction='none')
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out,y)
            _,pred = out.max(1)
            correct += (pred==y).sum().item()
            total += x.size(0)
            losses.append(loss.cpu())
    return correct/total, torch.cat(losses)
            
    


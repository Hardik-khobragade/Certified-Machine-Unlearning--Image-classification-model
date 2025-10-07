import torch, time
from train_eval import get_dataloaders, build_model, train_one_epoch, evaluate
from utils import set_seed, device


def retrain(dataset="CIFAR10", remove_idx=None, epochs=50, seed=0, save_path=None):
    set_seed(seed)
    dev = device()
    train_loader, test_loader = get_dataloaders(dataset, remove_idx=remove_idx)
    model = build_model().to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    schedular = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss()
    start = time.time()
    for e in range(epochs):
        train_one_epoch(model, train_loader, opt, criterion, dev)
        schedular.step()
        
    t= time.time() - start
    acc,_= evaluate(model,test_loader,dev)
    if save_path: torch.save(model.state_dict(),save_path)
    return model, acc, t

if __name__ == "__main__":
    model, acc, t = retrain(epochs=5)
    print("Acc", acc, "Time(s)", t)
    
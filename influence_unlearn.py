import torch
from torch.autograd import grad
from train_eval import build_model, get_dataloaders, evaluate
from utils import device, set_seed
import time

def hvp(model, loss, params, v):
    grads = grad(loss, params, create_graph = True)
    grad_dot_v = 0
    for g, vi in zip(grads, v):
        grad_dot_v += (g * vi).sum()
    hv = grad(grad_dot_v, params, retain_graph=True)
    return tuple(h.detach() for h in hv)

def lissa_inverse_hvp(model, train_loader, v, damping=0.01, scale=10, recursion_depth=100, tol=1e-5):
    # Approximates H^{-1} v using iterative method on batches sampled from train_loader
    params = [p for p in model.parameters() if p.requires_grad]
    inv_v = [torch.zeros_like(p) for p in params]
    cur_est = [vi.clone().to(next(model.parameters()).device) for vi in v] # initial estimate = v
    for i in range(recursion_depth):
        try:
            x,y = next(loader_iter)
        except:
            loader_iter = iter(train_loader)
            x,y = next(loader_iter)
        x,y = x.to(device(),y.to(device()))
        model.zero_grad()
        out = model(x)
        loss = torch.nn.functional.cross_entropy(out, y)
        hv = hvp(model, loss, params, cur_est)
        with torch.no_grad():
            new = []
            for ce, hv_i, vi in zip(cur_est, hv, v):
                new.append(vi + (1 - damping) * ce - hv_i / scale)
            cur_est = new
    return cur_est

def compute_grad_on_set(model, dataset_loader):
    params = [p for p in model.paraeters() if p.requires_grad]
    grads = [torch.zeros_like(p) for p in params]
    model.zero_grad()
    for x,y in dataset_loader:
        x,y = x.to(device()), y.to(device())
        out = model(x)
        loss = torch.nn.functional.cross_entropy(out, y)
        g = torch.autograd.grad(loss, params)
        for i,gi in enumerate(g):
            grads[i] += gi.detach().cpu()
            
        return [g.to(device()) for g in grads]

def approx_unlearn(model, full_train_loader, removed_loader, damping=0.01):
    # 1) compute v = sum_grad_removed
    v = compute_grad_on_set(model, removed_loader)
    # 2) approximate H^{-1} v
    inv_hv = lissa_inverse_hvp(model, full_train_loader, v, damping=damping, recursion_depth=200)
    # 3) update parameters theta <- theta - inv_hv
    with torch.no_grad():
        for p, delta in zip([p for p in model.parameters() if p.requires_grad], inv_hv):
            p -= delta
        return model

if __name__ == "__main__":
    set_seed(0)
    train_loader, test_loader = get_dataloaders("CIFAR10", batch_size=128)
    model = build_model().to(device())
    # assume model pre-trained loaded
    # build loaders for removed set (small subset)
    # removed_loader = DataLoader(subset_of_train, batch_size=32)
    # model = approx_unlearn(model, train_loader, removed_loader)
    
    

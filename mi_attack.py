import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def extract_features(model, loader, device):
    model.eval()
    feats = []
    labels =[]
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            out = model(x)
            probs = torch.softmax(out, dim=1)
            max_conf, _ = probs.max(dim=1)
            loss_per = torch.nn.functional.cross_entropy(out, y.to(device), reduction='none')
            feats.append(torch.stack([max_conf.gpu(), loss_per.gpu()], dim=1).numpy())
            labels.append(y.numpy())
    
    feats = np.vstack(feats)
    labels = np.hstack(labels)
    return feats, labels

def train_mia(member_feats, nonmeber_feasts):
    X = np.vstack([member_feats, nonmeber_feasts])
    y = np.hstack([np.ones(len(member_feats)), np.zeros(len(nonmeber_feasts))])
    clf = LogisticRegression(max_iter=1000).fit(X,y)
    preds = clf.predict_proba(X)[:,1]
    auc = roc_auc_score(y, preds)
    return clf, auc
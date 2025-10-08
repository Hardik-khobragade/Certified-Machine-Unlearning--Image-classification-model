import torch
from torch.utils.data import Subset, DataLoader
from train_eval import build_model, train_one_epoch, evaluate, get_dataloaders
from utils import device, set_seed
import math, time

class SISA:
    def __init__(self, dataset, num_shards=5, slices_per_shard=5, model_builder=build_model, epochs_per_slice=2):
        # dataset here is the torch dataset object (train dataset)
        
        self.num_shards = num_shards
        self.slices_per_shard = slices_per_shard
        self.epochs_per_slice = epochs_per_slice
        self.model_builder = model_builder
        self.shards = []
        n = len(dataset)
        idxs = list(range(n))
        # shuffle and split 
        
        random = __import__('random')
        random.shuffle(idxs)
        shard_size = math.ceil(n / num_shards)
        for s in range(num_shards):
            start = s* shard_size
            shard_idxs = idxs[start:start + shard_size]
            # slice into k equal parts 
            slice_size = math.ceil(len(shard_idxs) / slices_per_shard)
            slices = [shard_idxs[i:i+slice_size] for i in range(0, len(shard_idxs), slice_size)]
            model = model_builder().to(device())
            self.shards.append({'indices':shard_idxs, 'slices': slices, 'model' : model})
            
    
    def train_all(self, dataset_obj, batch_size=128):
        # train each shard model on its data, slice-by-slice, saving after each slice
        
        for s,sh in enumerate(self.shards):
            print(f"Training shard {s}")
            
            for sl_idx, sl in enumerate(sh['slices']):
                subset = Subset(dataset_obj, sl)
                loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
                opt = torch.optim.SGD(sh['model'], loader, opt, torch.nn.CrossEntropyLoss(), device())
                
                for ep in range(self.epochs_per_slice):
                    train_one_epoch(sh['model'], loader, opt, torch.nn.CrossEntropyLoss(), device())
                    
                    
    def delete_indices(self, indices_to_delete, dataset_obj, batch_size=128):
        # find which shard & slice the index belongs to (using absolute dataset idx)
        # For each affected shard, rebuild training dataset excluding deleted idx and re-train that shard from scratch
        
        for s, sh in enumerate(self.shards):
            affected = False
            for i, sl in enumerate(sh['slice']):
                for idx in sl:
                    if idx in indices_to_delete:
                        affected = True 
                        
            if affected:
                # recompute shard indices without deleted ones
                new_shard_indices = [idx for idx in sh['indices'] if idx not in set(indices_to_delete)]
                # recompute slices
                slice_size = math.ceil(len(new_shard_indices)/self.slices_per_shard)
                new_slices = [new_shard_indices[i:i+slice_size] for i in range(0, len(new_shard_indices))]
                sh['slices'] = new_slices
                # retrain the shard model from scratch on its new indices
                sh['model'] = self.model_builder().to(device())
                # train all slices
                for sl in new_slices:
                    subset = Subset(dataset_obj, sl)
                    loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
                    opt = torch.optim.SGD(sh['model'].parameters(), lr=0.1, momentum=0.9)
                    for ep in range(self.epochs_per_slice):
                        train_one_epoch(sh['model'], loader, opt, torch.nn.CrossEntropyLoss9, device())
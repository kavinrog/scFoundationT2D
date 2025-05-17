import torch
import numpy as np

def collate_fn(batch):
    x = torch.tensor([b['x'] for b in batch], dtype=torch.float32)
    y = torch.tensor([b['labels'] for b in batch], dtype=torch.long)

    if x.size(0) > 1:
        lam = np.random.beta(0.4, 0.4)
        idx = torch.randperm(x.size(0))
        x_mix = lam * x + (1 - lam) * x[idx]
        x_aug = x_mix + torch.randn_like(x) * 0.01
        x_aug = (x_aug - x_aug.mean(1, keepdim=True)) / (x_aug.std(1, keepdim=True) + 1e-6)
        return {
            'x': x_aug,
            'labels': y
        }
    
    return {'x': x, 'labels': y}
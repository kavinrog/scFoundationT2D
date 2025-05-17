import torch
import torch.nn as nn
import math

class Base(nn.Module):
    def __init__(self, in_size, hd):
        super().__init__()
        self.l1 = nn.Linear(in_size, hd)
        self.l2 = nn.Linear(hd, hd)
        self.act = nn.ReLU()
        self.out = nn.Linear(hd, in_size)

    def forward(self, x):
        x = x.to(next(self.parameters()).device) 
        return self.out(self.act(self.l2(self.act(self.l1(x)))))

def load_base_model(input_size, hidden_dim, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('state_dict', ckpt)
    base = Base(input_size, hidden_dim)
    base.load_state_dict(state, strict=False)
    base.eval()
    return base
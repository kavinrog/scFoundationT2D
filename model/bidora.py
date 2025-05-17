import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BiDoRALinear(nn.Module):
    def __init__(self, in_f, out_f, r):
        super().__init__()
        self.register_parameter('base_weight', nn.Parameter(torch.empty(out_f, in_f), requires_grad=False))
        self.A = nn.Parameter(torch.randn(r, in_f) * 0.01)
        self.B = nn.Parameter(torch.randn(out_f, r) * 0.01)
        self.m = nn.Parameter(torch.ones(out_f, 1))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

    def forward(self, x):
        W = self.base_weight + self.m * (self.B @ self.A)
        return F.linear(x, W)
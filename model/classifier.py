import torch.nn as nn
from .bidora import BiDoRALinear
from .loss import FocalLoss
import torch

class CustomHead(nn.Module):
    def __init__(self, in_f, hid, num_labels, r):
        super().__init__()
        self.fc1 = nn.Linear(in_f, hid)
        self.fc2 = nn.Linear(hid, hid)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.bidora = BiDoRALinear(hid, num_labels, r)

    def forward(self, x):
        x1 = self.act(self.fc1(x))
        x2 = self.act(self.fc2(x1))
        return self.bidora(self.drop(x2 + x1))

class BiDoRAClassifier(nn.Module):
    def __init__(self, base, input_size, num_labels, r=4, hid=256):
        super().__init__()
        self.base = base
        self.head = CustomHead(input_size, hid, num_labels, r)
        self.loss_fn = FocalLoss()

    def forward(self, x, labels=None):
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(next(self.parameters()).device)
        feat = self.base(x)
        logits = self.head(feat)
        if labels is not None:
            return {'loss': self.loss_fn(logits, labels), 'logits': logits}
        return {'logits': logits}
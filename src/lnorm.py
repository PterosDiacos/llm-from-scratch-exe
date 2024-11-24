import torch
import torch.nn as nn


class LayerMinmaxNorm(nn.Module):
    def __init__(self, emb_dim, *args):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(emb_dim) * 2.)
        self.shift = nn.Parameter(torch.ones(emb_dim) * -1.)

    def forward(self, x):
        lo = x.min(dim=-1, keepdim=True).values
        hi = x.max(dim=-1, keepdim=True).values
        return (x - lo) / (hi - lo) * self.scale + self.shift


class LayerStdNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mu = x.mean(dim=-1, keepdim=True)
        sig = x.var(dim=-1, keepdim=True, unbiased=False) + self.eps
        return (x - mu) / torch.sqrt(sig) * self.scale + self.shift

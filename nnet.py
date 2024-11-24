import torch
import torch.nn as nn
import torch.nn.functional as F
from attn import *
from hashenc import HPEmbedding
from lnorm import LayerStdNorm as LayerNorm
from utils import export


registry = []


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg['emb_dim'], cfg['ff_upscale'] * cfg['emb_dim']),
            nn.GELU(approximate='tanh'),
            nn.Linear(cfg['ff_upscale'] * cfg['emb_dim'], cfg['emb_dim']))

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MHA(
            d_in=cfg['emb_dim'],
            d_out=cfg['emb_dim'],
            context_len=cfg['context_len'],
            n_heads=cfg['n_heads'], 
            dropout=cfg['drop_attn_rate'],
            qkv_bias=cfg['qkv_bias'])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        self.drop_skip = nn.Dropout(cfg['drop_skip_rate'])

    def forward(self, x):
        x = x + self.drop_skip(self.att(self.norm1(x)))
        x = x + self.drop_skip(self.ff(self.norm2(x)))
        return x


@export(registry)
class GPTishModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_len'], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_emb_rate'])
        self.trf_blocks = nn.Sequential(*(TransformerBlock(cfg) for _ in range(cfg['n_layers'])))
        self.final_norm = LayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)

    def forward(self, inp):
        batch_size, seq_len = inp.shape
        x = self.tok_emb(inp) + self.pos_emb(torch.arange(seq_len, device=inp.device))
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        return self.out_head(self.final_norm(x))

    def loss_fn(self, out, tar):
        return F.cross_entropy(out.flatten(0, 1), tar.flatten())

    def to_probs(self, out, tau=1.0):
        return torch.softmax(out / tau, dim=-1)


@export(registry)
class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = nn.Linear(in_dim, rank, bias=False)
        self.B = nn.Linear(rank, out_dim, bias=False)
        nn.init.zeros_(self.B.weight)
        self.alpha = alpha

    def forward(self, x):
        return self.alpha * self.B(self.A(x))


@export(registry)
class LoRALinear(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )
    
    def forward(self, x):
        return self.linear(x) + self.lora(x)


@export(registry)
class GPTishBinEnc(nn.Module):
    '''... to explore the loss function'''
    def __init__(self, cfg, hmatrix, pmatrix):
        super().__init__()
        self.register_buffer('hmatrix', hmatrix)
        self.emb = HPEmbedding(cfg['emb_dim'], hmatrix, pmatrix)
        self.drop_emb = nn.Dropout(cfg['drop_emb_rate'])
        self.trf_blocks = nn.Sequential(*(TransformerBlock(cfg) for _ in range(cfg['n_layers'])))
        self.final_norm = LayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'], hmatrix.size(1), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp):
        x = self.drop_emb(self.emb(inp))
        x = self.trf_blocks(x)
        x = self.out_head(self.final_norm(x))
        return self.sigmoid(x)

    def loss_fn(self, out, tar):
        tar = self.hmatrix[tar.flatten()]
        return F.binary_cross_entropy(out.flatten(0, 1), tar, reduction='mean')
    
    def to_probs(self, out, tau=1.0):
        sim = F.normalize(out, dim=-1) @ F.normalize(self.hmatrix, dim=-1).T
        return torch.softmax(sim / tau, dim=-1)


__all__ = registry

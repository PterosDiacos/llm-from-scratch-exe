import torch
import torch.nn as nn


class MHA(nn.Module):
    def __init__(self, d_in, d_out, 
                 context_len, dropout, n_heads, qkv_bias=False, **kwargs):
        super().__init__()
        assert d_out % n_heads == 0
        self.d_out = d_out
        self.n_heads = n_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask',
            torch.triu(torch.ones(context_len, context_len), diagonal=1)
        )

    def forward(self, x):
        b, n_tokens, d_in = x.shape
        queries = self.W_query(x).view(b, n_tokens, self.n_heads, -1).transpose(1, 2)
        keys = self.W_key(x).view(b, n_tokens, self.n_heads, -1).transpose(1, 2)
        values = self.W_value(x).view(b, n_tokens, self.n_heads, -1).transpose(1, 2)
        attn_scores = queries @ keys.transpose(-1, -2)

        mask_bool = self.mask.bool()[:n_tokens, :n_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = self.out_proj(context_vec.reshape(b, n_tokens, self.d_out))
        return context_vec


class MHASparse(nn.Module):
    def __init__(self, d_in, d_out, 
                 context_len, dropout, n_heads, qkv_bias=False, **kwargs):
        super().__init__()
        assert d_out % n_heads == 0
        self.d_out = d_out
        self.n_heads = n_heads
        self.W_key   = nn.Linear(d_in, d_in, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask',
            torch.triu(torch.ones(context_len, context_len), diagonal=1)
        )

    def forward(self, x):
        b, n_tokens, d_in = x.shape
        queries = x.view(b, n_tokens, self.n_heads, -1).transpose(1, 2)
        keys = self.W_key(x).view(b, n_tokens, self.n_heads, -1).transpose(1, 2)
        values = self.W_value(x).view(b, n_tokens, self.n_heads, -1).transpose(1, 2)
        attn_scores = queries @ keys.transpose(-1, -2)

        mask_bool = self.mask.bool()[:n_tokens, :n_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = self.out_proj(context_vec.reshape(b, n_tokens, self.d_out))
        return context_vec


class MHAPosSemidef(MHASparse):
    def forward(self, x):
        b, n_tokens, d_in = x.shape
        queries = self.W_key(x).view(b, n_tokens, self.n_heads, -1).transpose(1, 2)
        keys = self.W_key(x).view(b, n_tokens, self.n_heads, -1).transpose(1, 2)
        values = self.W_value(x).view(b, n_tokens, self.n_heads, -1).transpose(1, 2)
        attn_scores = queries @ keys.transpose(-1, -2)

        mask_bool = self.mask.bool()[:n_tokens, :n_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = self.out_proj(context_vec.reshape(b, n_tokens, self.d_out))
        return context_vec

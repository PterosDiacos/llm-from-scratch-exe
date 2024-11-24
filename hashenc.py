import hashlib
import tiktoken
import torch
import torch.nn as nn


def bytes2hash(b, n=32) -> bytes:
    return hashlib.shake_256(b).digest(n // 8)


def str2hash(s, n=32) -> bytes:
    return bytes2hash(s.encode('utf-8'), n)


def ints2bytes(xs, n=32):
    return [int(x).to_bytes(n // 8, 'big') for x in xs]


def hash2bi(bs) -> str:
    return ''.join(format(b, '08b') for b in bs)


def bytes2hashbi(bs, n=32):
    return [hash2bi(bytes2hash(b, n)) for b in bs]


def str2arr(s) -> list:
    return list(map(int, s))


def get_idx2bi(idx2bytes, n=32):
    def idx2bi(xs):
        bs = idx2bytes(xs)
        hs = bytes2hashbi(bs, n=n)
        return [str2arr(h) for h in hs]
    return idx2bi


def get_hp_matrices(tok, idx_bits, pos_bits, context_len):
    idx2bi = get_idx2bi(tok.decode_tokens_bytes, n=idx_bits)
    pos2bi = get_idx2bi(ints2bytes, n=pos_bits)
    hmatrix = torch.tensor(
        idx2bi(list(range(tok.n_vocab))), 
        dtype=torch.float)
    pmatrix = torch.tensor(
        pos2bi(list(range(context_len))), 
        dtype=torch.float)
    return hmatrix, pmatrix


class HPEmbedding(nn.Module):
    '''Hash + position embedding'''
    def __init__(self, emb_dim, hmatrix, pmatrix):
        super().__init__()
        self.emb = nn.Linear(hmatrix.size(1) + pmatrix.size(1), emb_dim, bias=False)
        self.register_buffer('hmatrix', hmatrix)
        self.register_buffer('pmatrix', pmatrix)

    def forward(self, xss):
        *m, n = xss.shape
        inp = self.hmatrix[xss]
        pos = self.pmatrix[torch.arange(n).expand(*m, n)]
        return self.emb(torch.cat([inp, pos], dim=-1))


def self_test():
    tok = tiktoken.get_encoding('gpt2')
    hmatrix, pmatrix = get_hp_matrices(tok, 32, 32, 128)
    emb = HPEmbedding(768, hmatrix, pmatrix)
    ss = torch.tensor([tok.encode('Hello, do you see?')] * 2)
    return emb(ss).shape


if __name__ == '__main__':
    print(self_test())

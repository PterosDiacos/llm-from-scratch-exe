import torch
from math import ceil
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from prompts import alpaca_style
from utils import export


registry = []


@export(registry)
class InstrDataset(Dataset):
    def __init__(self, data, tok):
        self.encoded = tok.encode_batch(''.join(alpaca_style(d)) for d in data)

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        return self.encoded[idx]
    
    @staticmethod
    def _collate(batch, max_len=None, pad_id=0, ignore_index=-100):
        padded = pad_sequence(map(torch.tensor, batch), 
            batch_first=True, 
            padding_value=pad_id)
        
        eos = torch.full((len(batch), 1), pad_id)
        padded = torch.cat([padded, eos], dim=1)
        inp, tar = padded[:, :-1], padded[:, 1:].clone()
        
        mask = tar == pad_id
        first = mask.clone()
        first[:, 1:] = mask[:, 1:] & ~mask[:, :-1]
        tar[mask & ~first] = ignore_index
        return inp[:, :max_len], tar[:, :max_len]


@export(registry)
class SmsDataset(Dataset):
    def __init__(self, df, tok, max_len=None, pad_id=0):
        self.label = torch.tensor(df['label'].tolist())
        encoded = tok.encode_batch(df['text'])
        encoded = pad_sequence(map(torch.tensor, encoded), 
                batch_first=True, 
                padding_value=pad_id)[:, :max_len]
        if max_len and max_len > encoded.size(1):
            e = torch.full((encoded.size(0), max_len - encoded.size(1)), pad_id)
            encoded = torch.cat([encoded, e], dim=1)
        else:
            max_len = encoded.size(1)
        self.max_len = max_len
        self.encoded = encoded

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.encoded[idx], self.label[idx]


@export(registry)
class TxtDataset(Dataset):
    def __init__(self, txt: str, enc, max_len: int, stride: int):
        self.max_len = max_len
        self.stride = stride
        self.tok_ids = torch.tensor(enc(txt))
        self.n_chunks = ceil((len(self.tok_ids) - max_len) / stride)

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        lo = idx * self.stride
        hi = lo + self.max_len
        return self.tok_ids[lo:hi], self.tok_ids[lo + 1:hi + 1]


__all__ = registry

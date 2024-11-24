#%%
import json
import pandas as pd
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from collections import namedtuple
from dataload import *
from hashenc import get_hp_matrices
from nnet import *
from torch.utils.data import DataLoader
from train import *
from utils import *


def text_generation_demo(model):
    model.eval()
    inp = torch.tensor([enc(starter)])
    s = gen_text_argmax(model=model, 
        idx=inp, max_n_tokens=10, context_len=cfg['context_len'] )
    print(dec(s.flatten().tolist()))


def get_pretraining_dataloaders(path):
    args = {'enc':enc, 
            'max_len':cfg['context_len'], 
            'stride':cfg['context_len'] }
    args1 = {'shuffle': True, 
             'drop_last': True, 
             'batch_size': 2}
    Data = namedtuple('Data', ['train', 'val'])
    text = Data(*train_val_split(open(path).read()))
    return Data(
        DataLoader(TxtDataset(text.train, **args), **args1),
        DataLoader(TxtDataset(text.val, **args), **args1)
    )


def model_pretraining_demo(
        path='texts/the-verdict.txt', 
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
    loader = get_pretraining_dataloaders(path)
    model = GPTishModel(cfg)
    model.to(device)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=4e-4,
        weight_decay=cfg['weight_decay'])
    tracking = train_model_simple(
        model, loader,
        optimizer=optim,
        n_epochs=10, eval_freq=5, eval_iter=5,
        callbacks=[print_gen_sample],
        enc=enc, dec=dec, starter=starter)
    return model, tracking


def binenc_pretraining_expriment(
        path='texts/the-verdict.txt', 
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
    loader = get_pretraining_dataloaders(path)
    mats = get_hp_matrices(tok, 128, 128, cfg['context_len'])
    model = GPTishBinEnc(cfg, *mats)
    model.to(device)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=4e-4,
        weight_decay=cfg['weight_decay'])
    tracking = train_model_simple(
        model, loader,
        optimizer=optim,
        n_epochs=10, eval_freq=5, eval_iter=5,
        callbacks=[print_gen_sample],
        enc=enc, dec=dec, starter=starter)
    return model, tracking


def get_classification_dataloaders(path):
    pad_id, = tok._special_tokens.values()
    args = {'pad_id': pad_id, 'tok': tok}
    args1 = {'batch_size': 8}

    Data = namedtuple('Data', ['train', 'val', 'test'])
    df = pd.read_csv(path, sep='\t')
    take = lambda s: df[df.split == s]
    train = SmsDataset(take('train'), max_len=None, **args)
    data = Data(train, 
        SmsDataset(take('val'), max_len=train.max_len, **args),
        SmsDataset(take('test'), max_len=train.max_len, **args))
    return Data(DataLoader(data.train, shuffle=True, drop_last=True, **args1),
                DataLoader(data.val, **args1),
                DataLoader(data.test, **args1))


def get_classification_model(model, n_clss):
    set_req_grad(model, False)
    set_req_grad(model.final_norm, True)
    set_req_grad(model.trf_blocks[-1], True)
    model.out_head = nn.Linear(model.out_head.in_features, n_clss)
    model.loss_fn = lambda y, t: F.cross_entropy(y[:, -1 ,:], t)
    return model


def classification_finetuning_demo(
        path='texts/sms-spam/sms-spam-balanced.tsv', 
        model_state='gpt2-124m.pth',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
    loader = get_classification_dataloaders(path)
    model = GPTishModel(cfg)
    assign_weight(model, 
        torch.load(model_state, weights_only=True),
        n_layers=cfg['n_layers'],
        attn_d_out=cfg['emb_dim'])
    model = get_classification_model(model, 2)
    model.to(device)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=5e-5,
        weight_decay=cfg['weight_decay'])
    tracking = train_model_simple(
        model, loader,
        optimizer=optim,
        n_epochs=5, eval_freq=50, eval_iter=5,
        callbacks=[print_epoch_acc], n_batches=5)
    return model, tracking


def get_instruction_dataloaders(path):
    pad_id, = tok._special_tokens.values()
    args = {'batch_size': 8,
            'collate_fn': lambda b: InstrDataset._collate(
                b, max_len=cfg['context_len'], pad_id=pad_id
            )}
    
    Data = namedtuple('Data', ['train', 'val', 'test'])
    data = Data(*train_val_split(json.load(open(path)), 0.85, 0.1))
    ds = Data(InstrDataset(data.train, tok),
              InstrDataset(data.val, tok),
              InstrDataset(data.test, tok))
    return Data(DataLoader(ds.train, shuffle=True, drop_last=True, **args),
                DataLoader(ds.val, **args),
                DataLoader(ds.test, **args))


def instruction_finetuning_demo(
        path='texts/instructions.json',
        model_state='gpt2-355m.pth',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
    loader = get_instruction_dataloaders(path)
    model = GPTishModel(cfg)
    assign_weight(model,
        torch.load(model_state, weights_only=True),
        n_layers=cfg['n_layers'],
        attn_d_out=cfg['emb_dim'])
    model.to(device)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=5e-5,
        weight_decay=cfg['weight_decay'])
    tracking = train_model_simple(
        model, loader,
        optimizer=optim,
        n_epochs=2, eval_freq=5, eval_iter=5,
        callbacks=[print_gen_sample],
        enc=enc, dec=dec, starter=dec(loader.val.dataset[0][:45]))
    return model, tracking


def get_lora_classification_model(model, n_clss):
    model.out_head = nn.Linear(model.out_head.in_features, n_clss)
    model.loss_fn = lambda y, t: F.cross_entropy(y[:, -1 ,:], t)
    set_req_grad(model, False)
    replace_linear_with_lora(model, LoRALinear, rank=16, alpha=16)
    return model


if __name__ == '__main__':
    torch.manual_seed(123)
    starter = 'Every effort moves you'
    cfg = yaml.safe_load(open('cfg.yml'))['GPT_124M']
    tok = tiktoken.get_encoding('gpt2')
    enc = lambda s: tok.encode(s, allowed_special='all')
    dec = tok.decode

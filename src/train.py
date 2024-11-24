import torch
from utils import export, gen_text_argmax


registry = []


def get_device(model):
    return next(model.parameters()).device


def running_loss(avg, loss, i):
    return avg + (loss.item() - avg) / (i + 1) 


def one_batch_loss(inp, tar, model):
    device = get_device(model)
    inp = inp.to(device)
    tar = tar.to(device)      
    return model.loss_fn(model(inp), tar)


def avg_batch_loss(dataloader, model, n_batches=-1):
    avg_loss = 0.
    n_batches = len(dataloader) if n_batches < 0 else min(n_batches, len(dataloader))
    for i, (inp, tar) in enumerate(dataloader):
        if i >= n_batches: break
        loss = one_batch_loss(inp, tar, model)
        avg_loss = running_loss(avg_loss, loss, i)
    return avg_loss


def avg_batch_acc(model, dataloader, n_batches=-1):
    model.eval()
    device = get_device(model)
    n_hits, n_examples = 0, 0
    n_batches = len(dataloader) if n_batches < 0 else min(n_batches, len(dataloader))
    for i, (inp, tar) in enumerate(dataloader):
        if i >= n_batches: break
        inp = inp.to(device)
        tar = tar.to(device)
        with torch.no_grad():
            out = model(inp)[:, -1, :].argmax(dim=-1)
        n_examples += out.shape[0]
        n_hits += (out == tar).sum().item()
    return n_hits / n_examples


def eval_model(model, loader, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = avg_batch_loss(loader.train, model, eval_iter)
        val_loss = avg_batch_loss(loader.val, model, eval_iter)
    model.train()
    return train_loss, val_loss


@export(registry)
def train_model_simple(model, loader,
                       optimizer, 
                       n_epochs, eval_freq, eval_iter: int, 
                       callbacks=[], **kwargs):
    args = {k:v for k,v in vars().items() if k not in ['callbacks', 'kwargs']}
    train_loss, val_loss = [], []
    token_cnt, example_cnt = [0], [0]
    global_step = -1

    for epoch in range(n_epochs):
        model.train()
        for inp, tar in loader.train:
            optimizer.zero_grad()
            loss = one_batch_loss(inp, tar, model)
            loss.backward()
            optimizer.step()
            
            token_cnt[-1] += inp.numel()
            example_cnt[-1] += inp.size(0)
            global_step += 1
            if global_step % eval_freq == 0:
                res = eval_model(model, loader, eval_iter)
                train_loss.append(res[0])
                val_loss.append(res[1])
                token_cnt.append(token_cnt[-1])
                example_cnt.append(example_cnt[-1])
                print( f'Ep {epoch+1} (Step {global_step:06d}): '
                       f'Train loss {res[0]}, '
                       f'Val loss {res[1]}' )

        for g in callbacks: g(**args, **kwargs)

    return dict(train_loss=train_loss, 
                val_loss=val_loss, 
                token_count=token_cnt[:-1],
                example_count=example_cnt[:-1])


@export(registry)
def print_epoch_acc(model, loader, 
                    n_batches=-1, **kwargs):
    for dl, split in zip([loader.train, loader.val], ['Train', 'Val']):
        acc = avg_batch_acc(model, dl, n_batches)
        print(f'{split} accuracy: {acc:.3f}', end=' ')
    print('')


@export(registry)
def print_gen_sample(model, 
                     enc, dec, starter, 
                     max_n_tokens=50, **kwargs):
    model.eval()
    context_len = model.trf_blocks[0].att.mask.size(0)
    inp = torch.tensor([enc(starter)]).to(get_device(model))
    with torch.no_grad():
        tok_ids = gen_text_argmax(
            model=model, idx=inp, 
            max_n_tokens=max_n_tokens,
            context_len=context_len
        ).flatten().tolist()
    print(dec(tok_ids).replace('\n', ' '))
    model.train()


__all__ = registry

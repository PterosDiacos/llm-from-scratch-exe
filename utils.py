import re
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


registry = []
def export(dest):
    def reg(obj):
        dest.append(obj.__name__)
        return obj
    return reg


@export(registry)
def naive_tokenizer(raw_text):
    ss = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    ss = [s.strip() for s in ss if s.strip()]
    return ss


@export(registry)
def train_val_split(data, train_ratio=0.9, test_ratio=0.0): 
    cut = int(len(data) * train_ratio)
    if test_ratio > 0.0:
        cut1 = int(len(data) * test_ratio)
        return data[:cut], data[cut:-cut1], data[-cut1:]
    else:
        return data[:cut], data[cut:]


@export(registry)
def paramcount(module) -> int:
    return sum(p.numel() for p in module.parameters())


@export(registry)
def set_req_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


@export(registry)
def is_spam(text, model, tok, max_len=120):
    max_len = min(max_len, model.pos_emb.weight.size(1))
    pad_id, = tok._special_tokens.values()

    inp = tok.encode(text)[:max_len]
    inp += [pad_id] * (max_len - len(inp))
    inp = torch.tensor([inp]).to(next(model.parameters()).device)

    model.eval()
    with torch.no_grad():
        out = model(inp)[:, -1, :]
    return bool(out.argmax(dim=-1).item())


@export(registry)
def gen_text_argmax(model, idx, 
                    max_n_tokens, context_len): 
    for _ in range(max_n_tokens):
        with torch.no_grad():
            out = model(idx[:, -context_len:])[:, -1, :]

        probs = model.to_probs(out)
        succ = probs.argmax(dim=-1, keepdim=True)
        idx = torch.cat((idx, succ), dim=-1)
    return idx


def keep_topk(t, k=10):
    _, idxs = torch.topk(t, k)
    mask = torch.ones_like(t, dtype=bool)
    mask[torch.arange(idxs.size(0))[:, None], idxs] = 0
    t[mask] = -torch.inf
    return t


@export(registry)
def gen_text(model, idx, 
             max_n_tokens, context_len,
             argmax=True,
             temperature=1.0, top_k=None, eos_id=None):
    for _ in range(max_n_tokens):
        with torch.no_grad():
            out = model(idx[:, -context_len:])[:, -1, :]
        if top_k is not None:
            out = keep_topk(out, k=top_k)
        probs = model.to_probs(out, temperature)        
        succ = ( probs.argmax(dim=-1, keepdim=True)
            if argmax else probs.multinomial(num_samples=1) )
        if succ == eos_id:
            break
        else:
            idx = torch.cat((idx, succ), dim=1)
    return idx


@export(registry)
def assign_weight(gpt2, state, n_layers=12, attn_d_out=768):
    gpt2.out_head.load_state_dict({'weight': state['wte.weight']})
    gpt2.tok_emb.load_state_dict({'weight': state['wte.weight']})
    gpt2.pos_emb.load_state_dict({'weight': state['wpe.weight']})
    gpt2.final_norm.load_state_dict({'scale': state['ln_f.weight']
                                    ,'shift': state['ln_f.bias']})
    for i in range(n_layers):
        ws = torch.split(state[f'h.{i}.attn.c_attn.weight'], attn_d_out, dim=-1)
        bs = torch.split(state[f'h.{i}.attn.c_attn.bias'], attn_d_out, dim=-1)

        gpt2.trf_blocks[i].att.W_query.load_state_dict(
            {'weight': ws[0].T, 'bias': bs[0]})
        gpt2.trf_blocks[i].att.W_key.load_state_dict(
            {'weight': ws[1].T, 'bias': bs[1]})
        gpt2.trf_blocks[i].att.W_value.load_state_dict(
            {'weight': ws[2].T, 'bias': bs[2]})
        gpt2.trf_blocks[i].att.out_proj.load_state_dict(
            {'weight': state[f'h.{i}.attn.c_proj.weight'].T
            ,'bias': state[f'h.{i}.attn.c_proj.bias']})
        
        gpt2.trf_blocks[i].ff.layers[0].load_state_dict(
            {'weight': state[f'h.{i}.mlp.c_fc.weight'].T
            ,'bias': state[f'h.{i}.mlp.c_fc.bias']})
        gpt2.trf_blocks[i].ff.layers[2].load_state_dict(
            {'weight': state[f'h.{i}.mlp.c_proj.weight'].T
            ,'bias': state[f'h.{i}.mlp.c_proj.bias']})
        
        gpt2.trf_blocks[i].norm1.load_state_dict(
            {'scale': state[f'h.{i}.ln_1.weight']
            ,'shift': state[f'h.{i}.ln_1.bias']})
        gpt2.trf_blocks[i].norm2.load_state_dict(
            {'scale': state[f'h.{i}.ln_2.weight']
            ,'shift': state[f'h.{i}.ln_2.bias']})


@export(registry)
def replace_linear_with_lora(module, LoRA, rank, alpha):
    for n, m in module.named_children():
        if isinstance(m, torch.nn.Linear):
            setattr(module, n, LoRA(m, rank, alpha))
        else:
            replace_linear_with_lora(m, LoRA, rank, alpha)


@export(registry)
def plot_losses(n_epochs, tracking, metric='loss', upper_x='token_count'):
    x = torch.linspace(0, n_epochs, len(tracking[f'train_{metric}']))
    fig, ax1 = plt.subplots(figsize=(5, 3))
    
    ax1.plot(x, tracking[f'train_{metric}'], label=f'Train {metric}')
    ax1.plot(x, tracking[f'val_{metric}'], label=f'Val {metric}', linestyle='-.')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twiny()
    ax2.plot(tracking[upper_x], tracking[f'train_{metric}'], alpha=0)
    ax2.set_xlabel(upper_x.capitalize().replace('_', ' '))
    fig.tight_layout()
    plt.show()


__all__ = registry

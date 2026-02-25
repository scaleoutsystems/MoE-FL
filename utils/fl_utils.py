import torch
import numpy as np
from torch.utils.data import SubsetRandomSampler, DataLoader
from dataclasses import dataclass

@dataclass
class Client:
    cid: int
    subset: object                 
    num_samples: int
    loader: DataLoader 

#Allows for batchnorm (not ideal in FL setup)
@torch.no_grad()
def fedavg_batchnorm(client_states, client_num_samples):
    total = sum(client_num_samples)
    weights = [n / total for n in client_num_samples]

    agg = {}
    for k in client_states[0].keys():
        t0 = client_states[0][k]
        if not torch.is_floating_point(t0):
            agg[k] = t0.detach().clone()
            continue

        agg_k = t0.detach().clone() * weights[0]
        for w, sd in zip(weights[1:], client_states[1:]):
            agg_k += sd[k].detach() * w
        agg[k] = agg_k

    return agg

#Assumes we are not using BatchNorm (which is true for convnext)
@torch.no_grad()
def fedavg(client_states, client_num_samples, device=None):
    total = float(sum(client_num_samples))

    weights = [n / total for n in client_num_samples]

    #Ensure work is done on correct device
    device = torch.device(device)

    agg = {}
    for k, t0 in client_states[0].items():
        acc = torch.zeros_like(t0, device=device, dtype=torch.float32)
        for w, sd in zip(weights, client_states):
            acc.add_(sd[k].to(device=device, dtype=torch.float32), alpha=w)
        agg[k] = acc.to(dtype=t0.dtype)

    return agg

#Naive EMA in federated setting
@torch.no_grad()
def ema_update_model(ema_model, model, decay):
    ema_sd = ema_model.state_dict()
    sd = model.state_dict()

    for k, ema_v in ema_sd.items():
        v = sd[k].to(ema_v.device)
        if ema_v.dtype.is_floating_point:
            ema_v.mul_(decay).add_(v, alpha=1.0 - decay)
        else:
            ema_v.copy_(v)
   
def lr_schedule(base_lr, warmup_rounds, total_rounds, start_lr):
    r = np.arange(total_rounds, dtype=np.float32)

    warm_frac = (r + 1) / warmup_rounds
    warm_lr = base_lr * (start_lr + warm_frac * (1.0 - start_lr))

    cos_frac = (r - warmup_rounds) / max(1, (total_rounds - warmup_rounds))
    cos_lr = base_lr * 0.5 * (1.0 + np.cos(np.pi * cos_frac))

    lrs = np.where(r < warmup_rounds, warm_lr, cos_lr)
    return lrs

def set_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg["lr"] = lr

def init_clients(dataset, num_clients, batch_size, dl_kwargs, seed=42, shuffle=True):
    n = len(dataset)
    indices = np.arange(n)

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    splits = np.array_split(indices, num_clients)

    clients = []
    for cid, idxs in enumerate(splits):
        # idxs is disjoint by construction
        sampler = SubsetRandomSampler(idxs)  

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,      
            shuffle=False,     
            **dl_kwargs
        )

        clients.append(Client(
            cid=cid,
            subset=idxs,          
            num_samples=len(idxs),
            loader=loader
        ))

    return clients
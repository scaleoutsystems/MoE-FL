import torch
import numpy as np
from copy import deepcopy
from torch.utils.data import Subset, DataLoader
from dataclasses import dataclass, field

@dataclass
class Client:
    cid: int
    subset: object                 
    num_samples: int
    loader: DataLoader
    opt_kwargs: dict
    opt_state: dict | None = None  
    metrics: dict = field(default_factory=dict)

@torch.no_grad()
def fedavg(client_states, client_num_samples):
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

def init_clients(dataset,num_clients,batch_size,
                opt_kwargs,dl_kwargs,seed=42,shuffle=True):
    n = len(dataset)
    indices = np.arange(n)

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    splits = np.array_split(indices, num_clients)

    clients = []
    for cid, idxs in enumerate(splits):
        subset = Subset(dataset, idxs.tolist())

        loader = DataLoader(subset,batch_size=batch_size,
                            shuffle=True,**dl_kwargs)

        clients.append(Client(cid=cid,subset=subset,
                              num_samples=len(idxs),loader=loader,
                              opt_kwargs=deepcopy(opt_kwargs)))

    return clients
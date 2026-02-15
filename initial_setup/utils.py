import torch
import torch.nn as nn
import numpy as np
import time
from copy import deepcopy
#import matplotlib.pyplot as plt
from torch.utils.data import Subset, DataLoader
from dataclasses import dataclass, field
from torchvision.transforms.v2 import MixUp, CutMix
from torchvision.transforms.v2 import RandomChoice
from fvcore.nn import FlopCountAnalysis, flop_count_table

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

mix_transform = RandomChoice([ 
    MixUp(num_classes=10,alpha=0.8),
    CutMix(num_classes=10,alpha=1.0)
])

def train_client(model, loader, device, optimizer, loss_fn, num_epochs=1, log=True, augment=True):
    model.train()
    history = []

    has_aux = hasattr(model, "get_aux_loss")

    for epoch in range(num_epochs):
        running_base = 0.0
        running_aux = 0.0
        running_total = 0.0
        correct = 0
        total = 0

        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if augment:
                x, y = mix_transform(x, y)

            optimizer.zero_grad(set_to_none=True)

            if has_aux:
                logits, aux = model(x, return_aux=True)
                base_loss = loss_fn(logits, y)
                total_loss = base_loss + aux
                aux_val = aux.detach()
            else:
                logits = model(x)
                base_loss = loss_fn(logits, y)
                total_loss = base_loss
                aux_val = None

            total_loss.backward()
            optimizer.step()

            bs = x.size(0)
            running_base += base_loss.item() * bs
            running_total += total_loss.item() * bs
            if aux_val is not None:
                running_aux += aux_val.item() * bs

            if y.ndim == 2:          # soft labels from MixUp/CutMix
                y_hard = y.argmax(dim=1)
            else:
                y_hard = y
            correct += (logits.argmax(dim=1) == y_hard).sum().item()
            total += bs

        epoch_base = running_base / total
        epoch_total = running_total / total
        epoch_acc = correct / total
        epoch_aux = (running_aux / total) if has_aux else None

        m = {
            "epoch": epoch + 1,
            "total_loss": epoch_total,          
            "acc": epoch_acc,
            "samples": total,
        }
        
        if has_aux:
            m["base_loss"] = epoch_base
            m["aux_loss"] = epoch_aux

        history.append(m)

        if log:
            if epoch_aux is None:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] - "
                    f"loss: {epoch_total:.4f} - acc: {epoch_acc:.4f}"
                )
            else:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] - "
                    f"base loss: {epoch_base:.4f} - aux loss: {epoch_aux:.6f} - "
                    f"total loss: {epoch_total:.4f} - acc: {epoch_acc:.4f}"
                )

    return history

@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = loss_fn(logits, y)

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += bs

    return {
        "loss": total_loss / total_samples,
        "acc": total_correct / total_samples,
        "samples": total_samples,
    }

@torch.no_grad()
def measure_ms_per_sample(model,loader,device,
                          warmup_batches=10,
                          timed_batches=50):
    model.to(device)
    model.eval()

    # Warm-up
    for i, (x, _) in enumerate(loader):
        if i >= warmup_batches:
            break
        x = x.to(device, non_blocking=True)
        model(x)

    times = []
    
    # Timed runs
    for i, (x, _) in enumerate(loader):
        if i >= timed_batches:
            break

        x = x.to(device, non_blocking=True)

        if device == "cuda":
            torch.cuda.synchronize()
        start = time.time()

        model(x)

        if device == "cuda":
            torch.cuda.synchronize()
        end = time.time()

        batch_time = end - start
        times.append(batch_time / x.size(0))  # per-sample time

    times = np.array(times) * 1000  # ms
    return times.mean(), times.std()

def model_param_bytes(model, bytes_per_param=4):
    P = sum(p.numel() for p in model.parameters())
    return P * bytes_per_param, P
    
def fedavg_comm_cost_mb(model, num_clients, num_rounds, bytes_per_param=4):
    model_bytes, P = model_param_bytes(model, bytes_per_param)

    downlink_per_round = num_clients * model_bytes
    uplink_per_round   = num_clients * model_bytes
    total_per_round    = downlink_per_round + uplink_per_round
    total_training     = num_rounds * total_per_round

    to_mb = lambda x: x / 1e6  

    return {
        "Parameters": P,
        "model_MB": to_mb(model_bytes),
        "downlink_per_round_MB": to_mb(downlink_per_round),
        "uplink_per_round_MB": to_mb(uplink_per_round),
        "total_per_round_MB": to_mb(total_per_round),
        "total_training_MB": to_mb(total_training),
    }

def count_flops(model,inputs,show_table=False,max_depth=4,):
    model.eval()
    with torch.no_grad():
        flops = FlopCountAnalysis(model, inputs)
        #This returns MACs (FLOP not well defined according to docs)
        macs = flops.total()
        #Convert to FLOPs
        flops = 2*macs

        if show_table:
            print(flop_count_table(flops, max_depth=max_depth))

    return macs, flops
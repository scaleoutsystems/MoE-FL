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

mix_transform = RandomChoice([
    MixUp(num_classes=10),
    CutMix(num_classes=10)
])

@dataclass
class Client:
    cid: int
    subset: object                 
    num_samples: int
    loader: DataLoader
    opt_kwargs: dict
    opt_state: dict | None = None  # stores optimizer.state_dict() across rounds
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

def train_client(model, loader, device, optimizer, loss_fn, num_epochs=1, log=True, augment=True):
    model.train()
    history = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            if augment:
                x, y = mix_transform(x, y)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            running_loss += loss.item() * bs
            if y.ndim == 2:          # soft labels from MixUp/CutMix: [B, C]
                y_hard = y.argmax(dim=1)
            else:                    # hard labels: [B]
                y_hard = y
            correct += (logits.argmax(dim=1) == y_hard).sum().item()
            total += bs

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        m = {"epoch": epoch + 1, "loss": epoch_loss, "acc": epoch_acc, "samples": total}
        history.append(m)

        if log:
            print(f"Epoch [{epoch+1}/{num_epochs}] - loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f}")

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

def fedavg_comm_cost_bytes(model, num_clients, num_rounds, bytes_per_param=4):
    model_bytes, P = model_param_bytes(model, bytes_per_param)

    downlink_per_round = num_clients * model_bytes
    uplink_per_round   = num_clients * model_bytes
    total_per_round    = downlink_per_round + uplink_per_round

    total_training = num_rounds * total_per_round

    return {
        "Parameters": P,
        "model_bytes": model_bytes,
        "downlink_per_round": downlink_per_round,
        "uplink_per_round": uplink_per_round,
        "total_per_round": total_per_round,
        "total_training": total_training,
    }
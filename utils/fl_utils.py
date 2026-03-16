import torch
import threading
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
import ctypes

#We have to force free the RAM
def release_memory():
    libc = ctypes.CDLL("libc.so.6")
    libc.malloc_trim(0)

#Allows for batchnorm (not ideal in FL setup)
@torch.no_grad()
def fedavg_batchnorm(client_states, client_num_samples,device):
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
#With this implementation most reasonable to do model aggregation
#on CPU as that is where we store the local models (while not training)
@torch.no_grad()
def fedavg(client_states, client_num_samples, device="cpu"):
    total = float(sum(client_num_samples))
    weights = [n / total for n in client_num_samples]
    #Ensure work is done on correct device
    device = torch.device(device)

    agg = {}
    for k, t0 in client_states[0].items():
        acc = torch.zeros_like(t0, device=device, dtype=torch.float32)
        for w, sd in zip(weights, client_states):
            acc.add_(sd[k], alpha=w) 
        agg[k] = acc.to(dtype=t0.dtype)

    return agg


def expert_weighted_fedavg(client_states, client_num_samples, moe_stats, device="cpu"):
    
    #Data per client weights
    total = float(sum(client_num_samples))
    weights = torch.tensor([n / total for n in client_num_samples])
    device = torch.device(device)
    
    #Normalize expert stats: (num_clients, num_layers, num_experts)
    eps = 1e-8
    expert_weights = torch.stack([torch.stack(cs) for cs in moe_stats])
    expert_weights.div_(expert_weights.sum(dim=-1, keepdim=True) + eps)
    
    combined_weights = expert_weights * weights.view(-1, 1, 1)
    #Renormalize across clients so weights sum to 1 per layer-expert
    combined_weights.div_(combined_weights.sum(dim=0, keepdim=True) + eps)

    # Build layer to index mapping for experts
    layer_order = []
    for k in client_states[0].keys():
        if "experts" in k.split("."):
            parts = k.split(".")
            ei = parts.index("experts")
            layer_id = f"{parts[ei-2]}.{parts[ei-1]}"
            if layer_id not in layer_order:
                layer_order.append(layer_id)
    layer_to_idx = {l: i for i, l in enumerate(layer_order)}

    #Build (layer_idx, expert_idx) lookup
    key_info = {}
    for k in client_states[0].keys():
        if "experts" in k.split("."):
            parts = k.split(".")
            ei = parts.index("experts")
            layer_id = f"{parts[ei-2]}.{parts[ei-1]}"
            key_info[k] = (layer_to_idx[layer_id], int(parts[ei+1]))

    agg = {}
    for k, t0 in client_states[0].items():
        acc = torch.zeros_like(t0, device=device, dtype=torch.float32)
        if k in key_info:
            layer_idx, expert_idx = key_info[k]
            for client_idx, (w, sd) in enumerate(zip(weights, client_states)):
                w = combined_weights[client_idx, layer_idx, expert_idx].item()
                acc.add_(sd[k], alpha=w)
        else:
            for w, sd in zip(weights, client_states):
                acc.add_(sd[k], alpha=w)
        agg[k] = acc.to(dtype=t0.dtype)
        
    return agg

@torch.no_grad()
def snapshot_params_fp32(model, device):
    return [p.detach().float().clone().to(device) for p in model.parameters() if p.requires_grad]

def compute_prox_term(model, global_params_fp32, mu):
    # model params (could be fp16/bf16); compute in fp32
    params = [p for p in model.parameters() if p.requires_grad]
    params_fp32 = [p if p.dtype == torch.float32 else p.to(torch.float32) for p in params]

    diffs = torch._foreach_sub(params_fp32, global_params_fp32)
    sq = torch._foreach_mul(diffs, diffs)
    # sum all tensors
    prox = sum(t.sum() for t in sq)
    return 0.5 * mu * prox

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

def lr_schedule(base_lr, start_lr, warmup_rounds, total_rounds, end_lr=0.0):
    r = np.arange(total_rounds + 1, dtype=np.float32)
    lrs = np.zeros_like(r)

    # Warmup
    if warmup_rounds > 0:
        lrs[:warmup_rounds + 1] = np.linspace(start_lr, base_lr, warmup_rounds + 1)

    # Cosine
    T_max = total_rounds - warmup_rounds
    t = np.arange(T_max + 1) / T_max
    lrs[warmup_rounds:] = end_lr + 0.5 * (base_lr - end_lr) * (1 + np.cos(np.pi * t))

    return lrs

def set_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg["lr"] = lr
        
class RawImageSubset(Dataset):
    #Reads raw PIL images from disk, bypassing the dataset's transform
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path, label = self.dataset.imgs[self.indices[idx]]
        img = self.dataset.loader(path)
        return img, label


class InMemoryDataset(Dataset):
    #Holds raw PIL images in RAM, applies transform fresh on every access
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        return self.transform(img), int(label)


class Client:
    def __init__(self, cid, subset, num_samples, dataset, transform, dl_kwargs, batch_size):
        self.cid = cid
        self.subset = subset
        self.num_samples = num_samples
        self._dataset = dataset
        self._transform = transform
        self._dl_kwargs = dl_kwargs
        self._batch_size = batch_size
        self._samples = None
        self._thread = None
        self._error = None

    def prefetch(self):
        self._samples = None
        self._error = None

        def _load():
            try:
                raw_subset = RawImageSubset(self._dataset, self.subset)
                tmp_loader = DataLoader(
                    raw_subset,
                    batch_size=256,
                    num_workers=2,
                    pin_memory=False,
                    prefetch_factor=2,
                    persistent_workers=False,
                    shuffle=False,
                    collate_fn=lambda x: x,
                )
                samples = []
                for batch in tmp_loader:
                    samples.extend(batch)
                self._samples = samples
            except Exception as e:
                self._error = e

        self._thread = threading.Thread(target=_load, daemon=True)
        self._thread.start()

    def get_loader(self):
        self._thread.join()
        if self._error:
            raise RuntimeError(f"Client {self.cid} prefetch failed: {self._error}")
        release_memory()
        ram_dl_kwargs = {
            **self._dl_kwargs,
            "num_workers": 9,
            "prefetch_factor": 2,
            "persistent_workers": True,
        }
        return DataLoader(
            InMemoryDataset(self._samples, self._transform),
            batch_size=self._batch_size,
            shuffle=True,
            **ram_dl_kwargs
        )
        #uncomment for local testing instead of code above
        # subset = Subset(self._dataset, self.subset)
        # return DataLoader(subset, batch_size=self._batch_size, shuffle=True, **self._dl_kwargs)

    def free(self):
        self._samples = None

def init_clients(dataset, num_clients, batch_size, dl_kwargs, transform,
                 seed=42, shuffle=True):
    n = len(dataset)
    indices = np.arange(n)

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    splits = np.array_split(indices, num_clients)

    clients = []
    for cid, idxs in enumerate(splits):
        clients.append(Client(
            cid=cid,
            subset=idxs,
            num_samples=len(idxs),
            dataset=dataset,
            transform=transform,
            dl_kwargs=dl_kwargs,
            batch_size=batch_size,
        ))

    return clients
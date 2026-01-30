import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
#import matplotlib.pyplot as plt
import random
from torchvision.datasets import Imagenette
from torchvision.models import convnext_tiny 
from torchvision.models.convnext import ConvNeXt, CNBlock
from torchvision.ops import StochasticDepth, Permute
from torchvision import transforms
from copy import deepcopy
from utils import fedavg, init_clients, train_client, evaluate, measure_ms_per_sample, fedavg_comm_cost_bytes, lr_schedule, ema_update_model
from timm.loss import SoftTargetCrossEntropy
from functools import partial
from typing import Callable, Optional

from torch.nn import functional as F

#NOTE config convnext
#Last-2
#Experts: 4 or 8
#Top-1 routing (1x1 conv routing)
#MLP ratio 2 or 4
#Load balancing as in paper

class MoEConvNeXtWrapper(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self._aux_modules = [m for m in self.base.modules() if hasattr(m, "aux_loss")]

    def forward(self, x, return_aux: bool = False):
        logits = self.base(x)
        if not return_aux:
            return logits
        return logits, self.get_aux_loss()

    def get_aux_loss(self):
        aux = None
        for m in self._aux_modules:
            a = m.aux_loss
            if a is None:
                continue
            aux = a if aux is None else aux + a
        if aux is None:
            aux = next(self.base.parameters()).new_zeros(())
        return aux
    
#TODO load balancing loss Correct?
#TODO batch prioritized routing correct?
class MoECNBlock(nn.Module): #(Shazeer 2017 aux loss) TODO: should it be swapped to switch loss instead? (for k=1)
    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int,
        mlp_ratio: int,
        layer_scale: float,
        stochastic_depth_prob: float,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True),
            Permute([0, 2, 3, 1]),
            norm_layer(dim),
        )

        self.router = Router(dim, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(dim, mlp_ratio) for _ in range(num_experts)])
        self.permute_back = Permute([0, 3, 1, 2])
        self.aux_loss = torch.tensor(0.0)
        self.capacity_ratio = 0.8

        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        
    def _moe_route(self, x_flat, topi, weights, order):
        T, C = x_flat.shape
        K = topi.shape[1]
        E = len(self.experts)
        device = x_flat.device

        # Build flattened routes in BPR order
        token_idx = order.repeat(K) 

        # expert ids in round order (rank-0 for all tokens, then rank-1, ...)
        expert_id = topi[order].transpose(0, 1).reshape(-1)  

        # gate weights aligned with expert_id
        gate_w = weights[order].transpose(0, 1).reshape(-1, 1) 

        # Group by expert id while preserving BPR order inside each expert
        sort_perm = torch.argsort(expert_id, stable=True)
        token_idx = token_idx[sort_perm]
        expert_id = expert_id[sort_perm]
        gate_w = gate_w[sort_perm]

        Be = int(round((K * T * self.capacity_ratio) / E))
        assert(Be > 0)

        # Find start index of each expert chunk
        counts = torch.bincount(expert_id, minlength=E)
        offsets = torch.cumsum(counts, dim=0)
        starts = offsets - counts

        # Position of each route within its expert chunk
        M = expert_id.numel()
        pos_in_chunk = torch.arange(M, device=device) - starts[expert_id]

        # Keep only first Be routes per expert (highest priority due to BPR ordering)
        keep = pos_in_chunk < Be

        token_idx = token_idx[keep]
        expert_id = expert_id[keep]
        gate_w = gate_w[keep]

        out = x_flat.new_zeros(T, C)

        # Recompute chunk boundaries after dropping overflow routes
        counts = torch.bincount(expert_id, minlength=E)
        offsets = torch.cumsum(counts, dim=0)

        start = 0
        for e, expert in enumerate(self.experts):
            end = int(offsets[e])
            if end == start:
                continue

            idx_e = token_idx[start:end]        # tokens kept for this expert
            x_e = x_flat.index_select(0, idx_e) # gather tokens
            y_e = expert(x_e) * gate_w[start:end]
            out.index_add_(0, idx_e, y_e)
            start = end

        return out

    def forward(self, input):
        x = self.dwconv(input)  # (N, H, W, C)
        N, H, W, C = x.shape
        x_flat = x.reshape(-1, C)  # [T, C]

        topi, weights, priority, aux_loss = self.router(x_flat, return_aux=self.training)
        order = torch.argsort(priority, descending=True)
        out = self._moe_route(x_flat, topi, weights, order)  # [T, C]
        
        if aux_loss is None:
            self.aux_loss = x_flat.new_zeros(())
        else:
            self.aux_loss = aux_loss

        out = out.view(N, H, W, C)
        out = self.permute_back(out)  # (N, C, H, W)

        out = self.layer_scale * out
        out = self.stochastic_depth(out)
        return input + out
    
class Expert(nn.Module):
    def __init__(self, dim, mlp_ratio):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * dim, dim),
        )

    def forward(self, x):
        return self.block(x)
    
def _phi(z):
    # Standard normal CDF
    return 0.5 * (1.0 + torch.erf(z / np.sqrt(2.0)))

class Router(nn.Module):
    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int = 2,
        importance_coeff: float = 0.01,
        load_coeff: float = 0.01,
        eps: float = 1e-9,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.importance_coeff = importance_coeff
        self.load_coeff = load_coeff
        self.eps = eps

        # W_g and W_noise #TODO init as zeros?
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.noise = nn.Linear(dim, num_experts, bias=False)

    @staticmethod
    def _cv_squared(x, eps = 1e-9):
        mean = x.mean()
        var = (x - mean).pow(2).mean()  
        return var / (mean * mean + eps)

    def _importance_loss(self, logits):
        probs = torch.softmax(logits, dim=-1)  
        importance = probs.sum(dim=0)         
        return self.importance_coeff * self._cv_squared(importance, self.eps)

    def _load_loss(self, x, logits):
        B, E = logits.shape
        k = self.top_k

        noise_std = F.softplus(self.noise(x)) + self.eps 

        # kth and (k+1)th largest per token
        top_vals, top_idx = logits.topk(k + 1, dim=-1)    
        v_k = top_vals[:, k - 1]                          
        v_kplus = top_vals[:, k]                         

        # membership mask for top-k (not k+1)
        in_topk = torch.zeros(B, E, device=logits.device, dtype=torch.bool)
        in_topk.scatter_(1, top_idx[:, :k], True)

        kth_excl = torch.where(
            in_topk,
            v_kplus.unsqueeze(-1).expand_as(logits),  # if in top-k, exclude => use (k+1)th
            v_k.unsqueeze(-1).expand_as(logits),      # else use kth
        )

        z = (logits - kth_excl) / noise_std
        P = _phi(z)              
        load = P.sum(dim=0)       
        return self.load_coeff * self._cv_squared(load, self.eps)

    def forward(self, x, return_aux = True):
        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1])     

        logits = self.gate(x2)                  

        probs = torch.softmax(logits, dim=-1) # routing weights w_{e,p}
        priority = probs.max(dim=-1).values     

        topv, topi = torch.topk(logits, k=self.top_k, dim=-1) 
        weights = torch.softmax(topv, dim=-1)                  

        if return_aux and (self.importance_coeff != 0.0 or self.load_coeff != 0.0):
            aux = logits.new_zeros(())
            if self.importance_coeff != 0.0:
                aux = aux + self._importance_loss(logits)
            if self.load_coeff != 0.0:
                aux = aux + self._load_loss(x2, logits)
            aux_loss = aux
        else:
            aux_loss = None

        leading = orig_shape[:-1]
        topi = topi.view(*leading, self.top_k)
        weights = weights.view(*leading, self.top_k)

        
        return topi, weights, priority, aux_loss
    
def make_last_block_moe_factory(num_experts=4, top_k=1, mlp_ratio=4):
    # convnext_tiny stage dims and number of blocks
    stage_depth = {96: 3, 192: 3, 384: 9, 768: 3}

    # which stages should get MoE
    moe_dims = {384, 768}  # last 2 stages

    # Blocks created for each dim
    seen = {d: 0 for d in stage_depth}

    def block(dim, layer_scale, stochastic_depth_prob, norm_layer=None):
        # increment count for this stage dim
        if dim not in seen:
            raise ValueError(f"Unexpected dim={dim}. If not convnext_tiny, update stage_depth.")
        seen[dim] += 1

        is_last_block_in_stage = (seen[dim] == stage_depth[dim])
        use_moe = (dim in moe_dims) and is_last_block_in_stage

        if use_moe:
            return MoECNBlock(
                dim=dim,
                num_experts=num_experts,
                top_k=top_k,
                mlp_ratio=mlp_ratio,
                layer_scale=layer_scale,
                stochastic_depth_prob=stochastic_depth_prob,
                norm_layer=norm_layer,
            )
        else:
            return CNBlock(
                dim=dim,
                layer_scale=layer_scale,
                stochastic_depth_prob=stochastic_depth_prob,
                norm_layer=norm_layer,
            )

    return block

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Running on:", device)

# Hyperparameters
num_clients = 10
num_rounds = 300
local_epochs = 1

train_bs = 32
eval_bs = 64

base_lr = 4e-3
start_lr = 1e-3
warmup_rounds = 20

opt_kwargs = {
    "lr": base_lr,              
    "betas": (0.9, 0.999),
    "weight_decay": 0.05,
}

dl_kwargs = {
    # "num_workers": 4,
    "pin_memory": (device == "cuda"),
    # "persistent_workers": True,  
}

label_smoothing = 0.1

val_transform = transforms.Compose([
    transforms.Resize(236),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

#TODO papers may use more regularization
train_transform = transforms.Compose([
    transforms.Resize(236),
    transforms.CenterCrop(32),
    transforms.RandAugment(num_ops=9, magnitude=5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    transforms.RandomErasing(p=0.25),
])

train = Imagenette(root="data", split="train", download=True, transform=train_transform)
val   = Imagenette(root="data", split="val",   download=True, transform=val_transform)

# Global eval loaders
train_eval_loader = DataLoader(train, batch_size=eval_bs, shuffle=False, **dl_kwargs)
val_loader        = DataLoader(val,   batch_size=eval_bs, shuffle=False, **dl_kwargs)

# Client loaders
clients = init_clients(
    dataset=train,
    num_clients=num_clients,
    batch_size=train_bs,
    opt_kwargs=opt_kwargs,
    dl_kwargs=dl_kwargs,
    seed=seed,          
    shuffle=True,
)

def set_lr(optimizer, lr: float):
    for pg in optimizer.param_groups:
        pg["lr"] = lr

lr_sched = lr_schedule(
    base_lr=base_lr,
    warmup_rounds=warmup_rounds,
    total_rounds=num_rounds,
    start_lr=start_lr,
)

augment = True
if augment:
    train_loss_fn = SoftTargetCrossEntropy()
else:
    train_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
eval_loss_fn  = nn.CrossEntropyLoss()

# Model factory 
def model_fn():
    block_factory = make_last_block_moe_factory(
        num_experts=4,
        top_k=1,
        mlp_ratio=4,
    )
    return MoEConvNeXtWrapper(convnext_tiny(weights=None, num_classes=10, block=block_factory))

global_model = model_fn().to(device)
global_params = deepcopy(global_model.state_dict())

# EMA model
ema_decay = 0.995
ema_model = model_fn().to(device)
ema_model.load_state_dict(global_model.state_dict())
ema_model.eval()

for r in range(num_rounds):
    print(f"\n=== Round {r} ===")
    lr_r = float(lr_sched[r])

    client_states = []
    client_ns = []

    for client in clients:
        # Local model starts from current global
        local_model = model_fn().to(device)
        local_model.load_state_dict(global_params)

        # Optimizer 
        optimizer = torch.optim.AdamW(local_model.parameters(), **client.opt_kwargs)
        if client.opt_state is not None:
            optimizer.load_state_dict(client.opt_state)

        set_lr(optimizer, lr_r)

        # Train
        hist = train_client(
            local_model,
            client.loader,
            device,
            optimizer,
            train_loss_fn,
            num_epochs=local_epochs,
            log=True,
            augment=augment,
        )

        client.metrics[f"round_{r}"] = hist
        client.opt_state = deepcopy(optimizer.state_dict())

        client_states.append(deepcopy(local_model.state_dict()))
        client_ns.append(client.num_samples)

    # Aggregate
    global_params = fedavg(client_states, client_ns)
    global_model.load_state_dict(global_params)

    # EMA update on server
    ema_update_model(ema_model, global_model, ema_decay)

    # Periodic eval
    if (r % 10) == 0 or (r == num_rounds - 1):
        tr = evaluate(global_model, train_eval_loader, device, loss_fn=eval_loss_fn)
        va = evaluate(global_model, val_loader, device, loss_fn=eval_loss_fn)
        va_ema = evaluate(ema_model, val_loader, device, loss_fn=eval_loss_fn)

        print(
            f"[Server Eval] lr {lr_r:.6f} | "
            f"train loss {tr['loss']:.4f} acc {tr['acc']:.4f} | "
            f"val loss {va['loss']:.4f} acc {va['acc']:.4f} | "
            f"EMA val loss {va_ema['loss']:.4f} acc {va_ema['acc']:.4f}"
        )

# Final eval
tr = evaluate(global_model, train_eval_loader, device, loss_fn=eval_loss_fn)
va = evaluate(global_model, val_loader, device, loss_fn=eval_loss_fn)
print(f"\nFinal Aggregated Model Train Loss: {tr['loss']:.4f}, Train Acc: {tr['acc']:.4f}")
print(f"Final Aggregated Model Val   Loss: {va['loss']:.4f}, Val   Acc: {va['acc']:.4f}")

lat_mean, lat_std = measure_ms_per_sample(global_model, val_loader, device)
print(f"Latency: {lat_mean:.3f} +/- {lat_std:.3f} ms/sample (batch_size = {eval_bs})")

print(fedavg_comm_cost_bytes(global_model, num_clients=num_clients, num_rounds=num_rounds))
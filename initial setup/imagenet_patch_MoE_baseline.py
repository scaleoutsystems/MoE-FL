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
    
#MOE ConvNeXt base block for ConvNeXt tiny
#TODO missing load balancing loss
class MoECNBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_experts,
        top_k,
        mlp_ratio,
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
        self.router = Router(dim,num_experts,top_k)
        
        self.experts = nn.ModuleList([Expert(dim,mlp_ratio) for _ in range(num_experts)])
        self.permute_back = Permute([0, 3, 1, 2])
        
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, input):
        # input: (N, C, H, W)
        x = self.dwconv(input)  # (N, H, W, C)
        N, H, W, C = x.shape
        x_flat = x.reshape(-1, C)  # (T, C), T = N*H*W

        topi, weights = self.router(x_flat)  # (T, K), (T, K)
        T, K = topi.shape

        out = x_flat.new_zeros(T, C)

        #TODO highly innefficient with nested python loops
        #Improve routing code
        for k in range(K):
            ids_k = topi[:, k]                   # (T,)
            w_k = weights[:, k].unsqueeze(1)     # (T, 1)

            # For each expert, gather its tokens in one shot, run expert once, scatter back
            for e, expert in enumerate(self.experts):
                idx = (ids_k == e).nonzero(as_tuple=False).squeeze(1)  # (n_e,)
                if idx.numel() == 0:
                    continue

                y = expert(x_flat.index_select(0, idx))                # (n_e, C)
                out.index_add_(0, idx, y * w_k.index_select(0, idx))   # scatter-add

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
    
class Router(nn.Module):
    def __init__(self, dim: int, num_experts: int, top_k: int = 1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x):  # x: (T, C)
        logits = self.gate(x)                       # (T, E)
        topv, topi = torch.topk(logits, k=self.top_k, dim=-1)  # (T, K), (T, K)

        # Turn top-k logits into weights that sum to 1 for each token
        weights = torch.softmax(topv, dim=-1)       # (T, K)

        return topi, weights
    
def make_last_block_moe_factory(num_experts=4, top_k=1, mlp_ratio=4):
    # convnext_tiny: stage dim -> number of blocks in that stage
    stage_depth = {96: 3, 192: 3, 384: 9, 768: 3}

    # which stages should get "end-of-stage MoE"
    moe_dims = {384, 768}  # last 2 stages

    # how many blocks we've created so far for each dim
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
    # "persistent_workers": True,  # optional
}

label_smoothing = 0.1

val_transform = transforms.Compose([
    transforms.Resize(236),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

#TODO papers may use more regularization
train_transform = transforms.Compose([
    transforms.Resize(236),
    transforms.CenterCrop(64),
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

# ---- Model factory (MoE ConvNeXt Tiny) ----
def model_fn():
    block_factory = make_last_block_moe_factory(
        num_experts=4,
        top_k=1,
        mlp_ratio=4,
    )
    return convnext_tiny(weights=None, num_classes=10, block=block_factory)

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

        # Optimizer (persist across rounds via client.opt_state)
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

        # Save metrics + persist optimizer state
        client.metrics[f"round_{r}"] = hist
        client.opt_state = deepcopy(optimizer.state_dict())

        # Collect for FedAvg
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
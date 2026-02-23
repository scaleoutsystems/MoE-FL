import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import random
from torchvision.datasets import Imagenette
from torchvision.models import convnext_tiny
from torchvision import transforms
from timm.loss import SoftTargetCrossEntropy
from copy import deepcopy
from utils.fl_utils import fedavg, init_clients, ema_update_model, lr_schedule
from utils.training_utils import train_client, evaluate
from utils.metric_utils import  fedavg_comm_cost_mb, count_flops
from utils.experiment_tracking import init_run, log_and_checkpoint
from torchvision.transforms.v2 import MixUp, CutMix, RandomChoice

#Reproducability and GPU
seed=42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Running on:", device)

#Hyperparameters
num_clients = 10
num_rounds = 0#300
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
    #"num_workers": 4,
    "pin_memory": (device == "cuda"),
    # "persistent_workers": True, 
}

ema_decay = 0.995
label_smoothing = 0.1


#Transforms 
val_transform = transforms.Compose([
    transforms.Resize(236),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

num_ops = 9
magnitude = 5
rand_erase_p = 0.25


train_transform = transforms.Compose([
    transforms.Resize(236),
    transforms.CenterCrop(224),
    #transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandAugment(num_ops=num_ops, magnitude=magnitude),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    transforms.RandomErasing(p=rand_erase_p),
])

cutmix_alpha = 1.0
mixup_alpha = 0.8

mix_transform = RandomChoice([ 
    MixUp(num_classes=10,alpha=mixup_alpha),
    CutMix(num_classes=10,alpha=cutmix_alpha)
])

config = {
    "seed": seed,
    "num_clients": num_clients,
    "num_rounds": num_rounds,
    "local_epochs": local_epochs,
    "train_bs": train_bs,
    "eval_bs": eval_bs,
    "base_lr": base_lr,
    "start_lr": start_lr,
    "warmup_rounds": warmup_rounds,
    "opt_kwargs": opt_kwargs,
    "ema_decay": ema_decay,
    "mixup/cutmix": {"mixup_alpha": mixup_alpha, "cutmix_alpha": cutmix_alpha},
    "RandAugment": {"num_ops": num_ops, "magnitude": magnitude},
    "RandomErasing_p": rand_erase_p,
    "label_smoothing": label_smoothing,
}

ctx = init_run("imagenette_convnext_fl", config)
print("Run dir:", ctx["run_dir"])

train = Imagenette(root='data',split='train',download=True, transform=train_transform)
val = Imagenette(root='data',split='val',download=True, transform=val_transform)

#Global eval loaders 
train_eval_loader = DataLoader(train, batch_size=eval_bs, shuffle=False, **dl_kwargs)
val_loader = DataLoader(val, batch_size=eval_bs, shuffle=False, **dl_kwargs)

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

def set_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg["lr"] = lr
        
lr_sched = lr_schedule(base_lr=base_lr, warmup_rounds=warmup_rounds, total_rounds=num_rounds, start_lr=start_lr)

if mix_transform != None:
    train_loss_fn = SoftTargetCrossEntropy()
else:
    train_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
eval_loss_fn  = nn.CrossEntropyLoss()

def model_fn():
    return convnext_tiny(weights=None,num_classes=10)

global_model = model_fn().to(device)
global_params = deepcopy(global_model.state_dict())

ema_model = model_fn().to(device)
ema_model.load_state_dict(global_model.state_dict()) 
ema_model.eval()

# Federated loop
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
            mix_transform=mix_transform,
        )

        client.metrics[f"round_{r}"] = hist
        client.opt_state = deepcopy(optimizer.state_dict())

        client_states.append(deepcopy(local_model.state_dict()))
        client_ns.append(client.num_samples)

    print("now")
    # Aggregate
    global_params = fedavg(client_states, client_ns)
    global_model.load_state_dict(global_params)
    
    ema_update_model(ema_model, global_model, ema_decay)
        
    if (r % 10) == 0 or (r == num_rounds - 1):
        tr = evaluate(global_model, train_eval_loader, device, loss_fn=eval_loss_fn)
        va = evaluate(global_model, val_loader, device, loss_fn=eval_loss_fn)
        va_ema = evaluate(ema_model, val_loader, device, loss_fn=eval_loss_fn)

        print(f"[Server Eval] train loss {tr['loss']:.4f} acc {tr['acc']:.4f} | "
            f"val loss {va['loss']:.4f} acc {va['acc']:.4f} | "
            f"EMA val loss {va_ema['loss']:.4f} acc {va_ema['acc']:.4f}")
        
        metrics = {
            "lr": lr_r,
            "train_loss": float(tr["loss"]),
            "train_acc": float(tr["acc"]),
            "val_loss": float(va["loss"]),
            "val_acc": float(va["acc"]),
            "val_ema_loss": float(va_ema["loss"]),
            "val_ema_acc": float(va_ema["acc"]),
        }

        saved = log_and_checkpoint(ctx,r,metrics,global_model,
                                   ema_model,clients,
                                   best_key="val_acc", mode="max",)

        if saved:
            print(f"New best at round {r}: {metrics['val_acc']:.4f}")

# tr = evaluate(global_model, train_eval_loader, device, loss_fn=eval_loss_fn)
# va = evaluate(global_model, val_loader, device, loss_fn=eval_loss_fn)
# print(f"\nFinal Aggregated Model Train Loss: {tr['loss']:.4f}, Train Acc: {tr['acc']:.4f}")
# print(f"Final Aggregated Model Val   Loss: {va['loss']:.4f}, Val   Acc: {va['acc']:.4f}")

# print(fedavg_comm_cost_mb(global_model, num_clients=num_clients, num_rounds=num_rounds))

x, _ = next(iter(val_loader))
x = x[:1].contiguous()  # [1, C, H, W]
x = x.to(device)
global_model = global_model.to(device).eval()

macs = count_flops(global_model, x, show_table=True)
print(macs)

#TODO Data augmentation (paper might use more?)
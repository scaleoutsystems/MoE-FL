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
from utils import fedavg, init_clients, train_client, evaluate, measure_ms_per_sample, fedavg_comm_cost_mb, lr_schedule, ema_update_model

#Reproducability and GPU
seed=42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Running on:", device)

#Hyperparameters
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
    #"num_workers": 4,
    "pin_memory": (device == "cuda"),
    # "persistent_workers": True, 
}

label_smoothing = 0.1


#Transforms 
val_transform = transforms.Compose([
    transforms.Resize(236),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

train_transform = transforms.Compose([
    transforms.Resize(236),
    transforms.CenterCrop(64),
    #transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandAugment(num_ops=9, magnitude=5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    transforms.RandomErasing(p=0.25),
])

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

augment = True
if augment:
    train_loss_fn = SoftTargetCrossEntropy()
else:
    train_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
eval_loss_fn  = nn.CrossEntropyLoss()

def model_fn():
    return convnext_tiny(weights=None,num_classes=10)

global_model = model_fn().to(device)
global_params = deepcopy(global_model.state_dict())

ema_decay = 0.995
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
            augment=augment,
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

tr = evaluate(global_model, train_eval_loader, device, loss_fn=eval_loss_fn)
va = evaluate(global_model, val_loader, device, loss_fn=eval_loss_fn)
print(f"\nFinal Aggregated Model Train Loss: {tr['loss']:.4f}, Train Acc: {tr['acc']:.4f}")
print(f"Final Aggregated Model Val   Loss: {va['loss']:.4f}, Val   Acc: {va['acc']:.4f}")

lat_mean, lat_std = measure_ms_per_sample(global_model, val_loader, device)
print(f"Latency: {lat_mean:.3f} +/- {lat_std:.3f} ms/sample (batch_size = {eval_bs})")

print(fedavg_comm_cost_mb(global_model, num_clients=num_clients, num_rounds=num_rounds))

#TODO Data augmentation (paper might use more?)
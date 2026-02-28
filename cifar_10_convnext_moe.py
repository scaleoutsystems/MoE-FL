import random
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms

from utils.fl_utils import init_clients, lr_schedule
from utils.training_utils import evaluate, fl_loop
from utils.experiment_tracking import init_run
from custom_modules.convnext_moe import convnext_moe_model_fn

#Reproducability and GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Running on:", device)

seed=42
num_clients = 10
num_rounds = 300
local_epochs = 1
client_frac = 0.5

batch_size = 128
base_lr = 4e-3
start_lr = 1e-3
warmup_rounds = 20

label_smoothing = 0.1

fedprox = True
mu = 1e-3

num_experts = 4
top_k = 1
mlp_ratio = 2
capacity_ratio = 1.0

pad_size=4
random_crop_size = 32
random_flip_prob = 0.5

opt_kwargs = {
    "lr": base_lr,
    "betas": (0.9, 0.999),
    "weight_decay": 0.05,
}

dl_kwargs = {
    #"num_workers": 8,
    "pin_memory": (device == "cuda"),
    #"prefetch_factor": 4,
    "drop_last": True,
}


cfg = {
    "seed": seed,
    "fed": {
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "client_frac": client_frac,
        "local_epochs": local_epochs,
        "fedprox": fedprox,
        "mu": mu,
    },
    "optim": {
        "sched" : "cosine_annealing",
        "base_lr": base_lr,
        "kwargs": opt_kwargs,
    },
    "data": {
        "batch_size": batch_size,
    },
    "aug": {
        "const_pad_size" : pad_size,
        "random_crop_size": random_crop_size,
        "random_flip_prob": random_flip_prob,
    },
    "moe" : {
        "routing": "noisy_top_k",
        "expert_placement": "last-2",
        "num_experts": num_experts,
        "top_k": top_k,
        "mlp_ratio": mlp_ratio,
        "capacity_ratio": capacity_ratio,
    },
}

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

ctx = init_run("cifar10_convnext_moe_fl", cfg)
print("Run dir:", ctx["run_dir"])

lr_sched = lr_schedule(base_lr=base_lr,start_lr=0,warmup_rounds=0,
                       total_rounds=num_rounds)

train_transform = transforms.Compose([
    transforms.Pad(pad_size, padding_mode="constant"),   
    transforms.RandomCrop(random_crop_size),                     
    transforms.RandomHorizontalFlip(random_flip_prob),             
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),     
        std=(0.2023, 0.1994, 0.2010)       
    ),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
    ),
])

print("Loading data...")
train = CIFAR10(root='data',train=True, transform=train_transform)
val = CIFAR10(root='data',train=False, transform=val_transform)

#Global eval loaders 
train_eval_loader = DataLoader(train, batch_size=batch_size, shuffle=False, **dl_kwargs)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, **dl_kwargs)
print("Data loaded")

# Client loaders 
clients = init_clients(
    dataset=train,
    num_clients=num_clients,
    batch_size=batch_size,
    dl_kwargs=dl_kwargs,
    seed=seed,
    shuffle=True,
)
    
train_loss_fn = nn.CrossEntropyLoss()
eval_loss_fn  = nn.CrossEntropyLoss()

def opt_fn(model, opt_kwargs):
    return torch.optim.AdamW(model.parameters(), **opt_kwargs)

def model_fn():
    return convnext_moe_model_fn(num_experts,top_k,mlp_ratio,capacity_ratio,num_classes=10)

global_model = fl_loop(clients=clients, 
                       model_fn=model_fn,
                       opt_fn = opt_fn,
                       train_loss_fn=train_loss_fn,
                       eval_loss_fn=eval_loss_fn,
                       lr_sched=lr_sched,
                       mix_transform=None,
                       val_loader=val_loader,
                       device=device,
                       ctx=ctx,
                       fl_kwargs=cfg['fed'],
                       opt_kwargs=opt_kwargs)

tr = evaluate(global_model, train_eval_loader, device, loss_fn=eval_loss_fn)
va = evaluate(global_model, val_loader, device, loss_fn=eval_loss_fn)
print(f"\nFinal Aggregated Model Train Loss: {tr['loss']:.4f}, Train Acc: {tr['acc']:.4f}")
print(f"Final Aggregated Model Val   Loss: {va['loss']:.4f}, Val   Acc: {va['acc']:.4f}")


# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# import numpy as np
# import random
# from torchvision.datasets import CIFAR10
# from torchvision import transforms
# from copy import deepcopy
# from utils.fl_utils import fedavg, init_clients
# from utils.training_utils import train_client, evaluate
# from utils.metric_utils import measure_ms_per_sample, fedavg_comm_cost_mb
# from custom_modules.convnext_moe import convnext_moe_model_fn

# seed = 42
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print("Running on:", device)

# # Hyperparameters
# num_clients = 10
# num_rounds = 60
# local_epochs = 1

# train_bs = 32
# eval_bs = 64
# lr = 1e-3

# opt_kwargs = {
#     "lr": lr,              
#     "betas": (0.9, 0.999),
#     "weight_decay": 0.05,
# }

# dl_kwargs = {
#     # "num_workers": 4,
#     "pin_memory": (device == "cuda"),
#     # "persistent_workers": True,  
# }

# num_experts = 4
# top_k = 1
# mlp_ratio = 1

# train = CIFAR10(root="data", train=True, download=True, transform=transforms.ToTensor())
# val = CIFAR10(root="data", train=False,   download=True, transform=transforms.ToTensor())

# # Global eval loaders
# train_eval_loader = DataLoader(train, batch_size=eval_bs, shuffle=False, **dl_kwargs)
# val_loader        = DataLoader(val,   batch_size=eval_bs, shuffle=False, **dl_kwargs)

# # Client loaders
# clients = init_clients(
#     dataset=train,
#     num_clients=num_clients,
#     batch_size=train_bs,
#     opt_kwargs=opt_kwargs,
#     dl_kwargs=dl_kwargs,
#     seed=seed,          
#     shuffle=True,
# )

# loss_fn  = nn.CrossEntropyLoss()

# global_model = convnext_moe_model_fn(mlp_ratio=mlp_ratio).to(device)
# local_model = convnext_moe_model_fn(mlp_ratio=mlp_ratio).to(device)
# global_params = deepcopy(global_model.state_dict())

# for r in range(num_rounds):
#     print(f"\n=== Round {r} ===")

#     client_states = []
#     client_ns = []

#     for client in clients:
#         # Local model starts from current global
#         local_model.load_state_dict(global_params)

#         # Optimizer 
#         optimizer = torch.optim.AdamW(local_model.parameters(), **client.opt_kwargs)
#         if client.opt_state is not None:
#             optimizer.load_state_dict(client.opt_state)

#         # Train
#         hist = train_client(
#             local_model,
#             client.loader,
#             device,
#             optimizer,
#             loss_fn,
#             num_epochs=local_epochs,
#             log=True,
#             mix_transform=None,
#         )

#         client.metrics[f"round_{r}"] = hist
#         client.opt_state = deepcopy(optimizer.state_dict())

#         client_states.append(deepcopy(local_model.state_dict()))
#         client_ns.append(client.num_samples)

#     # Aggregate
#     global_params = fedavg(client_states, client_ns)
#     global_model.load_state_dict(global_params)

#     # Periodic eval
#     if (r % 10) == 0 or (r == num_rounds - 1):
#         tr = evaluate(global_model, train_eval_loader, device, loss_fn=loss_fn)
#         va = evaluate(global_model, val_loader, device, loss_fn=loss_fn)

#         print(
#             f"train loss {tr['loss']:.4f} acc {tr['acc']:.4f} | "
#             f"val loss {va['loss']:.4f} acc {va['acc']:.4f} | "
#         )

# # Final eval
# tr = evaluate(global_model, train_eval_loader, device, loss_fn=loss_fn)
# va = evaluate(global_model, val_loader, device, loss_fn=loss_fn)
# print(f"\nFinal Aggregated Model Train Loss: {tr['loss']:.4f}, Train Acc: {tr['acc']:.4f}")
# print(f"Final Aggregated Model Val   Loss: {va['loss']:.4f}, Val   Acc: {va['acc']:.4f}")
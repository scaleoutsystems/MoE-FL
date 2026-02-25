
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import random
from torchvision import transforms
from copy import deepcopy
from torchvision.datasets import CIFAR10
from utils.fl_utils import fedavg, init_clients
from utils.training_utils import train_client, evaluate
from custom_modules.resnet20 import ResNet20
    
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

train_bs = 128
eval_bs = 128

base_lr = 4e-3

opt_kwargs = {
    "lr": base_lr,
    #"betas": (0.9, 0.999),
    "weight_decay": 0.05,
}

dl_kwargs = {
    #"num_workers": 4,
    "pin_memory": (device == "cuda"),
    # "persistent_workers": True, 
}

label_smoothing = 0.1

train = CIFAR10(root='data',train=True,download=True,transform=transforms.ToTensor())
val = CIFAR10(root='data',train=False,download=True,transform=transforms.ToTensor())

#Global eval loaders 
train_eval_loader = DataLoader(train, batch_size=eval_bs, shuffle=False, **dl_kwargs)
val_loader = DataLoader(val, batch_size=eval_bs, shuffle=False, **dl_kwargs)

# Client loaders 
clients = init_clients(
    dataset=train,
    num_clients=num_clients,
    batch_size=train_bs,
    dl_kwargs=dl_kwargs,
    seed=seed,
    shuffle=True,
)

train_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
eval_loss_fn  = nn.CrossEntropyLoss()

def model_fn():
    return ResNet20()

global_model = model_fn().to(device)
local_model = model_fn().to(device)
global_params = deepcopy(global_model.state_dict())

# Federated loop
for r in range(num_rounds):
    print(f"\n=== Round {r} ===")
    client_states = []
    client_ns = []

    for client in clients:
        # Local model starts from current global
        local_model.load_state_dict(global_params)

        # Optimizer 
        optimizer = torch.optim.SGD(local_model.parameters(), **opt_kwargs)

        # Train 
        hist = train_client(
            local_model,
            client.loader,
            device,
            optimizer,
            train_loss_fn,
            num_epochs=local_epochs,
            log=True,
            mix_transform=None,
        )

        client.metrics[f"round_{r}"] = hist

        client_states.append(deepcopy(local_model.state_dict()))
        client_ns.append(client.num_samples)

    # Aggregate
    global_params = fedavg(client_states, client_ns,device=device)
    global_model.load_state_dict(global_params)
    
    if (r % 10) == 0:
        tr = evaluate(global_model, train_eval_loader, device, loss_fn=eval_loss_fn)
        va = evaluate(global_model, val_loader, device, loss_fn=eval_loss_fn)

        print(f"[Server Eval] train loss {tr['loss']:.4f} acc {tr['acc']:.4f} | "
            f"val loss {va['loss']:.4f} acc {va['acc']:.4f}")

tr = evaluate(global_model, train_eval_loader, device, loss_fn=eval_loss_fn)
va = evaluate(global_model, val_loader, device, loss_fn=eval_loss_fn)
print(f"\nFinal Aggregated Model Train Loss: {tr['loss']:.4f}, Train Acc: {tr['acc']:.4f}")
print(f"Final Aggregated Model Val   Loss: {va['loss']:.4f}, Val   Acc: {va['acc']:.4f}")

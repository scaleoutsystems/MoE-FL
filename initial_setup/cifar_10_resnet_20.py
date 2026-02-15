
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import random
from torchvision import transforms
from copy import deepcopy
from torchvision.datasets import CIFAR10
from utils import fedavg, init_clients, train_client, evaluate

#Simple CIFAR-10 style ResNet-20 implementation
class BasicBlock(nn.Module):
        def __init__(self, in_c, out_c, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
            self.bn1   = nn.BatchNorm2d(out_c)
            self.conv2 = nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False)
            self.bn2   = nn.BatchNorm2d(out_c)

            self.proj = None
            if stride != 1 or in_c != out_c:
                self.proj = nn.Sequential(
                    nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_c),
                )

        def forward(self, x):
            y = F.relu(self.bn1(self.conv1(x)))
            y = self.bn2(self.conv2(y))
            x = x if self.proj is None else self.proj(x)
            return F.relu(x + y)

class ResNet20(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_c = 16

        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(16, blocks=3, stride=1)
        self.layer2 = self._make_layer(32, blocks=3, stride=2)
        self.layer3 = self._make_layer(64, blocks=3, stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(64, num_classes)

    def _make_layer(self, out_c, blocks, stride):
        layers = [BasicBlock(self.in_c, out_c, stride)]
        self.in_c = out_c
        for _ in range(blocks - 1):
            layers.append(BasicBlock(self.in_c, out_c, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)
    
    
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
# val_transform = transforms.Compose([
#     transforms.Resize(236),
#     transforms.CenterCrop(64),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225],
#     ),
# ])

# train_transform = transforms.Compose([
#     transforms.Resize(236),
#     transforms.CenterCrop(64),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225],
#     ),
#     transforms.RandomErasing(p=0.25),
# ])

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
    opt_kwargs=opt_kwargs,
    dl_kwargs=dl_kwargs,
    seed=seed,
    shuffle=True,
)

train_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
eval_loss_fn  = nn.CrossEntropyLoss()

def model_fn():
    return ResNet20()

global_model = model_fn().to(device)
global_params = deepcopy(global_model.state_dict())

# Federated loop
for r in range(num_rounds):
    print(f"\n=== Round {r} ===")
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

        # Train 
        hist = train_client(
            local_model,
            client.loader,
            device,
            optimizer,
            train_loss_fn,
            num_epochs=local_epochs,
            log=True,
            augment=False,
        )

        client.metrics[f"round_{r}"] = hist
        client.opt_state = deepcopy(optimizer.state_dict())

        client_states.append(deepcopy(local_model.state_dict()))
        client_ns.append(client.num_samples)

    # Aggregate
    global_params = fedavg(client_states, client_ns)
    global_model.load_state_dict(global_params)
    
    if (r % 10) == 0 or (r == num_rounds - 1):
        tr = evaluate(global_model, train_eval_loader, device, loss_fn=eval_loss_fn)
        va = evaluate(global_model, val_loader, device, loss_fn=eval_loss_fn)

        print(f"[Server Eval] train loss {tr['loss']:.4f} acc {tr['acc']:.4f} | "
            f"val loss {va['loss']:.4f} acc {va['acc']:.4f}")

tr = evaluate(global_model, train_eval_loader, device, loss_fn=eval_loss_fn)
va = evaluate(global_model, val_loader, device, loss_fn=eval_loss_fn)
print(f"\nFinal Aggregated Model Train Loss: {tr['loss']:.4f}, Train Acc: {tr['acc']:.4f}")
print(f"Final Aggregated Model Val   Loss: {va['loss']:.4f}, Val   Acc: {va['acc']:.4f}")

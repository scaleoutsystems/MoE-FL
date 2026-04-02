import random
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
#from torchvision.datasets import ImageNet
from torchvision import transforms

from timm.loss import SoftTargetCrossEntropy
from timm.data import create_transform, Mixup
from timm.data.constants import IMAGENET_DEFAULT_STD, IMAGENET_DEFAULT_MEAN

from utils.fl_utils import init_clients, lr_schedule, expert_weighted_fedavg
from utils.training_utils import evaluate, fl_loop
from utils.experiment_tracking import init_run
from utils.data_utils import ImageNetSubset
from custom_modules.convnext_moe import convnext_moe_model_fn

#Reproducability and GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Running on:", device)

#There are graph breaks in the MoE model this allows torch.compile to work better
torch._dynamo.config.cache_size_limit = 64

seed=42
num_clients = 20
num_rounds = 600
local_epochs = 5
client_frac = 0.2

batch_size = 64
base_lr = 0.025
start_lr = 0.005
warmup_rounds = 30

label_smoothing = 0.1

fedprox = False
mu = 0

num_experts = 4
top_k = 1
mlp_ratio = 2
capacity_ratio = 1.0

#Comments are full imagenet augments we reduce for the subset/FL setting
auto_augment ="rand-m4-mstd0.5-inc1" #"rand-m9-mstd0.5-inc1"
rand_erase_p = 0.15 #0.25
rand_erase_mode="pixel"
rand_erase_count=1
cutmix_alpha = 0.5 #1.0
mixup_alpha = 0.2 #0.8
mix_prob = 0.3 #0.6
mix_mode = "batch"
mix_switch_prob = 0.5
color_jitter = 0.4
interpolation = "bicubic"

opt_kwargs = {
    "lr": base_lr,
    "momentum": 0.9,
    "weight_decay": 3e-4,
    "nesterov": True,
}

dl_kwargs = {
    "num_workers": 8,
    "pin_memory": (device == "cuda"),
    "prefetch_factor": 4,
    "drop_last": True,
}

val_dl_kwargs = {**dl_kwargs, "drop_last": False}

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
        "sched" : "lin_warmup_cosine_annealing",
        "base_lr": base_lr,
        "start_lr": start_lr,
        "warmup_rounds": warmup_rounds,
        "kwargs": opt_kwargs,
    },
    "data": {
        "batch_size": batch_size,
    },
    "aug": {
        "mixup/cutmix" : {
            "mixup_alpha": mixup_alpha,
            "cutmix_alpha": cutmix_alpha,
            "mix_prob": mix_prob,
            "mix_mode": mix_mode,
            "mix_switch_prob": mix_switch_prob,
        },
        "auto_augment": auto_augment,
        "random_erasing": {
            "random_erasing_p": rand_erase_p,
            "random_erase_count": rand_erase_count,
            "random_erase_mode": rand_erase_mode,
        },
        "interpolation": interpolation,
    },
    "reg": {
        "label_smoothing": label_smoothing,
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

#Transforms 
val_transform = transforms.Compose([
    transforms.Resize(236,interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    ),
])

train_transform = create_transform(
    input_size=224,
    is_training=True,
    color_jitter=color_jitter,
    auto_augment=auto_augment,
    interpolation=interpolation,
    re_prob=rand_erase_p,
    re_mode=rand_erase_mode,
    re_count=rand_erase_count,
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
)

mix_transform = Mixup(
    mixup_alpha=mixup_alpha,
    cutmix_alpha=cutmix_alpha,
    prob=mix_prob,
    switch_prob=mix_switch_prob,
    mode=mix_mode,
    label_smoothing=label_smoothing,
    num_classes=100, 
)

if mix_transform is not None:
    train_loss_fn = SoftTargetCrossEntropy()
else:
    train_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
eval_loss_fn  = nn.CrossEntropyLoss()

def opt_fn(model, opt_kwargs):
    return torch.optim.SGD(model.parameters(), **opt_kwargs)

def model_fn():
    return convnext_moe_model_fn(num_experts,top_k,mlp_ratio,capacity_ratio,
                                num_classes=100,collect_expert_stats=True)

if __name__ == "__main__":
    ctx = init_run("imagenet_convnext_moe_fl", cfg)
    print("Run dir:", ctx["run_dir"])

    print("Loading data...") 
    train = ImageNetSubset(root='data',split='train', transform=train_transform)
    val = ImageNetSubset(root='data',split='val', transform=val_transform)

    #Global eval loaders 
    val_loader = DataLoader(val, batch_size=256, shuffle=False,persistent_workers=True, **val_dl_kwargs)
    print("Data loaded")

    # Client loaders 
    clients = init_clients(
        dataset=train,
        num_clients=num_clients,
        batch_size=batch_size,
        dl_kwargs=dl_kwargs,
        seed=seed,
        shuffle=True,
        transform=train_transform
    )
            
    lr_sched = lr_schedule(base_lr=base_lr, warmup_rounds=warmup_rounds, 
                        total_rounds=num_rounds, start_lr=start_lr)

    global_model = fl_loop(clients=clients, 
                        model_fn=model_fn,
                        opt_fn = opt_fn,
                        train_loss_fn=train_loss_fn,
                        eval_loss_fn=eval_loss_fn,
                        lr_sched=lr_sched,
                        mix_transform=mix_transform,
                        val_loader=val_loader,
                        device=device,
                        ctx=ctx,
                        fl_kwargs=cfg['fed'],
                        opt_kwargs=opt_kwargs,
                        num_gpus=4,
                        agg_fn=expert_weighted_fedavg)

    va = evaluate(global_model, val_loader, device, loss_fn=eval_loss_fn)
    print(f"Final Aggregated Model Val   Loss: {va['loss']:.4f}, Val   Acc: {va['acc']:.4f}")
import os
import random
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torchvision import transforms
from torchvision.models import convnext_tiny

from timm.loss import SoftTargetCrossEntropy
from timm.data import create_transform, Mixup
from timm.data.constants import IMAGENET_DEFAULT_STD, IMAGENET_DEFAULT_MEAN

from utils.fl_utils import init_clients, lr_schedule, fedadam
from utils.training_utils import evaluate, fl_loop, train_centralized_ddp
from utils.experiment_tracking import init_run
from utils.data_utils import ImageNetSubset, WeatherImageNetSubset
from custom_modules.convnext_moe import convnext_moe_model_fn

#--setting fl --method {dense,moe}, --setting centralized --method {dense,moe}
#--data standard: ImageNet-100 subset, --data weather: the same subset with weather augmentation
parser = argparse.ArgumentParser()
parser.add_argument("--setting", choices=["fl", "centralized"], required=True)
parser.add_argument("--method", choices=["dense", "moe"], required=True)
parser.add_argument("--data", choices=["standard", "weather"], default="standard")
args = parser.parse_args()
fl = args.setting == "fl"
moe = args.method == "moe"

DATA_ROOT = "/project/home/p201222/imnet_subset" if args.data == "standard" else "/project/home/p201222/imnet_subset_weather_aug"
DataSet = ImageNetSubset if args.data == "standard" else WeatherImageNetSubset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Running on:", device)

#There are graph breaks in the MoE model this allows torch.compile to work better
torch._dynamo.config.cache_size_limit = 64

seed = 42
label_smoothing = 0.1
interpolation = "bicubic"
rand_erase_mode = "pixel"
rand_erase_count = 1
mix_mode = "batch"
mix_switch_prob = 0.5

#Federated
num_clients = 20
num_rounds = 500
local_epochs = 5
client_frac = 0.4
server_lr = 5e-3
server_betas = (0.9, 0.99)
server_eps = 1e-5
fedprox = False
mu = 0

batch_size = 64 if fl else 512  # centralized: per GPU
base_lr = 0.07
start_lr = 0.007
end_lr = 1e-6
warmup_rounds = 50

#Centralized (torchrun sets WORLD_SIZE = GPUs)
world_size = int(os.environ.get("WORLD_SIZE", 1))
num_epochs = 300
lr = 4e-3 * (batch_size * world_size) / 4096  # = 2e-3 at 4 GPUs
weight_decay = 0.05
warmup_epochs = 20

#MoE
num_experts = 4
top_k = 1
mlp_ratio = 2
capacity_ratio = 1.0

#Augmentation. Federated uses a reduced version of the full ImageNet one (the centralized values), weaker for dense than MoE
if not fl:
    aug = dict(auto_augment="rand-m9-mstd0.5-inc1", rand_erase_p=0.25, cutmix_alpha=1.0, mixup_alpha=0.8, mix_prob=1.0, color_jitter=0.4)
elif moe:
    aug = dict(auto_augment="rand-m9-mstd0.5-inc1", rand_erase_p=0.2, cutmix_alpha=0.8, mixup_alpha=0.6, mix_prob=0.5, color_jitter=0.5)
else:
    aug = dict(auto_augment="rand-m7-mstd0.5-inc1", rand_erase_p=0.15, cutmix_alpha=0.65, mixup_alpha=0.4, mix_prob=0.4, color_jitter=0.4)

if fl:
    opt_kwargs = {"lr": base_lr, "momentum": 0.9, "weight_decay": 0.0, "nesterov": True}
else:
    opt_kwargs = {"lr": lr, "weight_decay": weight_decay, "betas": (0.9, 0.999)}

dl_kwargs = {
    "num_workers": 8,
    "pin_memory": (device == "cuda"),
    "prefetch_factor": 4,
    "drop_last": True,
}

val_dl_kwargs = {**dl_kwargs, "drop_last": False, "persistent_workers": True}

cfg = {
    "seed": seed,
    "setting": args.setting,
    "method": args.method,
    "data": {"name": args.data, "root": DATA_ROOT, "batch_size": batch_size},
    "optim": {
        "name": "sgd_nesterov" if fl else "adamw",
        "kwargs": opt_kwargs,
        **({"sched": "lin_warmup_cosine_annealing", "base_lr": base_lr, "start_lr": start_lr, "end_lr": end_lr,
            "warmup_rounds": warmup_rounds} if fl else
           {"sched": "warmup_cosine", "lr": lr, "weight_decay": weight_decay, "warmup_epochs": warmup_epochs}),
    },
    "aug": {
        "mixup/cutmix": {
            "mixup_alpha": aug["mixup_alpha"],
            "cutmix_alpha": aug["cutmix_alpha"],
            "mix_prob": aug["mix_prob"],
            "mix_mode": mix_mode,
            "mix_switch_prob": mix_switch_prob,
        },
        "auto_augment": aug["auto_augment"],
        "random_erasing": {
            "random_erasing_p": aug["rand_erase_p"],
            "random_erase_count": rand_erase_count,
            "random_erase_mode": rand_erase_mode,
        },
        "interpolation": interpolation,
        "color_jitter": aug["color_jitter"],
    },
    "reg": {"label_smoothing": label_smoothing},
}
if fl:
    cfg["fed"] = {
        "agg": "fedadam",
        "server_lr": server_lr,
        "server_betas": server_betas,
        "server_eps": server_eps,
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "client_frac": client_frac,
        "local_epochs": local_epochs,
        "fedprox": fedprox,
        "mu": mu,
    }
else:
    cfg["training"] = {"num_epochs": num_epochs, "batch_size": batch_size, "effective_batch_size": batch_size * world_size}
if moe:
    cfg["moe"] = {
        "routing": "noisy_top_k",
        "expert_placement": "last-2",
        "num_experts": num_experts,
        "top_k": top_k,
        "mlp_ratio": mlp_ratio,
        "capacity_ratio": capacity_ratio,
    }

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

val_transform = transforms.Compose([
    transforms.Resize(236, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
])

train_transform = create_transform(
    input_size=224,
    is_training=True,
    color_jitter=aug["color_jitter"],
    auto_augment=aug["auto_augment"],
    interpolation=interpolation,
    re_prob=aug["rand_erase_p"],
    re_mode=rand_erase_mode,
    re_count=rand_erase_count,
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
)

mix_transform = Mixup(
    mixup_alpha=aug["mixup_alpha"],
    cutmix_alpha=aug["cutmix_alpha"],
    prob=aug["mix_prob"],
    switch_prob=mix_switch_prob,
    mode=mix_mode,
    label_smoothing=label_smoothing,
    num_classes=100,
)

train_loss_fn = SoftTargetCrossEntropy()
eval_loss_fn = nn.CrossEntropyLoss()

def opt_fn(model, opt_kwargs):
    if fl:
        return torch.optim.SGD(model.parameters(), **opt_kwargs)
    return torch.optim.AdamW(model.parameters(), **opt_kwargs)

def model_fn():
    if moe:
        return convnext_moe_model_fn(num_experts, top_k, mlp_ratio, capacity_ratio,
                                     num_classes=100, collect_expert_stats=True)
    return convnext_tiny(weights=None, num_classes=100)

def scheduler_fn(optimizer):
    return SequentialLR(optimizer, schedulers=[
        LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs),
        CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-6),
    ], milestones=[warmup_epochs])

if __name__ == "__main__":
    name = f"imagenet_convnext{'_moe' if moe else ''}_{args.setting}" + ("_weather" if args.data == "weather" else "")
    ctx = init_run(name, cfg)
    print("Run dir:", ctx["run_dir"])

    print("Loading data...")
    train = DataSet(root=DATA_ROOT, split='train', transform=train_transform)
    val = DataSet(root=DATA_ROOT, split='val', transform=val_transform)
    print("Data loaded")

    if fl:
        val_loader = DataLoader(val, batch_size=128, shuffle=False, **val_dl_kwargs)
        #Train transform is handled by init_clients
        clients = init_clients(
            dataset=train,
            num_clients=num_clients,
            batch_size=batch_size,
            dl_kwargs=dl_kwargs,
            seed=seed,
            shuffle=True,
            transform=train_transform,
        )
        lr_sched = lr_schedule(base_lr=base_lr, warmup_rounds=warmup_rounds,
                               total_rounds=num_rounds, start_lr=start_lr, end_lr=end_lr)

        global_model = fl_loop(clients=clients,
                               model_fn=model_fn,
                               opt_fn=opt_fn,
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
                               eval_every=5,
                               agg_fn=fedadam)

        va = evaluate(global_model, val_loader, device, loss_fn=eval_loss_fn)
        print(f"Final Aggregated Model Val   Loss: {va['loss']:.4f}, Val   Acc: {va['acc']:.4f}")
    else:
        train_centralized_ddp(
            model_fn=model_fn,
            opt_fn=opt_fn,
            dataset=train,
            val_dataset=val,
            loss_fn=train_loss_fn,
            eval_loss_fn=eval_loss_fn,
            scheduler_fn=scheduler_fn,
            mix_transform=mix_transform,
            ctx=ctx,
            opt_kwargs=opt_kwargs,
            num_epochs=num_epochs,
            eval_every=5,
            amp_dtype=torch.bfloat16,
            batch_size=batch_size,
            **dl_kwargs,
        )

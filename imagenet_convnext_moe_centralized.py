import random
import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torchvision import transforms

from timm.loss import SoftTargetCrossEntropy
from timm.data import create_transform, Mixup
from timm.data.constants import IMAGENET_DEFAULT_STD, IMAGENET_DEFAULT_MEAN

from custom_modules.convnext_moe import convnext_moe_model_fn

from utils.training_utils import train_centralized_ddp
from utils.experiment_tracking import init_run
from utils.data_utils import ImageNetSubset

seed          = 42
num_epochs    = 300
batch_size    = 512        # per GPU, 2048 effective
lr            = 4e-3 * (batch_size * 4) / 4096   # = 2e-3
weight_decay  = 0.05
warmup_epochs = 20

label_smoothing  = 0.1
auto_augment     = "rand-m9-mstd0.5-inc1"
rand_erase_p     = 0.25
rand_erase_mode  = "pixel"
rand_erase_count = 1
cutmix_alpha     = 1.0
mixup_alpha      = 0.8
mix_prob         = 1.0
mix_mode         = "batch"
mix_switch_prob  = 0.5
color_jitter     = 0.4
interpolation    = "bicubic"

num_experts = 4
top_k = 1
mlp_ratio = 2
capacity_ratio = 1.0

opt_kwargs = {
    "lr": lr,
    "weight_decay": weight_decay,
    "betas": (0.9, 0.999),
}

dl_kwargs = {
    "num_workers": 8,
    "pin_memory": True,
    "prefetch_factor": 4,
    "drop_last": True,
}

cfg = {
    "seed": seed,
    "training": {
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "effective_batch_size": batch_size * 4,
    },
    "optim": {
        "name": "adamw",
        "lr": lr,
        "weight_decay": weight_decay,
        "warmup_epochs": warmup_epochs,
        "sched": "warmup_cosine",
        "kwargs": opt_kwargs,
    },
    "aug": {
        "mixup/cutmix": {
            "mixup_alpha": mixup_alpha,
            "cutmix_alpha": cutmix_alpha,
            "mix_prob": mix_prob,
            "mix_switch_prob": mix_switch_prob,
        },
        "auto_augment": auto_augment,
        "random_erasing": {
            "random_erasing_p": rand_erase_p,
            "random_erase_count": rand_erase_count,
            "random_erase_mode": rand_erase_mode,
        },
        "interpolation": interpolation,
        "color_jitter": color_jitter,
    },
    "reg": {
        "label_smoothing": label_smoothing,
    },
    "moe": {
        "num_experts": num_experts,
        "top_k": top_k,
        "mlp_ratio": mlp_ratio,
        "capacity_ratio": capacity_ratio,
    },
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

train_loss_fn = SoftTargetCrossEntropy()
eval_loss_fn  = nn.CrossEntropyLoss()

def opt_fn(model, opt_kwargs):
    return torch.optim.AdamW(model.parameters(), **opt_kwargs)

def model_fn():
    return convnext_moe_model_fn(num_experts,top_k,mlp_ratio,capacity_ratio,
                                num_classes=100,collect_expert_stats=True)

def scheduler_fn(optimizer):
    return SequentialLR(optimizer, schedulers=[
        LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs),
        CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-6),
    ], milestones=[warmup_epochs])

if __name__ == "__main__":
    ctx = init_run("imagenet_convnext_moe_centralized", cfg)

    print("Loading data...")
    train = ImageNetSubset(root='/project/home/p201222/imnet_subset', split='train', transform=train_transform)
    val   = ImageNetSubset(root='/project/home/p201222/imnet_subset', split='val',   transform=val_transform)
    print("Data loaded")

    model = train_centralized_ddp(
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
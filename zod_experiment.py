import os
import time
import random
import argparse
from collections import Counter
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import LambdaLR

from utils.fl_utils import init_clients_zod, fedadam, context_expert_fedadam
from utils.training_utils import fl_loop, train_centralized_ddp, train_client, compile_model, zod_context_counts
from utils.experiment_tracking import init_run
from utils.data_utils import ZOD
from custom_modules.convnext_detection import convnext_detector_fn, convnext_moe_detector_fn

#Runs from NOTES.md: --setting fl --method {dense,moe,context_aware,context_distributed},
#--setting centralized --method {dense,moe,context_aware}
parser = argparse.ArgumentParser()
parser.add_argument("--setting", choices=["fl", "centralized"], required=True)
parser.add_argument("--method", choices=["dense", "moe", "context_aware", "context_distributed"], required=True)
#Pilot values (marked (pilot) in NOTES.md)
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--rounds", type=int, default=100)
parser.add_argument("--epochs", type=int, default=36)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--ckpt", default=None, help="run dir to resume from")
parser.add_argument("--benchmark", action="store_true", help="pilot step 1: memory and images/s per batch size")
args = parser.parse_args()
assert not (args.method == "context_distributed" and args.setting == "centralized"), "context_distributed has no centralized version"

#Paths: devkit COCO export resized by zod_resize.py (see NOTES.md step 0)
ZOD_ROOT   = "data"  # relative to the repo (git-ignored), scripts are run from the repo root
TRAIN_JSON = f"{ZOD_ROOT}/coco_1333/train.json"
VAL_JSON   = f"{ZOD_ROOT}/coco_1333/val.json"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#There are graph breaks in the MoE model this allows torch.compile to work better
torch._dynamo.config.cache_size_limit = 64

seed = 42
batch_size = args.batch_size

#Federated
num_clients = 20
client_frac = 0.4
#A weather is a client's context (and gives it that expert) with at least this many frames
min_context_frames = 1
local_epochs = 1
num_rounds = args.rounds
server_lr = 5e-3
server_betas = (0.9, 0.99)
server_eps = 1e-5
#0.07 at batch 64 in the ImageNet-100 FL runs, linearly scaled (also Faster R-CNN's 0.02 at batch 16)
base_lr = args.lr if args.lr is not None else 0.07 * batch_size / 64

#Centralized
num_epochs = args.epochs
#torchrun sets WORLD_SIZE (GPUs); effective batch = batch_size per GPU x GPUs
world_size = int(os.environ.get("WORLD_SIZE", 1))
#ConvNeXt detection recipe: 2e-4 at batch 16, linearly scaled to the effective batch
central_lr = args.lr if args.lr is not None else 2e-4 * (batch_size * world_size) / 16
weight_decay = 0.05

#LR schedule, Detectron2 3x standard: linear warmup from 0.001x over 1000 iterations,
#then x0.1 at 210k and 250k of 270k iterations (78% and 93% of training), in rounds or epochs.
#Federated has no standard: per-iteration warmup over the first 1% of rounds (~1% of training, as centralized)
warmup_iters = 1000
warmup_rounds = 0.01 * num_rounds
lr_steps = (210 / 270, 250 / 270)
lr_gamma = 0.1
num_steps = num_rounds if args.setting == "fl" else num_epochs
milestones = [round(f * num_steps) for f in lr_steps]

#MoE: one expert per weather
num_experts = len(ZOD.weathers)
top_k = 1
mlp_ratio = 2
capacity_ratio = 1.0
bpr = args.method == "moe"
context_aware = args.method in ("context_aware", "context_distributed")

if args.setting == "fl":
    opt_kwargs = {"lr": base_lr, "momentum": 0.9, "weight_decay": 0.0, "nesterov": True}
else:
    opt_kwargs = {"lr": central_lr, "weight_decay": weight_decay, "betas": (0.9, 0.999)}

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
    "fed": {
        "agg": "context_expert_fedadam" if args.method == "context_distributed" else "fedadam",
        "server_lr": server_lr,
        "server_betas": server_betas,
        "server_eps": server_eps,
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "client_frac": client_frac,
        "local_epochs": local_epochs,
        "fedprox": False,
        "mu": 0,
    },
    "optim": {
        "name": "sgd_nesterov" if args.setting == "fl" else "adamw",
        "sched": "detectron2_warmup_multistep",
        "base_lr": base_lr if args.setting == "fl" else central_lr,
        "warmup": {"rounds": warmup_rounds} if args.setting == "fl" else {"iters": warmup_iters},
        "warmup_factor": 0.001,
        "milestones": milestones,
        "gamma": lr_gamma,
        "num_epochs": num_epochs if args.setting == "centralized" else None,
        "kwargs": opt_kwargs,
    },
    "data": {
        "dataset": "zod_frames_full_blur",
        "train_json": TRAIN_JSON,
        "val_json": VAL_JSON,
        "contexts": ZOD.weathers,
        "classes": ZOD.classes,
        "batch_size": batch_size,
        "world_size": world_size if args.setting == "centralized" else None,
        "effective_batch_size": batch_size * world_size if args.setting == "centralized" else batch_size,
        "train_short_side": [640, 800],
        "test_short_side": 800,
        "max_size": 1333,
        "hflip": 0.5,
    },
    "model": {
        "detector": "faster_rcnn_fpn_p2-p6",
        "backbone": "convnext_tiny_scratch",
        "moe": None if args.method == "dense" else {
            "routing": "context_logit_ce" if context_aware else "noisy_top_k",
            "expert_placement": "last block of stages 3 and 4",
            "num_experts": num_experts,
            "top_k": top_k,
            "mlp_ratio": mlp_ratio,
            "capacity_ratio": capacity_ratio,
            "bpr": bpr,
            "aux_coeff": 0.1,
        },
    },
    "amp": "bf16",
}

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def opt_fn(model, opt_kwargs):
    if args.setting == "fl":
        return torch.optim.SGD(model.parameters(), **opt_kwargs)
    return torch.optim.AdamW(model.parameters(), **opt_kwargs)

def model_fn():
    if args.method == "dense":
        return convnext_detector_fn(num_classes=len(ZOD.classes) + 1)
    return convnext_moe_detector_fn(num_experts, top_k, mlp_ratio, capacity_ratio,
                                    num_classes=len(ZOD.classes) + 1, bpr=bpr, context_aware=context_aware)

def lr_factor(step):
    return lr_gamma ** sum(step >= m for m in milestones)

def scheduler_fn(optimizer):
    #The warmup runs per iteration inside train_client. Closed form (not MultiStepLR, which scales
    #the current LR), so a resumed optimizer with an already decayed LR is not decayed twice
    return LambdaLR(optimizer, lr_factor)

def client_table(train, clients):
    rows = []
    for c in clients:
        days = [train.days[i] for i in c.subset]
        counts = Counter(train.contexts[i] for i in c.subset)
        rows.append({"cid": c.cid, "car": train.cars[c.subset[0]], "period": [min(days), max(days)], "frames": len(c.subset),
                     "weather_frames": {w: counts[i] for i, w in enumerate(ZOD.weathers)},
                     "contexts": [ZOD.weathers[i] for i in c.contexts]})
    return rows

def benchmark(train, steps=50):
    model = compile_model(model_fn().to(device))
    optimizer = opt_fn(model, opt_kwargs)
    for bs in (20, 24, 28, 32):
        try:
            #Warm-up pass compiles for the new shapes (the random image scales give several padded sizes,
            #so it is long enough that few recompiles are left for the timed pass), then a timed pass
            warmup = DataLoader(Subset(train, range(bs * 20)), batch_size=bs, collate_fn=ZOD.collate, **dl_kwargs)
            timed = DataLoader(Subset(train, range(bs * steps)), batch_size=bs, collate_fn=ZOD.collate, **dl_kwargs)
            train_client(model, warmup, device, optimizer, None, None, torch.bfloat16, log=False)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.time()
            train_client(model, timed, device, optimizer, None, None, torch.bfloat16, log=False)
            torch.cuda.synchronize()
            print(f"batch {bs:2d}: {bs * steps / (time.time() - t):.1f} images/s, "
                  f"peak memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB allocated, "
                  f"{torch.cuda.max_memory_reserved() / 1e9:.1f} GB reserved (closer to nvidia-smi)")
        except torch.cuda.OutOfMemoryError:
            print(f"batch {bs:2d}: out of memory")
            break

if __name__ == "__main__":
    train = ZOD(TRAIN_JSON, ZOD_ROOT, train=True)
    val = ZOD(VAL_JSON, ZOD_ROOT)

    if args.benchmark:
        benchmark(train)
        raise SystemExit

    clients = None
    if args.setting == "fl":
        clients = init_clients_zod(train, num_clients=num_clients, batch_size=batch_size,
                                   dl_kwargs=dl_kwargs, min_frames=min_context_frames)
        cfg["data"]["clients"] = client_table(train, clients)
    cfg["data"]["context_counts"] = {"train": zod_context_counts(train), "val": zod_context_counts(val)}
    name = f"zod_{args.setting}_{args.method}"
    run_id = f"{name}_{os.environ['SLURM_JOB_ID']}" if "SLURM_JOB_ID" in os.environ else None
    #torchrun starts one process per GPU but only rank 0 logs and checkpoints, so only it creates the run dir
    #(RANK is not set for single-process runs)
    is_rank0 = int(os.environ.get("RANK", 0)) == 0
    ctx = init_run(name, cfg, run_id=run_id) if is_rank0 and not args.ckpt else None

    if args.setting == "fl":
        val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, collate_fn=ZOD.collate, **val_dl_kwargs)
        lr_sched = [base_lr * lr_factor(r) for r in range(num_rounds)]

        fl_loop(clients=clients,
                model_fn=model_fn,
                opt_fn=opt_fn,
                train_loss_fn=None,
                eval_loss_fn=None,
                lr_sched=lr_sched,
                mix_transform=None,
                val_loader=val_loader,
                device=device,
                ctx=ctx,
                fl_kwargs=cfg["fed"],
                opt_kwargs=opt_kwargs,
                num_gpus=2,
                eval_every=5,
                agg_fn=context_expert_fedadam if args.method == "context_distributed" else fedadam,
                ckpt_path=args.ckpt,
                warmup_rounds=warmup_rounds)
    else:
        train_centralized_ddp(
            model_fn=model_fn,
            opt_fn=opt_fn,
            dataset=train,
            val_dataset=val,
            loss_fn=None,
            eval_loss_fn=None,
            scheduler_fn=scheduler_fn,
            mix_transform=None,
            ctx=ctx,
            opt_kwargs=opt_kwargs,
            num_epochs=num_epochs,
            eval_every=2,
            amp_dtype=torch.bfloat16,
            ckpt_path=args.ckpt,
            warmup_iters=warmup_iters,
            batch_size=batch_size,
            **dl_kwargs,
        )

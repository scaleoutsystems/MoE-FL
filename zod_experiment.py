import os
import time
import random
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from utils.fl_utils import init_clients_zod, lr_schedule, fedadam, context_expert_fedadam
from utils.training_utils import fl_loop, train_centralized_ddp, train_client, compile_model, zod_context_counts
from utils.experiment_tracking import init_run
from utils.data_utils import ZOD
from custom_modules.convnext_detection import convnext_detector_fn, convnext_moe_detector_fn

#Runs from NOTES.md: --setting fl --method {D,S,A,B} --c {2,5}, --setting centralized --method {D,S,A}
parser = argparse.ArgumentParser()
parser.add_argument("--setting", choices=["fl", "centralized"], required=True)
parser.add_argument("--method", choices=["D", "S", "A", "B"], required=True)
parser.add_argument("--c", type=int, choices=[2, 3, 5], default=2, help="contexts per client (fl only)")
#Pilot values (marked (pilot) in NOTES.md)
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--rounds", type=int, default=100)
parser.add_argument("--epochs", type=int, default=36)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--ckpt", default=None, help="run dir to resume from")
parser.add_argument("--benchmark", action="store_true", help="pilot step 1: memory and images/s per batch size")
args = parser.parse_args()
assert not (args.method == "B" and args.setting == "centralized"), "B has no centralized version"
assert not (args.method == "B" and args.c == 5), "B equals A at c = 5"

#Paths (devkit COCO export, see NOTES.md step 0; file names depend on the devkit's Python version)
ZOD_ROOT   = "/project/home/p201222/zod"
TRAIN_JSON = f"{ZOD_ROOT}/coco/zod_full_Anonymization.BLUR_train.json"
VAL_JSON   = f"{ZOD_ROOT}/coco/zod_full_Anonymization.BLUR_val.json"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#There are graph breaks in the MoE model this allows torch.compile to work better
torch._dynamo.config.cache_size_limit = 64

seed = 42
batch_size = args.batch_size

#Federated
num_clients = 20
client_frac = 0.4
local_epochs = 1
num_rounds = args.rounds
server_lr = 5e-3
server_betas = (0.9, 0.99)
server_eps = 1e-5
base_lr = args.lr if args.lr is not None else 0.01
start_lr = base_lr / 10
end_lr = 1e-6
warmup_rounds = 5

#Centralized
num_epochs = args.epochs
central_lr = args.lr if args.lr is not None else 1e-4
weight_decay = 0.05
warmup_epochs = 1

#MoE: one expert per road type
num_experts = len(ZOD.road_types)
top_k = 1
mlp_ratio = 2
capacity_ratio = 1.0
bpr = args.method == "S"
context_aware = args.method in ("A", "B")

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
    "contexts_per_client": args.c if args.setting == "fl" else None,
    "fed": {
        "agg": "context_expert_fedadam" if args.method == "B" else "fedadam",
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
        "sched": "lin_warmup_cosine_annealing",
        "base_lr": base_lr if args.setting == "fl" else central_lr,
        "start_lr": start_lr,
        "end_lr": end_lr,
        "warmup": warmup_rounds if args.setting == "fl" else warmup_epochs,
        "num_epochs": num_epochs if args.setting == "centralized" else None,
        "kwargs": opt_kwargs,
    },
    "data": {
        "dataset": "zod_frames_full_blur",
        "train_json": TRAIN_JSON,
        "val_json": VAL_JSON,
        "contexts": ZOD.road_types,
        "classes": ZOD.classes,
        "batch_size": batch_size,
        "train_short_side": [640, 800],
        "test_short_side": 800,
        "max_size": 1333,
        "hflip": 0.5,
    },
    "model": {
        "detector": "faster_rcnn_fpn_p2-p6",
        "backbone": "convnext_tiny_scratch",
        "moe": None if args.method == "D" else {
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
    if args.method == "D":
        return convnext_detector_fn(num_classes=len(ZOD.classes) + 1)
    return convnext_moe_detector_fn(num_experts, top_k, mlp_ratio, capacity_ratio,
                                    num_classes=len(ZOD.classes) + 1, bpr=bpr, context_aware=context_aware)

def scheduler_fn(optimizer):
    return SequentialLR(optimizer, schedulers=[
        LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs),
        CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=end_lr),
    ], milestones=[warmup_epochs])

def benchmark(train, steps=50):
    model = compile_model(model_fn().to(device))
    optimizer = opt_fn(model, opt_kwargs)
    for bs in (4, 8, 12, 16):
        try:
            #Warm-up pass compiles for the new shapes, then a timed pass
            warmup = DataLoader(Subset(train, range(bs * 5)), batch_size=bs, collate_fn=ZOD.collate, **dl_kwargs)
            timed = DataLoader(Subset(train, range(bs * steps)), batch_size=bs, collate_fn=ZOD.collate, **dl_kwargs)
            train_client(model, warmup, device, optimizer, None, None, torch.bfloat16, log=False)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.time()
            train_client(model, timed, device, optimizer, None, None, torch.bfloat16, log=False)
            torch.cuda.synchronize()
            print(f"batch {bs:2d}: {bs * steps / (time.time() - t):.1f} images/s, "
                  f"peak memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
        except torch.cuda.OutOfMemoryError:
            print(f"batch {bs:2d}: out of memory")
            break

if __name__ == "__main__":
    train = ZOD(TRAIN_JSON, ZOD_ROOT, train=True)
    val = ZOD(VAL_JSON, ZOD_ROOT)

    if args.benchmark:
        benchmark(train)
        raise SystemExit

    cfg["data"]["context_counts"] = {"train": zod_context_counts(train), "val": zod_context_counts(val)}
    name = f"zod_{args.setting}_{args.method}" + (f"_c{args.c}" if args.setting == "fl" else "")
    #All torchrun ranks need the same run dir, so name it after the Slurm job when there is one
    run_id = f"{name}_{os.environ['SLURM_JOB_ID']}" if "SLURM_JOB_ID" in os.environ else None
    ctx = None if args.ckpt else init_run(name, cfg, run_id=run_id)

    if args.setting == "fl":
        clients = init_clients_zod(train, c=args.c, num_clients=num_clients, batch_size=batch_size,
                                   dl_kwargs=dl_kwargs, seed=seed)
        val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, collate_fn=ZOD.collate, **val_dl_kwargs)
        lr_sched = lr_schedule(base_lr=base_lr, warmup_rounds=warmup_rounds,
                               total_rounds=num_rounds, start_lr=start_lr, end_lr=end_lr)

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
                num_gpus=4,
                eval_every=5,
                agg_fn=context_expert_fedadam if args.method == "B" else fedadam,
                ckpt_path=args.ckpt)
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
            batch_size=batch_size,
            **dl_kwargs,
        )

import math
import torch
import time
import numpy as np
from fvcore.nn import FlopCountAnalysis, flop_count_table
from .fl_utils import _expert_id_of_key

@torch.no_grad()
def measure_ms_per_sample(model,loader,device,
                          warmup_batches=10,
                          timed_batches=50):
    model.to(device)
    model.eval()

    # Warm-up
    for i, (x, _) in enumerate(loader):
        if i >= warmup_batches:
            break
        x = x.to(device, non_blocking=True)
        model(x)

    times = []
    
    # Timed runs
    for i, (x, _) in enumerate(loader):
        if i >= timed_batches:
            break

        x = x.to(device, non_blocking=True)

        if device == "cuda":
            torch.cuda.synchronize()
        start = time.time()

        model(x)

        if device == "cuda":
            torch.cuda.synchronize()
        end = time.time()

        batch_time = end - start
        times.append(batch_time / x.size(0))  # per-sample time

    times = np.array(times) * 1000  # ms
    return times.mean(), times.std()

def model_param_bytes(model, bytes_per_param=4):
    P = sum(p.numel() for p in model.parameters())
    return P * bytes_per_param, P
    
def fedavg_comm_cost_gb(model, num_clients, num_rounds, participation, bytes_per_param=2):
    model_bytes, P = model_param_bytes(model, bytes_per_param)

    downlink_per_round = num_clients * model_bytes * participation
    uplink_per_round   = num_clients * model_bytes * participation
    total_per_round    = downlink_per_round + uplink_per_round
    total_training     = num_rounds * total_per_round

    to_gb = lambda x: x / 1e9

    return {
        "Parameters": P,
        "model_GB": to_gb(model_bytes),
        "downlink_per_round_GB": to_gb(downlink_per_round),
        "uplink_per_round_GB": to_gb(uplink_per_round),
        "total_per_round_GB": to_gb(total_per_round),
        "total_training_GB": to_gb(total_training),
    }

def context_expert_comm_cost_gb(model, client_contexts, num_rounds, participation, bytes_per_param=2,
                                download="held", sampled=None):
    shared, expert = 0, {}
    for name, p in model.named_parameters():
        e = _expert_id_of_key(name)
        if e is None:
            shared += p.numel()
        else:
            expert[e] = expert.get(e, 0) + p.numel()
    full = shared + sum(expert.values())
    held = np.array([shared + sum(expert[e] for e in ctx) for ctx in client_contexts], dtype=float)
    up = held * bytes_per_param
    down = (held if download == "held" else np.full(len(held), full, dtype=float)) * bytes_per_param
    cost = up + down  # one client, one round, bytes

    N = len(held)
    m = max(1, math.ceil(participation * N))
    #Sum of m clients drawn without replacement: mean m * mean(cost), variance with the finite-population factor
    var_round = m * cost.var() * (N - m) / (N - 1) if N > 1 else 0.0
    to_gb = lambda x: x / 1e9
    result = {
        "Parameters": full,
        "shared_parameters": shared,
        "client_parameters_min_mean_max": (int(held.min()), int(held.mean()), int(held.max())),
        "model_GB": to_gb(full * bytes_per_param),
        "downlink_per_round_GB": to_gb(m * down.mean()),
        "uplink_per_round_GB": to_gb(m * up.mean()),
        "total_per_round_GB": to_gb(m * cost.mean()),
        "total_per_round_std_GB": to_gb(math.sqrt(var_round)),
        "total_training_GB": to_gb(num_rounds * m * cost.mean()),
        "total_training_std_GB": to_gb(math.sqrt(num_rounds * var_round)),
        "fedavg_total_training_GB": to_gb(num_rounds * m * 2 * full * bytes_per_param),
    }
    if sampled is not None:
        result["realized_rounds"] = len(sampled)
        result["realized_total_training_GB"] = to_gb(sum(cost[c] for round_clients in sampled for c in round_clients))
    return result

def count_flops(model,inputs,show_table=False,max_depth=4):
    model.eval()
    with torch.no_grad():
        flops = FlopCountAnalysis(model, inputs)
        #This returns MACs (FLOP not well defined according to docs)
        macs = flops.total()

        if show_table:
            print(flop_count_table(flops, max_depth=max_depth))

    return macs

def print_expert_stats(stats, expert_label_print=False, class_names=None):
    for layer_name, s in stats.items():
        print(f"\n{'='*60}")
        print(f"Layer: {layer_name}")
        print(f"{'='*60}")

        pct = s["expert_activation_pct"]
        E = pct.shape[0]
        print(f"\nOverall expert activation:")
        for e in range(E):
            print(f"  Expert {e}: {pct[e]:.1f}%")
        
        if expert_label_print:
            print(f"\nPer-class expert activation:")
            cls_pct = s["class_expert_activation_pct"]
            header = "".join(f"{'E'+str(e):>8}" for e in range(E))
            print(f"  {'Class':<15}{header}")
            print(f"  {'-'*15}{'-'*8*E}")
            for c in range(cls_pct.shape[0]):
                name = class_names[c] if class_names else str(c)
                row = "".join(f"{cls_pct[c, e]:>7.1f}%" for e in range(E))
                print(f"  {name:<15}{row}")
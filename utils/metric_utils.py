import torch
import time
import numpy as np
from fvcore.nn import FlopCountAnalysis, flop_count_table

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
    
def fedavg_comm_cost_mb(model, num_clients, num_rounds, bytes_per_param=4):
    model_bytes, P = model_param_bytes(model, bytes_per_param)

    downlink_per_round = num_clients * model_bytes
    uplink_per_round   = num_clients * model_bytes
    total_per_round    = downlink_per_round + uplink_per_round
    total_training     = num_rounds * total_per_round

    to_mb = lambda x: x / 1e6  

    return {
        "Parameters": P,
        "model_MB": to_mb(model_bytes),
        "downlink_per_round_MB": to_mb(downlink_per_round),
        "uplink_per_round_MB": to_mb(uplink_per_round),
        "total_per_round_MB": to_mb(total_per_round),
        "total_training_MB": to_mb(total_training),
    }

def count_flops(model,inputs,show_table=False,max_depth=4,):
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
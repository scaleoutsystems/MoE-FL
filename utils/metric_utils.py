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
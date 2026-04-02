import torch
import math
import random
import gc
import ray
import logging
from copy import deepcopy
from functools import partial
from torch.amp import autocast, GradScaler
from .fl_utils import compute_prox_term, snapshot_params_fp32, set_lr,fedavg
from .experiment_tracking import log_and_checkpoint, log_client_metrics, load_checkpoint

def train_client(model,loader,device,optimizer,loss_fn, scaler, amp_dtype,
                num_epochs = 1,log = True,mix_transform=None,
                fedprox = False,mu=0.0, snap = None):
    #snap is a frozen global model only needed when fedprox enabled
    model.train()
    history = []

    has_aux = hasattr(model, "get_aux_loss")

    for epoch in range(num_epochs):
        running_base  = torch.zeros((), device=device)
        running_aux   = torch.zeros((), device=device)
        running_prox  = torch.zeros((), device=device)
        running_total = torch.zeros((), device=device)
        correct = torch.zeros((), device=device, dtype=torch.long)
        total   = torch.zeros((), device=device, dtype=torch.long)

        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if mix_transform is not None:
                x, y = mix_transform(x, y)

            optimizer.zero_grad(set_to_none=True)

            #Tesla T4 uses f16
            #A100 uses bfloat16
            with autocast(device_type="cuda", dtype=amp_dtype):
                if has_aux:
                    logits, aux = model(x, return_aux=True)
                    base_loss = loss_fn(logits, y)
                    total_loss = base_loss + aux
                else:
                    logits = model(x)
                    base_loss = loss_fn(logits, y)
                    total_loss = base_loss

            prox_term = None
            if fedprox and mu > 0.0:
                prox_term = compute_prox_term(model,snap,mu)
                total_loss = total_loss + prox_term
            
            if scaler is None:
                total_loss.backward()
                optimizer.step()
            else:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

            bs = x.size(0)
            running_base  += base_loss.detach() * bs
            running_total += total_loss.detach() * bs
            if has_aux:
                running_aux += aux.detach() * bs
            if fedprox and mu > 0:
               running_prox += prox_term.detach() * bs

            # accuracy (handles MixUp/CutMix soft labels)
            y_hard = y.argmax(dim=1) if y.ndim == 2 else y
            correct += (logits.argmax(dim=1) == y_hard).sum()
            total += bs

        epoch_base  = (running_base / total).item()
        epoch_total = (running_total / total).item()
        epoch_acc   = (correct / total).item()
        epoch_aux   = (running_aux / total).item() if has_aux else None
        epoch_prox  = (running_prox / total).item() if (fedprox and mu > 0.0) else None

        m = {
            "epoch": epoch + 1,
            "total_loss": epoch_total,
            "acc": epoch_acc,
        }
        
        if has_aux:
            m["aux_loss"] = epoch_aux

        if fedprox and mu > 0.0:
            m["prox_loss"] = epoch_prox
            
        if has_aux or fedprox:
            m["base_loss"] = epoch_base

        history.append(m)

        if log:
            parts = [f"Epoch [{epoch+1}/{num_epochs}]"]
            
            if has_aux or fedprox:
                parts.append(f"base loss: {epoch_base:.4f}")
            
            if has_aux:
                parts.append(f"aux loss: {epoch_aux:.6f}")

            if fedprox:
                parts.append(f"Prox loss: {epoch_prox:.6f}")
                parts.append(f'Prox percentage: {epoch_prox/epoch_total:.4f}')

            # Always show total + acc
            parts.append(f"total loss: {epoch_total:.4f}")
            parts.append(f"acc: {epoch_acc:.4f}")
            print(" - ".join(parts))

    return history
    
@torch.no_grad()
def evaluate(model, loader, device, loss_fn, num_classes=100, moe_stats=False):
    model.eval()

    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), device=device, dtype=torch.long)
    total_samples = torch.zeros((), device=device, dtype=torch.long)
    
    if moe_stats:
        E = model._moe_modules[0].num_experts
        layer_names = [n for n, m in model.base.named_modules() if m in model._moe_modules]
        tokens_kept = {n: torch.zeros(E, dtype=torch.long) for n in layer_names}
        class_hits  = {n: torch.zeros(num_classes, E, dtype=torch.long) for n in layer_names}

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = loss_fn(logits, y)

        bs = y.size(0)
        total_loss += loss.detach() * bs
        total_correct += (logits.argmax(dim=1) == y).sum()
        total_samples += bs
        
        if moe_stats:
            labels_cpu = y.cpu()
            for name, m in ((n, mo) for n, mo in model.base.named_modules() if mo in model._moe_modules):
                expert_id, token_idx, T = m._last_routing
                image_idx = token_idx // (T // labels_cpu.shape[0])
                tokens_kept[name].scatter_add_(0, expert_id, torch.ones_like(expert_id))
                idx = labels_cpu[image_idx] * E + expert_id
                class_hits[name].view(-1).scatter_add_(0, idx, torch.ones_like(idx))

    result = {
        "loss": (total_loss / total_samples).item(),
        "acc":  (total_correct.float() / total_samples).item(),
    }

    if moe_stats:
        result["moe_stats"] = {
            name: {
                "tokens_kept":             tokens_kept[name],
                "expert_activation_pct":       (tokens_kept[name] / (tokens_kept[name].sum() + 1e-9)) * 100,
                "class_expert_activation_pct": (class_hits[name]  / (class_hits[name].sum(1, keepdim=True) + 1e-9)) * 100,
            }
            for name in layer_names
        }
        
    return result


@ray.remote(num_gpus=1)
class ClientWorker:
    def __init__(self, model_fn, amp_dtype):
        self.device = torch.device("cuda:0")  # Ray assigns 1 GPU per actor
        self.model = torch.compile(model_fn().to(self.device))
        self.amp_dtype = amp_dtype
        self.scaler = GradScaler(device=self.device) if amp_dtype == torch.float16 else None
        
    def set_params(self, params):
        #params arrives as a regular dict, Ray serializes it once via the call itself
        self.model.load_state_dict({k: v.to(self.device) for k, v in params.items()})

    def train(self, client,  opt_fn, opt_kwargs, lr_r,
              train_loss_fn, local_epochs, mix_transform,
              fedprox, mu, snap_cpu, collect_expert_stats):

        optimizer = opt_fn(self.model, opt_kwargs)
        set_lr(optimizer, lr_r)
        snap = [p.to(self.device) for p in snap_cpu] if snap_cpu else None
        loader = client.get_loader(in_memory=False)

        hist = train_client(
            self.model, loader, self.device, optimizer,
            train_loss_fn, self.scaler, self.amp_dtype,
            num_epochs=local_epochs, log=False,
            mix_transform=mix_transform,
            fedprox=fedprox, mu=mu, snap=snap,
        )

        result = {
            "cid": client.cid,
            "num_samples": client.num_samples,
            "state": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            "history": hist,
        }

        if collect_expert_stats:
            result["moe_stats"] = self.model.pop_expert_stats()

        torch.cuda.empty_cache()
        return result
    
def run_clients_multi_gpu(selected_clients, workers, global_params, opt_fn, opt_kwargs,
                    lr_r, train_loss_fn, local_epochs, mix_transform,
                    fedprox, mu, snap, collect_expert_stats, num_gpus, **_):

    snap_cpu = [p.detach().cpu() for p in snap] if (fedprox and snap is not None) else None
    results = []
    
    global_ref = ray.put(global_params)
    ray.get([w.set_params.remote(global_ref) for w in workers])
    del global_ref
    snap_ref = ray.put(snap_cpu)

    for batch_start in range(0, len(selected_clients), num_gpus):
        batch = selected_clients[batch_start : batch_start + num_gpus]

        futures = [
            workers[rank].train.remote(
                client, opt_fn, opt_kwargs, lr_r,
                train_loss_fn, local_epochs, mix_transform,
                fedprox, mu, snap_ref, collect_expert_stats
            )
            for rank, client in enumerate(batch)
        ]

        batch_results = ray.get(futures)
        results.extend(batch_results)
    
    del snap_ref

    return results

#This should be used for single gpu with I/O bottleneck
def run_clients_io_bottleneck(selected_clients, local_model, global_params, opt_fn, opt_kwargs,
                               lr_r, train_loss_fn, local_epochs, mix_transform,
                               fedprox, mu, snap, scaler, amp_dtype, device,
                               collect_expert_stats, **_):
    
    results = []
    selected_clients[0].prefetch()

    for i, client in enumerate(selected_clients):
        print(f"Client {client.cid}")
        loader = client.get_loader(in_memory=True)
        if i + 1 < len(selected_clients):
            selected_clients[i + 1].prefetch()

        local_model.load_state_dict(global_params)
        optimizer = opt_fn(local_model, opt_kwargs)
        set_lr(optimizer, lr_r)

        hist = train_client(
            local_model, loader, device, optimizer,
            train_loss_fn, scaler, amp_dtype,
            num_epochs=local_epochs, log=True,
            mix_transform=mix_transform,
            fedprox=fedprox, mu=mu, snap=snap
        )
        
        result = {
            "cid": client.cid,
            "num_samples": client.num_samples,
            "state": {k: v.detach().cpu() for k, v in local_model.state_dict().items()},
            "history": hist,
        }

        if collect_expert_stats:
            result["moe_stats"] = local_model.pop_expert_stats()

        results.append(result)
        del loader #Delete the loader to not go over number of CPU cores available
        gc.collect()
        client.free() #release RAM

    return results
          

def run_clients_single_gpu(selected_clients,local_model, global_params, opt_fn, opt_kwargs,
                            lr_r, train_loss_fn, local_epochs, mix_transform,
                            fedprox, mu, snap, scaler, amp_dtype, device,
                            collect_expert_stats, **_):
    results = []

    for client in selected_clients:
        print(f"Client {client.cid}")
        loader = client.get_loader(in_memory=False)

        local_model.load_state_dict(global_params)
        optimizer = opt_fn(local_model, opt_kwargs)
        set_lr(optimizer, lr_r)

        hist = train_client(
            local_model, loader, device, optimizer,
            train_loss_fn, scaler, amp_dtype,
            num_epochs=local_epochs, log=True,
            mix_transform=mix_transform,
            fedprox=fedprox, mu=mu, snap=snap
        )
        
        result = {
            "cid": client.cid,
            "num_samples": client.num_samples,
            "state": {k: v.detach().cpu() for k, v in local_model.state_dict().items()},
            "history": hist,
        }

        if collect_expert_stats:
            result["moe_stats"] = local_model.pop_expert_stats()

        results.append(result)

    return results

def run_clients(selected_clients, num_gpus, io_bottleneck,workers, **kwargs):
    
    if num_gpus > 1:
        return run_clients_multi_gpu(selected_clients,workers=workers,num_gpus=num_gpus, **kwargs)
    if num_gpus == 1 and io_bottleneck:
        return  run_clients_io_bottleneck(selected_clients, **kwargs)
    else:
        return run_clients_single_gpu(selected_clients, **kwargs)

def fl_loop(clients,model_fn,opt_fn,
            train_loss_fn,eval_loss_fn,
            lr_sched,mix_transform,
            val_loader,device,
            ctx,fl_kwargs,opt_kwargs,
            eval_every=1, agg_fn=fedavg,
            amp_dtype=torch.bfloat16, 
            num_gpus=1, io_bottleneck=False,
            ckpt_path=None):
    
    client_frac = fl_kwargs['client_frac']
    num_clients = fl_kwargs['num_clients']
    num_rounds = fl_kwargs['num_rounds']
    local_epochs = fl_kwargs['local_epochs']
    fedprox = fl_kwargs['fedprox']
    mu = fl_kwargs['mu']
    
    global_model = model_fn()
    
    if ckpt_path is not None:
       start_round, ctx, global_model, _ = load_checkpoint(ckpt_path,global_model,None)
       global_model = torch.compile(global_model).to(device)
    else:
        start_round = 0
        global_model = torch.compile(global_model).to(device)
    
    if num_gpus == 1:
        local_model = torch.compile(model_fn().to(device))
    else:
        local_model = None
    global_params = deepcopy(global_model.state_dict())
    
    collect_expert_stats = getattr(global_model, "collect_expert_stats", False)
    
    if amp_dtype == torch.float16:
        scaler = GradScaler(device=device)
    else:
        scaler = None
        
    if num_gpus > 1:
        ray.init(include_dashboard=False,logging_level=logging.ERROR)
        workers = [ClientWorker.remote(model_fn, amp_dtype) for _ in range(num_gpus)]
        print(f"Started {num_gpus} persistent Ray workers")
    else:
        workers = None

    # Federated loop
    for r in range(start_round, num_rounds):
        print(f"\n=== Round {r} ===")
        
        lr_r = float(lr_sched[r])
        m = int(max(1, math.ceil(client_frac * num_clients)))
        selected_clients = random.sample(clients,k=m)
        
        #Global model snapshot for fedprox
        if fedprox:
            snap = snapshot_params_fp32(global_model,device)
        else:
            snap = None

        results = run_clients(
            selected_clients=selected_clients,
            num_gpus=num_gpus,
            workers=workers,
            io_bottleneck=io_bottleneck,
            local_model=local_model,
            global_params=global_params,
            opt_fn=opt_fn,
            opt_kwargs=opt_kwargs,
            lr_r=lr_r,
            train_loss_fn=train_loss_fn,
            local_epochs=local_epochs,
            mix_transform=mix_transform,
            fedprox=fedprox, mu=mu, snap=snap,
            scaler=scaler,
            amp_dtype=amp_dtype,
            device=device,
            collect_expert_stats=collect_expert_stats
        )

        client_states = [r["state"] for r in results]
        client_ns     = [r["num_samples"] for r in results]
        client_metrics = [{"cid": r["cid"], "num_samples": r["num_samples"],
                           "history": r["history"]} for r in results]

        if collect_expert_stats:
            client_expert_stats = [r["moe_stats"] for r in results]
            round_agg_fn = partial(agg_fn, moe_stats=client_expert_stats) 
        else:
            round_agg_fn = agg_fn

        log_client_metrics(ctx,r,client_metrics)
        
        # Aggregate
        global_params = round_agg_fn(client_states, client_ns,device="cpu")
        global_model.load_state_dict(global_params)
            
        if (r % eval_every) == 0 or (r == num_rounds-1):
            va = evaluate(global_model, val_loader, device, loss_fn=eval_loss_fn)

            print(f"[Server Eval] val loss {va['loss']:.4f} acc {va['acc']:.4f}")
            
            metrics = {
                "lr": lr_r,
                "val_loss": float(va["loss"]),
                "val_acc": float(va["acc"]),
            }

            saved = log_and_checkpoint(ctx,r,metrics,global_model, 
                                    ema_model=None, best_key="val_acc", mode="max",)

            if saved:
                print(f"New best at round {r}: {metrics['val_acc']:.4f}")
                
    if num_gpus > 1:
        ray.shutdown()
        
    return global_model
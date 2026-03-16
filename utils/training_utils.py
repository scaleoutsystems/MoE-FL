import torch
import math
import random
from copy import deepcopy
from functools import partial
from torch.amp import autocast, GradScaler
from .fl_utils import compute_prox_term, snapshot_params_fp32, set_lr,fedavg
from .experiment_tracking import log_and_checkpoint, log_client_metrics
import gc

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
    
def fl_loop(clients,model_fn,opt_fn,
            train_loss_fn,eval_loss_fn,
            lr_sched,mix_transform,
            val_loader,device,
            ctx,fl_kwargs,opt_kwargs,
            eval_every=1, agg_fn=fedavg,
            amp_dtype=torch.bfloat16):
    
    client_frac = fl_kwargs['client_frac']
    num_clients = fl_kwargs['num_clients']
    num_rounds = fl_kwargs['num_rounds']
    local_epochs = fl_kwargs['local_epochs']
    fedprox = fl_kwargs['fedprox']
    mu = fl_kwargs['mu']
    
    global_model = torch.compile(model_fn().to(device))
    local_model = torch.compile(model_fn().to(device))
    global_params = deepcopy(global_model.state_dict())
    
    collect_expert_stats = getattr(global_model, "collect_expert_stats", False)
    
    if amp_dtype == torch.float16:
        scaler = GradScaler(device=device)
    else:
        scaler = None

    # Federated loop
    for r in range(num_rounds):
        print(f"\n=== Round {r} ===")
        lr_r = float(lr_sched[r])
        client_states, client_ns, client_metrics = [], [], []
        
        if collect_expert_stats:
            client_expert_stats = []
        
        m = int(max(1, math.ceil(client_frac * num_clients)))
        selected_clients = random.sample(clients,k=m)
        
        #Global model snapshot for fedprox
        if fedprox:
            snap = snapshot_params_fp32(global_model,device)
        else:
            snap = None

        selected_clients[0].prefetch()

        for i, client in enumerate(selected_clients):
            print(f"Client {client.cid}")
            loader = client.get_loader()
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
            
            metric = {"cid": int(client.cid),
                "num_samples": int(client.num_samples),
                "history": hist,}
            
            if collect_expert_stats:
                moe_stats = local_model.get_reset_expert_stats()
                metric["moe_stats"] = moe_stats
                client_expert_stats.append(moe_stats)

            client_metrics.append(metric)
            client_states.append({k: v.detach().cpu() for k, v in local_model.state_dict().items()})
            client_ns.append(client.num_samples)

            del loader #Delete the loader to not go over number of CPU cores available
            gc.collect()
            client.free()  #release RAM

        log_client_metrics(ctx,r,client_metrics)
        
        if collect_expert_stats:
            agg_fn = partial(agg_fn, moe_stats=client_expert_stats)
        
        # Aggregate
        global_params = agg_fn(client_states, client_ns,device="cpu")
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
        
    return global_model


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
                "expert_activation_pct":       (tokens_kept[name] / (tokens_kept[name].sum() + 1e-9)) * 100,
                "class_expert_activation_pct": (class_hits[name]  / (class_hits[name].sum(1, keepdim=True) + 1e-9)) * 100,
            }
            for name in layer_names
        }
        
    return result
    
#For local testing
# def fl_loop_local(clients,model_fn,opt_fn,
#             train_loss_fn,eval_loss_fn,
#             lr_sched,mix_transform,
#             val_loader,train_eval_loader,device,
#             ctx,fl_kwargs,opt_kwargs,
#             eval_every=1, agg_fn=fedavg,
#             amp_dtype=torch.float16):
    
#     client_frac = fl_kwargs['client_frac']
#     num_clients = fl_kwargs['num_clients']
#     num_rounds = fl_kwargs['num_rounds']
#     local_epochs = fl_kwargs['local_epochs']
#     fedprox = fl_kwargs['fedprox']
#     mu = fl_kwargs['mu']
    
#     global_model = model_fn().to(device)
#     local_model = model_fn().to(device)
#     global_params = deepcopy(global_model.state_dict())
    
#     collect_stats = getattr(global_model, "collect_expert_stats", False)
    
#     if amp_dtype == torch.float16:
#         scaler = GradScaler(device=device)
#     else:
#         scaler = None

#     # Federated loop
#     for r in range(num_rounds):
#         print(f"\n=== Round {r} ===")
#         lr_r = float(lr_sched[r])
#         client_states, client_ns, client_metrics = [], [], []
        
#         if collect_stats:
#             client_expert_stats = []
        
#         m = int(max(1, math.ceil(client_frac * num_clients)))
#         selected_clients = random.sample(clients,k=m)
        
#         #Global model snapshot for fedprox
#         if fedprox:
#             snap = snapshot_params_fp32(global_model,device)
#         else:
#             snap = None

#         for client in selected_clients:

#             local_model.load_state_dict(global_params)
#             optimizer = opt_fn(local_model, opt_kwargs)
#             set_lr(optimizer, lr_r)

#             hist = train_client(
#                 local_model, client.get_loader(), device, optimizer,
#                 train_loss_fn, scaler, amp_dtype,
#                 num_epochs=local_epochs, log=True,
#                 mix_transform=mix_transform,
#                 fedprox=fedprox, mu=mu, snap=snap
#             )

#             metric = {"cid": int(client.cid),
#                 "num_samples": int(client.num_samples),
#                 "history": hist,}
            
#             if collect_stats:
#                 moe_stats = local_model.get_reset_expert_stats()
#                 metric["moe_stats"] = moe_stats
#                 client_expert_stats.append(moe_stats)

#             client_metrics.append(metric)
#             client_states.append({k: v.detach().cpu() for k, v in local_model.state_dict().items()})
#             client_ns.append(client.num_samples)

#         if collect_stats:
#             agg_fn = partial(agg_fn, moe_stats=client_expert_stats)
        
#         # Aggregate
#         global_params = agg_fn(client_states, client_ns, device="cpu")
#         global_model.load_state_dict(global_params)
            
#         if (r % eval_every) == 0 or (r == num_rounds-1):
#             va = evaluate(global_model, val_loader, device, loss_fn=eval_loss_fn)
#             tr = evaluate(global_model, train_eval_loader, device, loss_fn=eval_loss_fn)

#             print(f"[Server Eval] val loss {va['loss']:.4f} acc {va['acc']:.4f} train loss {tr['loss']:.4f} acc {tr['acc']:.4f}")
 
#     return global_model
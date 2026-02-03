import json, time, datetime, os
import torch
from pathlib import Path

def log_metrics(ctx, round_idx, metrics):
    record = {
        "t": time.time(),
        "round": int(round_idx),
    }
    record.update(metrics)

    with ctx["metrics_path"].open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

def init_run(name, config, run_root = "runs", run_id = None):
    
    if run_id is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{name}_{ts}"

    run_dir = Path(run_root) / run_id
    ckpt_dir = run_dir / "checkpoints"
    metrics_path = run_dir / "metrics.jsonl"

    # Create directories
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot once 
    if config is not None:
        with (run_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, sort_keys=True)

    # Write header line to metrics 
    if not metrics_path.exists():
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), "event": "init", "run_id": run_id}) + "\n")

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "ckpt_dir": ckpt_dir,
        "metrics_path": metrics_path,
        "best": None,  
    }

def save_checkpoint(ctx, round_idx, global_model, ema_model, clients, metrics=None):
    payload = {
        "round": int(round_idx),
        "global_model": global_model.state_dict(),
        "ema_model": ema_model.state_dict() if ema_model is not None else None,
        "clients_opt_state": [c.opt_state for c in clients],
        "metrics": metrics or {},
    }

    ckpt_path = ctx["ckpt_dir"] / "last.pt"
    tmp_path = ckpt_path.with_suffix(".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, ckpt_path)
    
    
def save_best_checkpoint(ctx, round_idx, score, global_model, ema_model, clients, metrics=None, mode="max"):
    if mode not in ("max", "min"):
        raise ValueError("mode must be 'max' or 'min'")

    score = float(score)
    best = ctx.get("best", None)

    improved = (best is None) or ((score > best) if mode == "max" else (score < best))
    if not improved:
        return False

    ctx["best"] = score

    payload = {
        "round": int(round_idx),
        "best_score": score,
        "global_model": global_model.state_dict(),
        "ema_model": ema_model.state_dict() if ema_model is not None else None,
        "clients_opt_state": [c.opt_state for c in clients],
        "metrics": metrics or {},
    }

    ckpt_path = ctx["ckpt_dir"] / "best.pt"
    tmp_path = ckpt_path.with_suffix(".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, ckpt_path)
    return True

def log_and_checkpoint(ctx,round_idx,metrics,
                       global_model,ema_model,
                       clients,best_key="val_acc",mode="max",):

    log_metrics(ctx, round_idx, metrics)
    save_checkpoint(ctx, round_idx, global_model, ema_model, clients, metrics)

    score = metrics[best_key]
    saved = save_best_checkpoint(
        ctx,
        round_idx,
        score=score,
        global_model=global_model,
        ema_model=ema_model,
        clients=clients,
        metrics=metrics,
        mode=mode,
    )

    return saved
    
def load_checkpoint(ctx, global_model, ema_model, clients, map_location="cpu"):
    ckpt_path = ctx["ckpt_dir"] / "last.pt"
    if not ckpt_path.exists():
        return 0

    ckpt = torch.load(ckpt_path, map_location=map_location)

    global_model.load_state_dict(ckpt["global_model"])
    if ema_model is not None and ckpt.get("ema_model") is not None:
        ema_model.load_state_dict(ckpt["ema_model"])

    opt_states = ckpt.get("clients_opt_state")
    if opt_states is not None:
        for c, st in zip(clients, opt_states):
            c.opt_state = st

    # resume at next round
    return int(ckpt["round"]) + 1
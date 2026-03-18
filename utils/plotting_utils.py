import json
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import matplotlib.cm as cm
import numpy as np

def plot_training(jsonl_path: str) -> plt.Figure:
    rounds, val_acc, val_loss = [], [], []

    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            if "round" in r:
                rounds.append(r["round"])
                val_acc.append(r["val_acc"] * 100)
                val_loss.append(r["val_loss"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Training Metrics", fontsize=14, fontweight="bold")

    for ax, values, label, color, fmt, loc in [
    (ax1, val_acc,  "Validation Accuracy (%)", "#4C72B0", "{:.2f}%", "lower right"),
    (ax2, val_loss, "Validation Loss",         "#DD8452", "{:.4f}",  "upper right"),
    ]:
        ax.plot(rounds, values, linewidth=2, color=color)

        box = AnchoredText(
            f"Final: {fmt.format(values[-1])}",
            loc=loc,  
            prop=dict(size=10, fontweight="bold", color=color),
            frameon=True,
            pad=0.5,
            borderpad=0.8,
        )
        box.patch.set(edgecolor=color, linewidth=1.2, facecolor="white", alpha=0.9)
        ax.add_artist(box)

        ax.set_xlabel("Round")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig("figs/run_2_global.png",dpi=800)
    
def plot_fl_client_training(log_path: str, metric: str = "acc", figsize=(10, 5)):
    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "round" not in obj or "cid" not in obj:
                continue
            last_epoch = obj["history"][-1]
            records.append({
                "round": obj["round"] + 1,
                "cid":   obj["cid"],
                "value": last_epoch[metric],
            })

    clients = sorted(set(r["cid"] for r in records))
    colors  = cm.tab20(np.linspace(0, 1, len(clients)))
    cmap    = {cid: colors[i] for i, cid in enumerate(clients)}

    all_rounds = sorted(set(r["round"] for r in records))

    fig, ax = plt.subplots(figsize=figsize)

    for cid in clients:
        pts = sorted((r["round"], r["value"]) for r in records if r["cid"] == cid)
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=cmap[cid], linewidth=0.8)

    ax.set_xlabel("Communication round")
    ax.set_ylabel("Client Accuracy")
    ax.set_xticks(range(0, all_rounds[-1] + 1, 50))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    plt.savefig("figs/run_2_clients.png", dpi=800)


plot_training("imagenet_convnext_fl_20260316_144213/metrics.jsonl")
plot_fl_client_training("imagenet_convnext_fl_20260316_144213/client_metrics.jsonl")
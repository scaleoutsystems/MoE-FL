import json
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import matplotlib.cm as cm
import numpy as np
import os
import seaborn as sns
from pathlib import Path

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
    plt.savefig(os.path.join(os.path.dirname(jsonl_path), "global.png"),dpi=800)
    
def plot_fl_client_training(jsonl_path: str, metric: str = "acc", figsize=(10, 5)):
    records = []
    with open(jsonl_path) as f:
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
    plt.savefig(os.path.join(os.path.dirname(jsonl_path), "clients.png"),dpi=800)

def plot_expert_activation_barplot(save_path, va, layer_name, moe_index):
    save_path = Path(save_path)
    stats = va["moe_stats"][layer_name]
    data = np.array(stats["expert_activation_pct"].tolist())
    n_experts = len(data)

    fig, ax = plt.subplots(figsize=(max(6, n_experts * 0.8), 4))
    bars = ax.bar(
        [f"E{i}" for i in range(n_experts)],
        data,
        color=sns.color_palette("Blues_d", n_experts),
        edgecolor="white",
        linewidth=0.5,
    )
    # for bar, val in zip(bars, data):
    #     ax.text(
    #         bar.get_x() + bar.get_width() / 2,
    #         bar.get_height() + 1.0,
    #         f"{val:.1f}%",
    #         ha="center", va="bottom", fontsize=9,
    #     )
    ax.set_title(f"Expert activation — MoE Layer {moe_index}", fontsize=13)
    ax.set_xlabel("Expert")
    ax.set_ylabel("Activation %")
    #ax.set_ylim(0, 110)
    #ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.axhline(y=25, color="red", linestyle="--", linewidth=1.0)
    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path / f"expert_activation_moe_layer_{moe_index}.png", dpi=800)
    plt.close()

def plot_class_expert_heatmap(save_path, va, layer_name, moe_index):
    save_path = Path(save_path)
    stats = va["moe_stats"][layer_name]
    data = np.array(stats["class_expert_activation_pct"].tolist()).T
    n_experts, n_classes = data.shape

    fig, ax = plt.subplots(figsize=(max(6, n_classes * 0.08), max(4, n_experts * 0.8)))
    sns.heatmap(
        data,
        ax=ax,
        annot=False,
        fmt=".1f",
        cmap="rocket",
        linewidths=0.0,
        #linecolor="white",
        #xticklabels=[str(i) if i % 10 == 0 else "" for i in range(n_classes)],
        xticklabels=[str(i) if i % 10 == 0 else "" for i in range(1, n_classes + 1)],
        yticklabels=[f"E{i}" for i in range(n_experts)],
        cbar_kws={"label": "Activation %"},
    )
    ax.set_title(f"Class and expert activation — MoE Layer {moe_index}", fontsize=13)
    ax.set_xlabel("Class")
    ax.set_ylabel("Expert")
    ax.tick_params(left=False, bottom=False)
    plt.tight_layout()
    plt.savefig(save_path / f"class_expert_activation_moe_layer_{moe_index}.png", dpi=800)
    plt.close()
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

def plot_training_comparison(runs_dir: str, base_lrs: list[float]) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Metrics Comparison", fontsize=14, fontweight="bold")

    run_dirs = sorted([
        os.path.join(runs_dir, d)
        for d in os.listdir(runs_dir)
        if os.path.isdir(os.path.join(runs_dir, d))
    ])

    lines = []
    for run_dir, lr in zip(run_dirs, base_lrs):
        jsonl_path = os.path.join(run_dir, "metrics.jsonl")
        rounds, val_acc, val_loss = [], [], []
        with open(jsonl_path) as f:
            for line in f:
                r = json.loads(line)
                if "round" in r and "val_acc" in r:
                    rounds.append(r["round"])
                    val_acc.append(r["val_acc"] * 100)
                    val_loss.append(r["val_loss"])

        label = f"lr={lr}"
        l, = ax1.plot(rounds, val_acc, linewidth=2, label=label)
        ax2.plot(rounds, val_loss, linewidth=2, label=label, color=l.get_color())
        lines.append((lr, val_acc, val_loss, l.get_color()))

    for ax, label, loc in [
        (ax1, "Validation Accuracy (%)", "lower right"),
        (ax2, "Validation Loss",         "upper right"),
    ]:
        ax.set_xlabel("Round")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc=loc)

    # Text box with stats per run
    acc_text  = "\n".join([
        f"lr={lr}: final={acc[-1]:.2f}%"
        for lr, acc, _, _ in lines
    ])
    loss_text = "\n".join([
        f"lr={lr}: final={loss[-1]:.4f}"
        for lr, _, loss, _ in lines
    ])

    for ax, text in [(ax1, acc_text), (ax2, loss_text)]:
        ax.text(
            0.02, 0.02, text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="grey", alpha=0.8),
            family="monospace",
        )

    plt.tight_layout()
    plt.savefig(os.path.join(runs_dir, "comparison.png"), dpi=800)
    return fig

def plot_training_comparison_aug(runs_dir: str, labels: list[str], filter_str: str = "") -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Metrics Comparison", fontsize=14, fontweight="bold")

    run_dirs = sorted([
        os.path.join(runs_dir, d)
        for d in os.listdir(runs_dir)
        if os.path.isdir(os.path.join(runs_dir, d)) and filter_str in d
    ])

    lines = []
    for run_dir, label in zip(run_dirs, labels):
        jsonl_path = os.path.join(run_dir, "metrics.jsonl")
        rounds, val_acc, val_loss = [], [], []
        with open(jsonl_path) as f:
            for line in f:
                r = json.loads(line)
                if "round" in r and "val_acc" in r:
                    rounds.append(r["round"])
                    val_acc.append(r["val_acc"] * 100)
                    val_loss.append(r["val_loss"])

        l, = ax1.plot(rounds, val_acc, linewidth=2, label=label)
        ax2.plot(rounds, val_loss, linewidth=2, label=label, color=l.get_color())
        lines.append((label, val_acc, val_loss))

    for ax, ylabel, loc in [
        (ax1, "Validation Accuracy (%)", "lower right"),
        (ax2, "Validation Loss",         "upper right"),
    ]:
        ax.set_xlabel("Round")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc=loc)

    acc_text  = "\n".join([f"{lbl}: final={acc[-1]:.2f}%"  for lbl, acc, _    in lines])
    loss_text = "\n".join([f"{lbl}: final={loss[-1]:.4f}"  for lbl, _,   loss in lines])

    for ax, text in [(ax1, acc_text), (ax2, loss_text)]:
        ax.text(
            0.02, 0.02, text,
            transform=ax.transAxes, fontsize=8, verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="grey", alpha=0.8),
            family="monospace",
        )

    plt.tight_layout()
    plt.savefig(os.path.join(runs_dir, "comparison_aug.png"), dpi=800)
    return fig
    
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

def plot_class_expert_heatmap_final(save_path, va, layer_names, moe_indices):
    save_path = Path(save_path)
    n_layers = len(layer_names)

    all_data = []
    for layer_name in layer_names:
        data = np.array(va["moe_stats"][layer_name]["class_expert_activation_pct"].tolist()).T
        all_data.append(data)

    fig, axes = plt.subplots(
        n_layers, 1,
        figsize=(max(10, 100 * 0.08), max(3, 6) * n_layers)
    )
    if n_layers == 1:
        axes = [axes]

    for ax, data, layer_name, moe_index in zip(axes, all_data, layer_names, moe_indices):
        n_experts, n_classes = data.shape
        vmin = data.min()
        vmax = data.max()

        sns.heatmap(
            data,
            ax=ax,
            annot=False,
            cmap="rocket",
            vmin=vmin,
            vmax=vmax,
            linewidths=0.0,
            cbar=True,
            cbar_kws={"shrink": 0.6, "label": "Routing fraction"},
            xticklabels=[str(i) if i % 10 == 0 or i == n_classes - 1 else "" for i in range(0, n_classes)],
            yticklabels=[f"E{i}" for i in range(n_experts)],
        )
        ax.set_title(f"Class and expert activation — MoE Layer {moe_index}", fontsize=26)
        ax.set_xlabel("Class", fontsize=22)
        ax.set_ylabel("Expert", fontsize=22)
        ax.tick_params(axis="both", labelsize=18, left=False, bottom=False)

        cbar = ax.collections[0].colorbar
        cbar.set_label("Routing fraction", fontsize=20)
        cbar.ax.tick_params(labelsize=16)

    plt.tight_layout()
    plt.savefig(save_path / "class_expert_activation_combined.png", dpi=800, bbox_inches="tight")
    plt.close()

def plot_spatial_expert_heatmap_final(save_path, va, layer_names, moe_indices, grid_sizes):
    save_path = Path(save_path)
    n_layers = len(layer_names)

    all_data = []
    all_ranges = []
    for layer_name in layer_names:
        data = np.array(va["moe_stats"][layer_name]["spatial_expert_activation"].tolist())
        token_sums = data.sum(axis=0, keepdims=True)
        data = data / (token_sums + 1e-9)
        all_data.append(data)
        all_ranges.append((data.min(), data.max()))

    n_experts = all_data[0].shape[0]
    fig, axes = plt.subplots(n_layers, n_experts, figsize=(n_experts * 3, n_layers * 3))
    fig.subplots_adjust(right=0.88)

    if n_layers == 1:
        axes = axes[np.newaxis, :]
    if n_experts == 1:
        axes = axes[:, np.newaxis]

    for row, (data, layer_name, moe_index, (grid_h, grid_w), (vmin, vmax)) in enumerate(
        zip(all_data, layer_names, moe_indices, grid_sizes, all_ranges)
    ):
        for e in range(n_experts):
            ax = axes[row, e]
            spatial = data[e].reshape(grid_h, grid_w)
            sns.heatmap(
                spatial,
                ax=ax,
                annot=False,
                cmap="rocket",
                vmin=vmin,
                vmax=vmax,
                cbar=False,
                xticklabels=False,
                yticklabels=False,
                linewidths=0.0,
            )
            if row == 0:
                ax.set_title(f"E{e}", fontsize=18)

        axes[row, 0].set_ylabel(f"MoE Layer {moe_index}", fontsize=16)

        cax = fig.add_axes([
            0.90,
            axes[row, -1].get_position().y0,
            0.02,
            axes[row, -1].get_position().height
        ])
        sm = plt.cm.ScalarMappable(cmap="rocket", norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax, label="Routing fraction")
        cbar.set_label("Routing fraction", fontsize=14)
        cbar.ax.tick_params(labelsize=12)

    plt.savefig(save_path / "spatial_expert_activation_combined.png", dpi=800, bbox_inches="tight")
    plt.close()

def plot_training_final(jsonl_paths: list[str], labels: list[str], colors: list[str] = None, linestyles: list[str] = None, save_path: str = None) -> plt.Figure:
    default_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
    if colors is None:
        colors = default_colors
    if linestyles is None:
        linestyles = ["-"] * len(jsonl_paths)

    fig, ax = plt.subplots(figsize=(10, 6))

    for jsonl_path, label, color, ls in zip(jsonl_paths, labels, colors, linestyles):
        rounds, val_acc = [], []
        with open(jsonl_path) as f:
            for line in f:
                r = json.loads(line)
                if "round" in r:
                    rounds.append(r["round"])
                    val_acc.append(r["val_acc"] * 100)

        ax.plot(rounds, val_acc, linewidth=2, color=color, linestyle=ls, label=f"{label} ({val_acc[-1]:.2f}%)")

    ax.set_xlabel("Round", fontsize=16)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(loc="lower right", fontsize=14, frameon=True, framealpha=0.9, edgecolor="grey")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

    out_path = save_path or os.path.join(os.path.dirname(jsonl_paths[0]), "global_combined.png")
    plt.savefig(out_path, dpi=800)
    plt.close()
    return fig

def plot_load_balance_final(save_path, logs_list: list[list[dict]], labels: list[str], colors: list[str] = None):
    save_path = Path(save_path)
    default_colors = ["#4C72B0", "#DD8452", "#55A868"]
    if colors is None:
        colors = default_colors

    layer_names = list(logs_list[0][0]["global_load_balance"].keys())
    n_layers = len(layer_names)
    n_experts = len(logs_list[0][0]["global_load_balance"][layer_names[0]])
    n_rows = len(logs_list)

    fig, axes = plt.subplots(n_rows, n_layers, figsize=(6 * n_layers, 5 * n_rows), sharey=True)
    axes = np.array(axes).reshape(n_rows, n_layers)

    expert_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"][:n_experts]

    for row, (logs, label) in enumerate(zip(logs_list, labels)):
        rounds = [l["round"] for l in logs]
        for col, layer in enumerate(layer_names):
            ax = axes[row, col]
            data = np.array([l["global_load_balance"][layer] for l in logs])

            for expert_idx in range(n_experts):
                ax.plot(rounds, data[:, expert_idx], color=expert_colors[expert_idx],
                        linewidth=1.8, label=f"Expert {expert_idx}")

            ax.axhline(1 / n_experts, color="black", linewidth=0.8, linestyle="--",
                       alpha=0.5, label="Uniform")
            ax.set_title(f"MoE Layer {col}", fontsize=18)
            ax.set_xlabel("Round", fontsize=16)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="both", labelsize=14)

        axes[row, 0].set_ylabel(f"{label}\nExpert load fraction", fontsize=16)

    axes[0, 0].legend(fontsize=13)
    #fig.suptitle("Expert load fractions over rounds", fontsize=20)
    plt.tight_layout()
    plt.savefig(save_path / "load_balance.png", dpi=800, bbox_inches="tight")
    plt.close()

def plot_routing_agreement_final(save_path, logs_list: list[list[dict]], labels: list[str], colors: list[str] = None):
    save_path = Path(save_path)
    default_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
    if colors is None:
        colors = default_colors

    first_log = logs_list[0][0]
    per_layer = isinstance(first_log["mean_routing_agreement"], dict)

    fig, ax = plt.subplots(figsize=(6, 4))

    for logs, label, color in zip(logs_list, labels, colors):
        rounds = [l["round"] for l in logs]

        if per_layer:
            layer_names = list(first_log["mean_routing_agreement"].keys())
            for i, layer in enumerate(layer_names):
                values = [l["mean_routing_agreement"][layer] for l in logs]
                ax.plot(
                    rounds, values,
                    color=color,
                    linewidth=2,
                    linestyle=["-", "--", ":", "-."][i % 4],
                    label=f"{label} — MoE Layer {i}",
                )
        else:
            values = [l["mean_routing_agreement"] for l in logs]
            ax.plot(rounds, values, color=color, linewidth=2, label=label)

    ax.set_xlabel("Round", fontsize=16)
    ax.set_ylabel("Routing Agreement", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path / "routing_agreement.png", dpi=800, bbox_inches="tight")
    plt.close()
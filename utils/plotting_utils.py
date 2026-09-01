import json
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import matplotlib.cm as cm
import numpy as np
import os
import seaborn as sns
from pathlib import Path

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
    plt.savefig(os.path.join(runs_dir, "comparison.png"), dpi=600)
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
    plt.savefig(os.path.join(runs_dir, "comparison_aug.png"), dpi=600)
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
    plt.savefig(os.path.join(os.path.dirname(jsonl_path), "clients.png"),dpi=600)
    
# ---- shared constants ----
PANEL_H = 1.9
FS_LABEL, FS_TICK, FS_ANNOT = 9, 8, 6.5
CLASS_W, SPATIAL_W, WEATHER_W = 2.8, 2.1, 2.5
CBAR_KWS = {"shrink": 0.6, "aspect": 12, "pad": 0.02}


def _style_cbar(cbar):
    cbar.set_label("Routing fraction", fontsize=FS_LABEL, labelpad=2)
    cbar.ax.tick_params(labelsize=FS_TICK, length=0, pad=1)
    cbar.outline.set_visible(False)


def plot_class_expert_heatmap_final(save_path, va, layer_names, moe_indices):
    save_path = Path(save_path)
    for layer_name, moe_index in zip(layer_names, moe_indices):
        data = np.array(va["moe_stats"][layer_name]["class_expert_activation_pct"].tolist()).T
        n_experts, n_classes = data.shape

        fig, ax = plt.subplots(figsize=(CLASS_W, PANEL_H), layout="constrained")
        sns.heatmap(
            data, ax=ax, annot=False, cmap="rocket",
            vmin=0.1, vmax=0.5, linewidths=0.0, cbar_kws=CBAR_KWS,
            xticklabels=[str(i) if i % 20 == 0 or i == n_classes - 1 else ""
                         for i in range(n_classes)],
            yticklabels=[f"$E_{{{i}}}$" for i in range(n_experts)],
        )
        ax.set_xlabel("Class", fontsize=FS_LABEL, labelpad=2)
        ax.set_ylabel("Expert", fontsize=FS_LABEL, labelpad=2)
        ax.tick_params(axis="both", labelsize=FS_TICK, left=False, bottom=False, pad=1)
        ax.tick_params(axis="x", rotation=0)
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([0.1, 0.3, 0.5])
        _style_cbar(cbar)

        fig.savefig(save_path / "class_expert_activation_combined.png", dpi=600)
        plt.close(fig)


def plot_weather_expert_heatmap(save_path, va, layer_names, moe_indices, weather_types):
    save_path = Path(save_path)
    for layer_name, moe_index in zip(layer_names, moe_indices):
        data = np.array(va["moe_stats"][layer_name]["weather_expert_activation_pct"].tolist()).T / 100.0
        n_experts = data.shape[0]
        uniform = 1 / n_experts

        fig, ax = plt.subplots(figsize=(WEATHER_W, PANEL_H), layout="constrained")
        sns.heatmap(
            data, ax=ax, annot=True, fmt=".2f", annot_kws={"size": FS_ANNOT},
            cmap="coolwarm", center=uniform, vmin=0.01, vmax=0.95,
            linewidths=0.5, linecolor="white", cbar_kws=CBAR_KWS,
            xticklabels=weather_types,
            yticklabels=[f"$E_{{{i}}}$" for i in range(n_experts)],
        )
        ax.set_xlabel("Weather", fontsize=FS_LABEL, labelpad=2)
        ax.set_ylabel("Expert", fontsize=FS_LABEL, labelpad=2)
        ax.tick_params(axis="both", labelsize=FS_TICK, left=False, bottom=False, pad=1)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([0.1, 0.5, 0.9])
        _style_cbar(cbar)

        fig.savefig(save_path / "weather_expert_activation_combined.png", dpi=600)
        plt.close(fig)


def plot_spatial_expert_heatmap_final(save_path, va, layer_names, moe_indices, grid_sizes,
                                      n_experts=4, cmap="rocket", **_ignored):
    save_path = Path(save_path)
    for layer_name, moe_index, (grid_h, grid_w) in zip(layer_names, moe_indices, grid_sizes):
        data = np.array(va["moe_stats"][layer_name]["spatial_expert_activation"].tolist())
        data = data / (data.sum(axis=0, keepdims=True) + 1e-9)
        vmin, vmax = 0.1, 0.4

        fig, axes = plt.subplots(2, 2, figsize=(SPATIAL_W, PANEL_H), layout="constrained")
        fig.get_layout_engine().set(w_pad=0.04, h_pad=0.04, wspace=0.06, hspace=0.08)

        for e in range(n_experts):
            ax = axes[e // 2, e % 2]
            im = ax.imshow(data[e].reshape(grid_h, grid_w),
                           cmap=cmap, vmin=vmin, vmax=vmax,
                           interpolation="nearest", aspect="equal")
            #ax.set_title(f"$E_{{{e}}}$", fontsize=FS_TICK, pad=2)
            ax.set_xlabel(f"$E_{{{e}}}$", fontsize=FS_TICK, labelpad=2)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)

        cbar = fig.colorbar(im, ax=axes.ravel().tolist(),
                            orientation="vertical", shrink=0.5, aspect=16, pad=0.03)
        cbar.set_ticks([0.1, 0.2, 0.3, 0.4])
        _style_cbar(cbar)

        fig.savefig(save_path / "spatial_expert_activation_combined.png", dpi=600)
        plt.close(fig)

def plot_training(jsonl_paths: list[str], labels: list[str], colors: list[str] = None, linestyles: list[str] = None, save_path: str = None) -> plt.Figure:
    default_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
    if colors is None:
        colors = default_colors
    if linestyles is None:
        linestyles = ["-"] * len(jsonl_paths)

    fig, ax = plt.subplots(figsize=(6, 4))

    for jsonl_path, label, color, ls in zip(jsonl_paths, labels, colors, linestyles):
        rounds, val_acc = [], []
        with open(jsonl_path) as f:
            for line in f:
                r = json.loads(line)
                if "round" in r:
                    rounds.append(r["round"])
                    val_acc.append(r["val_acc"] * 100)

        ax.plot(rounds, val_acc, linewidth=2, color=color, linestyle=ls, label=f"{label} ({val_acc[-1]:.2f}%)")

    ax.set_xlabel("Epoch", fontsize=16)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(loc="lower right", fontsize=14, frameon=True, framealpha=0.9, edgecolor="grey")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

    out_path = save_path or os.path.join(os.path.dirname(jsonl_paths[0]), "global_combined.png")
    plt.savefig(out_path, dpi=600)
    plt.close()
    return fig

def plot_load_balance(save_path, logs_list: list[list[dict]], labels: list[str],
                            filenames: list[str] = None,
                            place_width=3.4, place_height=2.3, scale=1.0,
                            fs_label=11, fs_tick=10, fs_legend=9,
                            xticks=(0, 250, 500), legend_on=0,
                            shared_ylabel=False):
    """
    Creates one figure per log set with identical axis limits and identical
    axes geometry, so panels align perfectly when placed side by side.

    shared_ylabel: if True, only the first panel gets a y-axis label
                   (use when panels sit side by side in the paper).
                   Set False to label every panel.
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    layer_names = list(logs_list[0][0]["global_load_balance"].keys())
    layer = layer_names[1]
    n_experts = len(logs_list[0][0]["global_load_balance"][layer])
    expert_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"][:n_experts]

    if filenames is None:
        filenames = [f"load_balance_{i}.png" for i in range(len(logs_list))]

    all_data = [np.array([l["global_load_balance"][layer] for l in logs])
                for logs in logs_list]

    yhi = max(d.max() for d in all_data)
    ylim = (0.0, yhi * 1.05)
    xlim = (0, max(xticks))

    # Fixed axes rectangle -> identical drawable area in every figure,
    # regardless of whether a panel has a y-label or legend.
    #ax_rect = [0.17, 0.19, 0.80, 0.78]  # [left, bottom, width, height]
    ax_rect = [0.19, 0.22, 0.77, 0.75]

    paths = []
    for i, (logs, label, fname) in enumerate(zip(logs_list, labels, filenames)):
        rounds = [l["round"] for l in logs]
        data = all_data[i]

        fig = plt.figure(figsize=(place_width * scale, place_height * scale))
        ax = fig.add_axes(ax_rect)

        for e in range(n_experts):
            ax.plot(rounds, data[:, e], color=expert_colors[e],
                    linewidth=1.0 * scale, label=f"$E_{{{e}}}$")
        ax.axhline(1 / n_experts, color="black", linewidth=0.8 * scale,
                   linestyle="--", alpha=0.5, label="Uniform")

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks(list(xticks))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        #ax.grid(True, linestyle="--", alpha=0.4)
        #ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", labelsize=fs_tick * scale)
        ax.set_xlabel("Round", fontsize=fs_label * scale)

        if not shared_ylabel or i == 0:
            ax.set_ylabel("Expert load fraction", fontsize=fs_label * scale)

        if i == legend_on:
            ax.legend(fontsize=fs_legend * scale, frameon=True, ncol=2,
                      loc="upper right", handlelength=1.2, columnspacing=1.0)

        out = save_path / fname
       # fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(out, dpi=600)
        plt.close(fig)
        paths.append(out)

    return paths

def plot_routing_agreement(save_path, logs_list: list[list[dict]], labels: list[str], colors: list[str] = None):
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
    plt.savefig(save_path / "routing_agreement.png", dpi=600, bbox_inches="tight")
    plt.close()
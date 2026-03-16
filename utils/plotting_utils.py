import json
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

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
    plt.savefig("run_1.png",dpi=800)


plot_training("imagenet_convnext_fl_20260314_170908/metrics.jsonl")
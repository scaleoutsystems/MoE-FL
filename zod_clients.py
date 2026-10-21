"""Explore the natural ZOD client split (one car during one period, see split_by_car_and_time in utils/fl_utils.py)
and plot how local dataset sizes, weathers and countries are distributed over the clients.

    python zod_clients.py                                  # 20 clients on the train split
    python zod_clients.py --num-clients 30 --min-frames 50
    python zod_clients.py --split val                      # weather of the test set (no clients)

Reads metadata.json of every frame directly (no devkit needed) and caches it in
data/zod_metadata.json. Figures go to figs/ (git-ignored).
"""
import os
import argparse
from utils.fl_utils import split_by_car_and_time
from collections import Counter
from datetime import date
import numpy as np
import matplotlib.pyplot as plt

from utils.data_utils import ZOD

parser = argparse.ArgumentParser()
parser.add_argument("--root", default="data")
parser.add_argument("--split", choices=["train", "val"], default="train")
parser.add_argument("--num-clients", type=int, default=20)
parser.add_argument("--min-frames", type=int, default=1, help="frames needed for a weather to count for a client")
parser.add_argument("--out", default="figs")
args = parser.parse_args()

WEATHER_COLORS = {"clear": "#f2c94c", "cloudy": "#7f7f7f", "rain": "#1f4fbf", "fog": "#c9b3e6", "snow": "#8fe3f5"}

rows = [r for r in ZOD.load_metadata(args.root) if r["split"] == args.split]
weather_totals = Counter(r["weather"] for r in rows)
print(f"{args.split}: {len(rows)} frames | " + ", ".join(f"{w} {weather_totals[w]} ({weather_totals[w] / len(rows):.1%})" for w in ZOD.weathers))
if args.split == "val":
    raise SystemExit

splits = split_by_car_and_time([r["car"] for r in rows], [r["time"][:10] for r in rows], args.num_clients)
clients = [[rows[i] for i in idxs] for idxs in splits]
counts = np.array([[sum(r["weather"] == w for r in c) for w in ZOD.weathers] for c in clients])  # clients x weathers
sizes = counts.sum(1)
present = counts >= args.min_frames

print(f"{len(clients)} clients, {sizes.sum()} of {len(rows)} frames used ({len(rows) - sizes.sum()} dropped), "
      f"sizes {sizes.min()} / {int(np.median(sizes))} / {sizes.max()} (min / median / max)")
print(f"weathers per client (>= {args.min_frames} frames): " + ", ".join(f"{k}: {v}" for k, v in sorted(Counter(present.sum(1)).items())))
print(f"clients holding each weather: " + ", ".join(f"{w} {n}" for w, n in zip(ZOD.weathers, present.sum(0))))
print(f"\n{'#':>2s} {'car':6s} {'period':22s} {'frames':>6s}  " + " ".join(f"{w:>6s}" for w in ZOD.weathers) + "  countries")
for i, c in enumerate(clients):
    countries = "/".join(k for k, _ in Counter(r["country"] for r in c).most_common(3))
    shares = " ".join(f"{n / sizes[i]:6.0%}" for n in counts[i])
    print(f"{i:2d} {c[0]['car']:6s} {c[0]['time'][:10]}..{c[-1]['time'][:10]} {sizes[i]:6d}  {shares}  {countries}")

# --- figures
os.makedirs(args.out, exist_ok=True)
x = np.arange(len(clients))
labels = [f"{i} {c[0]['car']}" for i, c in enumerate(clients)]
fig, axes = plt.subplots(2, 2, figsize=(16, 9))

ax = axes[0, 0]
bottom = np.zeros(len(clients))
for j, w in enumerate(ZOD.weathers):
    ax.bar(x, counts[:, j], bottom=bottom, color=WEATHER_COLORS[w], label=w)
    bottom += counts[:, j]
ax.set(title="Frames per client", ylabel="frames", xticks=x, xticklabels=labels)
ax.legend(ncol=5, fontsize=8)

ax = axes[0, 1]
bottom = np.zeros(len(clients))
for j, w in enumerate(ZOD.weathers):
    ax.bar(x, counts[:, j] / sizes, bottom=bottom, color=WEATHER_COLORS[w], label=w)
    bottom += counts[:, j] / sizes
ax.set(title="Weather mix per client", ylabel="share of frames", xticks=x, xticklabels=labels, ylim=(0, 1))

ax = axes[1, 0]
k, n = np.unique(present.sum(1), return_counts=True)
ax.bar(k, n, color="#555")
ax.set(title=f"Weathers per client (>= {args.min_frames} frames)", xlabel="number of weathers", ylabel="clients", xticks=range(1, len(ZOD.weathers) + 1))

ax = axes[1, 1]
for j, w in enumerate(ZOD.weathers):
    share = np.sort(counts[:, j])[::-1] / max(counts[:, j].sum(), 1)
    ax.plot(np.arange(1, len(clients) + 1), np.cumsum(share), marker="o", ms=3, color=WEATHER_COLORS[w], label=w)
ax.set(title="How concentrated each weather is", xlabel="clients (largest share first)", ylabel="cumulative share of the weather's frames", ylim=(0, 1.02))
ax.legend(fontsize=8)

for ax in axes[0]:
    ax.tick_params(axis="x", labelrotation=90)
fig.tight_layout()
path = os.path.join(args.out, f"zod_clients_{len(clients)}.png")
fig.savefig(path, dpi=150)

# timeline: one row per car, frames per day, coloured by client
fig, ax = plt.subplots(figsize=(16, 4))
cars = sorted({c[0]["car"] for c in clients}, key=lambda car: -sum(s for c, s in zip(clients, sizes) if c[0]["car"] == car))
cmap = plt.get_cmap("tab20")
for i, c in enumerate(clients):
    per_day = Counter(r["time"][:10] for r in c)
    days = [date.fromisoformat(d) for d in per_day]
    ax.scatter(days, [cars.index(c[0]["car"])] * len(days), s=[3 + v / 10 for v in per_day.values()], color=cmap(i % 20))
    ax.annotate(str(i), (days[len(days) // 2], cars.index(c[0]["car"]) + 0.3), fontsize=8, ha="center")
ax.set(title="Client periods per car (dot size = frames that day)", yticks=range(len(cars)), yticklabels=cars, ylim=(-0.5, len(cars) - 0.3))
fig.tight_layout()
timeline = os.path.join(args.out, f"zod_clients_{len(clients)}_timeline.png")
fig.savefig(timeline, dpi=150)

# local dataset sizes: sorted per client (coloured by car) and their distribution
car_colors = {car: plt.get_cmap("Set2")(i) for i, car in enumerate(cars)}
order = np.argsort(sizes)[::-1]
fig, axes = plt.subplots(1, 2, figsize=(16, 4.5), gridspec_kw={"width_ratios": [3, 1]})
ax = axes[0]
ax.bar(np.arange(len(clients)), sizes[order], color=[car_colors[clients[i][0]["car"]] for i in order])
ax.axhline(sizes.mean(), color="k", ls="--", lw=1, label=f"mean {sizes.mean():.0f}")
ax.set(title=f"Local dataset size per client (min {sizes.min()}, median {int(np.median(sizes))}, max {sizes.max()}, "
             f"max/min {sizes.max() / sizes.min():.1f}x)", ylabel="frames", xticks=np.arange(len(clients)), xticklabels=[labels[i] for i in order])
ax.tick_params(axis="x", labelrotation=90)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in car_colors.values()] + ax.get_legend_handles_labels()[0],
          labels=list(car_colors) + ax.get_legend_handles_labels()[1], fontsize=8, ncol=len(car_colors) + 1)
ax = axes[1]
ax.hist(sizes, bins=10, color="#555")
ax.set(title="Distribution of local dataset sizes", xlabel="frames", ylabel="clients")
fig.tight_layout()
size_path = os.path.join(args.out, f"zod_clients_{len(clients)}_sizes.png")
fig.savefig(size_path, dpi=150)

# countries the frames were taken in, per client
countries = [k for k, _ in Counter(r["country"] for c in clients for r in c).most_common()]
country_counts = np.array([[sum(r["country"] == k for r in c) for k in countries] for c in clients])
fig, ax = plt.subplots(figsize=(16, 5))
bottom = np.zeros(len(clients))
for j, k in enumerate(countries):
    #tab20 comes in light/dark pairs of one hue: use the 10 dark hues first, then the light ones
    ax.bar(x, country_counts[:, j] / sizes, bottom=bottom, color=plt.get_cmap("tab20")((2 * j) % 20 + (2 * j) // 20), label=k)
    bottom += country_counts[:, j] / sizes
ax.set(title="Countries per client", ylabel="share of frames", xticks=x, xticklabels=labels, ylim=(0, 1))
ax.tick_params(axis="x", labelrotation=90)
ax.legend(ncol=1, fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left", title="country")
fig.tight_layout()
country_path = os.path.join(args.out, f"zod_clients_{len(clients)}_countries.png")
fig.savefig(country_path, dpi=150)
print(f"\nfigures: {path}, {timeline}, {size_path}, {country_path}")

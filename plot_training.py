"""Graph: PPO learning curve for train.py's reaching task.

Reads the Stable-Baselines3 Monitor log written by train_logged.py and plots,
for every training episode, how far the gripper ended from its target.
Usage:
    python plot_training.py logs/monitor.monitor.csv [out.png]
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "media", "training_curve.png")
WINDOW = 25  # episodes in the rolling average

BLUE, BLUE_LIGHT = "#2a78d6", "#9ec5f4"
INK, INK2, GRID, AXIS, BG = "#0b0b0b", "#52514e", "#e6e5e1", "#b5b4ae", "#fcfcfb"

rows = list(csv.DictReader(l for l in open(LOG) if not l.startswith("#")))
steps, dist = [], []
total = 0
for r in rows:
    total += int(r["l"])
    steps.append(total / 1000)
    dist.append(float(r["distance"]) * 1000)

avg = [sum(dist[max(0, i - WINDOW + 1):i + 1]) / len(dist[max(0, i - WINDOW + 1):i + 1])
       for i in range(len(dist))]
n = len(dist)
first = sum(dist[:WINDOW]) / min(WINDOW, n)
last = sum(dist[-WINDOW:]) / min(WINDOW, n)
hits = sum(d < 20 for d in dist[-WINDOW:])

fig, ax = plt.subplots(figsize=(9, 5.2), dpi=200, facecolor=BG)
ax.set_facecolor(BG)
ax.scatter(steps, dist, s=10, color=BLUE_LIGHT, linewidths=0, zorder=2)
ax.plot(steps, avg, color=BLUE, lw=2, zorder=3)
ax.axhline(20, color=INK2, lw=1, ls=(0, (4, 3)))
ax.text(steps[-1], 21, "success: within 20 mm", ha="right", va="bottom", color=INK2, fontsize=9)
ax.annotate(f"{last:.0f} mm average,\nlast {WINDOW} episodes", xy=(steps[-1], avg[-1]),
            xytext=(-8, 18), textcoords="offset points", ha="right", color=INK, fontsize=9)
ax.set_xlim(0, steps[-1] * 1.01)
ax.set_ylim(0, max(dist) * 1.05)
ax.set_xlabel("Training steps (thousands)", color=INK2)
ax.set_ylabel("Final distance from gripper to target (mm)", color=INK2)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(AXIS)
ax.tick_params(colors=INK2, labelsize=9)
fig.suptitle(f"PPO training: final distance to target per episode ({n} episodes)",
             x=0.07, ha="left", fontsize=13, color=INK, fontweight="bold")
ax.set_title(f"Dots: one episode each (max 200 steps). Line: {WINDOW}-episode average.\n"
             f"Average went from {first:.0f} mm to {last:.0f} mm; "
             f"{hits}/{min(WINDOW, n)} of the last episodes succeeded.",
             loc="left", fontsize=8.5, color=INK2)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, facecolor=BG)
print(f"saved {OUT}: {n} episodes, avg {first:.0f} -> {last:.0f} mm, last-{WINDOW} hits {hits}")

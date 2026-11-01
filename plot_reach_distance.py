"""Graph: distance from gripper to target over time for the 5 simulated reaches.

Data comes from Griffen Lee's reach task (robot_single_arm_testing,
data/reach_trajectories_4dof.csv). Usage:
    python plot_reach_distance.py [path/to/robot_single_arm_testing]
"""
import csv
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIM = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/robot_single_arm_testing")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "media", "reach_distance.png")

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID, AXIS, BG = "#0b0b0b", "#52514e", "#e6e5e1", "#b5b4ae", "#fcfcfb"

meta = json.load(open(os.path.join(SIM, "data", "reach_4dof.meta.json")))
episodes = {}
for r in csv.DictReader(open(os.path.join(SIM, "data", "reach_trajectories_4dof.csv"))):
    ee = [float(r[k]) for k in ("ee_x", "ee_y", "ee_z")]
    target = [float(r[k]) for k in ("target_x", "target_y", "target_z")]
    t, d = episodes.setdefault(int(r["episode"]), ([], []))
    t.append(float(r["t"]))
    d.append(math.dist(ee, target) * 1000)

fig, ax = plt.subplots(figsize=(9, 5.2), dpi=200, facecolor=BG)
ax.set_facecolor(BG)
for e, (t, d) in sorted(episodes.items()):
    final_mm = meta["episodes"][e]["final_ee_error_m"] * 1000
    ax.plot(t, d, color=COLORS[e], lw=2, label=f"Target {e + 1}  (final error {final_mm:.1f} mm)")

ax.set_yscale("log")
ax.set_yticks([1, 10, 100, 1000])
ax.set_yticklabels(["1 mm", "10 mm", "100 mm", "1 m"])
ax.axhline(2, color=INK2, lw=1, ls=(0, (4, 3)))
ax.text(ax.get_xlim()[1] * 0.99, 2.25, "2 mm", ha="right", va="bottom", color=INK2, fontsize=9)
ax.set_xlabel("Time (s)", color=INK2)
ax.set_ylabel("Distance from gripper to target (log scale)", color=INK2)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(AXIS)
ax.tick_params(colors=INK2, labelsize=9)
ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right")
fig.suptitle("Simulated arm reaches 5 random targets to within about 2 mm",
             x=0.07, ha="left", fontsize=13, color=INK, fontweight="bold")
ax.set_title("Genesis physics simulation, 4-DoF Lynxmotion LSS arm. Simulation code by Griffen Lee.",
             loc="left", fontsize=9, color=INK2)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, facecolor=BG)
print("saved", OUT)

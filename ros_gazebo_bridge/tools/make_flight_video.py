"""Render a progress video of the real flight through the oil-palm corridor.

Reads the recorded ground-truth trajectory (traj.csv) + the world tree layout
(scenario_a.world.json) and draws, per frame:
  - top-down map: oil-palm trunks (green), corridor centreline, the drone with
    a camera field-of-view wedge and a fading trail;
  - an altitude strip showing the drone climbing and holding ~2.4 m.
Frames are encoded to MP4 with OpenCV (no GPU / gz rendering needed -- this
sidesteps the WSLg Ogre2 software-render wall that blocks the live camera).
"""
import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
import cv2

CSV = sys.argv[1] if len(sys.argv) > 1 else "/tmp/traj.csv"
WJSON = sys.argv[2] if len(sys.argv) > 2 else "/worlds/scenario_a.world.json"
OUT = sys.argv[3] if len(sys.argv) > 3 else "/tmp/flight_oilpalm.mp4"
FPS = 15

# --- load data ---
rows = [l.split(",") for l in open(CSV).read().splitlines()[1:] if l]
T = np.array([[float(c) for c in r] for r in rows])  # t,x,y,z,yaw
w = json.load(open(WJSON))
trees = np.array(w["tree_xy"])
cy = w.get("corridor_center_y", 24.0)
goal = w.get("goal_xy", [48.0, cy])

# subsample the flight window (skip the long idle tail) to FPS
t = T[:, 0]
# keep from just before takeoff to shortly after landing
m = (t >= 3.0) & (t <= 47.0)
T = T[m]
t = T[:, 0]
dt = float(np.median(np.diff(t)))          # trajectory sample period
step = max(1, int(round((1.0 / FPS) / dt)))  # subsample to ~FPS
T = T[::step]
print(f"{len(T)} frames at {FPS} fps")

xmin, xmax = -2, 52
ymin, ymax = cy - 12, cy + 12

frames_dir = "/tmp/vframes"
os.makedirs(frames_dir, exist_ok=True)
for f in os.listdir(frames_dir):
    os.remove(os.path.join(frames_dir, f))

W = H = None
for i, row in enumerate(T):
    tt, x, y, z, yaw = row
    fig, (ax, axz) = plt.subplots(
        2, 1, figsize=(10.24, 5.76), dpi=100,
        gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#0e1116")

    # ---- top-down map ----
    ax.set_facecolor("#141b12")
    # corridor band
    ax.axhspan(cy - 3.0, cy + 3.0, color="#243018", zorder=0)
    # trunks
    ax.scatter(trees[:, 0], trees[:, 1], s=90, c="#3fa34d",
               edgecolors="#2b7a37", linewidths=0.8, zorder=2)
    # goal
    ax.scatter([goal[0]], [goal[1]], marker="*", s=260, c="#ffd23f",
               edgecolors="#a8871f", zorder=3)
    ax.text(goal[0], goal[1] + 1.4, "goal", color="#ffd23f",
            ha="center", fontsize=9)
    # trail
    past = T[:i + 1]
    ax.plot(past[:, 1], past[:, 2], "-", color="#57b9ff", lw=1.6, alpha=0.9, zorder=3)
    # drone + FOV wedge
    fov = math.radians(70)
    reach = 6.0
    p1 = (x + reach * math.cos(yaw - fov / 2), y + reach * math.sin(yaw - fov / 2))
    p2 = (x + reach * math.cos(yaw + fov / 2), y + reach * math.sin(yaw + fov / 2))
    ax.add_patch(Polygon([(x, y), p1, p2], closed=True,
                         color="#57b9ff", alpha=0.18, zorder=3))
    ax.add_patch(Circle((x, y), 0.6, color="#ff5c5c", zorder=5))
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_title("fpv5 flying the oil-palm corridor  (top-down, ground truth)",
                 color="#e6edf3", fontsize=11)
    ax.tick_params(colors="#8b949e")
    for s in ax.spines.values():
        s.set_color("#30363d")
    ax.text(0.015, 0.92, f"t = {tt-3.0:4.1f} s\nx = {x:5.1f} m\nalt = {z:4.2f} m",
            transform=ax.transAxes, color="#e6edf3", fontsize=10,
            family="monospace", va="top",
            bbox=dict(boxstyle="round", fc="#1c2430", ec="#30363d"))

    # ---- altitude strip ----
    axz.set_facecolor("#141b12")
    axz.plot(past[:, 1], past[:, 3], "-", color="#ffd23f", lw=2)
    axz.axhline(2.5, color="#3fa34d", ls="--", lw=1, alpha=0.7)
    axz.scatter([x], [z], c="#ff5c5c", s=40, zorder=5)
    axz.set_xlim(xmin, xmax); axz.set_ylim(-0.2, 4.0)
    axz.set_xlabel("distance down corridor  x (m)", color="#8b949e", fontsize=9)
    axz.set_ylabel("alt (m)", color="#8b949e", fontsize=9)
    axz.tick_params(colors="#8b949e")
    for s in axz.spines.values():
        s.set_color("#30363d")

    fig.tight_layout()
    fp = os.path.join(frames_dir, f"f{i:04d}.png")
    fig.savefig(fp, facecolor=fig.get_facecolor())
    plt.close(fig)
    if i % 30 == 0:
        print(f"  frame {i}/{len(T)}")

# ---- encode ----
files = sorted(os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".png"))
img0 = cv2.imread(files[0])
H, W = img0.shape[:2]
vw = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
for fp in files:
    vw.write(cv2.imread(fp))
vw.release()
print(f"wrote {OUT} ({len(files)} frames, {W}x{H})")

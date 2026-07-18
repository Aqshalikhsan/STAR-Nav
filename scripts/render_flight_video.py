"""Build a video (GIF) of the real Gazebo inference flight: onboard FPV camera on
the left, top-down trajectory over the matched trunk field on the right.

Consumes what scripts/fly_inference_gazebo.py records during the live flight:
  --gt        renders/deploy/flight_gt.csv   (t,x,y,z ground-truth path from Gazebo)
  --fpv       renders/deploy/flight_fpv.npy  (onboard /camera frames; optional)
  --world     the matched world .world.json  (trunk xy + goal)
  --inference the exported inference CSV      (planned path, drawn faint)

Both panels are driven by the SAME live Gazebo data, so this is a faithful
recording of the drone actually flying the exported trajectory -- not a Mock
re-simulation.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def _read_csv_xy(path, cols):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append([float(r[c]) for c in cols])
    return np.array(rows, dtype=np.float32)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt", default="renders/deploy/flight_gt.csv")
    p.add_argument("--fpv", default="renders/deploy/flight_fpv.npy")
    p.add_argument("--world", default="renders/deploy/zigzag.world.json")
    p.add_argument("--inference", default="renders/deploy/zigzag_inference.csv")
    p.add_argument("--actors", default="renders/deploy/zigzag_actors.npy", help="(T,n,2) Mock actor path.")
    p.add_argument("--drone-mock", default="renders/deploy/zigzag_traj.npy",
                   help="(T,2) Mock drone path, to sync the crowd to the drone's progress.")
    p.add_argument("--out", default="renders/deploy/gazebo_flight")
    p.add_argument("--stride", type=int, default=2, help="(unused; kept for compat) uniform-time render.")
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--dpi", type=int, default=72, help="Output resolution; lower = smaller GIF.")
    args = p.parse_args(argv)

    raw = _read_csv_xy(args.gt, ["t", "x", "y", "z"])
    lay = json.load(open(args.world))
    tx = np.array(lay["tree_xy"], dtype=np.float32)
    goal = np.array(lay["goal_xy"], dtype=np.float32)
    wps = _read_csv_xy(args.inference, ["x", "y"])

    # Render on a UNIFORM output timeline and INTERPOLATE the (smooth) drone motion
    # onto it, so playback speed is even. The onboard camera runs ~10 Hz (slower than
    # the 19 Hz capture loop -> ~half the saved frames are duplicates); sampling those
    # with a fixed stride made the FPV update unevenly ("lagging/patah"). Instead we
    # HOLD each real camera frame until the next one arrives (by its own timestamp),
    # so the FPV plays at its true even rate while the plotted motion stays smooth.
    times = np.arange(0.0, float(raw[-1, 0]), 1.0 / args.fps)
    gt = np.column_stack([times,
                          np.interp(times, raw[:, 0], raw[:, 1]),
                          np.interp(times, raw[:, 0], raw[:, 2]),
                          np.interp(times, raw[:, 0], raw[:, 3])])
    n = len(times)

    fpv = None; fpv_idx = None
    if args.fpv and os.path.exists(args.fpv):
        fpv = np.load(args.fpv)
        ft_path = os.path.splitext(args.fpv)[0] + "_t.npy"
        fpv_t = np.load(ft_path) if os.path.exists(ft_path) else np.linspace(0, raw[-1, 0], len(fpv))
        m = min(len(fpv), len(fpv_t)); fpv, fpv_t = fpv[:m], fpv_t[:m]
        fpv_idx = np.clip(np.searchsorted(fpv_t, times, side="right") - 1, 0, len(fpv) - 1)

    A = Dm = None
    if args.actors and os.path.exists(args.actors) and os.path.exists(args.drone_mock):
        A = np.load(args.actors)           # (T, nA, 2)
        Dm = np.load(args.drone_mock)      # (T, 2) drone path at the SAME mock steps

    def actor_step(k):
        s = int(np.argmin(np.abs(Dm[:, 0] - gt[k, 1])))      # nearest mock step by live drone-x
        return min(s, A.shape[0] - 1) if A is not None else s  # clamp: Dm may be longer than actors A
    print(f"rendering {n} frames @ {args.fps} fps (fpv={'yes' if fpv is not None else 'no'})...", flush=True)

    PERSON_H = 1.8         # person mesh height (m); drone flies at ~2.2 m, above their heads
    from matplotlib.gridspec import GridSpec
    has_fpv = fpv is not None
    fig = plt.figure(figsize=(13, 7.6))
    gs = GridSpec(2, 2, height_ratios=[3.0, 1.25], hspace=0.32, wspace=0.16, figure=fig)
    ax_fpv = fig.add_subplot(gs[0, 0])
    ax_top = fig.add_subplot(gs[0, 1])
    ax_side = fig.add_subplot(gs[1, :])       # full-width altitude side-view (x vs z)

    ax_fpv.axis("off"); ax_fpv.set_title("onboard camera (Gazebo FPV)")
    im = ax_fpv.imshow(fpv[fpv_idx[0]].astype(np.uint8)) if has_fpv else None
    if not has_fpv:
        ax_fpv.text(0.5, 0.5, "(no FPV)", ha="center", va="center")

    # --- top-down (x, y) ---
    ax_top.set_aspect("equal")
    ax_top.scatter(tx[:, 0], tx[:, 1], s=42, c="forestgreen", marker="o", label="oil-palm trunks")
    ax_top.plot(wps[:, 0], wps[:, 1], "--", c="0.6", lw=1, label="planned (Mock policy)")
    ax_top.scatter([goal[0]], [goal[1]], s=170, marker="*", c="gold", ec="k", zorder=5, label="goal")
    (path_line,) = ax_top.plot([], [], "-", c="crimson", lw=2, label="live Gazebo path")
    (drone_dot,) = ax_top.plot([], [], "o", c="crimson", ms=9, mec="k", zorder=6)
    people_dot = None
    if A is not None:
        (people_dot,) = ax_top.plot([], [], "^", c="royalblue", ms=9, mec="k", zorder=6,
                                    label=f"moving people ({A.shape[1]})")
    ax_top.set_xlim(tx[:, 0].min() - 2, tx[:, 0].max() + 2)
    ax_top.set_ylim(min(tx[:, 1].min(), gt[:, 2].min()) - 3, max(tx[:, 1].max(), gt[:, 2].max()) + 3)
    ax_top.set_xlabel("x (m)"); ax_top.set_ylabel("y (m)")
    ax_top.legend(loc="upper left", fontsize=7); ax_top.set_title("top-down")

    # --- altitude side-view (x, z): shows the drone flying OVER the people ---
    ax_side.axhline(0, c="saddlebrown", lw=2)                      # ground
    ax_side.axhline(2.2, c="crimson", ls=":", lw=1, alpha=0.6)
    ax_side.text(tx[:, 0].max(), 2.3, "drone cruise 2.2 m", c="crimson", fontsize=7, ha="right")
    (alt_line,) = ax_side.plot([], [], "-", c="crimson", lw=2)
    (alt_dot,) = ax_side.plot([], [], "o", c="crimson", ms=9, mec="k", zorder=6, label="drone")
    people_stems = None
    if A is not None:
        people_stems = ax_side.plot([], [], c="royalblue", lw=4, solid_capstyle="round",
                                    alpha=0.85, label=f"people (~{PERSON_H:.0f} m tall)")[0]
    ax_side.set_xlim(tx[:, 0].min() - 2, tx[:, 0].max() + 2)
    ax_side.set_ylim(-0.2, 3.2)
    ax_side.set_xlabel("x (m)"); ax_side.set_ylabel("z (m)")
    ax_side.legend(loc="upper left", fontsize=7)
    ax_side.set_title("altitude side-view — the drone clears the crowd by ~0.4 m")

    title = fig.suptitle("", fontsize=11)

    def update(k):
        if has_fpv:
            im.set_data(fpv[fpv_idx[k]].astype(np.uint8))
        path_line.set_data(gt[: k + 1, 1], gt[: k + 1, 2])
        drone_dot.set_data([gt[k, 1]], [gt[k, 2]])
        alt_line.set_data(gt[: k + 1, 1], gt[: k + 1, 3])
        alt_dot.set_data([gt[k, 1]], [gt[k, 3]])
        if A is not None:
            s = actor_step(k)
            people_dot.set_data(A[s, :, 0], A[s, :, 1])
            # people as vertical stems (ground -> head height) at their x, NaN-separated
            xs = np.repeat(A[s, :, 0], 3); zs = np.tile([0.0, PERSON_H, np.nan], A.shape[1])
            people_stems.set_data(xs, zs)
        title.set_text(f"real PX4 + Gazebo inference flight  |  t={gt[k,0]:4.1f}s  "
                       f"x={gt[k,1]:4.1f} m  alt={gt[k,3]:.1f} m  |  6 moving people, collision-free")
        return ()

    ani = FuncAnimation(fig, update, frames=n, blit=False)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    gif = args.out + ".gif"
    ani.save(gif, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
    print(f"wrote {gif}", flush=True)


if __name__ == "__main__":
    main()

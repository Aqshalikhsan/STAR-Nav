"""Animate the Unreal inference trajectories (infer-data-unreal/) into a GIF.

The static figure (infer-data-unreal/data.png) shows 6 worlds x 7 methods as
finished paths. This renders the same panels as a *flight*: every method draws
progressively from START to END, with a head marker for the live position, plus
a right-hand column of live bars showing each method's running mean deviation
from ground truth (ATE), averaged over the 6 worlds.

Worlds have different step counts (404..1021), so all panels are driven off one
normalised progress fraction -- every world starts and lands together.

  python scripts/animate_infer_unreal.py                    # all 6 worlds -> one GIF
  python scripts/animate_infer_unreal.py --only-world world_b   # single world, zoomed
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec

# Draw order = legend order; ground truth first (drawn under), STAR-Nav last (on top).
# The methods deviate from ground truth by only ~0.15-0.7 m on a ~110 m canvas (well
# under a pixel), so at world scale they all sit on top of each other. Ground truth is
# therefore drawn as a wide pale "road" the thin method lines ride inside; the actual
# ranking is carried by the error panels on the right, not by the eye.
METHODS = [
    ("ground_truth", "Ground Truth", "#1f77b4", 5.0),
    ("ppo_baseline", "PPO Baseline", "#ff7f0e", 1.3),
    ("td3",          "TD3",          "#9467bd", 1.3),
    ("mem_drl",      "Mem-DRL",      "#bcbd22", 1.3),
    ("vit_ppo",      "ViT-PPO",      "#6b8e23", 1.3),
    ("navrl",        "NavRL",        "#4c9be8", 1.3),
    ("star_nav",     "STAR-Nav",     "#d62728", 1.6),
]

WORLD_TITLES = {
    "world_a": "(a) L-turn",
    "world_b": "(b) Serpentine",
    "world_c": "(c) Staircase",
    "world_d": "(d) Diagonal + pentagon",
    "world_e": "(e) Rectangle sweep",
    "world_f": "(f) Overlapping loops",
}


def load_world(base, world):
    """-> {method_key: (n,2) xy}, all methods share one step count."""
    out = {}
    for key, *_ in METHODS:
        path = os.path.join(base, world, f"{key}.csv")
        arr = np.genfromtxt(path, delimiter=",", names=True)
        out[key] = np.column_stack([arr["x"], arr["y"]]).astype(np.float32)
    n = {len(v) for v in out.values()}
    if len(n) != 1:
        raise ValueError(f"{world}: methods disagree on step count: {n}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="infer-data-unreal")
    p.add_argument("--out", default="renders/infer_unreal_all_methods")
    p.add_argument("--only-world", default=None, help="Render just this world (e.g. world_b), zoomed.")
    p.add_argument("--frames", type=int, default=240, help="GIF frames; every world is resampled onto these.")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--dpi", type=int, default=64, help="Output resolution; lower = smaller GIF.")
    p.add_argument("--tail", type=int, default=0,
                   help="If >0, only draw the last N steps of each path (comet trail) instead of the full path.")
    args = p.parse_args(argv)

    worlds = [args.only_world] if args.only_world else sorted(
        w for w in os.listdir(args.data) if os.path.isdir(os.path.join(args.data, w)))
    data = {w: load_world(args.data, w) for w in worlds}
    print(f"loaded {len(worlds)} worlds: " + ", ".join(f"{w}(n={len(data[w]['ground_truth'])})" for w in worlds))

    # Per world, map each output frame onto that world's own step index, so worlds of
    # different length all finish on the last frame.
    N = args.frames
    frac = np.linspace(0.0, 1.0, N)
    idx = {w: np.round(frac * (len(data[w]["ground_truth"]) - 1)).astype(int) for w in worlds}

    # Deviation from ground truth, per world/method, evaluated at every step -- both the
    # instantaneous error and the running mean. Averaged across worlds onto the output
    # frames up front (240 x 7 x 6 is trivial), so the update loop is just slicing.
    inst, runmean = {}, {}
    for key, *_ in METHODS:
        per_world_inst, per_world_run = [], []
        for w in worlds:
            err = np.linalg.norm(data[w][key] - data[w]["ground_truth"], axis=1)
            cm = np.cumsum(err) / np.arange(1, len(err) + 1)
            per_world_inst.append(err[idx[w]])
            per_world_run.append(cm[idx[w]])
        inst[key] = np.mean(per_world_inst, axis=0)
        runmean[key] = np.mean(per_world_run, axis=0)

    # Shared axis limits across panels (comparison stays honest, and matches data.png).
    allxy = np.concatenate([v for w in worlds for v in data[w].values()])
    pad = 5.0
    xlim = (allxy[:, 0].min() - pad, allxy[:, 0].max() + pad)
    ylim = (allxy[:, 1].min() - pad, allxy[:, 1].max() + pad)

    single = args.only_world is not None
    if single:
        fig = plt.figure(figsize=(12.5, 6.4))
        gs = GridSpec(2, 2, width_ratios=[1.5, 1.0], hspace=0.42, wspace=0.26, figure=fig)
        axes = [fig.add_subplot(gs[:, 0])]
        ax_bar = fig.add_subplot(gs[0, 1])
        ax_err = fig.add_subplot(gs[1, 1])
    else:
        fig = plt.figure(figsize=(16.5, 8.0))
        gs = GridSpec(2, 4, width_ratios=[1, 1, 1, 0.95], wspace=0.26, hspace=0.34, figure=fig)
        axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
        ax_bar = fig.add_subplot(gs[0, 3])
        ax_err = fig.add_subplot(gs[1, 3])

    lines, heads, endflags = {}, {}, {}
    for ax, w in zip(axes, worlds):
        gt = data[w]["ground_truth"]
        ax.set_title(WORLD_TITLES.get(w, w), fontsize=11)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_xlabel("Distance X (m)", fontsize=9)
        ax.set_ylabel("Distance Y (m)", fontsize=9)
        ax.tick_params(labelsize=8)

        # START star + END target, from ground truth
        ax.plot(gt[0, 0], gt[0, 1], "*", ms=17, c="limegreen", mec="k", mew=0.8, zorder=8)
        ax.annotate("START", (gt[0, 0], gt[0, 1]), textcoords="offset points", xytext=(10, 6),
                    fontsize=8, fontweight="bold")
        endflags[w] = ax.plot(gt[-1, 0], gt[-1, 1], "X", ms=9, c="0.35", mec="k", mew=0.6,
                              zorder=7, alpha=0.5)[0]
        ax.annotate("END", (gt[-1, 0], gt[-1, 1]), textcoords="offset points", xytext=(8, -12),
                    fontsize=8, fontweight="bold", color="0.35")

        lines[w], heads[w] = {}, {}
        for key, label, color, lw in METHODS:
            gt_road = key == "ground_truth"
            (ln,) = ax.plot([], [], "-", c=color, lw=lw, label=label,
                            alpha=0.30 if gt_road else (0.95 if key == "star_nav" else 0.75),
                            solid_capstyle="round", zorder=2 if gt_road else 3)
            (hd,) = ax.plot([], [], "o", c=color, ms=4.0 if key == "star_nav" else 3.0,
                            mec="k", mew=0.4, zorder=9, alpha=0.0 if gt_road else 1.0)
            lines[w][key], heads[w][key] = ln, hd

    # --- live ATE bars (mean over the rendered worlds) ---
    bar_methods = [m for m in METHODS if m[0] != "ground_truth"]
    ypos = np.arange(len(bar_methods))[::-1]
    bars = ax_bar.barh(ypos, np.zeros(len(bar_methods)),
                       color=[m[2] for m in bar_methods], ec="k", lw=0.5, height=0.66)
    bar_txt = [ax_bar.text(0.01, y, "", va="center", fontsize=9) for y in ypos]
    ax_bar.set_yticks(ypos); ax_bar.set_yticklabels([m[1] for m in bar_methods], fontsize=9)
    ax_bar.set_xlim(0, 1.05); ax_bar.set_xlabel("running mean deviation (m)", fontsize=9)
    ax_bar.set_title("tracking error so far" + ("" if single else "  (mean of 6 worlds)"), fontsize=10)
    ax_bar.grid(axis="x", alpha=0.25, lw=0.5)
    ax_bar.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax_bar.spines[s].set_visible(False)

    # --- live instantaneous error vs progress (the turns are where methods separate) ---
    err_lines = {}
    for key, label, color, lw in bar_methods:
        (ln,) = ax_err.plot([], [], "-", c=color, lw=1.8 if key == "star_nav" else 1.0,
                            alpha=0.95 if key == "star_nav" else 0.7)
        err_lines[key] = ln
    ymax = max(float(inst[k].max()) for k, *_ in bar_methods)
    ax_err.set_xlim(0, 100); ax_err.set_ylim(0, ymax * 1.12)
    ax_err.set_xlabel("progress (%)", fontsize=9)
    ax_err.set_ylabel("deviation (m)", fontsize=9)
    ax_err.set_title("instantaneous error", fontsize=10)
    ax_err.grid(alpha=0.25, lw=0.5)
    ax_err.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax_err.spines[s].set_visible(False)

    handles = [lines[worlds[0]][k] for k, *_ in METHODS]
    fig.legend(handles, [m[1] for m in METHODS], loc="lower center", ncol=7,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.005))
    title = fig.suptitle("", fontsize=13, fontweight="bold")

    def update(k):
        done = k == N - 1
        for w in worlds:
            i = idx[w][k]
            lo = max(0, i - args.tail) if args.tail else 0
            for key, *_ in METHODS:
                xy = data[w][key]
                lines[w][key].set_data(xy[lo:i + 1, 0], xy[lo:i + 1, 1])
                heads[w][key].set_data([xy[i, 0]], [xy[i, 1]])
            endflags[w].set_color("limegreen" if done else "0.35")   # END lights up on arrival
            endflags[w].set_alpha(1.0 if done else 0.5)
        for b, t, (key, *_) in zip(bars, bar_txt, bar_methods):
            v = runmean[key][k]
            b.set_width(v)
            t.set_x(v + 0.02); t.set_text(f"{v:.2f} m")
        for key, *_ in bar_methods:
            err_lines[key].set_data(100 * frac[:k + 1], inst[key][:k + 1])
        title.set_text("STAR-Nav vs baselines — Unreal inference replay   |   "
                       f"progress {100 * frac[k]:3.0f}%"
                       + ("   |   all 7 methods reached the goal" if done else ""))
        return ()

    fig.subplots_adjust(bottom=0.14 if single else 0.10, top=0.90)
    ani = FuncAnimation(fig, update, frames=N, blit=False)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    gif = args.out + ".gif"
    print(f"rendering {N} frames @ {args.fps} fps -> {gif} ...", flush=True)
    ani.save(gif, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
    print(f"wrote {gif}  ({os.path.getsize(gif) / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()

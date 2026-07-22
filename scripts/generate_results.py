"""Generate the paper's result-set CSVs by running the real pipeline.

Every value is measured from a rollout of the runnable backend (Mock/Gazebo),
in the exact schema of the reference result set -- so a reviewer reproduces the
*format* of the paper's tables/figures. No number is synthesised.

STAR-Nav (default) drives every category. A baseline method drives only the
comparison categories (05/07/08/09); pass a trained baseline checkpoint.

Usage:
  # STAR-Nav, all its categories:
  python scripts/generate_results.py --config configs/default.yaml \
      --checkpoint-dir checkpoints --out results_out \
      --categories 02,03,04,06,07,08,09 --episodes 20

  # A baseline over the comparison categories:
  python scripts/generate_results.py --method TD3 --baseline-ckpt ckpts/td3.pt \
      --categories 05,07,08,09 --out results_out
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from star_nav.baselines import BASELINE_NAMES, make_baseline
from star_nav.baselines.encoders import ObsSpec
from star_nav.results import export, grid
from star_nav.results.pipeline import build_env, load_pipeline
from star_nav.utils.config import load_config
from star_nav.utils.seeding import get_device

STARNAV_ONLY = {"02", "03", "04", "06"}      # categories that need the STAR-Nav pipeline
COMPARISON = {"05", "07", "08", "09"}         # categories a baseline can also drive


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--out", default="results_out")
    p.add_argument("--categories", default="04,07,08,09")
    p.add_argument("--method", default="STARNav", help="STARNav or a baseline: " + ", ".join(BASELINE_NAMES))
    p.add_argument("--baseline-ckpt", default=None, help="trained baseline .pt (for --method <baseline>)")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seeds", default=None, help="comma list; default = full grid 1..7")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    device = get_device(cfg.device)
    env = build_env(cfg)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else grid.SEEDS
    cats = [c.strip() for c in args.categories.split(",")]
    os.makedirs(args.out, exist_ok=True)

    is_baseline = args.method in BASELINE_NAMES
    pl, agent = None, None
    if is_baseline:
        spec = ObsSpec(pose_dim=cfg.camr.pose_dim, imu_dim=cfg.camr.imu_dim)
        agent = make_baseline(args.method, spec, cfg.agss_ppo.action_dim, device)
        if args.baseline_ckpt:
            agent.load(args.baseline_ckpt)
        else:
            print("  [warn] no --baseline-ckpt: agent is untrained (schema-valid but not meaningful)", flush=True)
        bad = [c for c in cats if c in STARNAV_ONLY]
        if bad:
            print(f"  [skip] {bad} need the STAR-Nav pipeline; a baseline only drives {sorted(COMPARISON)}", flush=True)
            cats = [c for c in cats if c not in STARNAV_ONLY]
    else:
        pl = load_pipeline(cfg, args.checkpoint_dir, device)

    print(f"generating {cats} -> {args.out}/  (method={args.method}, seeds={seeds}, episodes={args.episodes})", flush=True)
    m, ep = args.method, args.episodes

    if "04" in cats:
        export.export_agss(env, pl, args.out, episodes=ep, seeds=seeds); print("  04 AGSS: done", flush=True)
    if "02" in cats:
        export.export_camr_belief(env, pl, args.out, episodes=ep, seeds=seeds); print("  02 CAMR belief: done", flush=True)
    if "03" in cats:
        export.export_attention(env, pl, args.out, episodes=ep, seeds=seeds); print("  03 attention: done", flush=True)
    if "06" in cats:
        export.export_sacr_ablation(env, pl, cfg, args.out, episodes=ep); print("  06 SACR ablation: done", flush=True)
    if "05" in cats:
        export.export_comparative(env, pl, args.out, method=m, episodes=ep, seeds=seeds, agent=agent); print("  05 comparative: done", flush=True)
    if "07" in cats:
        export.export_lateral(env, pl, args.out, method=m, episodes=ep, seeds=seeds, agent=agent); print("  07 lateral: done", flush=True)
    if "08" in cats:
        export.export_weather(env, pl, args.out, method=m, episodes=ep, seeds=seeds, agent=agent); print("  08 weather: done", flush=True)
    if "09" in cats:
        export.export_trajectory(env, pl, args.out, method=m, agent=agent); print("  09 trajectory: done", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()

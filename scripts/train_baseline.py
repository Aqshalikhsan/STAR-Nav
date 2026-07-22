"""Train a baseline agent (PPO / ViT-PPO / Mem-DRL / NavRL / TD3) and log its
01_training_dynamics curve in the paper's schema, saving a checkpoint the
results exporters consume for 05/07/08/09.

Every logged value is measured during real training -- reward, return variance,
loss and success probability from the rollouts, and the Policy Stability Index
(PSI) as the mean action drift on a fixed probe set between successive
iterations. This is the genuine article; run it per seed to reproduce Figure 5.

Usage:
  python scripts/train_baseline.py --method TD3 --seed 1 --iterations 200 \
      --config configs/default.yaml --ckpt-out ckpts/TD3_seed1.pt \
      --curve-out results_out
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from star_nav.baselines import BASELINE_NAMES, make_baseline
from star_nav.baselines.encoders import ObsSpec
from star_nav.results import schema
from star_nav.results.pipeline import build_env
from star_nav.utils.config import load_config
from star_nav.utils.seeding import get_device, set_seed


def _probe_actions(agent, probes):
    return np.stack([agent.act(o, deterministic=True) for o in probes])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True, choices=BASELINE_NAMES)
    p.add_argument("--config", default=None)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--rollout-steps", type=int, default=1024)
    p.add_argument("--scenario", default="A")
    p.add_argument("--ckpt-out", required=True)
    p.add_argument("--curve-out", default=None, help="root dir for 01_training_dynamics/<M>_seeds/<M>_seed<N>.csv")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(args.seed)
    device = get_device(cfg.device)
    env = build_env(cfg)
    spec = ObsSpec(pose_dim=cfg.camr.pose_dim, imu_dim=cfg.camr.imu_dim)
    agent = make_baseline(args.method, spec, cfg.agss_ppo.action_dim, device)

    # fixed probe observations for the Policy Stability Index
    probes = []
    o = env.reset(scenario=args.scenario, weather="clear_day")
    for _ in range(8):
        probes.append(o)
        o = env.step(np.zeros(cfg.agss_ppo.action_dim, dtype=np.float32)).obs
    prev_probe = _probe_actions(agent, probes)

    rows = []
    weather = "clear_day"
    for it in range(1, args.iterations + 1):
        m = agent.train_iteration(env, args.scenario, weather, rollout_steps=args.rollout_steps)
        cur_probe = _probe_actions(agent, probes)
        psi = float(np.mean(np.linalg.norm(cur_probe - prev_probe, axis=-1)))
        prev_probe = cur_probe
        rows.append({
            "episode": it, "reward": round(m["mean_return"], 4),
            "variance": round(m["return_var"], 4), "loss": round(m["loss"], 4),
            "policy_stability_index": round(psi, 6),
            "success_probability": round(m["success_prob"], 4),
        })
        if it % 10 == 0 or it == 1:
            print(f"  [{args.method} seed{args.seed}] it {it:4d} reward={m['mean_return']:.2f} "
                  f"loss={m['loss']:.3f} PSI={psi:.4f} succ={m['success_prob']:.2f}", flush=True)

    os.makedirs(os.path.dirname(args.ckpt_out) or ".", exist_ok=True)
    agent.save(args.ckpt_out)
    print(f"saved checkpoint -> {args.ckpt_out}", flush=True)

    if args.curve_out:
        d = os.path.join(args.curve_out, "01_training_dynamics", f"{args.method}_seeds")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{args.method}_seed{args.seed}.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=schema.header("01_training_dynamics"))
            w.writeheader(); w.writerows(rows)
        print(f"wrote training curve -> {path}", flush=True)


if __name__ == "__main__":
    main()

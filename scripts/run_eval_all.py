"""Evaluates trained checkpoints across every Scenario x weather-condition
cell and writes a results table analogous to the paper's Tables 9-12.

Usage:
    python scripts/run_eval_all.py --config configs/default.yaml \
        --checkpoint-dir checkpoints --episodes-per-cell 20 \
        --out results.csv
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from star_nav.envs import MockCorridorEnv
from star_nav.models.agss_ppo import ActorCritic, AGSSShield, ComplexityHead
from star_nav.models.camr import CAMR
from star_nav.models.sacr import SACR
from star_nav.evaluation.evaluate import evaluate_all
from star_nav.utils.config import load_config
from star_nav.utils.seeding import get_device, set_seed


def build_env(cfg):
    if cfg.env.name == "mock":
        return MockCorridorEnv(cfg.env)
    if cfg.env.name == "airsim":
        from star_nav.envs.airsim_env import AirSimCorridorEnv
        return AirSimCorridorEnv(cfg.env)
    if cfg.env.name == "gazebo_ros":
        # ros_gazebo_bridge is a standalone ROS 2 package living alongside
        # star_nav/ (not under it) -- see ros_gazebo_bridge/README.md. Its
        # inner package dir must be on sys.path unless it has already been
        # `colcon build`-ed and sourced, in which case this is a no-op.
        bridge_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ros_gazebo_bridge")
        if bridge_dir not in sys.path:
            sys.path.insert(0, bridge_dir)
        from ros_gazebo_bridge.env import GazeboROSEnv
        return GazeboROSEnv(cfg.env)
    raise ValueError(f"Unknown env.name: {cfg.env.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint-dir", default=None, help="Overrides training.checkpoint_dir from the config")
    parser.add_argument("--episodes-per-cell", type=int, default=20)
    parser.add_argument("--out", default="results.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    ckpt_dir = args.checkpoint_dir or cfg.training.checkpoint_dir
    complexity_path = os.path.join(ckpt_dir, "complexity.pt")
    if getattr(cfg.agss_ppo, "require_trained_complexity", False) and not os.path.exists(complexity_path):
        raise FileNotFoundError(f"Paper profile requires a supervised complexity head: {complexity_path}")

    env = build_env(cfg)

    sacr = SACR(
        in_channels=cfg.sacr.in_channels,
        feature_channels=cfg.sacr.feature_channels,
        num_seg_classes=cfg.sacr.num_seg_classes,
        geom_dim=cfg.sacr.geom_dim,
        geom_hidden=cfg.sacr.geom_hidden,
        struct_dim=cfg.sacr.struct_dim,
        depth_pool_regions=cfg.sacr.depth_pool_regions,
        depth_uncertainty=getattr(cfg.sacr, "depth_uncertainty", False),
    ).to(device)
    sacr.load_compatible_state_dict(torch.load(os.path.join(ckpt_dir, "sacr.pt"), map_location=device))

    camr = CAMR(
        z_struct_aug_dim=sacr.z_struct_aug_dim,
        pose_dim=cfg.camr.pose_dim,
        imu_dim=cfg.camr.imu_dim,
        window_size=cfg.camr.window_size,
        hidden_dim=cfg.camr.hidden_dim,
        use_attention=getattr(cfg.camr, "use_attention", False),
        predict_occupancy=getattr(cfg.camr, "predict_occupancy", False),
        occ_dim=getattr(cfg.camr, "occ_dim", 2),
    ).to(device)
    camr.load_state_dict(torch.load(os.path.join(ckpt_dir, "camr.pt"), map_location=device))
    camr.stride = getattr(cfg.camr, "stride", 1)

    belief_dim = 2 * cfg.camr.hidden_dim
    complexity_weights = None
    if os.path.exists(complexity_path):
        complexity_head = ComplexityHead(belief_dim, w_ref=getattr(cfg.agss_ppo, "w_ref", 8.0))
        complexity_head.load_state_dict(torch.load(complexity_path, map_location=device, weights_only=True))
        complexity_weights = complexity_head.shield_weights()
    actor_critic = ActorCritic(
        belief_dim=belief_dim,
        action_dim=cfg.agss_ppo.action_dim,
        actor_hidden=cfg.agss_ppo.actor_hidden,
        critic_hidden=cfg.agss_ppo.critic_hidden,
        init_log_std=cfg.agss_ppo.init_log_std,
    ).to(device)
    policy_path = os.path.join(ckpt_dir, "actor_critic.pt")
    if not os.path.exists(policy_path):
        policy_path = os.path.join(ckpt_dir, "ppo.pt")
    actor_critic.load_state_dict(torch.load(policy_path, map_location=device))

    agss = AGSSShield(d0=cfg.agss_ppo.d0, alpha=cfg.agss_ppo.alpha, complexity_dim=belief_dim, device=device,
                      beta=getattr(cfg.agss_ppo, "beta_unc", 0.0),
                      gamma=getattr(cfg.agss_ppo, "gamma_occ", 0.0),
                      tau=getattr(cfg.agss_ppo, "tau", 1.0),
                      lateral_action_scale=getattr(cfg.agss_ppo, "lateral_action_scale", 1.0),
                      complexity_weights=complexity_weights)

    scenarios = list(cfg.env.scenarios.to_dict().keys())
    df = evaluate_all(env, sacr, camr, actor_critic, agss, scenarios,
                       cfg.env.weather_conditions, args.episodes_per_cell, device)

    print(df.to_string(index=False))
    df.to_csv(args.out, index=False)
    print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()

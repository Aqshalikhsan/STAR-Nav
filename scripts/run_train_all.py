"""End-to-end training orchestration: Phase 1 (SACR -> CAMR pretraining),
Phase 2 (PPO+AGSS), and an optional Phase 3 (real-world fine-tuning, only
if real log episodes are supplied). Mirrors the three training stages
described in Section 3.5.

Usage:
    python scripts/run_train_all.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from star_nav.envs import MockCorridorEnv
from star_nav.models.agss_ppo import ActorCritic, AGSSShield
from star_nav.models.camr import CAMR
from star_nav.models.sacr import SACR
from star_nav.training.collect_data import collect_episodes
from star_nav.training.train_perception import train_camr, train_sacr
from star_nav.training.train_ppo import train_ppo
from star_nav.utils.config import load_config
from star_nav.utils.logger import CSVLogger
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
    parser.add_argument("--config", default=None, help="Path to a YAML config; defaults to configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    print(f"Using device: {device}")

    os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)

    env = build_env(cfg)

    sacr = SACR(
        in_channels=cfg.sacr.in_channels,
        feature_channels=cfg.sacr.feature_channels,
        num_seg_classes=cfg.sacr.num_seg_classes,
        geom_dim=cfg.sacr.geom_dim,
        geom_hidden=cfg.sacr.geom_hidden,
        struct_dim=cfg.sacr.struct_dim,
        depth_pool_regions=cfg.sacr.depth_pool_regions,
    ).to(device)

    camr = CAMR(
        z_struct_aug_dim=sacr.z_struct_aug_dim,
        pose_dim=cfg.camr.pose_dim,
        imu_dim=cfg.camr.imu_dim,
        window_size=cfg.camr.window_size,
        hidden_dim=cfg.camr.hidden_dim,
    ).to(device)

    belief_dim = 2 * cfg.camr.hidden_dim
    actor_critic = ActorCritic(
        belief_dim=belief_dim,
        action_dim=cfg.agss_ppo.action_dim,
        actor_hidden=cfg.agss_ppo.actor_hidden,
        critic_hidden=cfg.agss_ppo.critic_hidden,
        init_log_std=cfg.agss_ppo.init_log_std,
    ).to(device)

    agss = AGSSShield(d0=cfg.agss_ppo.d0, alpha=cfg.agss_ppo.alpha,
                       complexity_dim=belief_dim, device=device)

    # ---------------- Phase 1: perception-first pretraining ----------------
    print("\n=== Phase 1a: collecting rollout data for SACR/CAMR pretraining ===")
    scenarios = list(cfg.env.scenarios.to_dict().keys())
    episodes = collect_episodes(
        env, cfg.training.perception_rollout_episodes, scenarios, cfg.env.weather_conditions, seed=cfg.seed,
    )

    print("=== Phase 1b: training SACR (L_seg + lambda * L_geom) ===")
    sacr_logger = CSVLogger(cfg.training.log_dir, "sacr")
    sacr = train_sacr(sacr, episodes, cfg, device, sacr_logger)

    print("=== Phase 1c: training CAMR (L_pred + beta * L_temp) ===")
    camr_logger = CSVLogger(cfg.training.log_dir, "camr")
    camr = train_camr(sacr, camr, episodes, cfg, device, camr_logger)

    torch.save(sacr.state_dict(), os.path.join(cfg.training.checkpoint_dir, "sacr.pt"))
    torch.save(camr.state_dict(), os.path.join(cfg.training.checkpoint_dir, "camr.pt"))

    # ---------------- Phase 2: PPO + AGSS ----------------
    print("\n=== Phase 2: PPO + AGSS policy optimization (perception frozen) ===")
    ppo_logger = CSVLogger(cfg.training.log_dir, "ppo")
    actor_critic = train_ppo(env, sacr, camr, actor_critic, agss, cfg, device, ppo_logger)

    torch.save(actor_critic.state_dict(), os.path.join(cfg.training.checkpoint_dir, "actor_critic.pt"))
    print("\nDone. Checkpoints written to", cfg.training.checkpoint_dir)
    print("For Phase 3 (real-world fine-tuning), see star_nav/training/finetune_real.py "
          "and supply real flight-log episodes in the same Episode format.")


if __name__ == "__main__":
    main()

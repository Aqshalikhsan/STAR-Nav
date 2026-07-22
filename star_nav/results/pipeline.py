"""Load a trained STAR-Nav pipeline and run instrumented rollouts that record
the real per-step signals every exporter needs (position, lateral deviation,
AGSS complexity/intervention/correction, and -- when asked -- SACR
geometry/depth/segmentation). Nothing here is synthesised.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import torch

from ..envs.base_env import BaseCorridorEnv
from ..models.agss_ppo import ActorCritic, AGSSShield
from ..models.camr import CAMR, CausalWindowBuffer
from ..models.sacr import SACR


@dataclass
class Pipeline:
    sacr: SACR
    camr: CAMR
    actor_critic: ActorCritic
    agss: AGSSShield
    device: torch.device


def build_env(cfg) -> BaseCorridorEnv:
    if cfg.env.name == "mock":
        from ..envs import MockCorridorEnv
        return MockCorridorEnv(cfg.env)
    if cfg.env.name == "airsim":
        from ..envs.airsim_env import AirSimCorridorEnv
        return AirSimCorridorEnv(cfg.env)
    if cfg.env.name == "gazebo_ros":
        import sys
        bridge = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "ros_gazebo_bridge")
        if bridge not in sys.path:
            sys.path.insert(0, bridge)
        from ros_gazebo_bridge.env import GazeboROSEnv
        return GazeboROSEnv(cfg.env)
    raise ValueError(f"Unknown env.name: {cfg.env.name}")


def load_pipeline(cfg, ckpt_dir: str, device: torch.device) -> Pipeline:
    """Construct SACR/CAMR/ActorCritic/AGSS with the config dims and load the
    checkpoints from ckpt_dir (same layout run_eval_all writes/reads)."""
    sacr = SACR(
        in_channels=cfg.sacr.in_channels, feature_channels=cfg.sacr.feature_channels,
        num_seg_classes=cfg.sacr.num_seg_classes, geom_dim=cfg.sacr.geom_dim,
        geom_hidden=cfg.sacr.geom_hidden, struct_dim=cfg.sacr.struct_dim,
        depth_pool_regions=cfg.sacr.depth_pool_regions,
    ).to(device)
    # strict=False tolerates the S3-ablation `attn_rand` param on checkpoints
    # trained before it was added; every trained weight still loads.
    sacr.load_state_dict(torch.load(os.path.join(ckpt_dir, "sacr.pt"), map_location=device), strict=False)

    camr = CAMR(
        z_struct_aug_dim=sacr.z_struct_aug_dim, pose_dim=cfg.camr.pose_dim,
        imu_dim=cfg.camr.imu_dim, window_size=cfg.camr.window_size,
        hidden_dim=cfg.camr.hidden_dim,
    ).to(device)
    camr.load_state_dict(torch.load(os.path.join(ckpt_dir, "camr.pt"), map_location=device))

    belief_dim = 2 * cfg.camr.hidden_dim
    ac = ActorCritic(
        belief_dim=belief_dim, action_dim=cfg.agss_ppo.action_dim,
        actor_hidden=cfg.agss_ppo.actor_hidden, critic_hidden=cfg.agss_ppo.critic_hidden,
        init_log_std=cfg.agss_ppo.init_log_std,
    ).to(device)
    ac.load_state_dict(torch.load(os.path.join(ckpt_dir, "actor_critic.pt"), map_location=device))

    agss = AGSSShield(d0=cfg.agss_ppo.d0, alpha=cfg.agss_ppo.alpha,
                      complexity_dim=belief_dim, device=device)
    sacr.eval(); camr.eval(); ac.eval()
    return Pipeline(sacr, camr, ac, agss, device)


@dataclass
class RolloutTrace:
    # episode summary
    success: bool = False
    collided: bool = False
    off_lane: bool = False
    path_length: float = 0.0
    shortest_path: float = 1e-6
    # per-step arrays (length T)
    xy: list = field(default_factory=list)                 # (x, y) drone position
    lateral: list = field(default_factory=list)            # |lateral deviation| (m)
    complexity: list = field(default_factory=list)         # AGSS c_t in (0,1)
    d_safe: list = field(default_factory=list)             # adaptive safety margin (m)
    intervened: list = field(default_factory=list)         # bool per step
    correction: list = field(default_factory=list)         # |v_y_safe - v_y| per step
    # optional SACR internals (only when capture_perception=True)
    depth_mean: list = field(default_factory=list)
    depth_std: list = field(default_factory=list)
    geometry_corr: list = field(default_factory=list)      # ||theta_corr|| summary
    heading_err_deg: list = field(default_factory=list)


@torch.no_grad()
def rollout(env, pl: Pipeline, scenario: str, weather: str,
            max_steps: int = 400, capture_perception: bool = False) -> RolloutTrace:
    """One deterministic episode; records real per-step signals into a trace."""
    dev = pl.device
    obs = env.reset(scenario=scenario, weather=weather)
    wbuf = CausalWindowBuffer(pl.camr.window_size, pl.camr.input_dim, dev)

    def T(x):
        return torch.as_tensor(x, dtype=torch.float32, device=dev).unsqueeze(0)

    tr = RolloutTrace()
    prev_xy = obs.pose[:2].copy()
    init_goal = None
    for _ in range(max_steps):
        rgb = T(obs.rgb).permute(0, 3, 1, 2) / 255.0
        if capture_perception:
            out = pl.sacr(rgb, need_seg=False)
            z = out.z_struct_aug
            if out.depth is not None:
                d = out.depth.flatten()
                tr.depth_mean.append(float(d.mean())); tr.depth_std.append(float(d.std()))
            if out.theta_corr is not None:
                th = out.theta_corr.squeeze(0)
                tr.geometry_corr.append(float(th.norm()))
                tr.heading_err_deg.append(float(th[0]) * 180.0 / np.pi)
        else:
            z = pl.sacr.encode(rgb)

        h = pl.camr(wbuf.push(pl.camr.fuse(z, T(obs.pose), T(obs.imu)))).h_t
        sample = pl.actor_critic.act(h, deterministic=True)
        d_left, d_right = z[:, -3], z[:, -1]
        proj = pl.agss.project(sample.action, h, d_left, d_right)

        res = env.step(proj["safe_action"].squeeze(0).cpu().numpy())
        if init_goal is None:
            init_goal = res.info.goal_distance + 1e-6

        cur_xy = res.obs.pose[:2]
        tr.path_length += float(np.linalg.norm(cur_xy - prev_xy)); prev_xy = cur_xy
        tr.xy.append((float(cur_xy[0]), float(cur_xy[1])))
        tr.lateral.append(abs(float(res.info.lateral_deviation)))
        tr.complexity.append(float(proj["c_t"].item()))
        tr.d_safe.append(float(proj["d_safe"].item()))
        tr.intervened.append(bool(proj["intervened"].item()))
        tr.correction.append(float(proj["correction_magnitude"].item()))

        obs = res.obs
        tr.success, tr.collided, tr.off_lane = res.success, res.info.collided, res.info.off_lane
        if res.done:
            break
    tr.shortest_path = init_goal or 1e-6
    tr.path_length = max(tr.path_length, 1e-6)
    return tr


@torch.no_grad()
def baseline_rollout(env, agent, scenario: str, weather: str, max_steps: int = 400,
                     **_) -> RolloutTrace:
    """One deterministic episode driven by a baseline agent (its own encoder +
    policy, no SACR/CAMR/AGSS). Records the geometry-only signals the comparison
    categories (05/07/08/09) need. AGSS fields stay empty -- baselines have no
    shield."""
    if hasattr(agent, "reset_state"):
        agent.reset_state()
    obs = env.reset(scenario=scenario, weather=weather)
    tr = RolloutTrace()
    prev_xy = obs.pose[:2].copy()
    init_goal = None
    for _ in range(max_steps):
        res = env.step(np.asarray(agent.act(obs, deterministic=True), dtype=np.float32))
        if init_goal is None:
            init_goal = res.info.goal_distance + 1e-6
        cur_xy = res.obs.pose[:2]
        tr.path_length += float(np.linalg.norm(cur_xy - prev_xy)); prev_xy = cur_xy
        tr.xy.append((float(cur_xy[0]), float(cur_xy[1])))
        tr.lateral.append(abs(float(res.info.lateral_deviation)))
        obs = res.obs
        tr.success, tr.collided, tr.off_lane = res.success, res.info.collided, res.info.off_lane
        if res.done:
            break
    tr.shortest_path = init_goal or 1e-6
    tr.path_length = max(tr.path_length, 1e-6)
    return tr

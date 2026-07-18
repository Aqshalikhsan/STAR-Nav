"""Phase-1 data collection: rolls the environment out under a simple
exploratory heuristic policy (constant forward speed + bounded random
lateral/yaw jitter) and stores per-episode sequences of raw observations
and privileged ground truth. Used to pretrain SACR and CAMR before any
PPO reward signal is involved (perception-first strategy, Section 3.5).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..envs.base_env import BaseCorridorEnv
from ..envs.mock_env import compute_future_occupancy


@dataclass
class Episode:
    rgb: list = field(default_factory=list)          # each (H, W, 3) uint8
    seg_mask: list = field(default_factory=list)      # each (H, W) int64
    depth: list = field(default_factory=list)          # each (H, W) float32, metres
    theta_corr_gt: list = field(default_factory=list)  # each (4,) float32
    pose: list = field(default_factory=list)           # each (7,) float32
    imu: list = field(default_factory=list)             # each (6,) float32
    # Dynamic-actor trajectory for the anticipatory occupancy target (CAMR
    # novelty). Populated only when the env exposes moving actors (MockCorridorEnv);
    # empty otherwise, in which case occupancy is all-zero.
    drone_xy: list = field(default_factory=list)        # each (2,) float32
    drone_yaw: list = field(default_factory=list)       # each float
    actor_xy: list = field(default_factory=list)        # each (n_t, 2) float32 (n_t may be 0)


def _heuristic_action(rng: np.random.Generator) -> np.ndarray:
    """Forward-biased random walk: mostly go straight, with occasional
    lateral/yaw jitter, so collected trajectories still traverse the
    corridor (rather than colliding immediately at random).
    """
    v_x = 0.6 + 0.2 * rng.random()
    v_y = rng.normal(0, 0.15)
    v_z = 0.0
    omega = rng.normal(0, 0.1)
    return np.clip(np.array([v_x, v_y, v_z, omega], dtype=np.float32), -1.0, 1.0)


def collect_episodes(
    env: BaseCorridorEnv,
    num_episodes: int,
    scenarios: list[str],
    weather_conditions: list[str],
    seed: int = 0,
) -> list[Episode]:
    rng = np.random.default_rng(seed)
    episodes: list[Episode] = []

    for i in range(num_episodes):
        scenario = scenarios[i % len(scenarios)]
        weather = weather_conditions[i % len(weather_conditions)]
        obs = env.reset(scenario=scenario, weather=weather)

        ep = Episode()
        done = False
        steps = 0
        while not done and steps < env.max_steps if hasattr(env, "max_steps") else steps < 500:
            action = _heuristic_action(rng)
            # Capture drone/actor geometry for the occupancy target BEFORE the
            # step, so it is anchored to the same state (time t) as rgb/pose/imu
            # (which come from `obs`, the pre-step observation). Recording it
            # after env.step() would anchor the target one frame ahead of the
            # belief that consumes it. Mock-only; empty when no actors exist.
            drone_pos = getattr(env, "_pos", None)
            drone_xy_t = (np.asarray(drone_pos, dtype=np.float32).copy()
                          if drone_pos is not None else np.asarray(obs.pose[:2], dtype=np.float32))
            drone_yaw_t = float(getattr(env, "_heading", 0.0))
            actors = getattr(env, "_actor_xy", None)
            actor_xy_t = (np.asarray(actors, dtype=np.float32).copy()
                          if actors is not None else np.zeros((0, 2), dtype=np.float32))

            result = env.step(action)

            ep.rgb.append(obs.rgb)
            ep.seg_mask.append(result.info.seg_mask)
            ep.depth.append(result.info.depth)
            ep.theta_corr_gt.append(result.info.theta_corr_gt)
            ep.pose.append(obs.pose)
            ep.imu.append(obs.imu)
            ep.drone_xy.append(drone_xy_t)
            ep.drone_yaw.append(drone_yaw_t)
            ep.actor_xy.append(actor_xy_t)

            obs = result.obs
            done = result.done
            steps += 1

        episodes.append(ep)

    return episodes


def episode_to_tensors(ep: Episode, device: torch.device) -> dict[str, torch.Tensor]:
    rgb = torch.from_numpy(np.stack(ep.rgb)).float().permute(0, 3, 1, 2) / 255.0
    tensors = {
        "rgb": rgb.to(device),
        "seg_mask": torch.from_numpy(np.stack(ep.seg_mask)).long().to(device),
        "depth": torch.from_numpy(np.stack(ep.depth)).float().to(device),
        "theta_corr_gt": torch.from_numpy(np.stack(ep.theta_corr_gt)).float().to(device),
        "pose": torch.from_numpy(np.stack(ep.pose)).float().to(device),
        "imu": torch.from_numpy(np.stack(ep.imu)).float().to(device),
    }
    # Anticipatory occupancy target (T, 2). All-zero when no actor trajectory
    # was recorded (env without dynamic actors).
    if ep.drone_xy:
        occ = compute_future_occupancy(ep.drone_xy, ep.drone_yaw, ep.actor_xy)
    else:
        occ = np.zeros((len(ep.rgb), 2), dtype=np.float32)
    tensors["occ"] = torch.from_numpy(occ).float().to(device)
    return tensors

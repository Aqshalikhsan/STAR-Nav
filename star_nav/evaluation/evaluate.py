"""Runs the full frozen STAR-Nav pipeline (SACR + CAMR + AGSS-PPO) across
every Scenario x weather-condition cell and aggregates the metrics from
`metrics.py` into a table shaped like the paper's Tables 9-12.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..envs.base_env import BaseCorridorEnv
from ..models.agss_ppo import ActorCritic, AGSSShield
from ..models.camr import CAMR, CausalWindowBuffer
from ..models.sacr import SACR
from .metrics import EpisodeRecord, summarize


@torch.no_grad()
def run_episode(
    env: BaseCorridorEnv,
    sacr: SACR,
    camr: CAMR,
    actor_critic: ActorCritic,
    agss: AGSSShield,
    scenario: str,
    weather: str,
    device: torch.device,
) -> EpisodeRecord:
    obs = env.reset(scenario=scenario, weather=weather)
    window_buffer = CausalWindowBuffer(camr.window_size, camr.input_dim, device)

    def to_tensor(x, dtype=torch.float32):
        return torch.as_tensor(x, dtype=dtype, device=device).unsqueeze(0)

    path_length = 0.0
    prev_xy = obs.pose[:2].copy()
    lateral_deviations, interventions, corrections = [], [], []
    done = False
    success = collided = off_lane = False
    initial_goal_dist = None

    while not done:
        rgb_t = to_tensor(obs.rgb).permute(0, 3, 1, 2) / 255.0
        pose_t = to_tensor(obs.pose)
        imu_t = to_tensor(obs.imu)

        z_struct_aug = sacr.encode(rgb_t)
        x_t = camr.fuse(z_struct_aug, pose_t, imu_t)
        window = window_buffer.push(x_t)
        h_t = camr(window).h_t

        sample = actor_critic.act(h_t, deterministic=True)
        d_left, d_right = z_struct_aug[:, -3], z_struct_aug[:, -1]
        projection = agss.project(sample.action, h_t, d_left, d_right)

        result = env.step(projection["safe_action"].squeeze(0).cpu().numpy())

        if initial_goal_dist is None:
            initial_goal_dist = result.info.goal_distance + 1e-6

        cur_xy = result.obs.pose[:2]
        path_length += float(np.linalg.norm(cur_xy - prev_xy))
        prev_xy = cur_xy

        lateral_deviations.append(abs(result.info.lateral_deviation))
        interventions.append(bool(projection["intervened"].item()))
        corrections.append(float(projection["correction_magnitude"].item()))

        obs = result.obs
        done = result.done
        success = result.success
        collided = result.info.collided
        off_lane = result.info.off_lane

    return EpisodeRecord(
        success=success,
        collided=collided,
        off_lane=off_lane,
        path_length=max(path_length, 1e-6),
        shortest_path=initial_goal_dist,
        lateral_deviations=lateral_deviations,
        agss_interventions=interventions,
        agss_corrections=corrections,
    )


def evaluate_all(
    env: BaseCorridorEnv,
    sacr: SACR,
    camr: CAMR,
    actor_critic: ActorCritic,
    agss: AGSSShield,
    scenarios: list[str],
    weather_conditions: list[str],
    episodes_per_cell: int,
    device: torch.device,
) -> pd.DataFrame:
    sacr.eval()
    camr.eval()
    actor_critic.eval()

    rows = []
    for scenario in scenarios:
        for weather in weather_conditions:
            records = [
                run_episode(env, sacr, camr, actor_critic, agss, scenario, weather, device)
                for _ in range(episodes_per_cell)
            ]
            metrics = summarize(records)
            metrics.update({"scenario": scenario, "weather": weather, "n_episodes": episodes_per_cell})
            rows.append(metrics)

    return pd.DataFrame(rows)

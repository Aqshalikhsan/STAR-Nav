"""Evaluation metrics matching Section 5's reported quantities:

  SR   - Success Rate: episode reaches the goal without collision/timeout.
  CR   - Collision Rate.
  OR   - Off-lane Rate: lateral deviation exceeds the corridor bound.
  SPL  - Success-weighted Path Length (Anderson et al.-style):
         SPL = (1/N) * sum success_i * (shortest_path_i / max(path_i, shortest_path_i))
  delta_lat / delta^2 - lateral displacement error and its variance
         relative to the corridor centerline.
  AGSS intervention rate & mean correction magnitude.
  PSI  - Policy Stability Index: mean action-mean drift between two
         successive policy checkpoints, evaluated on a fixed probe set of
         belief states (lower = more stable policy evolution).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class EpisodeRecord:
    success: bool
    collided: bool
    off_lane: bool
    path_length: float
    shortest_path: float
    lateral_deviations: list = field(default_factory=list)   # per-step |y - y_center|
    agss_interventions: list = field(default_factory=list)     # per-step bool
    agss_corrections: list = field(default_factory=list)       # per-step |v_y_safe - v_y|


def success_rate(records: list[EpisodeRecord]) -> float:
    return float(np.mean([r.success for r in records])) if records else 0.0


def collision_rate(records: list[EpisodeRecord]) -> float:
    return float(np.mean([r.collided for r in records])) if records else 0.0


def off_lane_rate(records: list[EpisodeRecord]) -> float:
    return float(np.mean([r.off_lane for r in records])) if records else 0.0


def success_weighted_path_length(records: list[EpisodeRecord]) -> float:
    if not records:
        return 0.0
    terms = [r.success * (r.shortest_path / max(r.path_length, r.shortest_path, 1e-6)) for r in records]
    return float(np.mean(terms))


def lateral_deviation_variance(records: list[EpisodeRecord]) -> float:
    """delta^2: variance of the lateral position relative to the corridor
    centerline, pooled across all steps of all episodes.
    """
    all_dev = np.concatenate([np.asarray(r.lateral_deviations) for r in records if r.lateral_deviations]) \
        if any(r.lateral_deviations for r in records) else np.array([0.0])
    return float(np.var(all_dev))


def agss_intervention_rate(records: list[EpisodeRecord]) -> float:
    all_interventions = np.concatenate([np.asarray(r.agss_interventions) for r in records if r.agss_interventions]) \
        if any(r.agss_interventions for r in records) else np.array([0.0])
    return float(np.mean(all_interventions))


def agss_mean_correction(records: list[EpisodeRecord]) -> float:
    all_corr = np.concatenate([np.asarray(r.agss_corrections) for r in records if r.agss_corrections]) \
        if any(r.agss_corrections for r in records) else np.array([0.0])
    return float(np.mean(all_corr))


def summarize(records: list[EpisodeRecord]) -> dict[str, float]:
    return {
        "SR": success_rate(records),
        "CR": collision_rate(records),
        "OR": off_lane_rate(records),
        "SPL": success_weighted_path_length(records),
        "delta2": lateral_deviation_variance(records),
        "AGSS_intervention_rate": agss_intervention_rate(records),
        "AGSS_mean_correction": agss_mean_correction(records),
    }


@torch.no_grad()
def policy_stability_index(actor_new, actor_old, probe_beliefs: torch.Tensor) -> float:
    """PSI = mean_i || mu_new(h_i) - mu_old(h_i) ||_2 over a fixed probe
    set of belief states, evaluated between two successive checkpoints of
    the same actor architecture.
    """
    mu_new, _ = actor_new(probe_beliefs)
    mu_old, _ = actor_old(probe_beliefs)
    return float((mu_new - mu_old).norm(dim=-1).mean().item())

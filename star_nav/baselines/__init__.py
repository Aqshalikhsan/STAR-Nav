"""Baseline navigation policies (paper Section 5.5): PPO, Mem-DRL, ViT-PPO, TD3,
NavRL. Each consumes the SAME monocular RGB + pose + IMU observation as STAR-Nav
and emits the same 4-DoF action, so it runs in MockCorridorEnv / GazeboROSEnv
and feeds the results exporters (05 comparative, 07/08, 09 trajectory).

Each is a dedicated implementation following its cited source work's formulation:

  PPO      [59] Schulman et al.        clipped-surrogate PPO + GAE, CNN encoder
  TD3      [42] Fujimoto et al.        twin critics, target smoothing, delayed actor
  Mem-DRL  [65] Haddad & Khudher       dual-memory recurrent (fast + slow LSTM) + PPO
  ViT-PPO  [66] Wei et al. (DTPPO)     dual-transformer (spatial + temporal) + PPO
  NavRL    [68] Xu et al.              PPO + velocity-obstacle safety shield

These are best-effort reimplementations from the published formulations (no
original source code was available offline); PPO/TD3 match their algorithms, the
others match each method's architecture and mechanism at comparable capacity.

  make_baseline(name, obs_spec, action_dim, device, **cfg) -> agent
"""
from __future__ import annotations

from .mem_drl import MemDRLAgent
from .navrl import NavRLAgent
from .ppo import PPOAgent
from .td3 import TD3Agent
from .vit_ppo import DTPPOAgent

_REGISTRY = {
    "PPO": PPOAgent,
    "TD3": TD3Agent,
    "MemDRL": MemDRLAgent,
    "ViTPPO": DTPPOAgent,
    "NavRL": NavRLAgent,
}

BASELINE_NAMES = list(_REGISTRY)


def make_baseline(name: str, obs_spec, action_dim: int, device, **override):
    if name not in _REGISTRY:
        raise ValueError(f"unknown baseline {name!r}; choices: {BASELINE_NAMES}")
    return _REGISTRY[name](obs_spec=obs_spec, action_dim=action_dim, device=device, name=name, **override)

"""On-policy rollout storage and Generalized Advantage Estimation.

  delta_t   = r_t + gamma * V(h_{t+1}) - V(h_t)
  A_hat_t   = sum_{k=0}^{T} (gamma * lambda_GAE)^k * delta_{t+k}
  return_t  = A_hat_t + V(h_t)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class RolloutBuffer:
    belief_dim: int
    action_dim: int
    capacity: int
    device: torch.device

    def __post_init__(self):
        self.beliefs = torch.zeros(self.capacity, self.belief_dim, device=self.device)
        self.actions = torch.zeros(self.capacity, self.action_dim, device=self.device)
        self.log_probs = torch.zeros(self.capacity, device=self.device)
        self.values = torch.zeros(self.capacity, device=self.device)
        self.rewards = torch.zeros(self.capacity, device=self.device)
        self.dones = torch.zeros(self.capacity, device=self.device)
        self.ptr = 0

    def add(self, belief, action, log_prob, value, reward, done) -> None:
        i = self.ptr
        self.beliefs[i] = belief
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.values[i] = value
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.ptr += 1

    def full(self) -> bool:
        return self.ptr >= self.capacity

    def reset(self) -> None:
        self.ptr = 0

    def compute_gae(self, last_value: torch.Tensor, gamma: float, lam: float):
        n = self.ptr
        advantages = torch.zeros(n, device=self.device)
        last_gae = 0.0
        for t in reversed(range(n)):
            next_value = last_value if t == n - 1 else self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + self.values[:n]
        return advantages, returns

    def get(self, last_value: torch.Tensor, gamma: float, lam: float):
        n = self.ptr
        advantages, returns = self.compute_gae(last_value, gamma, lam)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return {
            "beliefs": self.beliefs[:n],
            "actions": self.actions[:n],
            "log_probs": self.log_probs[:n],
            "values": self.values[:n],
            "advantages": advantages,
            "returns": returns,
        }


def minibatches(data: dict, batch_size: int, epochs: int, rng: np.random.Generator):
    n = data["beliefs"].shape[0]
    for _ in range(epochs):
        idx = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch_idx = idx[start:start + batch_size]
            yield {k: v[batch_idx] for k, v in data.items()}

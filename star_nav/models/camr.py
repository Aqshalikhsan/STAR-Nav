"""CAMR: Consistency-Aware Memory Representation (paper Section 3.3).

x_t = [z_struct_aug ; pose_t ; imu_raw_t]                      (fusion)
W_t = [x_{t-T+1}, ..., x_t]                                    (causal window)
h_fwd = LSTM_fwd(W_t)[-1]                                       (forward pass)
h_rev = LSTM_rev(reverse(W_t))[-1]                              (reverse pass)
h_t = [h_fwd ; h_rev] in R^{2*d_h}                              (belief state)

The window W_t contains only past and current observations (x_{t-T+1..t}),
so running a second LSTM over it in reverse order is still causal at the
outer timestep t -- it never touches x_{t+k}, k > 0. That is exactly the
"reverse-order traversal of the causal sliding window" described in the
paper, not a non-causal bidirectional encoder over the whole trajectory.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CAMROutput:
    h_t: torch.Tensor         # (B, 2*d_h) -- sole state consumed by PPO & AGSS (I3)
    h_fwd: torch.Tensor       # (B, d_h)
    h_rev: torch.Tensor       # (B, d_h)


class CAMR(nn.Module):
    def __init__(
        self,
        z_struct_aug_dim: int,
        pose_dim: int = 7,
        imu_dim: int = 6,
        window_size: int = 10,
        hidden_dim: int = 128,
        predict_occupancy: bool = False,
        occ_dim: int = 2,
    ):
        super().__init__()
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.input_dim = z_struct_aug_dim + pose_dim + imu_dim
        self.use_occupancy = predict_occupancy
        self.occ_dim = occ_dim

        self.lstm_fwd = nn.LSTM(self.input_dim, hidden_dim, batch_first=True)
        self.lstm_rev = nn.LSTM(self.input_dim, hidden_dim, batch_first=True)

        # f_pred: predictive projection head used only for L_pred during training
        self.f_pred = nn.Sequential(
            nn.Linear(2 * hidden_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, z_struct_aug_dim),
        )

        # Anticipatory occupancy head (novelty): from the belief h_t predict
        # whether a dynamic actor will occupy the [left, right] of the drone's
        # path over a short future horizon. Lets AGSS/policy react to where a
        # worker *will be*, not where it is now. Trained with BCE against
        # ground-truth future actor occupancy (Mock: self._actor_xy trajectory;
        # Gazebo: actor poses). Off by default -> no behavioural change.
        if predict_occupancy:
            self.occ_head = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, occ_dim),
            )

    def fuse(self, z_struct_aug: torch.Tensor, pose: torch.Tensor, imu_raw: torch.Tensor) -> torch.Tensor:
        """x_t = [z_struct_aug ; pose_t ; imu_raw_t]"""
        return torch.cat([z_struct_aug, pose, imu_raw], dim=-1)

    def forward(self, window: torch.Tensor) -> CAMROutput:
        """window: (B, T, input_dim), oldest-to-newest, T <= window_size,
        containing only observations up to and including the current step.
        """
        _, (h_n_fwd, _) = self.lstm_fwd(window)
        h_fwd = h_n_fwd[-1]                                   # (B, d_h), state at x_t

        reversed_window = torch.flip(window, dims=[1])
        _, (h_n_rev, _) = self.lstm_rev(reversed_window)
        h_rev = h_n_rev[-1]                                   # (B, d_h), state at x_{t-T+1}

        h_t = torch.cat([h_fwd, h_rev], dim=-1)               # (B, 2*d_h)
        return CAMROutput(h_t=h_t, h_fwd=h_fwd, h_rev=h_rev)

    def predict_next(self, h_t: torch.Tensor) -> torch.Tensor:
        return self.f_pred(h_t)

    def predict_occupancy(self, h_t: torch.Tensor) -> Optional[torch.Tensor]:
        """Future actor-occupancy logits (B, occ_dim) for [left, right, ...].
        None when the head is disabled. Apply sigmoid for probabilities.
        """
        if not self.use_occupancy:
            return None
        return self.occ_head(h_t)


class CausalWindowBuffer:
    """Fixed-length deque implementing W_t for online (single-episode)
    rollout. Not used during batched offline training, where windows are
    unfolded directly from a stored sequence (see training/train_camr.py).
    """

    def __init__(self, window_size: int, input_dim: int, device: torch.device):
        self.window_size = window_size
        self.input_dim = input_dim
        self.device = device
        self._buf: deque[torch.Tensor] = deque(maxlen=window_size)

    def reset(self) -> None:
        self._buf.clear()

    def push(self, x_t: torch.Tensor) -> torch.Tensor:
        """Appends x_t (B, input_dim) and returns the current window
        (B, T_cur, input_dim), left-padded by repeating the first
        observation so early-episode windows still have shape T=window_size.
        """
        self._buf.append(x_t)
        frames = list(self._buf)
        while len(frames) < self.window_size:
            frames.insert(0, frames[0])
        return torch.stack(frames, dim=1)


def camr_loss(
    output: CAMROutput,
    z_struct_aug_next: torch.Tensor,
    predicted_next: torch.Tensor,
    prev_h_t: Optional[torch.Tensor] = None,
    beta_temp: float = 0.5,
    occ_logits: Optional[torch.Tensor] = None,
    occ_target: Optional[torch.Tensor] = None,
    lambda_occ: float = 0.5,
) -> dict[str, torch.Tensor]:
    """L_CAMR = L_pred + beta * L_temp + lambda_occ * L_occ
       L_pred = ||f_pred(h_t) - z_struct_aug_{t+1}||^2
       L_temp = ||h_t - h_{t-1}||^2
       L_occ  = BCE(sigmoid(occ_logits), occ_target)  -- anticipatory head

    L_occ trains the belief to predict whether a dynamic actor will occupy the
    [left, right] of the drone's path over a short future horizon, so the shield
    and policy can anticipate moving workers. Backward compatible: omit
    ``occ_logits``/``occ_target`` and L_occ is 0.
    """
    l_pred = F.mse_loss(predicted_next, z_struct_aug_next)

    if prev_h_t is not None:
        l_temp = F.mse_loss(output.h_t, prev_h_t.detach())
    else:
        l_temp = torch.zeros((), device=output.h_t.device)

    if occ_logits is not None and occ_target is not None:
        l_occ = F.binary_cross_entropy_with_logits(occ_logits, occ_target)
    else:
        l_occ = torch.zeros((), device=output.h_t.device)

    l_camr = l_pred + beta_temp * l_temp + lambda_occ * l_occ
    return {"L_CAMR": l_camr, "L_pred": l_pred, "L_temp": l_temp, "L_occ": l_occ}

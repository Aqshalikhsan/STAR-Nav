"""Common environment interface shared by MockCorridorEnv (bundled,
dependency-free) and AirSimCorridorEnv (the actual Unreal Engine +
Microsoft AirSim setup described in Section 4 of the paper).

Deliberately not a `gym.Env` subclass -- STAR-Nav's observation is a
structured bundle (image + pose + IMU + privileged geometry), which maps
more directly onto a small dataclass than onto gym's Box/Dict spaces, and
this keeps the reference implementation free of a gymnasium dependency.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class EnvObservation:
    """Everything the perception stack (SACR) is allowed to see."""
    rgb: np.ndarray          # (H, W, 3) uint8
    pose: np.ndarray         # (7,) VIO position (3) + quaternion (4)
    imu: np.ndarray          # (6,) accel (3) + gyro (3)


@dataclass
class PrivilegedInfo:
    """Ground-truth quantities available only from the simulator, used to
    supervise SACR pretraining and to compute reward / termination. Never
    exposed to the policy at inference time.
    """
    seg_mask: np.ndarray         # (H, W) int class ids
    depth: np.ndarray            # (H, W) float32 metres
    theta_corr_gt: np.ndarray    # (4,) [phi, d_L, d_R, d_F]
    d_left: float
    d_right: float
    goal_distance: float
    collided: bool
    off_lane: bool
    lateral_deviation: float     # signed lateral offset from the corridor centerline (m)


@dataclass
class EnvStepResult:
    obs: EnvObservation
    reward: float
    done: bool
    success: bool
    info: PrivilegedInfo


class BaseCorridorEnv(ABC):
    """Scenario A/B/C plantation-corridor navigation task, POMDP-formulated
    per Section 3.1: continuous state, continuous 4-DoF action
    (v_x, v_y, v_z, omega), monocular RGB + IMU observation only.
    """

    action_dim = 4

    @abstractmethod
    def reset(self, scenario: str = "A", weather: str = "clear_day") -> EnvObservation:
        ...

    @abstractmethod
    def step(self, action: np.ndarray) -> EnvStepResult:
        ...

    @property
    @abstractmethod
    def max_forward_speed(self) -> float:
        ...

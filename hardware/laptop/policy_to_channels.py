"""policy_to_channels.py -- ground-station velocity loop between the AGSS-filtered
command and the RC link.

The policy (after AGSS) outputs a body-frame velocity and yaw-rate setpoint
a_safe = (v_x, v_y, v_z, omega) in physical units. The iNav flight controller
runs in ANGLE mode, where stick deflection sets an attitude target, so the
ground station closes a horizontal-velocity PID against the VINS-Mono velocity
estimate and outputs roll/pitch angle targets limited to +/- max_tilt_deg.
Yaw rate and the vertical command are passed as proportional stick channels.

    v_x error -> pitch target      v_y error -> roll target
    omega     -> yaw stick         v_z       -> throttle around hover

The resulting normalized channels go out through RCLink (Arduino -> PPM ->
ELRS TX -> ELRS RX -> flight controller over CRSF).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PID:
    kp: float
    ki: float
    kd: float
    limit: float                      # output saturation (deg)
    _integral: float = field(default=0.0, init=False)
    _prev_error: float = field(default=0.0, init=False)
    _first: bool = field(default=True, init=False)

    def reset(self) -> None:
        self._integral, self._prev_error, self._first = 0.0, 0.0, True

    def step(self, error: float, dt: float) -> float:
        derivative = 0.0 if self._first else (error - self._prev_error) / dt
        self._first = False
        self._prev_error = error
        candidate = self._integral + error * dt
        out = self.kp * error + self.ki * candidate + self.kd * derivative
        if abs(out) < self.limit:              # conditional integration (anti-windup)
            self._integral = candidate
        return float(np.clip(out, -self.limit, self.limit))


def world_to_body_xy(v_world: np.ndarray, yaw: float) -> np.ndarray:
    """Rotate a horizontal velocity from the VIO frame into the body frame."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([c * v_world[0] + s * v_world[1], -s * v_world[0] + c * v_world[1]])


def yaw_from_quaternion(q: np.ndarray) -> float:
    x, y, z, w = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


class VelocityToAttitude:
    """Horizontal-velocity PID -> roll/pitch angle targets (+/- max_tilt_deg)."""

    def __init__(self, gains, max_tilt_deg: float = 30.0):
        if gains is None or len(gains) != 3:
            raise ValueError("velocity PID gains [kp, ki, kd] must be set for the airframe")
        kp, ki, kd = gains
        self.max_tilt_deg = max_tilt_deg
        self.pid_x = PID(kp, ki, kd, max_tilt_deg)
        self.pid_y = PID(kp, ki, kd, max_tilt_deg)

    def reset(self) -> None:
        self.pid_x.reset(); self.pid_y.reset()

    def __call__(self, v_cmd_body: np.ndarray, v_meas_body: np.ndarray, dt: float) -> tuple[float, float]:
        pitch_deg = self.pid_x.step(v_cmd_body[0] - v_meas_body[0], dt)   # forward
        roll_deg = self.pid_y.step(v_cmd_body[1] - v_meas_body[1], dt)    # lateral (+ right)
        return roll_deg, pitch_deg


def make_velocity_sender(link, controller: VelocityToAttitude, hover_throttle: float = 0.0,
                         max_vertical_speed: float = 1.0, max_yaw_rate_deg: float = 120.0,
                         throttle_gain: float = 0.5):
    """Return send(a_safe, v_meas_body, dt, armed) mapping the filtered setpoint to RC sticks.

    a_safe: (v_x, v_y, v_z, omega) in m/s and deg/s (body frame).
    v_meas_body: measured horizontal body velocity (m/s) from the VIO estimate.
    """
    def send(a_safe, v_meas_body, dt, armed=True):
        vx, vy, vz, omega = (list(a_safe) + [0, 0, 0, 0])[:4]
        roll_deg, pitch_deg = controller(np.array([vx, vy]), np.asarray(v_meas_body), dt)
        link.send_norm(
            roll=roll_deg / controller.max_tilt_deg,
            pitch=pitch_deg / controller.max_tilt_deg,
            yaw=float(np.clip(omega / max_yaw_rate_deg, -1.0, 1.0)),
            throttle=float(np.clip(hover_throttle + throttle_gain * vz / max_vertical_speed, -1.0, 1.0)),
            arm=armed,
        )
        return roll_deg, pitch_deg

    return send

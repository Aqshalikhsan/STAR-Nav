"""Real-simulator backend: Microsoft AirSim + Unreal Engine 4.27, as
described in Section 4 ("Design of Simulation"). This module is a
structural template, not a bundled asset -- it requires:

  1. `pip install airsim`
  2. A running Unreal Engine 4.27 instance with the custom oil-palm
     plantation environment (procedurally placed trees per Table 6,
     AirSim plugin enabled) already loaded, listening on the default
     RPC port.
  3. An AirSim `settings.json` configuring a multirotor with a monocular
     RGB camera (640x480 @ 30 FPS) and IMU, GPS disabled.

It implements the same `BaseCorridorEnv` contract as `MockCorridorEnv`,
so `star_nav.training.train_ppo` does not need to know which backend is
in use -- only `configs/default.yaml`'s `env.name` changes.
"""
from __future__ import annotations

import numpy as np

try:
    import airsim
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "AirSimCorridorEnv requires the `airsim` package and a running "
        "Unreal Engine instance. Install with `pip install airsim`, or "
        "use env.name: mock in configs/default.yaml to run without it."
    ) from exc

from .base_env import BaseCorridorEnv, EnvObservation, EnvStepResult, PrivilegedInfo


class AirSimCorridorEnv(BaseCorridorEnv):
    def __init__(self, cfg):
        self.cfg = cfg
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self._max_forward_speed = cfg.get("max_forward_speed", 2.5) if hasattr(cfg, "get") else 2.5
        self._max_yaw_rate = np.deg2rad(120.0)
        self.dt = 0.2
        self.max_steps = cfg.episode_max_steps
        self._t = 0
        self._prev_goal_dist = 0.0
        self._prev_omega = 0.0
        self._prev_v = np.zeros(3)

        # Goal / scenario metadata should match the region actually built
        # into the loaded Unreal level for the requested scenario name.
        self._goal_xy = np.array([cfg.area_size_m - 2.0, 0.0], dtype=np.float32)

    @property
    def max_forward_speed(self) -> float:
        return self._max_forward_speed

    def reset(self, scenario: str = "A", weather: str = "clear_day") -> EnvObservation:
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        self._t = 0
        self._prev_omega = 0.0
        self._prev_v = np.zeros(3, dtype=np.float32)
        obs, info = self._observe()
        self._prev_goal_dist = info.goal_distance
        return obs

    def step(self, action: np.ndarray) -> EnvStepResult:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        v_x, v_y, v_z, omega_n = action
        v_x = float(v_x * self._max_forward_speed)
        v_y = float(v_y * self._max_forward_speed)
        v_z = float(v_z * self._max_forward_speed)
        yaw_rate_deg = float(omega_n * np.rad2deg(self._max_yaw_rate))

        self.client.moveByVelocityBodyFrameAsync(
            v_x, v_y, v_z, self.dt,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate_deg),
        ).join()
        self._t += 1

        obs, info = self._observe()

        r_progress = (self._prev_goal_dist - info.goal_distance) / self._max_forward_speed
        omega = np.deg2rad(yaw_rate_deg)
        d_omega = omega - self._prev_omega
        d_v = np.array([v_x, v_y, v_z]) - self._prev_v
        r_smooth = -abs(d_omega) - float(np.linalg.norm(d_v))
        reward = 1.0 * r_progress + 0.05 * r_smooth + 0.01 * 1.0

        self._prev_goal_dist = info.goal_distance
        self._prev_omega = omega
        self._prev_v = np.array([v_x, v_y, v_z])

        success = info.goal_distance < 1.5
        timeout = self._t >= self.max_steps
        done = bool(success or info.collided or timeout)

        return EnvStepResult(obs=obs, reward=float(reward), done=done, success=success, info=info)

    def _observe(self) -> tuple[EnvObservation, PrivilegedInfo]:
        responses = self.client.simGetImages([
            airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("front_center", airsim.ImageType.DepthPlanar, True),
            airsim.ImageRequest("front_center", airsim.ImageType.Segmentation, False, False),
        ])
        rgb_resp, depth_resp, seg_resp = responses

        rgb = np.frombuffer(rgb_resp.image_data_uint8, dtype=np.uint8).reshape(
            rgb_resp.height, rgb_resp.width, 3)
        depth = np.array(depth_resp.image_data_float, dtype=np.float32).reshape(
            depth_resp.height, depth_resp.width)
        seg = np.frombuffer(seg_resp.image_data_uint8, dtype=np.uint8).reshape(
            seg_resp.height, seg_resp.width, 3)[..., 0].astype(np.int64)

        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        ori = state.kinematics_estimated.orientation
        pose = np.array([pos.x_val, pos.y_val, pos.z_val,
                          ori.x_val, ori.y_val, ori.z_val, ori.w_val], dtype=np.float32)

        imu_data = self.client.getImuData()
        imu = np.array([
            imu_data.linear_acceleration.x_val, imu_data.linear_acceleration.y_val, imu_data.linear_acceleration.z_val,
            imu_data.angular_velocity.x_val, imu_data.angular_velocity.y_val, imu_data.angular_velocity.z_val,
        ], dtype=np.float32)

        w, h = depth.shape[1], depth.shape[0]
        d_left = float(depth[:, : w // 3].mean())
        d_right = float(depth[:, 2 * w // 3:].mean())
        d_center = float(depth[:, w // 3: 2 * w // 3].mean())

        collision_info = self.client.simGetCollisionInfo()
        goal_distance = float(np.linalg.norm(np.array([pos.x_val, pos.y_val]) - self._goal_xy))

        info = PrivilegedInfo(
            seg_mask=seg,
            depth=depth,
            theta_corr_gt=np.array([0.0, d_left, d_right, d_center], dtype=np.float32),
            d_left=d_left,
            d_right=d_right,
            goal_distance=goal_distance,
            collided=bool(collision_info.has_collided),
            off_lane=False,  # requires lane-centerline ground truth from the level itself
            lateral_deviation=0.0,  # populate from the level's corridor-centerline spline if available
        )
        return EnvObservation(rgb=rgb, pose=pose, imu=imu), info

"""Pose input for CAMR from monocular VINS-Mono.

VINS-Mono runs on the same camera stream as SACR together with the vehicle
IMU and publishes ``nav_msgs/Odometry`` (default ``/vins_estimator/odometry``).
CAMR consumes ``pose_t = [p - w_active ; q]`` (7-D): the position relative to
the active waypoint and the orientation quaternion (x, y, z, w). The VIO scale
is used for the pose only; SACR's metric range proxies are regressed from the
image and never rescaled by it.

``VinsOdometry`` keeps the latest message; ``waypoint_relative_pose`` builds the
CAMR input and advances the active waypoint once it is reached.
"""
from __future__ import annotations

import threading
from typing import Optional, Sequence

import numpy as np


class VinsOdometry:
    """Latest VINS-Mono odometry (position, quaternion, linear velocity)."""

    def __init__(self, topic: str = "/vins_estimator/odometry", node_name: str = "star_nav_vins_reader"):
        self._lock = threading.Lock()
        self._p = None
        self._q = None
        self._v = None
        self._stamp = None
        try:
            import rospy
            from nav_msgs.msg import Odometry
        except ImportError as exc:  # pragma: no cover
            raise ImportError("VinsOdometry needs a ROS environment with nav_msgs "
                              "(the VINS-Mono estimator publishes over ROS).") from exc
        if not rospy.core.is_initialized():
            rospy.init_node(node_name, anonymous=True, disable_signals=True)
        rospy.Subscriber(topic, Odometry, self._callback, queue_size=10)

    def _callback(self, msg) -> None:
        pp, oo, vv = msg.pose.pose.position, msg.pose.pose.orientation, msg.twist.twist.linear
        with self._lock:
            self._p = np.array([pp.x, pp.y, pp.z], dtype=np.float32)
            self._q = np.array([oo.x, oo.y, oo.z, oo.w], dtype=np.float32)
            self._v = np.array([vv.x, vv.y, vv.z], dtype=np.float32)
            self._stamp = msg.header.stamp.to_sec()

    def ready(self) -> bool:
        with self._lock:
            return self._p is not None

    def latest(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[float]]:
        """(position, quaternion xyzw, linear velocity, stamp) in the VIO frame."""
        with self._lock:
            if self._p is None:
                raise RuntimeError("No VINS-Mono odometry received yet (check initialization).")
            return self._p.copy(), self._q.copy(), self._v.copy(), self._stamp


class WaypointTracker:
    """Active-waypoint bookkeeping for the waypoint-relative CAMR pose."""

    def __init__(self, waypoints: Sequence[Sequence[float]], radius: float = 1.0):
        wp = np.asarray(waypoints, dtype=np.float32)
        if wp.ndim != 2 or wp.shape[1] not in (2, 3) or len(wp) == 0:
            raise ValueError("waypoints must be an (N, 2) or (N, 3) array")
        if wp.shape[1] == 2:
            wp = np.concatenate([wp, np.zeros((len(wp), 1), np.float32)], axis=1)
        self.waypoints = wp
        self.radius = radius
        self.index = 0

    @property
    def active(self) -> np.ndarray:
        return self.waypoints[self.index]

    def update(self, position: np.ndarray) -> np.ndarray:
        """Advance past reached waypoints (horizontal distance) and return the active one."""
        while (self.index < len(self.waypoints) - 1
               and np.linalg.norm(position[:2] - self.waypoints[self.index, :2]) < self.radius):
            self.index += 1
        return self.active


def waypoint_relative_pose(position: np.ndarray, quaternion: np.ndarray, waypoint: np.ndarray) -> np.ndarray:
    """pose_t = [p - w_active ; q] in R^7."""
    return np.concatenate([position - waypoint, quaternion]).astype(np.float32)

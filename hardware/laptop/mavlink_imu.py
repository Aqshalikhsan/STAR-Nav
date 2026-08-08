"""Flight-controller IMU over MAVLink telemetry.

Supplies imu_raw_t = [a_x, a_y, a_z, w_x, w_y, w_z] (m/s^2, rad/s) to CAMR and,
when ROS is available, republishes it as sensor_msgs/Imu for VINS-Mono, which
runs on the same camera stream on the ground station.

Accepted messages: HIGHRES_IMU (SI units), SCALED_IMU / SCALED_IMU2
(mG, mrad/s) and RAW_IMU (treated as SCALED units, as sent by iNav).
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

G = 9.80665


class MavlinkImu:
    def __init__(self, url: str = "udpin:0.0.0.0:14550", ros_topic: Optional[str] = "/imu0"):
        from pymavlink import mavutil
        self._conn = mavutil.mavlink_connection(url)
        self._lock = threading.Lock()
        self._imu: Optional[np.ndarray] = None
        self._stamp: Optional[float] = None
        self._pub = None
        if ros_topic:
            try:
                import rospy
                from sensor_msgs.msg import Imu
                if not rospy.core.is_initialized():
                    rospy.init_node("star_nav_mavlink_imu", anonymous=True, disable_signals=True)
                self._pub = rospy.Publisher(ros_topic, Imu, queue_size=200)
                self._imu_msg = Imu
                self._rospy = rospy
            except ImportError:
                self._pub = None
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _decode(self, msg) -> Optional[np.ndarray]:
        kind = msg.get_type()
        if kind == "HIGHRES_IMU":
            return np.array([msg.xacc, msg.yacc, msg.zacc, msg.xgyro, msg.ygyro, msg.zgyro], np.float32)
        if kind in ("SCALED_IMU", "SCALED_IMU2", "RAW_IMU"):
            acc = np.array([msg.xacc, msg.yacc, msg.zacc], np.float32) * 1e-3 * G
            gyr = np.array([msg.xgyro, msg.ygyro, msg.zgyro], np.float32) * 1e-3
            return np.concatenate([acc, gyr])
        return None

    def _loop(self) -> None:
        types = ["HIGHRES_IMU", "SCALED_IMU", "SCALED_IMU2", "RAW_IMU"]
        while not self._stop:
            msg = self._conn.recv_match(type=types, blocking=True, timeout=1.0)
            if msg is None:
                continue
            imu = self._decode(msg)
            if imu is None:
                continue
            now = time.time()
            with self._lock:
                self._imu, self._stamp = imu, now
            if self._pub is not None:
                m = self._imu_msg()
                m.header.stamp = self._rospy.Time.from_sec(now)
                m.header.frame_id = "imu"
                m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = map(float, imu[:3])
                m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = map(float, imu[3:])
                self._pub.publish(m)

    def latest(self) -> np.ndarray:
        with self._lock:
            if self._imu is None:
                raise RuntimeError("No IMU message received over MAVLink yet.")
            return self._imu.copy()

    def close(self) -> None:
        self._stop = True
        self._thread.join(timeout=2.0)

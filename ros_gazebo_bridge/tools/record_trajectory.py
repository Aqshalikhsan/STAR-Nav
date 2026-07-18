"""Record the vehicle's ground-truth trajectory to CSV while it flies.

Subscribes to the Gazebo ground-truth odometry (the OdometryPublisher output
bridged by ros_gz_bridge, e.g. /model/fpv5_0/odometry) and logs
`t,x,y,z,yaw` at the topic rate. Pair it with a flight (manual_control.py) to
capture a real run, then render it with make_flight_video.py.

Run inside the ros-bridge container (MAVROS + ros_gz_bridge up):
    python3 record_trajectory.py /tmp/traj.csv 55 /model/fpv5_0/odometry
Args: <out.csv> <duration_s> [odom_topic]
"""
import math
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from nav_msgs.msg import Odometry

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/traj.csv"
DUR = float(sys.argv[2]) if len(sys.argv) > 2 else 55.0
TOPIC = sys.argv[3] if len(sys.argv) > 3 else "/model/fpv5_0/odometry"


class Recorder(Node):
    def __init__(self):
        super().__init__("traj_recorder")
        qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                         history=QoSHistoryPolicy.KEEP_LAST, depth=10)
        self.create_subscription(Odometry, TOPIC, self._cb, qos)
        self.f = open(OUT, "w")
        self.f.write("t,x,y,z,yaw\n")
        self.t0 = time.monotonic()
        self.n = 0

    def _cb(self, m: Odometry):
        p = m.pose.pose.position
        q = m.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                         1 - 2 * (q.y * q.y + q.z * q.z))
        self.f.write(f"{time.monotonic()-self.t0:.3f},"
                     f"{p.x:.3f},{p.y:.3f},{p.z:.3f},{yaw:.4f}\n")
        self.n += 1


def main():
    rclpy.init()
    node = Recorder()
    end = time.monotonic() + DUR
    while rclpy.ok() and time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.05)
    node.f.close()
    print(f"recorded {node.n} samples to {OUT}")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

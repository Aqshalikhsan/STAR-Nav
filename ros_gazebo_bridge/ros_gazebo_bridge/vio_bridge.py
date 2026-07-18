"""Feed a VIO-quality odometry estimate into PX4's EKF (external vision).

Why this exists
---------------
STAR-Nav is a GPS-DENIED navigation system: the real drone holds position
from visual-inertial odometry (VIO), not GPS. But PX4's EKF2, left alone in
SITL, estimates position from the *simulated* GPS -- which drifts badly here
("GPS Horizontal Pos Drift too high"), so the EKF pose diverges from reality
(the vehicle flies correctly per Gazebo ground truth while the EKF estimate
wanders metres away, especially in z). Nothing in the stack was feeding a
visual pose to the flight controller: SACR/CAMR run on the RL *agent* side
and never reach PX4's EKF.

This node closes that gap by republishing the Gazebo ground-truth odometry
(a stand-in for a perfect onboard VIO -- exactly what the real hardware
provides) onto the MAVROS odometry-in topic, which forwards it to PX4 as an
external-vision ODOMETRY estimate. With PX4's EKF2 configured to fuse
external vision (EKF2_EV_CTRL) and to stop trusting GPS (EKF2_GPS_CTRL 0),
the EKF locks onto this clean pose and the flight becomes stable AND the sim
becomes faithful to the paper's GPS-denied premise.

It is deliberately a thin pass-through (ground truth == perfect VIO). To
model a realistic, noisier VIO later, add noise/latency/drift here.

Frames: the gz odometry is in the gz world ENU frame with a base_link child
frame; MAVROS's odom plugin converts ENU->NED for PX4 and PX4 aligns the
external-vision frame origin to its own local origin, so the world-vs-spawn
offset is handled by the EKF.

It feeds PX4 through MAVROS's mocap plugin (`~/mocap/tf`, a TransformStamped
that becomes an ATT_POS_MOCAP MAVLink message) rather than the vision_pose
plugin, because on this ros-humble-mavros build the mocap plugin is the
external-pose input actually loaded/advertised (vision_pose isn't). PX4
fuses ATT_POS_MOCAP as external vision exactly like VISION_POSITION_ESTIMATE
when EKF2_EV_CTRL is enabled -- and ground truth IS effectively a perfect
motion-capture pose, so mocap is the natural fit.

Usage (inside the ros-bridge container, MAVROS running):
    python3 -m ros_gazebo_bridge.vio_bridge \
        --gt-odom-topic /model/fpv5_0/odometry \
        --mocap-tf-topic /mavros/mavros/mocap/tf
"""
from __future__ import annotations

import argparse

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry


class VioBridge(Node):
    def __init__(self, args):
        super().__init__("star_nav_vio_bridge")
        self.args = args
        self._n = 0

        qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                         history=QoSHistoryPolicy.KEEP_LAST, depth=10)
        self._pub = self.create_publisher(TransformStamped, args.mocap_tf_topic, qos)
        self.create_subscription(Odometry, args.gt_odom_topic, self._on_odom, qos)
        self.get_logger().info(
            f"VIO bridge: {args.gt_odom_topic} -> {args.mocap_tf_topic} "
            f"(ATT_POS_MOCAP; frame_id={args.frame_id}, child={args.child_frame_id})")

    def _on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        # Offset into PX4's LOCAL frame. The gz odometry is in world
        # coordinates (e.g. y ~= 24 on the corridor centreline), but PX4's EKF
        # local origin is at the vehicle spawn point -- feeding raw world
        # coordinates makes the external-vision innovation huge (tens of
        # metres) so the EKF gates it out (or resets), and the estimate never
        # locks on. Subtracting the spawn origin puts the mocap pose in the
        # same frame the EKF expects (small innovation -> clean fusion). The
        # default origin matches PX4_GZ_MODEL_POSE for the corridor spawn.
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = self.args.frame_id
        tf.child_frame_id = self.args.child_frame_id
        tf.transform.translation.x = p.x - self.args.origin_x
        tf.transform.translation.y = p.y - self.args.origin_y
        tf.transform.translation.z = p.z - self.args.origin_z
        tf.transform.rotation = q
        self._pub.publish(tf)
        self._n += 1
        if self._n % 100 == 1:
            self.get_logger().info(
                f"forwarded {self._n} (last pos {p.x:.2f},{p.y:.2f},{p.z:.2f})")


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt-odom-topic", default="/model/fpv5_0/odometry",
                   help="Gazebo ground-truth odometry topic (ros_gz_bridge output)")
    p.add_argument("--mocap-tf-topic", default="/mavros/mavros/mocap/tf",
                   help="MAVROS mocap TransformStamped input (forwarded to PX4 as ATT_POS_MOCAP)")
    p.add_argument("--frame-id", default="map")
    p.add_argument("--child-frame-id", default="base_link")
    # Spawn origin in gz world coords (must match PX4_GZ_MODEL_POSE) so the fed
    # pose lands in PX4's local frame; see _on_odom.
    p.add_argument("--origin-x", type=float, default=1.0)
    p.add_argument("--origin-y", type=float, default=24.0)
    p.add_argument("--origin-z", type=float, default=0.3)
    return p


def main():
    args = build_argparser().parse_args()
    rclpy.init()
    node = VioBridge(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

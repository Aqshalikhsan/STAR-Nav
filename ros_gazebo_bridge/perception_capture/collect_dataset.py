"""Manual perception-dataset collector for the PX4-FREE capture path.

Pairs with capture_world_gen.py. Runs INSIDE the ros-bridge container (needs
rclpy + cv_bridge + the `gz` CLI, all present there; host networking). Glides
the camera rig down the corridor under a PHYSICS velocity command (the render
follows physics motion, unlike set_pose), weaving its heading a little for
viewpoint variety, and captures real rgb / seg / depth + the rig's true pose
at a fixed rate. Saves an npz with the fields SACR training expects:

    rgb            (N, H, W, 3) uint8
    seg_mask       (N, H, W)    int64   (0=sky 1=ground 2=trunk; 4=actor when
                                         --num-workers>0 -- see README.md)
    theta_corr_gt  (N, 4)       float32 [phi, d_L, d_R, d_C]
    pose           (N, 7)       float32 rig odom pose [x,y,z, qx,qy,qz,qw]
    depth          (N, H, W)    float32 metric depth map, clipped [0, DEPTH_FAR]
                                        (SACR L_depth + aleatoric L_unc target)

The full depth map (not just the pooled thirds in theta_corr_gt) is saved so
the novelty SACR can supervise its depth net (L_depth) and calibrate the
per-region aleatoric log-variance head (L_unc). Actor pixels in seg_mask (when
workers cross) are the anticipatory-occupancy signal the novelty CAMR learns.

This is the MANUAL data path, fully separate from the autonomy path
(star_nav.training + GazeboROSEnv). It never arms a vehicle and never touches
PX4/MAVROS, so it is unaffected by the SITL GPS-drift auto-disarm that blocks
armed flight.

Manual run recipe (two GPU containers from the existing images, host net):

  # 0) generate the labeled corridor + capture world (host):
  python3 -m ros_gazebo_bridge.world_gen --scenario A \
      --out ros_gazebo_bridge/worlds/scenario_a
  python3 ros_gazebo_bridge/perception_capture/capture_world_gen.py \
      --base-world ros_gazebo_bridge/worlds/scenario_a.sdf \
      --out ros_gazebo_bridge/perception_capture/capture_scenario_a.sdf

  # 1) render container (px4-gazebo image), standalone gz sim (NO PX4):
  docker run -d --name caprig --network host --gpus all \
    -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e __NV_PRIME_RENDER_OFFLOAD=1 -e __GLX_VENDOR_LIBRARY_NAME=nvidia \
    -e GZ_SIM_RESOURCE_PATH=/opt/PX4-Autopilot/Tools/simulation/gz/models \
    -v $PWD/ros_gazebo_bridge/perception_capture/capture_scenario_a.sdf:/w.sdf:ro \
    --entrypoint bash docker-px4-gazebo:latest -c 'gz sim -s -r /w.sdf'

  # 2) collector container (ros-bridge image) -- start it FRESH after (1) is up
  #    (gz-transport discovery is stale if it predates the gz server):
  docker run -d --name capcli --network host --gpus all \
    -v $PWD/ros_gazebo_bridge/perception_capture:/pc \
    -v $PWD/data:/out --entrypoint bash docker-ros-bridge:latest -c 'sleep infinity'
  docker exec capcli bash -lc 'source /opt/ros/humble/setup.bash; \
    source /ros2_ws/install/setup.bash; python3 /pc/collect_dataset.py --out /out/rig_frames.npz'

  # 3) SACR training happens on the HOST GPU from the npz (torch in the
  #    ros-bridge image is CPU-only). See sacr training script.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time

import numpy as np

CMD_VEL_TOPIC = "/model/capture_rig/cmd_vel"
DEPTH_FAR = 20.0

# NOTE: cmd_vel is driven via a ROS->gz parameter_bridge (published with rclpy),
# NOT the `gz topic -p` CLI. The ros-bridge image's `gz` CLI can be a different
# gz-transport major version than the px4-gazebo image's gz *server*, so a
# direct `gz topic -p` from here silently fails to reach the rig (the rig never
# moves). The gzgarden parameter_bridge pins transport12 and works both ways.


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="/out/rig_frames.npz", help="Output npz path.")
    p.add_argument("--speed", type=float, default=1.0, help="Forward glide speed (m/s).")
    p.add_argument("--yaw-amp", type=float, default=0.25,
                   help="Heading weave amplitude (rad/s) for viewpoint variety; 0 = straight.")
    p.add_argument("--yaw-period", type=float, default=6.0, help="Yaw weave period (s).")
    # VelocityControl linear is in the WORLD frame here (forward x stays
    # monotonic under yaw weave), so lateral/vertical weave add real y/z
    # viewpoint diversity within one pass. Keep amplitudes inside the lane.
    p.add_argument("--lateral-amp", type=float, default=0.5,
                   help="Lateral (world y) weave speed amplitude (m/s); 0 = none.")
    p.add_argument("--lateral-period", type=float, default=9.0, help="Lateral weave period (s).")
    p.add_argument("--vert-amp", type=float, default=0.25,
                   help="Vertical (world z) weave speed amplitude (m/s); 0 = none.")
    p.add_argument("--vert-period", type=float, default=7.0, help="Vertical weave period (s).")
    p.add_argument("--capture-hz", type=float, default=4.0, help="Frame capture rate.")
    p.add_argument("--max-x", type=float, default=40.0, help="Stop when rig odom x exceeds this.")
    p.add_argument("--max-frames", type=int, default=200, help="Hard cap on frames.")
    # Dynamic CLASS_ACTOR workers (for CAMR). Each /model/worker<i> is driven
    # laterally (world y) in a sine so it walks back and forth across the lane;
    # per-worker phase offsets desynchronise them. Requires the capture world to
    # have been generated with --worker-xs (capture_world_gen.py).
    p.add_argument("--num-workers", type=int, default=0, help="Number of crossing workers to drive.")
    p.add_argument("--worker-speed", type=float, default=0.9, help="Worker crossing speed amplitude (m/s).")
    p.add_argument("--worker-period", type=float, default=8.0, help="Worker crossing period (s).")
    p.add_argument("--rgb-topic", default="/rig_camera")
    p.add_argument("--depth-topic", default="/rig_depth")
    p.add_argument("--seg-topic", default="/rig_seg/labels_map")
    p.add_argument("--odom-topic", default="/model/capture_rig/odometry")
    args = p.parse_args(argv)

    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import Twist
    from cv_bridge import CvBridge

    # Our own bridge subprocess (setpgrp so we can reliably kill it at the end,
    # and so it isn't orphaned when this process exits). cmd_vel is bridged
    # ROS->gz (`]`), the sensors gz->ROS (`[`).
    bridge = subprocess.Popen(
        ["bash", "-lc",
         "source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash 2>/dev/null; "
         f"exec ros2 run ros_gz_bridge parameter_bridge "
         f"{args.rgb_topic}@sensor_msgs/msg/Image[gz.msgs.Image "
         f"{args.depth_topic}@sensor_msgs/msg/Image[gz.msgs.Image "
         f"{args.seg_topic}@sensor_msgs/msg/Image[gz.msgs.Image "
         f"{args.odom_topic}@nav_msgs/msg/Odometry[gz.msgs.Odometry "
         f"{CMD_VEL_TOPIC}@geometry_msgs/msg/Twist]gz.msgs.Twist "
         + " ".join(f"/model/worker{i}/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist"
                    for i in range(args.num_workers))],
        preexec_fn=os.setsid)
    time.sleep(7)

    rclpy.init()
    node = Node("perception_collect")
    br = CvBridge()
    L = {}
    node.create_subscription(Image, args.rgb_topic, lambda m: L.__setitem__("rgb", m), 1)
    node.create_subscription(Image, args.depth_topic, lambda m: L.__setitem__("depth", m), 1)
    node.create_subscription(Image, args.seg_topic, lambda m: L.__setitem__("seg", m), 1)
    node.create_subscription(Odometry, args.odom_topic, lambda m: L.__setitem__("od", m), 1)
    cmd_pub = node.create_publisher(Twist, CMD_VEL_TOPIC, 10)
    worker_pubs = [node.create_publisher(Twist, f"/model/worker{i}/cmd_vel", 10)
                   for i in range(args.num_workers)]

    def gz_cmd_vel(vx: float, wz: float, vy: float = 0.0, vz: float = 0.0) -> None:
        tw = Twist()
        tw.linear.x = float(vx)
        tw.linear.y = float(vy)
        tw.linear.z = float(vz)
        tw.angular.z = float(wz)
        cmd_pub.publish(tw)

    def drive_workers(t: float) -> None:
        # Each worker wanders in 2D so the group disperses ("berjalan menyebar"):
        # a lateral (world y) crossing plus a slower along-corridor (world x)
        # drift, each with its own phase and a different x/y period ratio so no
        # two move alike. Amplitudes stay modest to keep them in the scene.
        n = max(1, len(worker_pubs))
        for i, pub in enumerate(worker_pubs):
            ph = 2 * np.pi * i / n
            vy = args.worker_speed * np.sin(2 * np.pi * t / args.worker_period + ph)
            vx = 0.5 * args.worker_speed * np.sin(2 * np.pi * t / (args.worker_period * 1.7) + 1.3 * ph)
            tw = Twist(); tw.linear.x = float(vx); tw.linear.y = float(vy); pub.publish(tw)

    def spin_until(keys, timeout):
        t = time.time()
        while not set(keys) <= set(L) and time.time() - t < timeout:
            rclpy.spin_once(node, timeout_sec=0.05)
        return set(keys) <= set(L)

    spin_until(["rgb", "depth", "seg", "od"], 15.0)

    rgb_l, seg_l, theta_l, pose_l, depth_l = [], [], [], [], []
    dt = 1.0 / args.capture_hz
    t0 = time.time()
    gz_cmd_vel(args.speed, 0.0)
    try:
        while len(rgb_l) < args.max_frames:
            t = time.time() - t0
            # cosine yaw-RATE so the integrated heading (phi) oscillates
            # symmetrically around 0 (a sine rate integrates to a one-signed
            # heading -> phi never goes negative, biasing the geometry head).
            wz = args.yaw_amp * np.cos(2 * np.pi * t / args.yaw_period)
            vy = args.lateral_amp * np.sin(2 * np.pi * t / args.lateral_period)
            vz = args.vert_amp * np.sin(2 * np.pi * t / args.vert_period)
            gz_cmd_vel(args.speed, float(wz), vy=float(vy), vz=float(vz))
            drive_workers(t)

            L.clear()
            if not spin_until(["rgb", "depth", "seg", "od"], 3.0):
                print("timed out waiting for a frame set; stopping", flush=True)
                break
            rgb = br.imgmsg_to_cv2(L["rgb"], desired_encoding="rgb8").astype(np.uint8)
            seg = br.imgmsg_to_cv2(L["seg"], desired_encoding="rgb8")[:, :, 0].astype(np.int64)
            depth = np.asarray(br.imgmsg_to_cv2(L["depth"], desired_encoding="32FC1"), dtype=np.float32)
            depth = np.clip(np.nan_to_num(depth, nan=DEPTH_FAR, posinf=DEPTH_FAR, neginf=0.0), 0.0, DEPTH_FAR)
            w = depth.shape[1]
            dL = float(depth[:, :w // 3].mean())
            dC = float(depth[:, w // 3:2 * w // 3].mean())
            dR = float(depth[:, 2 * w // 3:].mean())
            pp = L["od"].pose.pose.position
            oo = L["od"].pose.pose.orientation
            phi = float(np.arctan2(2 * (oo.w * oo.z + oo.x * oo.y), 1 - 2 * (oo.y * oo.y + oo.z * oo.z)))
            rgb_l.append(rgb)
            seg_l.append(seg)
            theta_l.append(np.array([phi, dL, dR, dC], dtype=np.float32))
            pose_l.append(np.array([pp.x, pp.y, pp.z, oo.x, oo.y, oo.z, oo.w], dtype=np.float32))
            depth_l.append(depth.astype(np.float32))
            if len(rgb_l) % 10 == 0:
                print(f"frame {len(rgb_l):3d} x={pp.x:5.1f} phi={phi:+.2f} "
                      f"trunk%={100*(seg==2).mean():4.1f} actor%={100*(seg==4).mean():4.1f} "
                      f"d_L/C/R={dL:.1f}/{dC:.1f}/{dR:.1f}", flush=True)
            if pp.x > args.max_x:
                print(f"reached max-x ({args.max_x}); stopping", flush=True)
                break
            time.sleep(dt)
    finally:
        gz_cmd_vel(0.0, 0.0)  # stop the rig
        try:
            os.killpg(os.getpgid(bridge.pid), signal.SIGTERM)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()

    if not rgb_l:
        print("no frames captured -- check the render container + topics", flush=True)
        return
    rgb = np.stack(rgb_l); seg = np.stack(seg_l); theta = np.stack(theta_l); pose = np.stack(pose_l)
    depth = np.stack(depth_l)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez_compressed(args.out, rgb=rgb, seg_mask=seg, theta_corr_gt=theta, pose=pose, depth=depth)
    tf = (seg == 2).mean(axis=(1, 2))
    af = (seg == 4).mean(axis=(1, 2))
    print(f"\nsaved {args.out}: {len(rgb)} frames in {time.time()-t0:.1f}s", flush=True)
    print(f"  seg classes: {np.unique(seg).tolist()}", flush=True)
    print(f"  per-frame trunk%: min={100*tf.min():.1f} mean={100*tf.mean():.1f} max={100*tf.max():.1f}", flush=True)
    print(f"  per-frame ACTOR%: min={100*af.min():.2f} mean={100*af.mean():.2f} max={100*af.max():.2f}  "
          f"frames-with-actor={int((af>0).sum())}/{len(af)}", flush=True)
    print(f"  x range: [{pose[:,0].min():.1f}, {pose[:,0].max():.1f}]  "
          f"phi range: [{theta[:,0].min():+.2f}, {theta[:,0].max():+.2f}]  "
          f"depth finite: {bool(np.isfinite(theta[:,1:]).all())}", flush=True)


if __name__ == "__main__":
    main()

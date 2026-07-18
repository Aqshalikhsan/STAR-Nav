"""Capture-world generator for the PX4-FREE perception-data path (MANUAL).

This is deliberately SEPARATE from the autonomy path (star_nav training +
GazeboROSEnv + PX4 SITL). It takes an existing *labeled* corridor world SDF
(produced by ros_gazebo_bridge.world_gen -- the one with the gz-sim Label
plugins on trunks/ground) and injects a standalone, physics-movable camera
rig carrying rgb + depth + segmentation sensors. No PX4, no MAVROS, no
arming: a plain `gz sim -s` renders the rig, and the rig is driven by a
*physics* velocity command (gz VelocityControl), NOT the set_pose service.

Why velocity, not set_pose: in this gz-sim7 (Garden) build the one-shot
set_pose service moves a model's Pose component but does NOT update a
rendering sensor's camera viewpoint (verified: rig pose changes while the
render stays frozen at spawn). Only physics-driven motion updates the render,
so the rig glides down the corridor under a commanded velocity instead.

The rig also carries an OdometryPublisher so the collector can read the rig's
true pose per frame (ground-truth for theta_corr_gt's phi).

Usage (host, pure stdlib -- no ROS/gz needed just to generate the SDF):

    python3 ros_gazebo_bridge/perception_capture/capture_world_gen.py \
        --base-world ros_gazebo_bridge/worlds/scenario_a.sdf \
        --out ros_gazebo_bridge/perception_capture/capture_scenario_a.sdf

Then run it in a GPU container from the px4-gazebo image (has gz + oil_palm
model + rendering); see collect_dataset.py for the full manual run recipe.
"""
from __future__ import annotations

import argparse
import os

# Camera intrinsics -- match ros_gazebo_bridge/ros_gazebo_bridge/fpv5_gen.py so
# the captured frames look like what the fpv5 vehicle camera would see.
CAM_W, CAM_H = 640, 480
CAM_HFOV = 1.5708          # 90 deg (fpv5_gen CAMERA_HFOV_RAD)
DEPTH_FAR = 20.0           # fpv5_gen DEPTH_MAX_RANGE_M

# gz topic names the collector subscribes to / publishes on.
RGB_TOPIC = "rig_camera"
DEPTH_TOPIC = "rig_depth"
SEG_TOPIC = "rig_seg"                       # -> rig_seg/labels_map + rig_seg/colored_map
CMD_VEL_TOPIC = "/model/capture_rig/cmd_vel"
ODOM_TOPIC = "/model/capture_rig/odometry"


def build_rig(x0: float, y0: float, z0: float) -> str:
    """A gravity-off dynamic model with the 3 camera sensors, a VelocityControl
    plugin (so it can be glided under physics -> the render actually follows),
    and an OdometryPublisher (so the collector can read its true pose).
    """
    return f"""
    <model name="capture_rig">
      <pose>{x0} {y0} {z0} 0 0 0</pose>
      <!-- VelocityControl: drives the model at a commanded twist published on
           {CMD_VEL_TOPIC} (gz.msgs.Twist). Physics-driven motion, so the
           camera render follows (unlike the set_pose service). -->
      <plugin filename="gz-sim-velocity-control-system" name="gz::sim::systems::VelocityControl">
        <topic>{CMD_VEL_TOPIC}</topic>
      </plugin>
      <!-- OdometryPublisher: rig ground-truth pose (dimensions=3 for real z). -->
      <plugin filename="gz-sim-odometry-publisher-system" name="gz::sim::systems::OdometryPublisher">
        <dimensions>3</dimensions>
        <odom_topic>{ODOM_TOPIC}</odom_topic>
      </plugin>
      <link name="link">
        <gravity>false</gravity>
        <inertial>
          <mass>0.05</mass>
          <inertia><ixx>1e-3</ixx><ixy>0</ixy><ixz>0</ixz><iyy>1e-3</iyy><iyz>0</iyz><izz>1e-3</izz></inertia>
        </inertial>
        <sensor name="rig_camera" type="camera">
          <camera>
            <horizontal_fov>{CAM_HFOV}</horizontal_fov>
            <image><width>{CAM_W}</width><height>{CAM_H}</height></image>
            <clip><near>0.05</near><far>100</far></clip>
          </camera>
          <always_on>1</always_on>
          <update_rate>30</update_rate>
          <topic>{RGB_TOPIC}</topic>
        </sensor>
        <sensor name="rig_depth" type="depth_camera">
          <camera>
            <horizontal_fov>{CAM_HFOV}</horizontal_fov>
            <image><width>{CAM_W}</width><height>{CAM_H}</height><format>R_FLOAT32</format></image>
            <clip><near>0.1</near><far>{DEPTH_FAR}</far></clip>
          </camera>
          <always_on>1</always_on>
          <update_rate>30</update_rate>
          <topic>{DEPTH_TOPIC}</topic>
        </sensor>
        <sensor name="rig_seg" type="segmentation">
          <topic>{SEG_TOPIC}</topic>
          <camera>
            <segmentation_type>semantic</segmentation_type>
            <horizontal_fov>{CAM_HFOV}</horizontal_fov>
            <image><width>{CAM_W}</width><height>{CAM_H}</height></image>
            <clip><near>0.05</near><far>100</far></clip>
          </camera>
          <always_on>1</always_on>
          <update_rate>30</update_rate>
        </sensor>
      </link>
    </model>
"""


def build_worker(name: str, x: float, y: float, yaw: float) -> str:
    """A person_worker (human mesh, labelled CLASS_ACTOR in model.sdf) driven
    across the corridor at capture time via VelocityControl. Its cmd_vel and
    odometry topics are namespaced by model name.
    """
    return f"""
    <include>
      <uri>model://person_worker</uri>
      <name>{name}</name>
      <pose>{x} {y} 0 0 0 {yaw}</pose>
      <plugin filename="gz-sim-velocity-control-system" name="gz::sim::systems::VelocityControl">
        <topic>/model/{name}/cmd_vel</topic>
      </plugin>
      <plugin filename="gz-sim-odometry-publisher-system" name="gz::sim::systems::OdometryPublisher">
        <dimensions>3</dimensions>
        <odom_topic>/model/{name}/odometry</odom_topic>
      </plugin>
    </include>"""


def inject(base_world_sdf: str, x0: float, y0: float, z0: float,
           worker_xs=None, worker_y: float = 24.0) -> str:
    if "</world>" not in base_world_sdf:
        raise ValueError("base world SDF has no </world> tag -- is it a valid gz world?")
    blocks = [build_rig(x0, y0, z0)]
    # Scatter the workers across the lane width (deterministic spread) and give
    # each a different starting heading so they disperse rather than march in a
    # line -- "berjalan menyebar". The collector then drives each on its own 2D
    # wander (see collect_dataset.py drive_workers).
    y_off = [-3.5, 2.5, -1.5, 3.5, -2.5, 1.5, 0.0, -3.0, 3.0]
    yaws = [1.5708, -1.5708, 0.0, 3.1416, 0.7854, -0.7854, 2.356, -2.356, 1.0]
    for i, wx in enumerate(worker_xs or []):
        wy = worker_y + y_off[i % len(y_off)]
        blocks.append(build_worker(f"worker{i}", float(wx), wy, yaws[i % len(yaws)]))
    return base_world_sdf.replace("</world>", "".join(blocks) + "\n  </world>")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-world", required=True,
                   help="Path to a labeled corridor world SDF from ros_gazebo_bridge.world_gen "
                        "(the one with gz-sim Label plugins on trunks/ground).")
    p.add_argument("--out", required=True, help="Output capture-world SDF path.")
    p.add_argument("--rig-x0", type=float, default=2.0, help="Rig start x (down-corridor).")
    p.add_argument("--rig-y0", type=float, default=24.0, help="Rig start y (corridor centre).")
    p.add_argument("--rig-z0", type=float, default=2.2, help="Rig start z (flight-like height).")
    p.add_argument("--worker-xs", default="",
                   help="Comma-separated x positions for moving CLASS_ACTOR workers "
                        "(e.g. '12,22,32'); empty = no workers (static scene, SACR-style).")
    p.add_argument("--worker-y", type=float, default=24.0, help="Corridor centre the workers cross.")
    args = p.parse_args(argv)

    worker_xs = [float(v) for v in args.worker_xs.split(",") if v.strip()] if args.worker_xs else []

    with open(args.base_world, "r", encoding="utf-8") as f:
        base = f.read()
    out = inject(base, args.rig_x0, args.rig_y0, args.rig_z0, worker_xs, args.worker_y)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"wrote {args.out}")
    print(f"  rig start pose: ({args.rig_x0}, {args.rig_y0}, {args.rig_z0}), facing +x")
    print(f"  topics: rgb={RGB_TOPIC}  depth={DEPTH_TOPIC}  seg={SEG_TOPIC}/labels_map")
    print(f"  drive:  publish gz.msgs.Twist on {CMD_VEL_TOPIC}; read pose on {ODOM_TOPIC}")
    if worker_xs:
        print(f"  workers (CLASS_ACTOR, crossing): worker0..{len(worker_xs)-1} at x={worker_xs} "
              f"-> cmd_vel /model/worker<i>/cmd_vel")


if __name__ == "__main__":
    main()

"""Procedural Gazebo model generator for a "Pavo Femto" brushless whoop --
the actual real-world hardware STAR-Nav was flown on (see
``elsarticle-paper/sections/04_experimental_verification.tex``'s "Design
of Real-World Implementation" section), used here in place of PX4's
stock ``x500``/``x500_depth`` models so the simulated vehicle matches the
real one instead of a generic 500mm-class quad.

No mesh tooling (Blender etc.) is available in this environment, so the
frame is built entirely from SDF primitives (boxes/cylinders), including
a segmented-box approximation of the duct rings (Gazebo/SDF has no torus
primitive). This is a geometric/visual approximation, not a scanned or
CAD-exported model.

Real hardware specs used (user-provided, 2026-07-05 -- also recorded in
this session's project memory):
    Wheelbase (motor-to-motor diagonal): 75 mm
    Motor: LAVA 1102 brushless
    Frame: Pavo Femto Brushless Whoop Frame (ducted, 4 fused duct rings)
    Propeller: Gemfan 1611, 3-blade, 40 mm diameter
    Camera/VTX: DJI O4 (Lite) air unit, front-mounted
    Weight: 54.8 g dry (no battery) -- battery mass is intentionally
        *excluded* per explicit user instruction ("baterai diabaikan
        dulu"); revisit if/when battery modeling is wanted.

Reference photos (user-provided) additionally show:
    - The 4 duct rings are fused directly to each other (no separate
      skinny arms like a standard quad) -- the ring wall *is* the frame.
    - An elevated top plate (FC/ESC stack) sits above the duct-ring
      plane on standoffs, with a forward camera pod and a rear VTX
      antenna.
    - A small ground-contact skid hangs below the center.

Everything below marked "engineering estimate" is a reasonable a priori
guess (motor thrust constants, duct/frame dimensions not given exactly,
camera tilt), not measured bench data -- expect to retune after flight
testing in sim.
"""
from __future__ import annotations

import argparse
import math
import os

# ---------------------------------------------------------------------------
# Real/derived dimensions
# ---------------------------------------------------------------------------
WHEELBASE_M = 0.075  # motor-to-motor diagonal, per spec
ARM_OFFSET_M = WHEELBASE_M / (2 * math.sqrt(2))  # per-axis motor offset for an X layout, ~0.0265 m

PROP_DIAMETER_M = 0.040  # Gemfan 1611, "40mm" per spec
PROP_RADIUS_M = PROP_DIAMETER_M / 2.0
PROP_BLADE_COUNT = 3
PROP_BLADE_WIDTH_M = 0.006
PROP_BLADE_THICKNESS_M = 0.0015

# Duct ring: engineering estimate. Inner radius clears the prop tip with a
# small margin; outer radius is picked so adjacent ducts touch/slightly
# overlap (matches the reference photos, where the ring walls are fused
# into a single frame with no visible gap between neighboring ducts).
DUCT_INNER_R_M = PROP_RADIUS_M + 0.004
DUCT_OUTER_R_M = ARM_OFFSET_M * 1.06
DUCT_HEIGHT_M = 0.012
DUCT_SEGMENTS = 16  # box segments approximating the ring; higher = smoother

MOTOR_RADIUS_M = 0.0055  # LAVA 1102 -> 11mm stator diameter
MOTOR_HEIGHT_M = 0.016

# Top plate (FC/ESC stack), elevated above the duct-ring plane on standoffs.
TOP_PLATE_SIZE_M = (0.040, 0.026, 0.003)
STANDOFF_HEIGHT_M = 0.014
STANDOFF_RADIUS_M = 0.0015

# Front camera pod (DJI O4 Lite air unit) -- engineering estimate for
# position/size; tilt defaults to level (0 deg) since no exact tilt angle
# was given. Override via --camera-tilt-deg if the real mount angle is
# known.
CAMERA_POD_SIZE_M = (0.014, 0.014, 0.016)

# Rear VTX antenna (visual only, no RF simulated).
ANTENNA_RADIUS_M = 0.0015
ANTENNA_LENGTH_M = 0.03

# Ground-contact skid below the frame center.
SKID_SIZE_M = (0.006, 0.02, 0.012)

# ---------------------------------------------------------------------------
# Mass budget: 54.8 g dry, battery excluded per user instruction.
# ---------------------------------------------------------------------------
TOTAL_MASS_KG = 0.0548
ROTOR_MASS_KG = 0.005  # per motor+prop, engineering estimate for a 1102-class motor
BASE_MASS_KG = TOTAL_MASS_KG - 4 * ROTOR_MASS_KG  # frame + FC/ESC + camera + VTX + antenna + skid

# Motor-model constants for gz-sim-multicopter-motor-model-system.
# Engineering estimate for a LAVA1102 (~20-25kV kV class) + Gemfan 1611
# combo on 2S LiHV: target ~0.45N (~45 gf) max static thrust per motor
# (typical for this class, giving an overall thrust-to-weight ratio
# around 2-2.5 at the 84.4g flying weight including battery), reached at
# maxRotVelocity. Tiny motors spin up/down far faster than the 500-class
# x500's 5010 motors, hence much shorter time constants.
MOTOR_MAX_ROT_VELOCITY = 12000.0  # rad/s, engineering estimate
MOTOR_MAX_THRUST_N = 0.45
MOTOR_CONSTANT = MOTOR_MAX_THRUST_N / (MOTOR_MAX_ROT_VELOCITY ** 2)
MOTOR_MOMENT_CONSTANT = 0.02  # yaw-torque/thrust ratio, typical multirotor-prop ballpark
MOTOR_TIME_CONSTANT_UP = 0.005
MOTOR_TIME_CONSTANT_DOWN = 0.01
ROTOR_DRAG_COEFFICIENT = 1.2e-6  # scaled down from x500's 8.06e-5 by (prop-diameter ratio)^2
ROLLING_MOMENT_COEFFICIENT = 1.5e-8

# Camera specs: 640x480 @ 30 FPS per the paper's UAV/Simulation
# Configuration table. Horizontal FOV is not specified there -- reusing
# the 90 deg used by MockCorridorEnv (star_nav/envs/mock_env.py) for
# cross-backend consistency.
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_HFOV_RAD = math.radians(90.0)
DEPTH_HFOV_RAD = math.radians(90.0)
DEPTH_MAX_RANGE_M = 20.0


def _pose(x: float, y: float, z: float, roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0) -> str:
    return f"{x:.5f} {y:.5f} {z:.5f} {roll:.5f} {pitch:.5f} {yaw:.5f}"


def _duct_ring_visual(name_prefix: str, cx: float, cy: float, z_bottom: float) -> str:
    """A ring of ``DUCT_SEGMENTS`` boxes approximating a torus -- SDF has no
    native torus/tube primitive, so this is the standard low-effort
    workaround when no mesh tool is available."""
    mid_r = (DUCT_INNER_R_M + DUCT_OUTER_R_M) / 2.0
    radial_thickness = DUCT_OUTER_R_M - DUCT_INNER_R_M
    z_mid = z_bottom + DUCT_HEIGHT_M / 2.0
    arc_len = 2 * math.pi * mid_r / DUCT_SEGMENTS
    tangential_len = arc_len * 1.3  # slight overlap so segments read as a continuous ring

    parts = []
    for i in range(DUCT_SEGMENTS):
        angle = 2 * math.pi * i / DUCT_SEGMENTS
        px = cx + mid_r * math.cos(angle)
        py = cy + mid_r * math.sin(angle)
        parts.append(f"""
      <visual name="{name_prefix}_seg{i}">
        <pose>{_pose(px, py, z_mid, 0, 0, angle)}</pose>
        <geometry>
          <box><size>{radial_thickness:.5f} {tangential_len:.5f} {DUCT_HEIGHT_M:.5f}</size></box>
        </geometry>
        <material>
          <ambient>0.02 0.02 0.02 1</ambient>
          <diffuse>0.02 0.02 0.02 1</diffuse>
        </material>
      </visual>""")
    return "".join(parts)


def _prop_blades(link_name: str, spin_dir: int) -> str:
    """3 thin box "blades" -- visual only; the motor plugin models thrust
    as a rigid body regardless of blade count/shape."""
    parts = []
    for i in range(PROP_BLADE_COUNT):
        angle = 2 * math.pi * i / PROP_BLADE_COUNT
        cx = (PROP_RADIUS_M / 2.0) * math.cos(angle)
        cy = (PROP_RADIUS_M / 2.0) * math.sin(angle)
        parts.append(f"""
      <visual name="{link_name}_blade{i}">
        <pose>{_pose(cx, cy, 0, 0, 0, angle)}</pose>
        <geometry>
          <box><size>{PROP_RADIUS_M:.5f} {PROP_BLADE_WIDTH_M:.5f} {PROP_BLADE_THICKNESS_M:.5f}</size></box>
        </geometry>
        <material>
          <ambient>0.05 0.05 0.05 1</ambient>
          <diffuse>0.05 0.05 0.05 1</diffuse>
        </material>
      </visual>""")
    return "".join(parts)


def _rotor_link(index: int, cx: float, cy: float, z: float, turning_direction: str) -> str:
    spin_dir = 1 if turning_direction == "ccw" else -1
    # Small-rotor inertia estimate (thin-rod approximation about the prop
    # diameter) -- order-of-magnitude plausible for a sub-10g motor+prop,
    # not measured.
    ixx_iyy = ROTOR_MASS_KG * PROP_DIAMETER_M ** 2 / 12.0
    izz = ROTOR_MASS_KG * (PROP_RADIUS_M ** 2) / 2.0
    return f"""
    <link name="rotor_{index}">
      <gravity>true</gravity>
      <self_collide>false</self_collide>
      <velocity_decay/>
      <pose>{_pose(cx, cy, z)}</pose>
      <inertial>
        <mass>{ROTOR_MASS_KG:.6f}</mass>
        <inertia>
          <ixx>{ixx_iyy:.3e}</ixx>
          <iyy>{ixx_iyy:.3e}</iyy>
          <izz>{izz:.3e}</izz>
        </inertia>
      </inertial>
      <visual name="rotor_{index}_motor">
        <pose>0 0 {-MOTOR_HEIGHT_M / 2:.5f} 0 0 0</pose>
        <geometry>
          <cylinder><radius>{MOTOR_RADIUS_M:.5f}</radius><length>{MOTOR_HEIGHT_M:.5f}</length></cylinder>
        </geometry>
        <material>
          <ambient>0.3 0.2 0.05 1</ambient>
          <diffuse>0.5 0.35 0.05 1</diffuse>
          <specular>0.5 0.5 0.5 1</specular>
        </material>
      </visual>
      {_prop_blades(f"rotor_{index}", spin_dir)}
      <collision name="rotor_{index}_collision">
        <pose>0 0 0 0 0 0</pose>
        <geometry>
          <box><size>{PROP_DIAMETER_M:.5f} {PROP_BLADE_WIDTH_M * 2:.5f} {PROP_BLADE_THICKNESS_M * 2:.5f}</size></box>
        </geometry>
      </collision>
    </link>
    <joint name="rotor_{index}_joint" type="revolute">
      <parent>base_link</parent>
      <child>rotor_{index}</child>
      <axis>
        <xyz>0 0 1</xyz>
        <limit><lower>-1e+16</lower><upper>1e+16</upper></limit>
        <dynamics><spring_reference>0</spring_reference><spring_stiffness>0</spring_stiffness></dynamics>
      </axis>
    </joint>"""


def _motor_plugin(index: int, turning_direction: str) -> str:
    return f"""
    <plugin filename="gz-sim-multicopter-motor-model-system" name="gz::sim::systems::MulticopterMotorModel">
      <jointName>rotor_{index}_joint</jointName>
      <linkName>rotor_{index}</linkName>
      <turningDirection>{turning_direction}</turningDirection>
      <timeConstantUp>{MOTOR_TIME_CONSTANT_UP}</timeConstantUp>
      <timeConstantDown>{MOTOR_TIME_CONSTANT_DOWN}</timeConstantDown>
      <maxRotVelocity>{MOTOR_MAX_ROT_VELOCITY}</maxRotVelocity>
      <motorConstant>{MOTOR_CONSTANT:.6e}</motorConstant>
      <momentConstant>{MOTOR_MOMENT_CONSTANT}</momentConstant>
      <commandSubTopic>command/motor_speed</commandSubTopic>
      <motorNumber>{index}</motorNumber>
      <rotorDragCoefficient>{ROTOR_DRAG_COEFFICIENT:.3e}</rotorDragCoefficient>
      <rollingMomentCoefficient>{ROLLING_MOMENT_COEFFICIENT:.3e}</rollingMomentCoefficient>
      <rotorVelocitySlowdownSim>10</rotorVelocitySlowdownSim>
      <motorType>velocity</motorType>
    </plugin>"""


def build_model_sdf(camera_tilt_deg: float = 0.0) -> str:
    a = ARM_OFFSET_M
    duct_z_bottom = 0.0  # base_link origin sits at the duct-ring bottom plane
    motor_z = duct_z_bottom + DUCT_HEIGHT_M / 2.0
    top_plate_z = duct_z_bottom + DUCT_HEIGHT_M + STANDOFF_HEIGHT_M
    spawn_z = duct_z_bottom + 0.03  # ground clearance so the model doesn't spawn clipping the ground plane

    # Rotor layout mirrors PX4's x500 convention: (0,1) one diagonal pair
    # spinning one way, (2,3) the other -- standard X-quad yaw-cancellation.
    rotor_positions = [
        (0, a, -a, "ccw"),
        (1, -a, a, "ccw"),
        (2, a, a, "cw"),
        (3, -a, -a, "cw"),
    ]

    duct_visuals = "".join(
        _duct_ring_visual(f"duct{i}", cx, cy, duct_z_bottom) for i, cx, cy, _ in rotor_positions
    )
    rotor_links = "".join(_rotor_link(i, cx, cy, motor_z, td) for i, cx, cy, td in rotor_positions)
    motor_plugins = "".join(_motor_plugin(i, td) for i, _, _, td in rotor_positions)

    standoff_offsets = [(0.012, 0.008), (0.012, -0.008), (-0.012, 0.008), (-0.012, -0.008)]
    standoffs = "".join(f"""
      <visual name="standoff_{i}">
        <pose>{_pose(dx, dy, duct_z_bottom + DUCT_HEIGHT_M + STANDOFF_HEIGHT_M / 2)}</pose>
        <geometry>
          <cylinder><radius>{STANDOFF_RADIUS_M:.5f}</radius><length>{STANDOFF_HEIGHT_M:.5f}</length></cylinder>
        </geometry>
        <material><ambient>0.6 0.6 0.6 1</ambient><diffuse>0.7 0.7 0.7 1</diffuse></material>
      </visual>""" for i, (dx, dy) in enumerate(standoff_offsets))

    camera_x = a + DUCT_OUTER_R_M + CAMERA_POD_SIZE_M[0] / 2.0 + 0.006
    camera_z = top_plate_z + TOP_PLATE_SIZE_M[2] / 2.0 + CAMERA_POD_SIZE_M[2] / 2.0
    camera_pitch = -math.radians(camera_tilt_deg)  # positive tilt_deg = nose-down in this convention

    antenna_x = -(a + DUCT_OUTER_R_M * 0.6)
    antenna_z = top_plate_z + ANTENNA_LENGTH_M / 2.0
    antenna_pitch = math.radians(20.0)  # angled up and back, matching the reference photos

    # Base-link inertia: flat-disc approximation over the overall footprint
    # (duct outer radius + arm offset) and stack height -- engineering
    # estimate, not measured.
    footprint_r = a + DUCT_OUTER_R_M
    stack_h = DUCT_HEIGHT_M + STANDOFF_HEIGHT_M + TOP_PLATE_SIZE_M[2]
    izz = BASE_MASS_KG * footprint_r ** 2 / 2.0
    ixx_iyy = BASE_MASS_KG * (3 * footprint_r ** 2 + stack_h ** 2) / 12.0

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sdf version='1.9'>
  <model name='pavo_femto'>
    <pose>0 0 {spawn_z:.5f} 0 0 0</pose>
    <self_collide>false</self_collide>
    <static>false</static>
    <link name="base_link">
      <inertial>
        <mass>{BASE_MASS_KG:.6f}</mass>
        <inertia>
          <ixx>{ixx_iyy:.3e}</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>{ixx_iyy:.3e}</iyy>
          <iyz>0</iyz>
          <izz>{izz:.3e}</izz>
        </inertia>
      </inertial>
      <gravity>true</gravity>
      <velocity_decay/>
      {duct_visuals}
      {standoffs}
      <visual name="top_plate">
        <pose>{_pose(0, 0, top_plate_z)}</pose>
        <geometry>
          <box><size>{TOP_PLATE_SIZE_M[0]:.5f} {TOP_PLATE_SIZE_M[1]:.5f} {TOP_PLATE_SIZE_M[2]:.5f}</size></box>
        </geometry>
        <material><ambient>0.02 0.02 0.02 1</ambient><diffuse>0.03 0.03 0.03 1</diffuse></material>
      </visual>
      <visual name="camera_pod">
        <pose>{_pose(camera_x, 0, camera_z, 0, camera_pitch, 0)}</pose>
        <geometry>
          <box><size>{CAMERA_POD_SIZE_M[0]:.5f} {CAMERA_POD_SIZE_M[1]:.5f} {CAMERA_POD_SIZE_M[2]:.5f}</size></box>
        </geometry>
        <material><ambient>0.02 0.02 0.02 1</ambient><diffuse>0.02 0.02 0.02 1</diffuse></material>
      </visual>
      <visual name="vtx_antenna">
        <pose>{_pose(antenna_x, 0, antenna_z, 0, antenna_pitch, 0)}</pose>
        <geometry>
          <cylinder><radius>{ANTENNA_RADIUS_M:.5f}</radius><length>{ANTENNA_LENGTH_M:.5f}</length></cylinder>
        </geometry>
        <material><ambient>0.05 0.05 0.05 1</ambient><diffuse>0.1 0.1 0.1 1</diffuse></material>
      </visual>
      <visual name="skid">
        <pose>{_pose(0, 0, duct_z_bottom - SKID_SIZE_M[2] / 2)}</pose>
        <geometry>
          <box><size>{SKID_SIZE_M[0]:.5f} {SKID_SIZE_M[1]:.5f} {SKID_SIZE_M[2]:.5f}</size></box>
        </geometry>
        <material><ambient>0.02 0.02 0.02 1</ambient><diffuse>0.02 0.02 0.02 1</diffuse></material>
      </visual>
      <collision name="base_link_collision">
        <pose>{_pose(0, 0, top_plate_z / 2)}</pose>
        <geometry>
          <box><size>{2 * footprint_r:.5f} {2 * footprint_r:.5f} {stack_h:.5f}</size></box>
        </geometry>
        <surface>
          <contact><ode><min_depth>0.001</min_depth><max_vel>0</max_vel></ode></contact>
          <friction><ode/></friction>
        </surface>
      </collision>
      <sensor name="air_pressure_sensor" type="air_pressure">
        <always_on>1</always_on>
        <update_rate>50</update_rate>
        <air_pressure><pressure><noise type="gaussian"><mean>0</mean><stddev>0.01</stddev></noise></pressure></air_pressure>
      </sensor>
      <sensor name="imu_sensor" type="imu">
        <always_on>1</always_on>
        <update_rate>250</update_rate>
        <imu>
          <angular_velocity>
            <x><noise type="gaussian"><mean>0</mean><stddev>0.00018665</stddev><dynamic_bias_stddev>3.8785e-05</dynamic_bias_stddev><dynamic_bias_correlation_time>1000</dynamic_bias_correlation_time></noise></x>
            <y><noise type="gaussian"><mean>0</mean><stddev>0.00018665</stddev><dynamic_bias_stddev>3.8785e-05</dynamic_bias_stddev><dynamic_bias_correlation_time>1000</dynamic_bias_correlation_time></noise></y>
            <z><noise type="gaussian"><mean>0</mean><stddev>0.00018665</stddev><dynamic_bias_stddev>3.8785e-05</dynamic_bias_stddev><dynamic_bias_correlation_time>1000</dynamic_bias_correlation_time></noise></z>
          </angular_velocity>
          <linear_acceleration>
            <x><noise type="gaussian"><mean>0</mean><stddev>0.00186</stddev><dynamic_bias_stddev>0.006</dynamic_bias_stddev><dynamic_bias_correlation_time>300</dynamic_bias_correlation_time></noise></x>
            <y><noise type="gaussian"><mean>0</mean><stddev>0.00186</stddev><dynamic_bias_stddev>0.006</dynamic_bias_stddev><dynamic_bias_correlation_time>300</dynamic_bias_correlation_time></noise></y>
            <z><noise type="gaussian"><mean>0</mean><stddev>0.00186</stddev><dynamic_bias_stddev>0.006</dynamic_bias_stddev><dynamic_bias_correlation_time>300</dynamic_bias_correlation_time></noise></z>
          </linear_acceleration>
        </imu>
      </sensor>
      <sensor name="navsat_sensor" type="navsat">
        <always_on>1</always_on>
        <update_rate>30</update_rate>
      </sensor>
      <sensor name="camera" type="camera">
        <pose>{_pose(camera_x, 0, camera_z, 0, camera_pitch, 0)}</pose>
        <camera>
          <horizontal_fov>{CAMERA_HFOV_RAD:.5f}</horizontal_fov>
          <image><width>{CAMERA_WIDTH}</width><height>{CAMERA_HEIGHT}</height></image>
          <clip><near>0.05</near><far>100</far></clip>
        </camera>
        <always_on>1</always_on>
        <update_rate>{CAMERA_FPS}</update_rate>
        <visualize>true</visualize>
        <topic>camera</topic>
      </sensor>
      <sensor name="depth_camera" type="depth_camera">
        <pose>{_pose(camera_x, 0, camera_z, 0, camera_pitch, 0)}</pose>
        <camera>
          <horizontal_fov>{DEPTH_HFOV_RAD:.5f}</horizontal_fov>
          <image><width>{CAMERA_WIDTH}</width><height>{CAMERA_HEIGHT}</height><format>R_FLOAT32</format></image>
          <clip><near>0.1</near><far>{DEPTH_MAX_RANGE_M:.1f}</far></clip>
        </camera>
        <always_on>1</always_on>
        <update_rate>{CAMERA_FPS}</update_rate>
        <visualize>false</visualize>
        <topic>depth_camera</topic>
      </sensor>
      <!-- Real semantic segmentation, for SACR's segmentation-mask ground
           truth (env.ros.seg_topic -> ros_bridge_node.py's _on_seg). Class
           ids are assigned per-model/per-visual via the Label plugin in
           world_gen.py's to_sdf(), matching star_nav/envs/mock_env.py's
           CLASS_SKY=0/CLASS_GROUND=1/CLASS_TRUNK=2/CLASS_CANOPY=3/
           CLASS_ACTOR=4 scheme (unlabeled background, e.g. sky, defaults
           to 0). Publishes <topic>/labels_map (mono8 class-id image, what
           ros_bridge_node.py expects) and <topic>/colored_map. -->
      <sensor name="segmentation_camera" type="segmentation">
        <pose>{_pose(camera_x, 0, camera_z, 0, camera_pitch, 0)}</pose>
        <camera>
          <segmentation_type>semantic</segmentation_type>
          <horizontal_fov>{CAMERA_HFOV_RAD:.5f}</horizontal_fov>
          <image><width>{CAMERA_WIDTH}</width><height>{CAMERA_HEIGHT}</height></image>
          <clip><near>0.05</near><far>100</far></clip>
        </camera>
        <always_on>1</always_on>
        <update_rate>{CAMERA_FPS}</update_rate>
        <visualize>false</visualize>
        <topic>semantic_segmentation</topic>
      </sensor>
    </link>
    {rotor_links}
    {motor_plugins}
    <!-- Ground-truth pose for GazeboROSEnv/ros_gz_bridge's gt_odom_topic
         (env.ros.gt_odom_topic = /model/pavo_femto_0/odometry by
         default). Without this, nothing in Gazebo ever publishes that
         gz topic at all; ros_gz_bridge still advertises the ROS 2 side,
         but silently forwards zero messages (found the hard way:
         GazeboROSEnv.reset() hangs in wait_until_ready() with no error,
         waiting on a topic with a registered bridge but no actual
         source). See fpv5_gen.py for the same fix. -->
    <plugin filename="gz-sim-odometry-publisher-system" name="gz::sim::systems::OdometryPublisher">
      <odom_frame>world</odom_frame>
      <robot_base_frame>base_link</robot_base_frame>
      <odom_publish_frequency>30</odom_publish_frequency>
      <!-- dimensions=3 REQUIRED (plugin defaults to 2 = planar, z always 0);
           PX4's GZBridge feeds this as vehicle_visual_odometry, and a 2D feed
           reports altitude 0 forever, breaking vision-height fusion. -->
      <dimensions>3</dimensions>
    </plugin>
  </model>
</sdf>
"""


def build_model_config() -> str:
    return """<?xml version="1.0"?>
<model>
  <name>pavo_femto</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <author>
    <name>STAR-Nav project</name>
  </author>
  <description>Procedurally generated approximation of a Pavo Femto brushless whoop (75mm wheelbase, LAVA 1102 motors, Gemfan 1611 3-blade props, DJI O4 Lite front camera) -- the real-world hardware STAR-Nav was flown on. Built entirely from SDF primitives (no mesh assets available in this environment); duct rings are a segmented-box approximation of a torus. Physical constants not given by the hardware spec (motor thrust curve, inertia, exact camera tilt) are engineering estimates, not measured data -- see pavo_femto_gen.py for the full list of assumptions.</description>
</model>
"""


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="ros_gazebo_bridge/px4_models/pavo_femto",
                         help="Output model directory (will contain model.sdf and model.config)")
    parser.add_argument("--camera-tilt-deg", type=float, default=0.0,
                         help="Camera pitch tilt in degrees, positive = nose-down. "
                              "Defaults to level (0) since the real mount angle wasn't specified.")
    args = parser.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    sdf_path = os.path.join(args.out, "model.sdf")
    config_path = os.path.join(args.out, "model.config")
    with open(sdf_path, "w") as f:
        f.write(build_model_sdf(camera_tilt_deg=args.camera_tilt_deg))
    with open(config_path, "w") as f:
        f.write(build_model_config())
    print(f"Wrote {sdf_path} and {config_path}")


if __name__ == "__main__":
    main()

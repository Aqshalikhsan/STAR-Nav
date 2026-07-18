"""Procedural Gazebo world generator for the oil-palm corridor task.

Pure NumPy/stdlib -- no ROS 2, Gazebo Python bindings, or MAVLink stack
required, so this module (and the tree-layout math in particular) is
importable and testable without any of the ROS install described in
``ros_gazebo_bridge/README.md``.

It mirrors ``star_nav.envs.mock_env.MockCorridorEnv._generate_world``'s
procedural row/tree placement -- the same Table 6 spacing mean/std per
scenario (A/B/C -> low/medium/high aliasing via decreasing spacing
variance) -- so the *ground-truth geometry* that ``GazeboROSEnv`` reports
as ``PrivilegedInfo`` describes the same task structure as
``MockCorridorEnv``/``AirSimCorridorEnv``, even though the RGB/depth
observation now comes from a real Gazebo camera instead of a ray cast.

Unlike ``MockCorridorEnv``, which can regenerate a fresh procedural world
every ``reset()`` for free, a live Gazebo session cannot cheaply spawn or
delete hundreds of trunk models every episode. The intended workflow is
therefore "generate once, reset many times":

    python -m ros_gazebo_bridge.world_gen --config configs/default.yaml \\
        --scenario A --out ros_gazebo_bridge/worlds/scenario_a

This writes ``scenario_a.sdf`` (loaded by Gazebo, see
``launch/corridor_sim.launch.py``) and ``scenario_a.world.json`` (the
ground-truth trunk layout, loaded at runtime by ``ros_bridge_node.py`` so
collision/goal-distance/lateral-deviation stay in sync with whatever
world is actually loaded -- see ``load_world_layout``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np

TRUNK_RADIUS = 0.25
TRUNK_HEIGHT = 6.0

# Segmentation class ids for the Label plugin below -- duplicated from (must
# stay in sync with) star_nav/envs/mock_env.py's
# `CLASS_SKY, CLASS_GROUND, CLASS_TRUNK, CLASS_CANOPY, CLASS_ACTOR = range(5)`.
# Not imported directly: this module is deliberately pure NumPy/stdlib with
# no star_nav/torch dependency (see module docstring), same reasoning as
# TRUNK_RADIUS/TRUNK_HEIGHT already being duplicated against oil_palm's
# model.sdf. The oil_palm mesh (kelapa_sawit_tex.glb) is one glTF mesh
# object with 3 material primitives (trunk/frond/fruit) rather than
# separate named sub-meshes, and Gazebo's Label plugin labels a whole
# <visual>/<include>, not per-material primitives within one mesh -- so
# the whole tree is labeled CLASS_TRUNK; CLASS_CANOPY has no real pixels
# from this backend (see ros_gazebo_bridge/README.md Limitations).
SEG_CLASS_GROUND = 1
SEG_CLASS_TRUNK = 2


@dataclass
class CorridorWorld:
    tree_xy: np.ndarray          # (N, 2) trunk centers in metres
    trunk_radius: float
    corridor_center_y: float
    goal_xy: np.ndarray
    area_size_m: float
    scenario: str = ""
    weather_note: str = field(default=(
        "Gazebo backend does not simulate weather; this tag is informational only."
    ))


def build_corridor_world(scenario_cfg, area_size_m: float, scenario: str = "",
                          seed: "int | None" = None) -> CorridorWorld:
    """Same row/tree placement algorithm as MockCorridorEnv._generate_world,
    parameterized by Table 6's ``row_spacing_mean/std`` and
    ``tree_spacing_mean/std`` for the requested scenario.
    """
    rng = np.random.default_rng(seed)
    n_rows = max(2, int(area_size_m / scenario_cfg.row_spacing_mean))
    trees = []
    for r in range(n_rows):
        row_y = r * scenario_cfg.row_spacing_mean + rng.normal(0, scenario_cfg.row_spacing_std)
        n_trees = max(2, int(area_size_m / scenario_cfg.tree_spacing_mean))
        x = 0.0
        for _ in range(n_trees):
            x += max(1.0, rng.normal(scenario_cfg.tree_spacing_mean, scenario_cfg.tree_spacing_std))
            trees.append((x, row_y))
    tree_xy = np.array(trees, dtype=np.float32)
    corridor_center_y = (n_rows * scenario_cfg.row_spacing_mean) / 2.0
    goal_xy = np.array([area_size_m - 2.0, corridor_center_y], dtype=np.float32)
    return CorridorWorld(tree_xy=tree_xy, trunk_radius=TRUNK_RADIUS,
                          corridor_center_y=corridor_center_y, goal_xy=goal_xy,
                          area_size_m=area_size_m, scenario=scenario)


def to_sdf(world: CorridorWorld, world_name: str = "oil_palm_corridor", use_mesh_trunks: bool = True) -> str:
    """Render a CorridorWorld to a Gazebo (gz sim) SDF world: a ground
    plane, a sun, and one trunk model per tree position.

    ``use_mesh_trunks=True`` (default) spawns each trunk as an
    ``<include>`` of the ``oil_palm`` model (``px4_models/oil_palm/`` --
    a textured mesh with trunk + fruit clusters, provided by the user;
    see that model's own ``model.sdf`` for the mesh/scale/collision
    details). This requires ``oil_palm`` to be on Gazebo's resource path
    (baked into the px4-gazebo Docker image, same mechanism as the
    pavo_femto/fpv5 vehicle models). ``use_mesh_trunks=False`` falls back
    to the original plain brown cylinder (no external asset dependency --
    useful for lightweight tests, e.g. ``test_world_gen.py``, or if
    ``oil_palm`` isn't available in your Gazebo resource path).

    The drone itself is NOT spawned here -- PX4 SITL's own gz integration
    spawns the vehicle model (e.g. ``gz_x500_depth``, which bundles a
    depth camera) separately; see ``launch/corridor_sim.launch.py``.
    """
    models = []
    for i, (x, y) in enumerate(world.tree_xy):
        if use_mesh_trunks:
            # oil_palm/model.sdf owns both the mesh visual and a matching
            # collision cylinder (radius/height hardcoded there to the
            # same TRUNK_RADIUS/TRUNK_HEIGHT constants this module uses --
            # keep the two in sync if either ever changes).
            models.append(f"""
    <include>
      <uri>model://oil_palm</uri>
      <name>trunk_{i}</name>
      <pose>{x:.3f} {y:.3f} 0 0 0 0</pose>
      <plugin filename="gz-sim-label-system" name="gz::sim::systems::Label">
        <label>{SEG_CLASS_TRUNK}</label>
      </plugin>
    </include>""")
        else:
            models.append(f"""
    <model name="trunk_{i}">
      <static>true</static>
      <pose>{x:.3f} {y:.3f} {TRUNK_HEIGHT / 2:.2f} 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <cylinder><radius>{world.trunk_radius:.3f}</radius><length>{TRUNK_HEIGHT:.2f}</length></cylinder>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder><radius>{world.trunk_radius:.3f}</radius><length>{TRUNK_HEIGHT:.2f}</length></cylinder>
          </geometry>
          <material>
            <ambient>0.35 0.25 0.15 1</ambient>
            <diffuse>0.35 0.25 0.15 1</diffuse>
          </material>
          <plugin filename="gz-sim-label-system" name="gz::sim::systems::Label">
            <label>{SEG_CLASS_TRUNK}</label>
          </plugin>
        </visual>
      </link>
    </model>""")

    return f"""<?xml version="1.0"?>
<sdf version="1.9">
  <world name="{world_name}">
    <physics name="1ms" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-contact-system" name="gz::sim::systems::Contact"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <!-- Without these, Gazebo never actually simulates or publishes any
         sensor data -- the vehicle model's <sensor> tags (IMU, depth
         camera, etc.) sit inert with no system to drive them. PX4 then
         fails every preflight check (accel/gyro/baro/compass "missing"),
         EKF2 never initializes, and MAVROS has nothing to relay to
         /mavros/imu/data or /mavros/local_position/pose. Matches the set
         PX4's own Tools/simulation/gz/worlds/default.sdf loads. -->
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
    <plugin filename="gz-sim-air-pressure-system" name="gz::sim::systems::AirPressure"/>
    <plugin filename="gz-sim-apply-link-wrench-system" name="gz::sim::systems::ApplyLinkWrench"/>
    <plugin filename="gz-sim-navsat-system" name="gz::sim::systems::NavSat"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>

    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.3 0.3 -0.9</direction>
    </light>

    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>{world.area_size_m * 2:.1f} {world.area_size_m * 2:.1f}</size></plane></geometry>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>{world.area_size_m * 2:.1f} {world.area_size_m * 2:.1f}</size></plane></geometry>
          <material><ambient>0.3 0.5 0.3 1</ambient><diffuse>0.3 0.5 0.3 1</diffuse></material>
          <plugin filename="gz-sim-label-system" name="gz::sim::systems::Label">
            <label>{SEG_CLASS_GROUND}</label>
          </plugin>
        </visual>
      </link>
    </model>
{''.join(models)}
  </world>
</sdf>
"""


def write_world_bundle(world: CorridorWorld, out_prefix: str, world_name: str = "oil_palm_corridor",
                        use_mesh_trunks: bool = True) -> None:
    """Writes ``{out_prefix}.sdf`` (for Gazebo) and ``{out_prefix}.world.json``
    (the ground-truth layout ``ros_bridge_node.py`` loads at runtime).
    """
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    with open(out_prefix + ".sdf", "w", encoding="utf-8") as f:
        f.write(to_sdf(world, world_name, use_mesh_trunks=use_mesh_trunks))

    layout = {
        "world_name": world_name,
        "scenario": world.scenario,
        "area_size_m": world.area_size_m,
        "trunk_radius": world.trunk_radius,
        "corridor_center_y": world.corridor_center_y,
        "goal_xy": world.goal_xy.tolist(),
        "tree_xy": world.tree_xy.tolist(),
    }
    with open(out_prefix + ".world.json", "w", encoding="utf-8") as f:
        json.dump(layout, f, indent=2)


def load_world_layout(json_path: str) -> CorridorWorld:
    """Loads the ground-truth trunk layout written by ``write_world_bundle``.
    Used by ``GazeboROSEnv``/``ROSGazeboBridge`` at runtime -- the layout is
    *not* recomputed from ``build_corridor_world`` with a fresh seed, since
    that would desync from whatever trunks are actually spawned in the
    already-running Gazebo world.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        layout = json.load(f)
    return CorridorWorld(
        tree_xy=np.array(layout["tree_xy"], dtype=np.float32),
        trunk_radius=float(layout["trunk_radius"]),
        corridor_center_y=float(layout["corridor_center_y"]),
        goal_xy=np.array(layout["goal_xy"], dtype=np.float32),
        area_size_m=float(layout["area_size_m"]),
        scenario=layout.get("scenario", ""),
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to a YAML config; defaults to configs/default.yaml")
    parser.add_argument("--scenario", default="A", help="Scenario key from env.scenarios (A/B/C)")
    parser.add_argument("--out", required=True, help="Output path prefix, e.g. ros_gazebo_bridge/worlds/scenario_a")
    parser.add_argument(
        "--world-name", default=None,
        help="SDF <world name>. Defaults to --out's basename -- ros_bridge_node/env.py set "
             "PX4_GZ_WORLD to that same basename, and PX4 derives its Gazebo service/topic "
             "names (e.g. /world/<name>/create) from PX4_GZ_WORLD, so a mismatch here means "
             "PX4 waits forever for a world that never responds.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--cylinder-trunks", action="store_true",
        help="Use plain brown cylinders for trunks instead of the textured oil_palm mesh model "
             "(no external asset/GZ_SIM_RESOURCE_PATH dependency -- useful if oil_palm isn't "
             "available in your Gazebo resource path).")
    args = parser.parse_args(argv)
    if args.world_name is None:
        args.world_name = os.path.basename(args.out)

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, repo_root)
    from star_nav.utils.config import load_config  # noqa: E402

    cfg = load_config(args.config)
    scenario_cfg = cfg.env.scenarios[args.scenario]
    seed = args.seed if args.seed is not None else cfg.seed

    world = build_corridor_world(scenario_cfg, cfg.env.area_size_m, scenario=args.scenario, seed=seed)
    write_world_bundle(world, args.out, world_name=args.world_name, use_mesh_trunks=not args.cylinder_trunks)
    print(f"Wrote {args.out}.sdf and {args.out}.world.json "
          f"({len(world.tree_xy)} trunks, scenario {args.scenario}, seed {seed}, "
          f"trunks={'oil_palm mesh' if not args.cylinder_trunks else 'cylinder'})")


if __name__ == "__main__":
    main()

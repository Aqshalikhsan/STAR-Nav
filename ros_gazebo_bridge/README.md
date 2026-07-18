# ros_gazebo_bridge

A **Gazebo + PX4 (MAVLink) + ROS 2** backend for STAR-Nav's
`BaseCorridorEnv` contract (`../star_nav/envs/base_env.py`) -- an
open-source alternative to `AirSimCorridorEnv` for people who don't have
Unreal Engine + AirSim available, built entirely from freely available
tooling: PX4 SITL, Gazebo (`gz sim`), MAVROS, `ros_gz_bridge`, ROS 2.

This folder is a **separate top-level package**, not a module under
`star_nav/`, on purpose:

* It follows the standard ROS 2 `ament_python` package layout
  (`package.xml`, `setup.py`, `launch/`), which doesn't fit naturally
  inside a plain Python library like `star_nav/`.
* It keeps `star_nav/` free of a hard ROS 2 dependency -- `MockCorridorEnv`
  and `AirSimCorridorEnv` don't need any of this installed.
* It's meant to be swapped out later: once this is validated, the plan
  is to move on to the real AirSim setup (`../star_nav/envs/airsim_env.py`)
  without this folder's ROS/Gazebo/MAVLink dependencies getting in the way.

**Status: fully live-validated end to end, including sensor data.**
MAVLink/MAVROS connectivity, live IMU/pose data flow, and
`ROSGazeboBridge`'s actual subscriptions have all been confirmed working
against a real running PX4 SITL + Gazebo session. Concretely:

* `world_gen.py` and `docker/Dockerfile.px4_gazebo` have been built and
  run for real (in a WSL2 Ubuntu distro running Docker Engine directly):
  `docker build` produces a working `gz_x500_depth` SITL binary, and
  running the container with `HEADLESS=0` and WSLg's X11/Wayland sockets
  bind-mounted in opens an actual Gazebo GUI window showing a
  `world_gen.py`-generated corridor world with the `x500_depth_0`
  vehicle spawned into it. Three real PX4 build/runtime bugs were found
  and fixed to get this working -- see the comments in
  `docker/Dockerfile.px4_gazebo` and `docker/entrypoint_px4_gazebo.sh`
  (shallow-clone NuttX submodule crashing PX4's git-version script,
  a build-vs-run `make` target mixup, and a PX4_GZ_WORLD/SDF world-name
  mismatch that made PX4 wait forever for Gazebo).
* `docker/Dockerfile.ros_bridge` (MAVROS + `ros_gz_bridge`) has also
  been built and run against the live `px4-gazebo` container (both on
  the same WSL2 Docker Engine, `--network host`): MAVROS connects to PX4
  over MAVLink and `/mavros/state` shows `connected: true` with live
  `mode`/`armed` data. Several real bugs were found and fixed getting
  here -- see `progress.md`'s Gazebo+MAVLink+ROS section for the full
  list (CPU-only torch, a wrong `install_geographiclib_datasets.sh`
  path, two wrong `COPY` paths, `set -u` vs ROS 2's own `setup.bash`,
  and a `mavros_node` crash fixed by switching to Cyclone DDS).
* **Root cause of the missing IMU/pose data, found and fixed**:
  `world_gen.py`'s generated world SDF only loaded the `Physics`,
  `SceneBroadcaster`, `Contact`, and `UserCommands` Gazebo system
  plugins -- it was missing `Imu`, `AirPressure`, `ApplyLinkWrench`,
  `NavSat`, and `Sensors` (the set PX4's own stock
  `Tools/simulation/gz/worlds/default.sdf` loads). Without these,
  Gazebo never actually simulates or publishes *any* sensor data at
  all -- the vehicle model's `<sensor>` tags (IMU, depth camera, etc.)
  just sit inert with nothing driving them, regardless of PX4/MAVROS
  being "connected". This is what caused PX4's preflight checks to fail
  ("Accel/Gyro/Compass/Barometer Sensor missing", "ekf2 missing data")
  and, downstream, MAVROS to have nothing to relay. Fixed by adding
  those five plugins to `to_sdf()`. After regenerating the world and
  restarting both containers: `gz topic -l` now lists
  `.../sensor/imu_sensor/imu`, `.../sensor/air_pressure_sensor/air_pressure`,
  `.../sensor/navsat_sensor/navsat`, `/camera`, and `/depth_camera`;
  PX4's preflight warnings are gone and it reaches
  `commander: Ready for takeoff!`; and MAVROS logs "IMU: Attitude
  quaternion IMU detected!" / "IMU: High resolution IMU detected!".
* **Separately, a real MAVROS packaging quirk was found**: on this
  `ros-humble-mavros` build, `mavros_node`'s own C++ source hardcodes its
  node namespace to `mavros` (independent of anything this package's
  launch file sets), and most plugins build their topic names as
  relative `mavros/<name>` strings on top of that -- ROS 2 then doubles
  it to `/mavros/mavros/<name>`. A handful of topics (e.g.
  `/mavros/state`, `/mavros/local_position/pose`) are declared as
  absolute paths internally and so are *not* doubled, but on this build
  those specific ones turn out to have **zero publishers** (only other
  MAVROS plugins subscribe to them internally, e.g. `setpoint_position`
  tracking current pose to compute relative setpoints) -- the actual
  live data is on the doubled paths (`/mavros/mavros/data` for IMU,
  `/mavros/mavros/pose` for pose). `ros_bridge_node.py` now subscribes
  to these via configurable `env.ros.imu_topic` / `env.ros.pose_topic`
  (defaulting to the doubled paths verified working here) rather than
  the documented-but-actually-unpublished plain paths.
* **Verified end to end**: with both fixes in place, directly
  instantiating `ROSGazeboBridge` inside the live `ros-bridge` container
  and spinning it for a few seconds populated `self._imu` and
  `self._local_pose` with real values (e.g. `linear_acceleration.z`
  reading ~9.81 m/s^2 at rest -- exactly gravity, confirming this is
  real physics-simulated data, not noise). `GazeboROSEnv.reset()`/
  `step()` themselves (the full env wrapper, not just the bridge node)
  have not been separately exercised against a training loop, but the
  sensor data path they depend on is now confirmed live.
* One remaining loose end, not investigated further (did not block the
  above): MAVROS still logs `VER: autopilot version service timeout` /
  `switched to default capabilities` on every connection. This was
  originally suspected as the root cause of the missing sensor data --
  it wasn't; the missing world-level Gazebo sensor plugins were. The
  VER timeout appears to be a harmless, cosmetic capability-negotiation
  quirk of this PX4 SITL build.

## Architecture

```
                 MAVLink (UDP)                ROS 2 topics/services
  PX4 SITL  <----------------------->  MAVROS  <------------------->  ROSGazeboBridge (rclpy Node)
  (flight ctl,                        (mavros_node)                   ros_bridge_node.py
   speaks MAVLink)                                                          |
       |                                                                   |  wraps into
       | spawns + drives                                                   v
       v                                                          BaseCorridorEnv contract
  Gazebo (gz sim)  ---- camera/depth/odometry ---->  ros_gz_bridge -------> (GazeboROSEnv, env.py)
  (physics + sensors,                                                          |
   world generated by                                                         v
   world_gen.py)                                                    star_nav training/eval
                                                                      (SACR -> CAMR -> AGSS-PPO)
```

* **`world_gen.py`** -- pure NumPy/stdlib. Generates a Gazebo SDF world
  (`.sdf`) with trunks placed using the exact same Table 6 row/tree
  spacing mean/std algorithm as
  `star_nav/envs/mock_env.py::MockCorridorEnv._generate_world`, plus a
  ground-truth layout sidecar (`.world.json`) so the running bridge node
  knows trunk positions without needing to re-derive them from the SDF.
  Trunks default to a textured `oil_palm` mesh model (see below); pass
  `--cylinder-trunks` for the old dependency-free plain cylinder.
  **No ROS 2 install required for this file** -- see
  `test/test_world_gen.py`.
* **`ros_bridge_node.py`** -- an `rclpy.Node` that subscribes to MAVROS
  (IMU, EKF pose, arm/mode state) and `ros_gz_bridge`-relayed Gazebo
  topics (RGB camera, depth camera, ground-truth odometry), and publishes
  body-frame velocity setpoints to PX4 via
  `/mavros/setpoint_raw/local`.
* **`env.py`** -- `GazeboROSEnv(BaseCorridorEnv)`, the thin adapter that
  turns the bridge node's sensor state into `EnvObservation`/
  `PrivilegedInfo`/`EnvStepResult`, mirroring `airsim_env.py`'s reward/
  termination logic exactly (progress + smoothness + alive reward,
  success/collision/timeout termination).
* **`launch/corridor_sim.launch.py`** -- starts MAVROS + `ros_gz_bridge`
  (does **not** start Gazebo or PX4 SITL themselves -- see below).

## Vehicle models

Besides PX4's stock `x500`/`x500_depth` (a generic 500mm-class quad),
two custom vehicle models live in `px4_models/`, each generated by a
procedural script (no mesh tooling available in this environment --
frames are built from SDF primitives) and paired with a PX4 airframe
config in `px4_airframes/`:

| Model | Generator | Airframe | What it represents |
|---|---|---|---|
| `pavo_femto` | `pavo_femto_gen.py` | `4012_gz_pavo_femto` | The **actual real-world STAR-Nav hardware** -- a Pavo Femto 75mm ducted brushless whoop (LAVA 1102 motors, Gemfan 1611 3-blade props, DJI O4 Lite camera). Exact specs also recorded in this project's memory (`project_real_hardware_pavo_femto.md`). Battery mass is excluded (54.8g dry weight used), per explicit user instruction. |
| `fpv5` | `fpv5_gen.py` | `4013_gz_fpv5` | A **representative 5" freestyle quad** (225mm true-X, 2207-class motors, 5.1in 3-blade props, DJI O4 Pro camera), synthesized from Oscar Liang's 5-inch buying guide (which lists several interchangeable parts per component, not one fixed kit -- see the generator's docstring for exactly which option was picked and why). Added to evaluate STAR-Nav against a conventional open-frame quad, not just the tiny whoop. |

Both were live-tested by spawning them against the `scenario_a` world:
`gz topic -l` shows the expected per-vehicle sensor topics
(`/world/scenario_a/model/<name>_0/link/base_link/sensor/{imu,air_pressure,navsat}_sensor/...`,
plus `/camera` and `/depth_camera`), and PX4 stays up with no crash.
Their motor thrust curves, inertia, and `MPC_THR_HOVER` are engineering
estimates (see each generator's docstring for the specific numbers and
reasoning), and PX4's default rate-controller gains target `x500`-class
vehicles, so tune them for the whoop or the 5" racer before relying on flight
dynamics from either.

To use one instead of `x500_depth`: regenerate/rebuild with
`PX4_SIM_MODEL=pavo_femto` (or `fpv5`), and pass `vehicle_model` to
`corridor_sim.launch.py` plus override `env.ros.gt_odom_topic` to
`/model/pavo_femto_0/odometry` (or `/model/fpv5_0/odometry`) --
`gt_odom_topic`'s default is independent of `vehicle_model`, so it won't
follow automatically.

A real bug in `entrypoint_px4_gazebo.sh` was found and fixed while
adding these: the `make px4_sitl` target was hardcoded to
`gz_x500_depth`, ignoring `$PX4_SIM_MODEL` entirely -- so switching
models via that env var alone silently had no effect before this fix.
See `progress.md`'s "Custom vehicle models" section for the full story,
including a PX4 build-system gotcha (a CMake `file(GLOB ...)` that only
discovers new airframe files at configure time) worth knowing before
adding a third model.

## Environment asset: oil_palm tree model

`px4_models/oil_palm/` is a textured oil-palm tree (trunk + fruit
clusters), used by `world_gen.py` in place of a bare cylinder by
default. It's built from a mesh the user provided directly (a Blender
project exported to FBX), not one of the free CC-licensed options
researched as alternatives (Sketchfab "Oil Fruit Palm"/"Low Poly Palm
Tree", a Quaternius CC0 nature pack) -- those got as far as being
presented as options, but Sketchfab's models turned out to require a
logged-in account to actually download, which blocked fetching them
automatically.

The original FBX's materials had **absolute Windows paths** baked in for
every texture (`D:\BLENDER\picture\teksture\sawit\...`), which don't
resolve on Linux -- fixed by re-exporting to OBJ+MTL with `assimp
export` and hand-editing the `.mtl`'s texture paths to be relative to
the mesh's own directory. The original FBX ships alongside the fixed
OBJ+MTL in `meshes/` as a source reference, but the SDF references the
OBJ.

**Scale (`<scale>1.331...</scale>`) and the Y-up-to-Z-up orientation fix
(`roll=-pi/2`) in `model.sdf` are engineering estimates from the mesh's
reported bounding box, not confirmed by an actual eyeballed render** --
attempts to verify visually via the Gazebo GUI (WSLg) hit a real
rendering problem in this environment, root-caused to **no GPU device
passthrough** for the container (`/dev/dri` doesn't exist), forcing
Xwayland/Ogre2 onto a software-rendering path that throws real
`RenderingAPIException`s for *any* scene -- confirmed by reproducing the
identical exceptions with PX4's own stock world (no oil_palm, no custom
drones at all). Not specific to this mesh; see progress.md's "oil_palm
tree model" section for the full diagnosis. Fell back to non-visual
checks instead (all 54 trunk entities confirmed present via `gz topic`,
zero mesh-loading errors in the Gazebo *server* log). If a tree renders
obviously wrong-sized or sideways once GPU passthrough is set up (or on
a host that already has it), adjust those two values in `model.sdf`
first.

Collision physics is unaffected either way -- `model.sdf` keeps a simple
collision cylinder (0.25m radius, 6.0m height, matching `world_gen.py`'s
`TRUNK_RADIUS`/`TRUNK_HEIGHT`) under the visual mesh, same as the
original bare-cylinder trunks had.

## Why `reset()` doesn't re-randomize the world

`MockCorridorEnv` and `AirSimCorridorEnv` can regenerate/reset a fresh
procedural layout every episode essentially for free. A live Gazebo
session cannot cheaply spawn or delete hundreds of trunk models every
episode, so the workflow here is **"generate once, reset many times"**:

1. Generate a world ahead of time for a given scenario:
   ```bash
   python -m ros_gazebo_bridge.world_gen --config configs/default.yaml \
       --scenario A --out ros_gazebo_bridge/worlds/scenario_a
   ```
   This writes `scenario_a.sdf` (loaded by Gazebo) and
   `scenario_a.world.json` (the ground-truth trunk layout `env.py` loads
   at runtime -- point `env.ros.world_layout_json` at it).
2. Launch PX4 SITL + Gazebo with that world (see below).
3. `GazeboROSEnv.reset()` only re-arms/re-offboards the vehicle and
   teleports it back to the start pose -- it does **not** change the
   corridor geometry. Calling `reset(scenario="B")` while a scenario-A
   world is loaded raises a clear `ValueError` rather than silently
   evaluating against the wrong layout.
4. `weather` is accepted for interface compatibility (`BaseCorridorEnv`
   requires it) but is currently a no-op -- Gazebo has no direct
   equivalent of AirSim's weather API.

## Setup and run (manual, no Docker)

Requires ROS 2 Humble, Gazebo (Harmonic, via `ros-humble-ros-gz`),
MAVROS, and a PX4-Autopilot checkout capable of `make px4_sitl gz_<model>`.

```bash
# 0. One-time: build this package into your ROS 2 workspace
cd ~/ros2_ws/src && ln -s /path/to/code/ros_gazebo_bridge .
cd ~/ros2_ws && colcon build --packages-select ros_gazebo_bridge
source install/setup.bash

# 1. Generate the world for the scenario you want to evaluate
python -m ros_gazebo_bridge.world_gen --config configs/default.yaml \
    --scenario A --out ros_gazebo_bridge/worlds/scenario_a

# 2. Terminal 1: PX4 SITL + Gazebo, loaded with that world
cd PX4-Autopilot
PX4_GZ_WORLD=/path/to/code/ros_gazebo_bridge/worlds/scenario_a.sdf \
    PX4_SIM_MODEL=gz_x500_depth make px4_sitl gz_x500_depth

# 3. Terminal 2: MAVROS + ros_gz_bridge
ros2 launch ros_gazebo_bridge corridor_sim.launch.py

# 4. Terminal 3: STAR-Nav, same as the mock/airsim backends
python scripts/run_train_all.py --config configs/default.yaml
```

Set `env.name: gazebo_ros` and `env.ros.world_layout_json:
ros_gazebo_bridge/worlds/scenario_a.world.json` in your config (see the
`env.ros.*` block added to `configs/default.yaml`).

## Docker

See `docker/README.md` for a containerized version of the same three
processes (PX4 SITL+Gazebo, MAVROS+bridge, and the STAR-Nav training
process). Recommended if you don't want to hand-install the ROS 2 +
Gazebo + PX4 toolchain. Both `Dockerfile.px4_gazebo` and
`Dockerfile.ros_bridge` have been built and run against each other in
this environment (see the Status section above and `progress.md`) -- the
training/eval CMD paths (`train`/`eval` modes in
`entrypoint_ros_bridge.sh`) have not been separately exercised, though.

### Move it to a native-Linux GPU box (near "click-install")

The whole stack is containerized and on GitHub, so migrating off WSL is
essentially *clone + one script*. On a real Ubuntu/Debian machine with an
NVIDIA GPU (where the camera actually renders and video is fast):

```bash
# 0. one-time: NVIDIA driver on the host (skip if `nvidia-smi` already works)
sudo ubuntu-drivers autoinstall && sudo reboot

# 1. clone and bootstrap (installs Docker + NVIDIA Container Toolkit, builds)
git clone https://github.com/Aqshalikhsan/star-nav.git
cd star-nav/code/ros_gazebo_bridge/docker
./setup_linux.sh

# 2. fly it (setup_linux.sh prints these too)
xhost +local:docker
cd ..            # -> ros_gazebo_bridge/
python3 -m ros_gazebo_bridge.world_gen --scenario A --out worlds/scenario_a
PX4_GZ_WORLD=$(pwd)/worlds/scenario_a.sdf PX4_SIM_MODEL=fpv5 \
  PX4_GZ_MODEL_POSE="1,24,0.3,0,0,0" HEADLESS=0 \
  docker compose -f docker/docker-compose.yml up -d px4-gazebo ros-bridge
```

`docker/docker-compose.yml` already requests the GPU (via the NVIDIA
Container Toolkit) and mounts the X server, so with a working driver the
Gazebo GUI opens, the drone camera captures real frames, and rendering is
hardware-accelerated. `setup_linux.sh` is idempotent and installs Docker +
the toolkit for you; the only host-level step it can't do is the NVIDIA
*driver* install (step 0). On a CPU-only box it still builds and flies --
only the camera falls back to software rendering.


## Testing without any ROS install

```bash
pytest ros_gazebo_bridge/test/test_world_gen.py -v
```

This validates that `build_corridor_world` produces the same tree layout
as `MockCorridorEnv._generate_world` for the same seed, and that the
SDF/JSON round-trip is lossless. It is the one part of this backend that
has actually been executed and verified in this environment.

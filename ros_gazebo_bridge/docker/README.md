# Docker (Linux hosts)

Containerizes the two long-running processes the manual setup in
`../README.md` describes as separate terminals:

* **`px4-gazebo`** -- PX4 SITL + Gazebo, built from `Dockerfile.px4_gazebo`
  (clones `PX4-Autopilot`, runs its own dependency installer, pre-builds
  the `gz_x500_depth` SITL target).
* **`ros-bridge`** -- ROS 2 Humble + MAVROS + `ros_gz_bridge` + this
  repo's Python stack, built from `Dockerfile.ros_bridge`. Same image
  runs MAVROS/the bridge (`bridge` mode), generates worlds (`world`
  mode), or runs STAR-Nav training/eval (`train`/`eval` modes) --
  see `entrypoint_ros_bridge.sh`.

**Not built or run in this environment** (no internet-facing Docker or
PX4 toolchain available here) -- this is a documented starting point,
not a validated image. Expect to adjust package versions/toolchain steps
the first time you build it. Uses `network_mode: host`, so **Linux only**
(WSL2 or a Linux VM on Windows/Mac).

## Usage

```bash
cd ros_gazebo_bridge/docker
docker compose build

# 1. Generate a world (pure Python, runs inside ros-bridge's image)
docker compose run --rm ros-bridge world --config configs/default.yaml \
    --scenario A --out ros_gazebo_bridge/worlds/scenario_a

# 2. PX4 SITL + Gazebo, headless by default
PX4_GZ_WORLD=$(pwd)/../worlds/scenario_a.sdf docker compose up px4-gazebo

# 3. MAVROS + ros_gz_bridge (in another terminal)
docker compose up ros-bridge

# 4. STAR-Nav training/eval (in another terminal)
docker compose run --rm ros-bridge train --config configs/default.yaml
docker compose run --rm ros-bridge eval --config configs/default.yaml --out results.csv
```

`checkpoints/` and `logs/` are volume-mounted back to the repo root so
trained weights survive container restarts.

To watch the Gazebo GUI instead of running headless, set `HEADLESS=0`.
`docker-compose.yml` already forwards X11 (`/tmp/.X11-unix`, `DISPLAY`) and
requests the GPU; you still need `xhost +local:docker` once per host login
session. On hybrid/Optimus laptops (an NVIDIA GPU alongside an Intel iGPU)
the compose file also sets `__NV_PRIME_RENDER_OFFLOAD`/
`__GLX_VENDOR_LIBRARY_NAME=nvidia` so windowed rendering actually uses the
NVIDIA GPU instead of falling back to Mesa's `iris` driver — see
`progress.md`'s "First run on a native-Linux GPU laptop" section.

**See `MANUAL_FLIGHT_GUIDE.md` in this directory** for the full step-by-step
walkthrough: launching PX4+Gazebo and MAVROS, opening the camera view(s) in
`rqt_image_view`, flying by hand with keyboard teleop before touching any
autonomy, and landing/shutdown.

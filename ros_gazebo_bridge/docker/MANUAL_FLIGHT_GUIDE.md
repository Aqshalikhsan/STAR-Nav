# Manual flight guide

**One-time install (only once per machine, or after a `Dockerfile.*`
change):** `./setup_linux.sh` — installs Docker Engine + NVIDIA Container
Toolkit and builds both images. None of the scripts below ever install or
rebuild anything; they only start/stop containers that already exist, so
they're safe and fast to re-run any number of times.

Four scripts in this directory cover launch → camera view → keyboard
control → shutdown. Run them from `ros_gazebo_bridge/docker/`.

```bash
./start_sim.sh          # 1. launch PX4+Gazebo (GUI) + MAVROS, wait until ready
./camera.sh             # 2. open RGB camera window (run again with "depth" for depth)
./camera.sh depth
./fly_keyboard.sh        # 3. fly by hand (interactive terminal required)
./stop_sim.sh            # 4. stop everything when done
```

## Controls (`fly_keyboard.sh`)

| Key | Action |
|---|---|
| `t` | Take off |
| `w` / `s` | Pitch forward / back |
| `a` / `d` | Roll left / right |
| `q` / `e` | Yaw left / right |
| `r` / `f` | Climb / descend |
| `space` | Hover |
| `l` | Land + disarm |
| `x` / `Ctrl-C` | Quit (auto-lands) |

## Diagnostic tools (`rqt_tools.sh`)

```bash
./rqt_tools.sh console   # ROS/MAVROS log messages, real-time (vs grepping docker logs)
./rqt_tools.sh plot      # plot a numeric topic field over time (e.g. altitude drift)
./rqt_tools.sh topic     # topic Hz/bandwidth monitor
./rqt_tools.sh graph     # node <-> topic connection graph
./rqt_tools.sh service   # call a ROS service from a GUI form (arm/set_mode/etc.)
```

Run it again (same or different tool name) to open another window. Useful
topics to plug into `plot`/`topic`: `/mavros/mavros/pose`,
`/mavros/mavros/data` (IMU), `/camera`, `/depth_camera`.

## "container ... is not running" (e.g. from `camera.sh` or `fly_keyboard.sh`)

This means the containers are currently stopped — normal right after
`./stop_sim.sh`, or if they were never started this session. Fix: run
`./start_sim.sh` (no need for `--fresh` unless the EKF issue below applies),
then retry `./camera.sh` / `./fly_keyboard.sh`. `camera.sh`/`fly_keyboard.sh`
now check for this and print this same hint before failing.

Note `./stop_sim.sh` followed by `./start_sim.sh` is a normal, supported
cycle — containers just don't run in between, which is expected.

## If arming is denied after a previous landing

PX4's EKF doesn't self-recover after a flight+land cycle in this setup
(`Navigation failure!` → persistent `Preflight Fail: position estimate
error`). Fix: `./start_sim.sh --fresh` (force-recreates both containers),
then re-run `./camera.sh` (camera windows don't survive a container
recreate).

## Options

- Different vehicle: `PX4_SIM_MODEL=pavo_femto ./start_sim.sh` (default `fpv5`)
- `camera.sh <topic>` accepts any gz-bridged topic, not just `rgb`/`depth`
- `fly_keyboard.sh --takeoff-alt 3.0 --move-speed 1.5` etc. — flags pass
  straight through to `keyboard_control.py`
- Scripted (non-keyboard) flight instead: `sudo docker exec star_nav_ros_bridge
  bash -lc 'source /opt/ros/humble/setup.bash && source
  /ros2_ws/install/setup.bash && ros2 run ros_gazebo_bridge manual_control
  --maneuver allaxes'`

## Under the hood

Each script is a thin wrapper — read them if something doesn't match your
setup (world path, container names, env vars). Full technical background
(why the PRIME-offload env vars, the `PX4_GZ_WORLD` in-container-path
gotcha, etc.) is in `progress.md`'s "First run on a native-Linux GPU laptop"
section.

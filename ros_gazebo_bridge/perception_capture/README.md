# perception_capture — PX4-free perception-dataset capture (MANUAL path)

This folder is the **manual data-collection path**, kept deliberately
**separate from the autonomy path** (`star_nav/` training + `ros_gazebo_bridge`'s
`GazeboROSEnv` + PX4 SITL). Nothing here arms a vehicle or talks to PX4/MAVROS,
and **nothing here overwrites the autonomy code** — the two paths are
independent so a change/failure in one can't break the other.

## Why this exists

Training SACR (and the other perception modules) wants many *varied* corridor
viewpoints with real ground truth (rgb + segmentation + geometry). Getting
those from the full autonomy stack is shaped by two constraints of this sim:

- **Armed flight** auto-disarms on the SITL GPS-drift preflight check, so the
  vehicle never moves (see `progress.md` / project memory).
- **Teleporting** the camera with the gz `set_pose` service does **not** update
  a rendering sensor's viewpoint in this gz-sim7 (Garden) build — the pose
  moves but the render stays frozen. Verified conclusively.

The fix used here: a standalone, PX4-free camera **rig** that is glided down the
corridor under a **physics velocity** command (gz `VelocityControl`). Physics
motion *does* update the render, so the rig sees genuinely different viewpoints.

## Two files

- **`capture_world_gen.py`** — takes a *labeled* corridor world SDF (from
  `python -m ros_gazebo_bridge.world_gen ...`, the one with the gz-sim `Label`
  plugins on trunks/ground) and injects a movable camera rig (rgb + depth +
  segmentation sensors + `VelocityControl` + `OdometryPublisher`). Pure stdlib;
  run on the host.
- **`collect_dataset.py`** — runs inside the `ros-bridge` container. Glides the
  rig forward (weaving its heading for variety), capturing real rgb / seg /
  depth + the rig's true pose at a fixed rate → an `npz` with the fields SACR
  training expects (`rgb`, `seg_mask`, `theta_corr_gt`, `pose`).

## Manual run recipe

```bash
# 0) labeled corridor world + capture world (host)
python3 -m ros_gazebo_bridge.world_gen --scenario A \
    --out ros_gazebo_bridge/worlds/scenario_a
python3 ros_gazebo_bridge/perception_capture/capture_world_gen.py \
    --base-world ros_gazebo_bridge/worlds/scenario_a.sdf \
    --out ros_gazebo_bridge/perception_capture/capture_scenario_a.sdf

# 1) render container (px4-gazebo image), standalone gz sim, NO PX4
docker run -d --name caprig --network host --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e __NV_PRIME_RENDER_OFFLOAD=1 -e __GLX_VENDOR_LIBRARY_NAME=nvidia \
  -e GZ_SIM_RESOURCE_PATH=/opt/PX4-Autopilot/Tools/simulation/gz/models \
  -v $PWD/ros_gazebo_bridge/perception_capture/capture_scenario_a.sdf:/w.sdf:ro \
  --entrypoint bash docker-px4-gazebo:latest -c 'gz sim -s -r /w.sdf'

# 2) collector container (ros-bridge image) -- START IT FRESH AFTER (1) IS UP
#    (gz-transport discovery goes stale if the collector predates the gz server)
docker run -d --name capcli --network host --gpus all \
  -v $PWD/ros_gazebo_bridge/perception_capture:/pc:ro -v $PWD/data:/out \
  --entrypoint bash docker-ros-bridge:latest -c 'sleep infinity'
docker exec capcli bash -lc \
  'source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash; \
   python3 /pc/collect_dataset.py --out /out/rig_frames.npz --speed 1.2 --max-x 40'

# 3) train SACR on the HOST GPU from the npz (torch in the ros-bridge image is
#    CPU-only). rgb/seg_mask/theta_corr_gt map straight onto sacr_loss().
```

## Gotchas baked into the scripts (so you don't rediscover them)

- **`cmd_vel` is bridged ROS→gz** and published with rclpy, *not* `gz topic -p`:
  the ros-bridge image's `gz` CLI can be a different gz-transport major version
  than the px4-gazebo gz *server*, so a direct CLI publish silently never
  reaches the rig (it won't move). The gzgarden `parameter_bridge` pins the
  right version both ways.
- Start the collector container **fresh, after** the render container is up —
  a pre-existing container won't discover a gz server that started later.
- Segmentation labels: `0`=sky, `1`=ground, `2`=trunk. `canopy`/`actor` have no
  real pixels here (single-mesh oil-palm asset; no human model) — see the
  Limitations section in `ros_gazebo_bridge/README.md`.
- The gz camera `<save>`-to-disk path did **not** write files under `gz sim -s`
  here, which is why capture goes through the ros_gz bridge instead.

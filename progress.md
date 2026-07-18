# STAR-Nav reference implementation — project notes

This is a from-scratch reference implementation of the STAR-Nav paper
("Spatio-Temporal Adaptive Reinforcement Learning for Autonomous
Monocular UAV Navigation in GPS-Denied Repetitive Environment"), whose
manuscript source lives at `C:\Users\User\Documents\publikasi2\elsarticle-paper`.

See `README.md` for the full module-to-paper mapping, install/run
instructions, and design notes (Invariants I1-I5 enforcement, causal
sliding-window CAMR, mock vs. AirSim env backends).

## Status (as of the initial commit)

- Full pipeline implemented: SACR (`star_nav/models/sacr.py`, `enet.py`,
  `depth_net.py`), CAMR (`camr.py`), AGSS-PPO (`agss_ppo.py`), 3-phase
  training (`star_nav/training/`), evaluation (`star_nav/evaluation/`).
- Verified end-to-end via `configs/smoke_test.yaml`: data collection ->
  SACR pretraining -> CAMR pretraining -> PPO+AGSS training ->
  evaluation all run without error on CPU using `MockCorridorEnv` (the
  bundled dependency-free synthetic environment; no Unreal/AirSim
  required). `tests/test_shapes.py` passes.
- One real bug was found and fixed during verification: `ENetEncoder`
  had a channel-count mismatch between its two downsampling stages
  (hardcoded `stage1_channels=64` instead of `stage0_channels*2`) —
  fixed in `star_nav/models/enet.py`.
- Git repo initialized in this folder with one root commit. Author
  identity comes from the local git config — keep commit authorship as
  the local git user in future commits here too.
- `MockCorridorEnv` numbers (SR/CR/etc.) are NOT meant to reproduce the
  paper's reported results — it exists only to prove the pipeline is
  wired correctly. Do not treat smoke-test metrics as validation numbers.

## Gazebo + MAVLink + ROS backend (added after the initial commit)

Implemented as `ros_gazebo_bridge/` — a **separate top-level ROS 2
ament_python package**, deliberately not a module under `star_nav/`
(ROS package layout doesn't fit there, and it keeps `star_nav/` free of
a hard ROS 2 dependency). See `ros_gazebo_bridge/README.md` for the full
architecture, setup steps, and honest limitations list.

- `ros_gazebo_bridge/world_gen.py` — pure NumPy/stdlib, no ROS required.
  Generates a Gazebo SDF world using the exact same Table 6 row/tree
  spacing algorithm as `MockCorridorEnv._generate_world`, plus a
  ground-truth JSON layout sidecar. Verified via
  `ros_gazebo_bridge/test/test_world_gen.py` (matches MockCorridorEnv's
  tree layout for the same seed; SDF/JSON round-trip is lossless).
- `ros_gazebo_bridge/ros_bridge_node.py` + `env.py` — `GazeboROSEnv`, an
  rclpy-based `BaseCorridorEnv` implementation bridging PX4 SITL
  (MAVLink) via MAVROS and Gazebo sensors via `ros_gz_bridge`. Written
  against documented MAVROS/PX4/gz-sim APIs. **MAVROS<->PX4 connectivity
  is live-validated** -- `/mavros/state` shows `connected: true` with
  live `mode`/`armed` data. **`ros_bridge_node.py`'s actual IMU/pose
  subscriptions are now also confirmed against live data** (see the
  "Missing MAVROS IMU/local_position data" follow-up below for the root
  cause that was blocking this and the fix) -- instantiating
  `ROSGazeboBridge` directly and spinning it populates real IMU/pose
  values. `GazeboROSEnv.reset()`/`step()` as a full env against an actual
  training loop has not been separately exercised yet.
- Wired into `scripts/run_train_all.py`/`run_eval_all.py` via
  `env.name: gazebo_ros`, and into `configs/default.yaml`'s new
  `env.ros.*` block.
- **`ros_gazebo_bridge/docker/Dockerfile.px4_gazebo` (PX4 SITL + Gazebo)
  has been built and live-validated end to end**, in a local WSL2 Ubuntu
  distro (relocated from C: to `F:\WSL\Ubuntu` for disk space) running
  Docker Engine directly (not Docker Desktop). Confirmed working:
  `docker build` produces a working `gz_x500_depth` SITL binary; running
  the container with `HEADLESS=0` and WSLg's X11/Wayland sockets bind-
  mounted in shows a real Gazebo GUI window on the Windows desktop,
  loading a `world_gen.py`-generated corridor world (54 trunks, scenario
  A) with the `x500_depth_0` vehicle spawned into it. Three real PX4
  build/runtime bugs were found and fixed while getting this working
  (see `Dockerfile.px4_gazebo` and `entrypoint_px4_gazebo.sh` comments
  for the full explanation of each):
  1. Shallow (`--depth 1`) submodule clone leaves the NuttX submodule
     with no git tags, which crashes PX4's own
     `src/lib/version/px_update_git_header.py` (`git tag --sort=...`
     returns empty, `[-1]` on that empty list raises `IndexError`) --
     fixed by `git submodule deinit -f` on the NuttX submodules before
     building (NuttX is only needed for real-hardware firmware, never
     for the posix/SITL target this image builds).
  2. `make px4_sitl_default gz_x500_depth` (two make targets) is PX4's
     "build AND launch a live simulation" convenience form, not a
     build-only command -- using it as the Dockerfile's pre-build step
     hangs the image build forever inside a running `gz sim`+`bin/px4`
     session with no display. Fixed: build with `make px4_sitl_default`
     alone (no second target); the entrypoint script runs the real
     `make px4_sitl gz_x500_depth` (build+run) at container start,
     where a live session is actually wanted.
  3. `PX4_GZ_WORLD` must be a bare world *name*, not a path -- PX4
     resolves it to `Tools/simulation/gz/worlds/<name>.sdf` itself. It
     also becomes the Gazebo world *name* PX4 expects (used to build
     `/world/<name>/create` etc. service names), which must match the
     `<world name="...">` attribute actually inside that SDF file, or
     PX4 waits forever for a Gazebo world that never answers. Fixed in
     both `entrypoint_px4_gazebo.sh` (copies the mounted `.sdf` into
     PX4's expected worlds directory under its basename) and
     `world_gen.py` (`--world-name` now defaults to `--out`'s basename
     instead of a fixed `"oil_palm_corridor"`, so file name and internal
     SDF world name can't drift apart by default). Also needed
     `PX4_GZ_STANDALONE=1` (GZBridge otherwise gives up after a single
     1-second connection attempt) and to launch `gz sim` explicitly from
     the entrypoint script rather than relying on PX4 to auto-launch it
     (observed empirically as unreliable).
- The `pxh>` console used to spam an unbounded prompt-redraw escape
  sequence into `docker logs` once the startup script finished (no real
  TTY attached) -- fixed in `entrypoint_px4_gazebo.sh` by feeding it a
  bash process-substitution pipe that stays open but never produces data
  (`< <(sleep infinity)`), rather than `/dev/null` (immediate EOF, which
  it treats as "retry now" with no backoff -- a tight busy loop, worse
  than the redraw spam). Verified: log went from 77.7MB to ~3KB for the
  same run.
- **`ros_gazebo_bridge/docker/Dockerfile.ros_bridge` (ROS 2 Humble +
  MAVROS + `ros_gz_bridge`) has now been built and smoke-tested against
  the live `px4-gazebo` container** (host networking, both containers on
  the same WSL2 Docker Engine). MAVROS successfully connects to PX4 SITL
  over MAVLink UDP -- confirmed via `ros2 topic echo /mavros/state`
  showing `connected: true`, live `mode` (e.g. `AUTO.LOITER`) and `armed`
  state. Bugs found and fixed getting here:
  1. The default PyPI `torch` wheel pulls the full CUDA toolkit
     (~3-4GB of `nvidia-*` packages) for a container with no GPU
     passthrough configured -- a build attempt was killed after 40
     minutes still mid-download. Fixed: install the CPU-only wheel from
     PyTorch's own index (`--index-url .../whl/cpu`) instead; cut that
     step to under 3 minutes.
  2. `install_geographiclib_datasets.sh` isn't at the hardcoded path
     this `ros-humble-mavros` build assumes
     (`share/mavros/scripts/...`) -- found dynamically with `find`
     instead, so MAVROS's GPS coordinate-transform datasets actually
     get installed rather than silently no-op'ing behind `|| true`.
  3. Two `COPY` paths in `Dockerfile.ros_bridge` assumed a build context
     of `ros_gazebo_bridge/docker/`, but the real context (per
     `docker-compose.yml` and the direct `docker build` used to test
     this) is the repo root -- fixed to include the `ros_gazebo_bridge/`
     prefix.
  4. `entrypoint_ros_bridge.sh` used `set -u`, but ROS 2's own
     `setup.bash` references variables (e.g. `AMENT_CURRENT_PREFIX`)
     that are legitimately unset on first source -- dropped `-u` (kept
     `-e`/`pipefail`).
  5. **`mavros_node` crashed** (`rclcpp::exceptions::RCLError`: "invalid
     allocator") while loading plugins one by one, each dying on its own
     topic (`companion_process_status` on `mavros/status`, then
     `debug_value` on `mavros/send`, ...) -- this is a known rough edge
     between MAVROS and the default `rmw_fastrtps_cpp` on this build.
     Fixed by installing `ros-humble-rmw-cyclonedds-cpp` and setting
     `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` (the commonly recommended
     RMW for PX4/MAVROS setups anyway). `plugin_denylist:
     [companion_process_status]` is also set in
     `launch/corridor_sim.launch.py` (that plugin isn't needed here) and
     was left in even after the DDS switch fixed the crash.
  6. **WSL2 VM auto-shutdown killed both containers** between tool
     calls/conversation turns in this session (`wsl -l -v` showed the
     whole distro as `Stopped`, not just the containers exiting) --
     `%UserProfile%\.wslconfig`'s `vmIdleTimeout=-1` alone did not
     prevent this in testing; what reliably worked was holding one
     persistent `wsl.exe` session attached in the background
     (`wsl -d Ubuntu -- sleep <n>`) for the duration of the work. Worth
     remembering for any future long-running WSL2 Docker work in this
     environment -- don't rely on one-shot `wsl -d Ubuntu -- <cmd>`
     calls alone across a long session.
- **Missing MAVROS IMU/local_position data -- root-caused and fixed** (as
  a follow-up to the above, in a later session). The actual cause was
  *not* the "VER: autopilot version service timeout" originally
  suspected -- it was that `world_gen.py`'s generated world SDF only
  loaded the `Physics`/`SceneBroadcaster`/`Contact`/`UserCommands` Gazebo
  system plugins, missing `Imu`, `AirPressure`, `ApplyLinkWrench`,
  `NavSat`, and `Sensors` (the set PX4's own stock
  `Tools/simulation/gz/worlds/default.sdf` loads). Without these, Gazebo
  never simulates or publishes *any* sensor data at all, regardless of
  PX4/MAVROS connectivity -- explaining both the empty `gz topic -l`
  sensor list and PX4's preflight failures ("Accel/Gyro/Compass/
  Barometer Sensor missing", "ekf2 missing data"). Fixed by adding those
  five plugins to `to_sdf()` in `ros_gazebo_bridge/world_gen.py`. After
  regenerating the world and restarting both containers: PX4 reaches
  `Ready for takeoff!` with no preflight warnings, `gz topic -l` lists
  the IMU/air-pressure/navsat/camera/depth-camera topics, and MAVROS
  logs "IMU: Attitude quaternion IMU detected!" / "High resolution IMU
  detected!".
  - Separately (a real, still-present quirk, but not the blocker): many
    MAVROS plugin topics publish under a doubled `/mavros/mavros/<topic>`
    path instead of the documented `/mavros/<topic>` -- `mavros_node`'s
    C++ source hardcodes its own node namespace to `mavros`, and most
    plugins build relative `mavros/<name>` topic strings on top of that,
    doubling it. A few topics (e.g. `/mavros/local_position/pose`) are
    absolute internally and aren't doubled, but on this build those
    specific ones have **zero publishers** (only other MAVROS plugins
    subscribe to them internally, e.g. `setpoint_position` tracking
    current pose). Fixed by making `ros_bridge_node.py`'s IMU/pose
    subscriptions configurable (`env.ros.imu_topic` / `env.ros.pose_topic`
    in `configs/default.yaml`), defaulting to the verified-working
    doubled paths (`/mavros/mavros/data`, `/mavros/mavros/pose`).
  - Verified end to end: instantiating `ROSGazeboBridge` directly inside
    the live `ros-bridge` container and spinning it populated
    `self._imu`/`self._local_pose` with real values (IMU
    `linear_acceleration.z` ~9.81 m/s^2 at rest -- gravity, confirming
    real physics data). The `VER: autopilot version service timeout`
    warning is still present but turned out to be an unrelated, harmless
    cosmetic quirk of this PX4 SITL build, not a blocker.
  - Not separately exercised: `GazeboROSEnv.reset()`/`step()` as a full
    env against an actual training loop (only the underlying bridge
    node's sensor subscriptions were verified). **This was later
    attempted (2026-07-05) and is currently blocked** — see "First
    attempt to run the full GazeboROSEnv loop" below. Also noticed but
    not investigated: a NumPy 2.x/`cv_bridge` ABI warning
    (`AttributeError: _ARRAY_API not found`) on `cv_bridge` import in
    this environment -- didn't block the IMU/pose test, but wasn't
    checked against actual image conversion (RGB/depth camera topics).
- Key design choice worth remembering: unlike Mock/AirSim, `reset()`
  does **not** re-randomize the corridor layout (Gazebo can't cheaply
  respawn hundreds of trunk models per episode) — worlds are generated
  once via `python -m ros_gazebo_bridge.world_gen` and `reset()` only
  re-arms/teleports the vehicle within that fixed world.
- Next planned step per the user (2026-07-04): once this backend is
  fully validated (including the ROS/MAVROS half), move on to the real
  AirSim setup (`star_nav/envs/airsim_env.py`) — that's *why* this was
  kept in its own folder, to avoid entangling the two.

### Custom vehicle models: pavo_femto and fpv5 (added 2026-07-05)

Two custom Gazebo vehicle models were added alongside PX4's stock
`x500`/`x500_depth`, since the real STAR-Nav hardware is nothing like a
500mm-class quad:

- **`pavo_femto`** (`ros_gazebo_bridge/pavo_femto_gen.py`,
  `px4_models/pavo_femto/`, airframe `px4_airframes/4012_gz_pavo_femto`)
  — a procedurally generated model of the actual real-world drone (a
  Pavo Femto 75mm ducted brushless whoop; exact specs in project memory
  `project_real_hardware_pavo_femto.md` and in the generator's own
  docstring). Battery mass is intentionally excluded (user instruction,
  2026-07-05) — the model uses the 54.8g dry weight. No mesh tooling
  (Blender etc.) is available in this environment, so the ducted frame
  is built entirely from SDF primitives, including a 16-segment-box
  approximation of each duct ring (SDF has no torus primitive). Motor
  thrust curve, inertia, and camera tilt are engineering estimates, not
  measured bench data — see the generator's module docstring for the
  full reasoning behind every number.
- **`fpv5`** (`ros_gazebo_bridge/fpv5_gen.py`, `px4_models/fpv5/`,
  airframe `px4_airframes/4013_gz_fpv5`) — a second, structurally
  unrelated model: a representative *open-frame* 225mm true-X 5" FPV
  freestyle quad (2207-class motors, 5.1in 3-blade props, DJI O4 Pro
  camera), added so STAR-Nav can also be evaluated against a
  conventional racing/freestyle quad rather than only the tiny whoop.
  Synthesized from Oscar Liang's "Best 5 Inch FPV Drone And Its
  Components" buying guide (user-provided, 2026-07-05), which lists
  several interchangeable options per part rather than one fixed kit —
  see the generator's docstring for exactly which option was picked to
  represent each component and why.
- Both were live-tested in this session (headless spawn against the
  live `scenario_a` world in the existing `px4-gazebo` container, using
  bind-mounted models/airframes + a manual `cmake .` reconfigure rather
  than a full image rebuild, to iterate quickly): both spawn
  successfully, `gz topic -l` shows the expected per-vehicle IMU/
  air-pressure/navsat/camera/depth-camera topics
  (`/world/scenario_a/model/<name>_0/link/base_link/sensor/...`), and
  the PX4 process stays alive with no crash. **Not yet tested**: actual
  flight behavior (arming, takeoff, stable hover) — both airframes'
  `MPC_THR_HOVER` and PX4's default rate-controller gains are rough
  estimates/untouched defaults respectively, and are explicitly flagged
  in each airframe file as needing retuning once someone actually tries
  to fly them in sim. `fpv5` in particular has a very different (far
  more aggressive, ~9.5 T/W) thrust profile than `x500`/`pavo_femto`, so
  expect it to need the most retuning.
- **A real, previously-latent bug was found and fixed while testing
  this**: `entrypoint_px4_gazebo.sh` hardcoded `make px4_sitl
  gz_x500_depth` — the literal `gz_x500_depth` make target, not derived
  from `$PX4_SIM_MODEL` at all. Setting `PX4_SIM_MODEL=pavo_femto` (or
  any non-default model) had **no effect** until this was fixed to
  `make px4_sitl "gz_${PX4_SIM_MODEL}"`. This means the backend was
  never actually selectable beyond the one hardcoded default before this
  fix — worth knowing if anything upstream assumed `PX4_SIM_MODEL` alone
  was sufficient to switch vehicles.
- **PX4 build-system mechanism worth remembering for any future custom
  vehicle model**: PX4's `src/modules/simulation/gz_bridge/CMakeLists.txt`
  generates one `gz_<model>` CMake/ninja target per airframe file
  matching `ROMFS/px4fmu_common/init.d-posix/airframes/*_gz_*`, via a
  `file(GLOB ...)` that only runs at CMake *configure* time. Add a new
  vehicle's airframe file to an *already-configured* build and the new
  `gz_<model>` target silently won't exist (`ninja: error: unknown
  target`) until something forces a re-configure (`cmake .` inside the
  build dir, confirmed to pick up new airframes without a full rebuild).
  `Dockerfile.px4_gazebo` avoids ever hitting this for a fresh build by
  copying both models' `px4_models/*` and `px4_airframes/*` in *before*
  the first `make px4_sitl_default` (which configures once, from
  scratch, and this glob runs as part of that).
- `Dockerfile.px4_gazebo`'s build context changed from
  `ros_gazebo_bridge/docker/` to the repo root (matching
  `Dockerfile.ros_bridge`'s existing convention) to reach
  `px4_models/`/`px4_airframes/` via `COPY` — `docker-compose.yml`'s
  `px4-gazebo` service updated to match (`context: ../..`,
  `dockerfile: ros_gazebo_bridge/docker/Dockerfile.px4_gazebo`). **Not
  yet re-validated with an actual fresh image build** in this session —
  the mechanism above was validated piece-by-piece against the
  already-built image via bind-mounts + a manual `cmake .` reconfigure,
  not via a full `docker build` from a clean checkout. Do a full rebuild
  before fully trusting the committed Dockerfile end-to-end.
- `launch/corridor_sim.launch.py`'s `vehicle_model` launch arg already
  existed (`--default x500_depth`) but `gt_odom_topic`'s default
  (`/model/x500_depth_0/odometry`) is a separate, independently-set
  value, not derived from `vehicle_model` — switching to `pavo_femto` or
  `fpv5` means also manually overriding `gt_odom_topic` to
  `/model/pavo_femto_0/odometry` / `/model/fpv5_0/odometry`.

### oil_palm tree model (added 2026-07-05)

`world_gen.py`'s `to_sdf()` now spawns each trunk as an `<include>` of a
new `oil_palm` model (`px4_models/oil_palm/`) by default
(`use_mesh_trunks=True`), instead of the bare brown cylinder used before
— a textured mesh with trunk and fruit-cluster geometry, built from a
file the user provided directly (`kelapa sawit.fbx`, from their own
Blender project). Pass `--cylinder-trunks` to `world_gen.py`'s CLI (or
`use_mesh_trunks=False` to `to_sdf()`/`write_world_bundle()` directly) to
fall back to the old plain-cylinder trunks with no external asset
dependency.

- **Mesh format: use the GLB, not the OBJ or FBX.** Getting this tree to
  actually render took working through three stacked problems:
  1. *Absolute Windows texture paths.* The source FBX's materials
     referenced `D:\BLENDER\picture\teksture\sawit\...` (confirmed via
     `assimp info`'s `Texture Refs`), which don't resolve on Linux.
     First fix: re-export to OBJ+MTL (`assimp export kelapa_sawit.fbx
     kelapa_sawit.obj`) and hand-edit the `.mtl`'s `map_Kd`/`map_Ns`/
     `bump` paths to be relative (`sawit/batang/...`, `sawit/buah/...`).
  2. *OBJ materials don't render in gz-sim's Ogre2.* The OBJ **loaded
     without error but rendered completely invisible** — Ogre2's HLMS
     rejects the OBJ's plain-MTL materials (`OGRE EXCEPTION: Fixed
     Function pipeline is no longer allowed nor supported`), so the mesh
     draws nothing while primitive-based models (the fpv5/pavo_femto
     drones) render fine. Fix: convert the *fixed* OBJ to binary glTF
     (`assimp export kelapa_sawit.obj kelapa_sawit_tex.glb`), which
     produces proper PBR (`pbrMetallicRoughness`) materials Ogre2's PBS
     pipeline renders, and carries the OBJ's already-relative texture
     URIs into the glTF `images` array. **Converting the FBX straight to
     GLB does NOT work** — it re-embeds the original `D:\` paths; the
     conversion must go through the path-fixed OBJ.
  3. *Scale/orientation.* The mesh bounding box is ~450 units tall;
     `model.sdf` uses `<scale>0.01331>` to hit `world_gen.py`'s
     `TRUNK_HEIGHT` (6.0m), and `roll=+pi/2` to turn the mesh's Y-up into
     Gazebo's Z-up. (An earlier OBJ-based `model.sdf` had `scale 1.331` /
     `roll -pi/2` — both wrong by the time we could actually see a
     render: 1.331 would be ~100x too tall. The GLB values are the
     correct ones.)
  The `meshes/` dir keeps the original FBX and the intermediate fixed
  OBJ+MTL as source references, but `model.sdf`'s `<mesh><uri>` points at
  `kelapa_sawit_tex.glb`. Regenerate the GLB with the two `assimp export`
  commands above if the mesh ever changes.
- **Verified by an actual render** (not just topic existence). The
  interactive Gazebo GUI genuinely can't render in this environment
  (see the GPU/WSLg limitation two bullets down), so verification was
  done via an **offscreen camera-sensor render**: a world with a
  `type="camera"` sensor whose `<camera>` has `<save enabled="true">`,
  run headless with `gz sim -s` — gz-sim writes each rendered frame to
  disk as a PNG, using the offscreen EGL render path, which works fine
  even with no GPU/GUI. That confirmed the tree renders correctly: green
  fronds, brown trunk, red fruit clusters, upright, ~6m, casting a
  shadow (saved to `code/renders/oil_palm_textured.png`), and the fpv5
  drone renders correctly too (`code/renders/fpv5_corridor_view.png`).
  This `<save>`-camera trick is the way to get any visual out of this
  environment — remember it instead of fighting the GUI.
- **The interactive Gazebo GUI can't render here — no GPU passthrough.**
  Root-caused thoroughly (it is NOT about any asset): the GUI window
  renders solid black. `/root/.gz/rendering/ogre2.log` shows `OGRE
  EXCEPTION(3:RenderingAPIException)` / `Fixed Function pipeline is no
  longer allowed`, reproducible even with PX4's **stock** `default.sdf`
  (nothing custom). The chain: WSLg's weston doesn't expose GBM to
  Xwayland (`Xwayland glamor: GBM Wayland interfaces not available` /
  `Failed to initialize glamor, falling back to sw` in
  `/mnt/wslg/stderr.log`) → Xwayland runs software → the X screen isn't
  DRI3-capable (`screen 0 does not appear to be DRI3 capable`) → all
  GLX/OpenGL clients get `llvmpipe`. Confirmed identical in the **native**
  WSL distro (not just the container), and unchanged by a full `wsl
  --shutdown`/reboot. The host DOES have a working GPU (`nvidia-smi`
  shows an RTX 2050, `/dev/dxg` + `/usr/lib/wsl/lib` present) and mounting
  `--device=/dev/dxg -v /usr/lib/wsl/lib` into the container makes the
  GPU libs available — but it still falls to `llvmpipe` because the
  blocker is weston/Xwayland not exposing GBM, a WSLg-level issue, not a
  container one. A `wsl --update` to 2.7.10 (newer WSLg might help) failed
  with a Windows Installer conflict (exit 1618). **Bottom line: use the
  offscreen `<save>`-camera render path above for any visual; the live
  GUI is a dead end in this environment without a WSLg/driver fix.** The
  offscreen path itself uses software `llvmpipe` too (only the
  `EGL_MESA_device_software` device enumerates, not d3d12) but renders
  correctly — it's just slow (54 textured trees take minutes/frame).
- Collision physics is unaffected by any of this: `oil_palm/model.sdf`
  keeps a simple collision cylinder (radius 0.25m, height 6.0m, matching
  `world_gen.py`'s `TRUNK_RADIUS`/`TRUNK_HEIGHT` constants exactly) under
  the detailed visual mesh, same as the original bare-cylinder trunks
  had — the mesh only changes what gets rendered, not collision/physics
  behavior. Those two constants are duplicated (once in `world_gen.py`,
  once in `oil_palm/model.sdf`) rather than derived from one source —
  keep them in sync if either ever changes.
- Three candidate free CC-licensed oil-palm meshes (Sketchfab "Oil Fruit
  Palm", Sketchfab "Low Poly Palm Tree", Quaternius CC0 nature pack) were
  researched and presented to the user before this, but ultimately went
  unused once the user supplied their own asset directly — worth knowing
  Sketchfab requires a logged-in account to actually download any model
  (even free/CC-licensed ones; confirmed their download endpoint 404s
  without auth), which was the practical blocker for those.

### Full GazeboROSEnv loop — now working end to end (2026-07-05)

Exercised the whole chain — PX4+Gazebo → `ros_gz_bridge` → MAVROS →
`GazeboROSEnv.reset()`/`step()` — end to end for the first time (prior
sessions only validated the MAVROS half in isolation). **It now works**:
`reset()` completes in ~1s returning real observations (rgb 480×640×3,
imu `az≈9.77`, EKF pose), and `step()` runs with reward, done, and
privileged info (goal_distance/collided/lateral_deviation) all computed.
`GazeboROSEnv` is now a working `BaseCorridorEnv`, on par with the
Mock/AirSim backends for the env contract. Getting there took finding and
fixing **six** distinct bugs (all fixed; commit messages have the detail):

1. **No ground-truth odometry source in Gazebo.** Gazebo published no
   odometry for the vehicle at all — added the `OdometryPublisher`
   gz-sim plugin to `fpv5_gen.py`/`pavo_femto_gen.py`.
2. **gz-transport version mismatch (the main blocker).** PX4 v1.15 ships
   Gazebo Garden (`gz-transport12`), but the default `ros-humble-ros-gz`
   package is built against Fortress (`ignition-transport11`) — a
   different major version that **cannot discover Garden topics at all**,
   so `ros_gz_bridge` advertised the ROS 2 topics but forwarded zero
   messages (looked exactly like a DDS delivery failure, but was really a
   gz-transport discovery failure on the *input* side). Confirmed by
   installing `gz-transport12-cli` in the bridge container and watching it
   suddenly see/echo the PX4 topics that `ign-transport11` couldn't. Fixed
   by installing `ros-humble-ros-gzgarden-bridge` (Garden/gz-transport12)
   instead of `ros-humble-ros-gz` in `Dockerfile.ros_bridge` (+ the OSRF
   gazebo apt repo). **General rule: the ros_gz_bridge's gz-transport
   major version MUST match whatever Gazebo PX4 builds.**
3. **Camera topic-name mismatch.** Models publish gz `/camera` and
   `/depth_camera`; the launch/config defaulted to `/camera/image` etc.
   (no gz source). Aligned launch + `configs/default.yaml`.
4. **NumPy 2.x vs cv_bridge ABI.** cv_bridge (built against NumPy 1.x)
   segfaults converting images under NumPy 2.x — pinned `numpy<2` in
   `Dockerfile.ros_bridge`.
5. **Best-effort starvation.** MAVROS publishes imu/pose BEST_EFFORT (no
   retry). Under the default mutually-exclusive callback group — even with
   a `MultiThreadedExecutor` — the slow cv_bridge rgb/depth callbacks
   serialize ahead of imu/pose, dropping their messages; `reset()` timed
   out on *only* imu/local_pose while every reliable topic came through.
   Fixed with a `ReentrantCallbackGroup` + a background
   `MultiThreadedExecutor` spin thread in `ros_bridge_node.py` (spin_for/
   wait_until_ready/service-calls now poll cached state instead of driving
   `spin_once` inline).
6. **Arm service name.** `arm_and_offboard()` used `/mavros/cmd/arming`,
   which doesn't exist on this build — it's `/mavros/mavros/arming`
   (doubled namespace; `set_mode` is `/mavros/set_mode`, not doubled).
   Made the arm/mode service + setpoint topic names configurable
   (`env.ros.arm_service`/`mode_service`/`setpoint_topic`).

**Flight now works — the "doesn't visibly fly" issue is RESOLVED
(2026-07-05).** The suspected setpoint-topic doubling was exactly right:
MAVROS's setpoint_raw plugin subscribes to **`/mavros/mavros/local`**
(mavros_msgs/PositionTarget), NOT the documented
`/mavros/setpoint_raw/local` (which has ZERO subscribers on this build —
verified via `ros2 node info /mavros/mavros`). The env/config was
publishing setpoints to the documented name, so every setpoint was
silently dropped and PX4 never moved. Fixed the default in both
`ros_bridge_node.py` and `configs/default.yaml` (`env.ros.setpoint_topic`)
to `/mavros/mavros/local`. With that, arm + OFFBOARD + velocity setpoints
produce real motion (proven via `manual_control.py`, below). Spawn-vs-
corridor alignment was handled separately by passing
`PX4_GZ_MODEL_POSE="1,24,0.3,0,0,0"` to the px4-gazebo entrypoint so the
vehicle starts on the corridor centerline (y=24) instead of the world
origin.

### Manual (scripted) OFFBOARD control — `manual_control.py` (2026-07-05)

`ros_gazebo_bridge/ros_gazebo_bridge/manual_control.py` (+ `manual_control`
console-script entry point) is the "drive it by hand before autonomy"
tool: arms, switches PX4 to OFFBOARD, and flies a fixed human-authored
sequence (takeoff → forward → hover → optional yaw → land) by streaming
`PositionTarget` setpoints, logging EKF pose throughout. It loads no
STAR-Nav policy — it exists purely to prove the control path
(arm → OFFBOARD → setpoint → visible motion) works, so later non-motion is
known to be policy/tuning, not plumbing. **Live-validated end to end**:
`x500_depth` armed, entered OFFBOARD, climbed to 2.5 m, and flew ~20 m of
forward travel down the palm corridor (EKF x 0.15 → 19.6 m) holding
altitude ~2.2 m with minimal lateral drift, then held position. Run it
inside the ros-bridge container (MAVROS already up) with e.g.
`python3 -m ros_gazebo_bridge.manual_control --takeoff-alt 2.5
--forward-speed 1.5 --forward-time 20 --no-land`. Notes:
- **Model selection quirk — SOLVED (2026-07-05).** `PX4_SIM_MODEL=fpv5`
  used to launch `x500_depth` anyway, for TWO stacked reasons, both now
  fixed: (1) the running container's `/entrypoint.sh` was a stale copy that
  hardcoded `make px4_sitl gz_x500_depth` (the repo entrypoint's
  `gz_${PX4_SIM_MODEL}` fix had never been rebuilt into the image) — copy
  the current entrypoint in (or rebuild); (2) the custom airframes were
  copied into `ROMFS/.../airframes/` but never added to that dir's
  `CMakeLists.txt` `px4_add_romfs_files(...)` list, so PX4 never INSTALLED
  them into the rootfs and `PX4_SIM_MODEL=fpv5` found no airframe and fell
  back to x500_depth (SYS_AUTOSTART 4002). `Dockerfile.px4_gazebo` now has a
  `sed` step registering `4012_gz_pavo_femto`/`4013_gz_fpv5` in that
  CMakeLists before the build. With both fixed, `fpv5` spawns as
  `fpv5_0` with `SYS_AUTOSTART=4013`.
- **fpv5 now FLIES (2026-07-05).** Verified: stable takeoff to 2.5 m and a
  smooth, controlled ~9 m forward flight down the palm corridor (EKF x
  0.4 → 9.3 m, altitude held ~3 m). Getting there took fixing THREE real
  airframe bugs in `px4_airframes/4013_gz_fpv5`, in order of discovery:
  1. **Thrust ceiling:** `SIM_GZ_EC_MAX*` was the x500 default 1000 while
     the motor model's `<maxRotVelocity>` is 3300, so the gz bridge capped
     every motor at ~1000 rad/s → max thrust ~2.6 N vs ~2.9 N weight →
     effective T/W < 1, could not leave the ground (motors just idled
     ~500 rad/s). Fixed: `SIM_GZ_EC_MAX* = 3300` (must match maxRotVelocity).
  2. **Wrong hover point:** with the ceiling fixed, `MPC_THR_HOVER 0.15` was
     far too low (real hover = sqrt(1/9.5) ≈ 0.32 of range). Set to 0.32,
     plus gentler `MC_ROLLRATE_P/PITCHRATE_P`, `MPC_THR_MAX`, and takeoff
     caps for this twitchy 9.5-T/W frame.
  3. **Roll-axis inversion (the decisive one):** all `CA_ROTOR*_PY` signs
     were the model's raw gz-frame y, but the gz model frame is FLU
     (y-left) and PX4's control allocation is FRD (y-right) — the y axis
     flips between them, so every `CA_ROTOR_PY` had to be the *negative* of
     the model's `rotor_N` link y. With them un-negated the roll axis was
     inverted → the mixer rolled the wrong way → the vehicle flipped itself
     over and disarmed the instant it left the ground (with the thrust fix
     it flipped *violently*, shooting to ~9 m before tumbling). Negating all
     four PY signs is what finally gave stable flight.
  Still-open polish (NOT airframe-fundamental): OFFBOARD occasionally drops
  to ALTCTL mid-flight (setpoint stream lapsing under system load → the
  >2 Hz OFFBOARD requirement is briefly missed), and the velocity-mode
  altitude hold in `manual_control.py` drifts up ~0.5 m during forward
  flight. Both are stream-robustness / script tuning, not the airframe.
- **pavo_femto has the same two systematic bugs** (same generator wrote
  both airframes): `SIM_GZ_EC_MAX* = 1000` vs its `<maxRotVelocity>` 12000,
  and un-negated `CA_ROTOR*_PY`. Both are corrected in
  `px4_airframes/4012_gz_pavo_femto` by analogy to the flight-verified fpv5
  fix, but pavo flight is **NOT yet verified** — its hover throttle and
  rate gains are still unvalidated estimates.
- **x500_depth** remains the most reliable model (PX4-tuned) and is what the
  earlier control-path proof used; it has no `OdometryPublisher` though, so
  the env's ground-truth `gt_odom` needs an odometry-capable model (fpv5 /
  pavo_femto) or adding the plugin to x500.
- The setpoint streaming loop is **wall-clock timed**, not iteration-count
  timed: `rclpy.spin_once` returns as soon as it services a callback (well
  before its timeout when messages are waiting), so counting iterations
  made manoeuvres finish early (a `--forward-time 8` ran only ~5 s). Fixed
  to publish on a fixed real-time schedule and spin until the wall-clock
  duration actually elapses.
- **EKF position estimate is jumpy** in this SITL setup (observed the
  estimate jump e.g. 24 m → 0.4 m between 1-second samples during
  hold/hover). Velocity-setpoint flight is robust to it (forward travel was
  smooth and monotonic), but it will hurt position-hold and any autonomy
  that trusts absolute position — worth investigating (GPS drift / no VIO)
  before the learned policy relies on pose.
- **Live drone-POV camera is blank here** (uniform grey) — the same known
  WSLg no-GPU-passthrough limitation: the live gz camera sensor renders via
  software llvmpipe and produces an empty frame. Only the offscreen
  `<save>`-camera trick (separate headless gz) yields real imagery (those
  beauty shots already exist under `renders/`). Also note the **running**
  `ros_gz_bridge` was started from an older launch (topics `/camera/image`,
  `/camera/depth_image`) and, having outlived a gz-sim restart, is stale to
  the current gz instance (gz-transport doesn't always re-discover across a
  server restart) — a *fresh* `parameter_bridge` re-discovers and delivers
  frames. MAVROS is unaffected by gz restarts (it speaks MAVLink UDP, not
  gz-transport), which is why flight kept working.
- **Full `GazeboROSEnv` step()-moves-drone not re-verified this session:**
  the env needs the whole sensor suite (rgb/depth/gt_odom via ros_gz_bridge)
  healthy for `wait_until_ready`, which the stale bridge + `x500_depth`
  having no `OdometryPublisher` block. But `step()` publishes the identical
  `PositionTarget` to the identical `/mavros/mavros/local` topic that
  `manual_control.py` proved moves the drone, so the fix is sound; a clean
  re-verify just needs a fresh bridge + an odometry-capable model.

### OFFBOARD stream robustness + sensor fusion / VIO (2026-07-06)

**OFFBOARD mid-flight dropout — FIXED.** The vehicle would fall out of
OFFBOARD to ALTCTL a few seconds into forward flight. Cause: setpoints were
published from inside the flight/step loop, so any slow step (a service
call, MAVROS lagging under load, or the env's cv_bridge image decode between
steps) let the >2 Hz OFFBOARD gate lapse. Fixed in BOTH
`manual_control.py` and `ros_bridge_node.py` by moving setpoint publishing to
a dedicated **50 Hz background timer** (on the MultiThreadedExecutor) that
republishes the latest commanded setpoint independently of the flight/step
logic, and re-commands OFFBOARD if PX4 ever drops it. Verified: fpv5 now
holds OFFBOARD+armed for a full ~30 s flight (~13 m forward) with no dropout.
`manual_control.py`'s forward manoeuvre also switched to velocity-xy +
**position-z** (locked altitude) instead of a velocity-z altitude hold.

**Sensor fusion / why the EKF still wanders — and VIO.** PX4's EKF2 is a
multi-sensor fusion estimator; here it fuses **IMU** (prediction) + **GPS**
(EKF2_GPS_CTRL) + **barometer** (height) + **magnetometer** (heading), and it
is *already* configured to accept **external vision** (EKF2_EV_CTRL defaults
to 15) — but nothing was feeding vision, so it fell back on the SITL GPS,
which drifts badly. Result: the vehicle flies correctly (Gazebo ground truth
shows clean forward motion down the y=24 centreline) while the **EKF estimate
diverges**, especially in z (seen reading 9–12 m during a ~2.5 m flight).
This matters because STAR-Nav is a **GPS-denied** system — the real drone
holds position from VIO, not GPS; the SACR/CAMR modules run on the RL *agent*
side and never reach PX4's flight-control EKF.

The fix is to feed a VIO-quality pose into the EKF, simulating the onboard
VIO. Added **`ros_gazebo_bridge/vio_bridge.py`** (+ `vio_bridge` console
script): it republishes the Gazebo ground-truth odometry (a stand-in for a
perfect VIO) to MAVROS's **mocap** input (`/mavros/mavros/mocap/tf`, →
ATT_POS_MOCAP → PX4 external vision) — mocap is the external-pose plugin
actually loaded on this ros-humble-mavros build (vision_pose isn't). Two
integration gotchas found and encoded:
1. **Frame offset.** gz odometry is in *world* coords (y≈24 on the
   centreline) but the EKF local origin is at the *spawn* point, so raw world
   coords give a tens-of-metres innovation → the EKF gates/rejects it (and
   with EKF2_HGT_REF=Vision and no accepted data, the local-position estimate
   drops out entirely — observed). `vio_bridge.py` now subtracts the spawn
   origin (`--origin-x/y/z`, default the corridor's `PX4_GZ_MODEL_POSE`
   1,24,0.3) so the fed pose is in PX4's local frame with small innovation.
2. **EKF params are boot-time.** Setting EKF2_HGT_REF / EKF2_GPS_CTRL live
   does not re-init the EKF; they must be set at boot (airframe or saved
   params). For a clean GPS-denied config, boot with GPS off + VIO on with
   the bridge already streaming (so the EKF has horizontal aiding as it
   converges). NOTE: after any px4-gazebo restart, MAVROS does **not**
   auto-reconnect over MAVLink UDP — `docker restart star_nav_mavros` (or
   relaunch the node) to restore the connection.

**Status:** `vio_bridge.py` + the OFFBOARD fix are committed and the fusion
mechanism is confirmed (EKF2_EV_CTRL already on; mocap plugin present). The
full GPS-denied EKF-locks-onto-VIO run is **not yet closed end-to-end** — it
needs a boot with the GPS-off/VIO-on param set and the bridge running at
init, which is the clear next step.

### Clean stable flight via built-in visual odometry — SOLVED (2026-07-06)

The altitude estimate is now **accurate and stable** and the drone flies
cleanly in every direction (take-off, hover, forward/back, left/right,
up/down, yaw) with altitude tracking its setpoint. Two fixes, in order of
importance:

1. **`<dimensions>3</dimensions>` on the OdometryPublisher (the root cause).**
   The gz OdometryPublisher **defaults to 2 (planar x/y, z always 0)**. PX4's
   own GZBridge subscribes to `/model/<name>/odometry_with_covariance` and
   republishes it to the EKF as `vehicle_visual_odometry` (built-in — no
   MAVROS mocap/vision plugin needed; it does the ENU->NED / FLU->FRD
   conversion itself). With a 2D feed the vision **altitude is 0 forever**, so
   with vision height selected the EKF fuses a constant-0 height and the z
   estimate diverges/offsets (the drone believed it was ~5 m up while sitting
   on the ground, so it "descended" and never took off). `dimensions=3` gives
   real z. Added to fpv5/pavo_femto `model.sdf` **and** their `_gen.py`.
2. **`EKF2_HGT_REF=3` (vision height)** in both airframes. The default (1,
   GPS) drifts/diverges here (z climbing to 70+ m in a ~2 m flight). Vision
   height + the 3D odometry gives a clean, drift-free altitude. `EKF2_EV_CTRL`
   already defaults to 15 (EV horiz+vert pos + vel + yaw), so no change there.

Net: this is the sim's stand-in for the real drone's **VIO** (a GPS-denied
position source), reached through PX4's *built-in* gz visual-odometry path
rather than the earlier hand-rolled `vio_bridge.py` mocap feed (which still
works and is kept, but is no longer needed for stable flight). PX4 gz does
NOT support optical-flow or lidar/rangefinder sensors without recompiling its
C++ GZBridge (it only bridges IMU/baro/GPS/airspeed/odometry) — the odometry
path is the practical equivalent.

**Two control files (both "perfect control"; the STAR-Nav policy is a
separate future "research control" file):**
- `manual_control.py` — scripted. `--maneuver allaxes` (default) flies every
  direction in sequence (the clean 6-DOF demo); `--maneuver corridor` does the
  takeoff->forward->hover->land corridor run.
- `keyboard_control.py` (+ `keyboard_control` console script) — **live WASD
  teleop**: w/s pitch(fwd/back), a/d roll(left/right), q/e yaw, r/f throttle
  (up/down), space hover, t takeoff, l land, x quit. Must be run in an
  interactive TTY (`docker exec -it ...`) — it puts stdin in raw mode.

**Reminder: keep the containers stopped when not actively testing** — the
PX4+Gazebo container runs Gazebo lockstep physics at ~130% CPU
continuously. `docker stop star_nav_px4_gazebo star_nav_mavros` when idle;
`docker start` + re-launch the entrypoint when you need them again. Also:
**MAVROS needs `docker restart star_nav_mavros` after any px4-gazebo
restart** — it does not auto-reconnect the MAVLink UDP link.

### First run on a native-Linux GPU laptop (not WSL2) — Docker + real GPU rendering (2026-07-06)

First time this repo's Docker/Gazebo stack ran on an actual native Ubuntu
box (`aksalspace-Thin-15-B12UCX`, i5-12450H + NVIDIA RTX 2050 4GB, hybrid
Optimus graphics) rather than the WSL2 rig documented above. Python side
(`torch==2.5.1+cu121` in a `.venv --system-site-packages`, rest of
`requirements.txt`, `tests/test_shapes.py` all green) and the full
Docker/Gazebo/PX4/MAVROS stack were both bootstrapped from scratch here.

- **`Dockerfile.px4_gazebo` build kept failing/stalling on the
  `arm-none-eabi-gcc` NuttX toolchain download** (`Tools/setup/ubuntu.sh`
  fetching from `armkeil.blob.core.windows.net`) — repeated
  `ReadTimeoutError`s, then a run that stalled for 75+ minutes across 5
  internal retries at ~83% before giving up. This toolchain is only for
  compiling firmware for real flight controllers, never for the
  posix/SITL target this image builds (same reasoning the Dockerfile
  already documents for deiniting the NuttX submodules). Fixed by adding
  `--no-nuttx` to the `ubuntu.sh` invocation — confirmed via PX4's own
  script source that this flag exists and skips exactly this download.
  Rebuild succeeded promptly afterward with the rest of the layer cache
  intact.
- **GUI rendering: genuinely works here, unlike WSL2 — but needed one
  more fix for this specific hybrid-graphics laptop.** `docker-compose.yml`
  already requests the GPU + forwards X11; with that alone, `gz sim`'s
  GLX client picked the host X server's *default* GPU, which on this
  Optimus laptop is the Intel iGPU (`glxinfo` confirmed: default renderer
  is Intel, but `__NV_PRIME_RENDER_OFFLOAD=1
  __GLX_VENDOR_LIBRARY_NAME=nvidia glxinfo` switches it to the NVIDIA
  GPU) — so it tried Mesa's `iris` driver and failed (`MESA: error:
  Failed to query drm device`, no `/dev/dri` passthrough for that path).
  Fixed by adding those same two PRIME-offload env vars to the
  `px4-gazebo` service in `docker-compose.yml`. Confirmed via
  `ogre2.log` afterward: real `GL_VERSION = 4.6.0 NVIDIA 595.71.05`,
  `GL_RENDERER = NVIDIA GeForce RTX 2050` — genuine hardware-accelerated
  rendering, not the `llvmpipe` software fallback WSL2 was permanently
  stuck on. Any other Optimus/hybrid-graphics native-Linux host will
  likely need the same two env vars.
- Remaining, non-blocking GUI oddity: `ogre2.log` still logs `HLMS
  Library path '.../Hlms/Gz' has no piece files` (the piece files are
  actually present one level down, in `Hlms/Gz/{Pbs,SolidColor,
  SphericalClipMinDistance}/` — dpkg confirms `libgz-rendering7-ogre2-dev`
  installed them correctly) and `OGRE EXCEPTION: Fixed Function pipeline
  is no longer allowed` for a handful of materials (`Default/TransGreen`,
  a few numeric `scene::Material(N)` IDs) — these read as GUI-only
  overlay/decoration materials (e.g. collision-shape visualization), not
  the main world/vehicle rendering, since the earlier offscreen
  `<save>`-camera renders (tree, drone) already proved the core scene
  renders correctly under equivalent conditions. Not root-caused further.
- **`env.ros.gt_odom_topic` in `configs/default.yaml` still defaulted to
  `/model/x500_depth_0/odometry`** — updated to `/model/fpv5_0/odometry`
  to match `fpv5` (the model actually flown here; `x500_depth` has no
  `OdometryPublisher`).
- **`PX4_GZ_WORLD` must be the in-container path, not the host path.**
  `docker-compose.yml` mounts `../worlds:/worlds:ro` and passes
  `PX4_GZ_WORLD` straight from the host shell's env var into the
  container — set it to `/worlds/scenario_a.sdf`, not the host's
  absolute path, or `entrypoint_px4_gazebo.sh`'s `cp "$PX4_GZ_WORLD" ...`
  fails with `No such file or directory`.
- **`docker compose stop`/`rm` need `PX4_GZ_WORLD` set too, not just
  `up`.** Compose interpolates/validates every service's environment
  block regardless of which service subcommand targets, so omitting the
  (required, `:?`) var makes `stop`/`rm` silently no-op with an
  interpolation error — the old container is left running. This is easy
  to miss because a **crashed `gz sim` child process does not stop the
  container**: `entrypoint_px4_gazebo.sh` backgrounds `gz sim &` and its
  `trap ... EXIT` only fires when the entrypoint's own foreground process
  (the `make px4_sitl gz_<model>` / `bin/px4`) exits, so Docker still
  reports the container `Up` with PX4 alive even though Gazebo itself
  segfaulted and nothing is simulating anymore. Always check
  `ps aux | grep "gz sim"` *inside* the container, not just `docker ps`,
  if things seem frozen.
- **A real, intermittent `gz-sim` segfault** in
  `gz::sim::v7::systems::MulticopterMotorModel::PreUpdate` (crash deep in
  a `std::string` assign) was observed once, triggered during `fpv5`'s
  **LEFT (+y roll)** maneuver via `manual_control.py --maneuver allaxes`
  — froze the whole sim (PX4's own process stayed up per the note above)
  and cascaded ~30s later into MAVROS `CON: Lost connection, HEARTBEAT
  timed out` once heartbeats genuinely stopped, which is what actually
  froze/zeroed out every subsequent logged position, not a LEFT/RIGHT
  axis bug. Investigated: `fpv5_gen.py`'s inertia (`ixx_iyy` shared
  between both axes) and the airframe's `CA_ROTOR*_PX/PY` +
  `MC_ROLLRATE_*`/`MC_PITCHRATE_*` gains are all symmetric between roll
  and pitch — no config asymmetry found. Re-tested twice more on a fresh
  container: `--maneuver corridor` (no lateral) completed cleanly, and a
  full `--maneuver allaxes` retry **also completed cleanly including
  LEFT and RIGHT** (smooth ~2.7-2.9m lateral travel each way, altitude
  held throughout, full 6-DOF + landing all fine). So the crash is
  non-deterministic/intermittent, not a systematic roll-axis or
  fpv5-config problem — most likely a rare upstream `gz-sim` race
  condition in the motor-model plugin's ECS entity/component access.
  Not root-caused further (would need `gdb`/core-dump analysis on the
  compiled plugin). Worth knowing this can happen; a retry has so far
  always succeeded.
- **Separate quirk: EKF health does not self-recover after a flight+land
  cycle in this session.** After the `corridor` maneuver's landing, PX4
  logged `Navigation failure! Land and recalibrate sensors` followed by
  persistent `Preflight Fail: position estimate error`, which blocked
  re-arming for a subsequent test until `px4-gazebo` was recreated fresh.
  Not investigated further — avoid chaining multiple flight tests
  back-to-back in one container session without a restart in between.

## Working conventions for this repo

- Keep commit messages plain: no extra trailers or attributions beyond
  the local git author — the user wants a clean history in this repo.
- Match the paper's exact variable names and equations when touching
  `star_nav/models/*` — each file's docstring cites the section/equation
  it implements; keep new code consistent with that mapping.

#!/usr/bin/env bash
# ============================================================================
# DECOUPLED deploy: fly a Mock-exported trajectory as position inference in real
# PX4+Gazebo, over a trunk field that MATCHES the Mock world the path came from.
#
# Unlike deploy_best.sh (which runs perception+policy LIVE and wanders due to the
# sim2real belief gap), this decouples perception from control: the intelligence
# (the path through the zigzag) is baked offline in Mock by
# scripts/export_mock_trajectory.py, and Gazebo just position-tracks it -- which
# PX4 offboard does reliably. Because the world is byte-matched (same trunk xy),
# the flight stays collision-free. "samakan env nya biar gampang."
#
# Prereq (run once on the host, already committed under renders/deploy/ +
# ros_gazebo_bridge/worlds/zigzag_deploy.*):
#     python scripts/export_mock_trajectory.py --out renders/deploy/zigzag
#
# It:
#   1. brings up the sim with the MATCHED world (PX4_GZ_WORLD=zigzag_deploy.sdf),
#   2. FRESH-restarts PX4 (clean EKF -> arming allowed; see deploy_best.sh),
#   3. syncs the live repo code + inference into the ros-bridge container,
#   4. bridges ground-truth odometry for the spawned model,
#   5. runs scripts/fly_inference_gazebo.py (arm -> climb -> track inference).
#
# Usage:   ./scripts/deploy_inference.sh
#          LAND=1 ./scripts/deploy_inference.sh          # descend+disarm at goal
#          PX4_SIM_MODEL=pavo_femto ./scripts/deploy_inference.sh
# Monitor: ./scripts/monitor.sh deploy_wp
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"

COMPOSE="ros_gazebo_bridge/docker/docker-compose.yml"
PX4C="star_nav_px4_gazebo"
ROSC="star_nav_ros_bridge"
MODEL="${PX4_SIM_MODEL:-fpv5}"
INFERENCE="${INFERENCE:-renders/deploy/zigzag_inference.csv}"
PKG="/ros2_ws/install/ros_gazebo_bridge/lib/python3.10/site-packages/ros_gazebo_bridge"
WORLDS_PKG="/ros2_ws/install/ros_gazebo_bridge/lib/python3.10/ros_gazebo_bridge/worlds"
LAND_FLAG=""; [ "${LAND:-0}" = "1" ] && LAND_FLAG="--land"

export PX4_GZ_WORLD=/worlds/zigzag_deploy.sdf
export PX4_SIM_MODEL="${MODEL}"
# Spawn at the corridor start (x=1); the flyer measures the exact world->local
# offset after takeoff, so a small spawn/inference mismatch is absorbed.
export PX4_GZ_MODEL_POSE="${PX4_GZ_MODEL_POSE:-1,24,0.3,0,0,0}"
export HEADLESS="${HEADLESS:-1}"

SRC_PREFIX="${SRC_PREFIX:-renders/deploy/zigzag}"   # tracked canonical export

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
xhost +local:docker >/dev/null 2>&1 || true

# worlds/ is gitignored (generated artifacts), so install the matched world from
# the tracked export into the /worlds bind-mount before compose reads it.
# PX4's gz-bridge derives the gz world NAME from the .sdf filename (-w zigzag_deploy),
# so the SDF's <world name="..."> MUST equal the basename or PX4 waits for Gazebo
# forever (scenario_a.sdf works precisely because its world is named "scenario_a").
say "0/5  Installing matched world into worlds/ (from ${SRC_PREFIX}) ..."
[ -f "${SRC_PREFIX}.sdf" ] || { echo "missing ${SRC_PREFIX}.sdf -- run: python scripts/export_mock_trajectory.py --out ${SRC_PREFIX}"; exit 1; }
sed 's/<world name="[^"]*">/<world name="zigzag_deploy">/' "${SRC_PREFIX}.sdf" > ros_gazebo_bridge/worlds/zigzag_deploy.sdf
cp "${SRC_PREFIX}.world.json" ros_gazebo_bridge/worlds/zigzag_deploy.world.json

say "1/5  Bringing up the sim with the MATCHED zigzag world ..."
docker compose -f "$COMPOSE" up -d px4-gazebo ros-bridge >/dev/null

# person_worker is NOT baked into the px4-gazebo image (only oil_palm is), so
# model://person_worker fails to resolve and the workers don't spawn. Copy it into
# the container's gz models dir BEFORE the step-2 restart reloads the world. We
# install a COLLISION-STRIPPED copy: the crowd is a visual/matched replay (the
# policy already avoided them offline), so a physical person mesh must not knock
# the drone down -- with collision on, the drone clipped a worker at the left
# dogleg and crashed to the ground.
GZMODELS="/opt/PX4-Autopilot/Tools/simulation/gz/models"
if grep -q "<name>worker" ros_gazebo_bridge/worlds/zigzag_deploy.sdf; then
  PWTMP="$(mktemp -d)/person_worker"; cp -r "$REPO/ros_gazebo_bridge/px4_models/person_worker" "$PWTMP"
  python3 -c "import re,sys; p='$PWTMP/model.sdf'; s=open(p).read(); open(p,'w').write(re.sub(r'\s*<collision[\s\S]*?</collision>','',s))"
  docker cp "$PWTMP" "$PX4C:$GZMODELS/person_worker" 2>/dev/null \
    && echo "  installed collision-free person_worker model into px4-gazebo"
fi

say "2/5  Fresh-restarting PX4 (reset EKF so arming is allowed) ..."
docker restart "$PX4C" >/dev/null
for _ in $(seq 1 30); do
  docker logs "$PX4C" 2>&1 | grep -q "Ready for takeoff" && break; sleep 2
done
docker logs "$PX4C" 2>&1 | grep -q "Ready for takeoff" \
  || { echo "PX4 not ready -- check: docker logs $PX4C"; exit 1; }
docker restart "$ROSC" >/dev/null
sleep 8
for _ in $(seq 1 15); do
  docker exec "$ROSC" bash -lc 'source /opt/ros/humble/setup.bash; timeout 3 ros2 topic echo --once /mavros/state 2>/dev/null' \
    | grep -q 'connected: true' && break; sleep 3
done

say "3/5  Syncing live repo code + inference into $ROSC ..."
docker cp "$REPO/star_nav"  "$ROSC:/workspace/" 2>/dev/null
docker cp "$REPO/scripts"   "$ROSC:/workspace/" 2>/dev/null
docker cp "$REPO/configs"   "$ROSC:/workspace/" 2>/dev/null
docker exec "$ROSC" mkdir -p /workspace/renders/deploy 2>/dev/null
docker cp "$REPO/renders/deploy/." "$ROSC:/workspace/renders/deploy/" 2>/dev/null
docker cp "$REPO/ros_gazebo_bridge/ros_gazebo_bridge/env.py" "$ROSC:$PKG/env.py" 2>/dev/null
docker cp "$REPO/ros_gazebo_bridge/ros_gazebo_bridge/ros_bridge_node.py" "$ROSC:$PKG/ros_bridge_node.py" 2>/dev/null
docker exec "$ROSC" bash -lc "mkdir -p $WORLDS_PKG" 2>/dev/null
docker cp "$REPO/ros_gazebo_bridge/worlds/zigzag_deploy.world.json" "$ROSC:$WORLDS_PKG/zigzag_deploy.world.json" 2>/dev/null

say "4/5  Bridging ground-truth odometry + worker cmd_vel ..."
NW=$(grep -c "<name>worker" ros_gazebo_bridge/worlds/zigzag_deploy.sdf || echo 0)
WORKER_BRIDGES=""
for i in $(seq 0 $((NW - 1))); do
  [ "$NW" -gt 0 ] && WORKER_BRIDGES="$WORKER_BRIDGES /model/worker${i}/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist"
done
docker exec -d "$ROSC" bash -lc "source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash 2>/dev/null; \
  exec ros2 run ros_gz_bridge parameter_bridge \
    /model/${MODEL}_0/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry ${WORKER_BRIDGES} > /tmp/gt_odom_bridge.log 2>&1"
echo "  bridged gt_odom + ${NW} worker cmd_vel topics"
sleep 5

say "5/5  Flying the exported inference (${INFERENCE}) + recording ..."
mkdir -p logs renders/deploy
docker exec "$ROSC" bash -lc "source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash 2>/dev/null; \
  cd /workspace && python3 scripts/fly_inference_gazebo.py --inference ${INFERENCE} ${LAND_FLAG} \
    --actors ${SRC_PREFIX}_actors.npy --drone-path ${SRC_PREFIX}_traj.npy \
    --log-traj /workspace/renders/deploy/flight_gt.csv \
    --save-frames /workspace/renders/deploy/flight_fpv.npy 2>&1" 2>&1 \
  | grep -vE "FutureWarning|weights_only|RuntimeWarning|umr_sum" | tee logs/deploy_wp.log

say "Copying recordings out of the container ..."
docker cp "$ROSC:/workspace/renders/deploy/flight_gt.csv"  renders/deploy/flight_gt.csv  2>/dev/null || echo "  (no flight_gt.csv)"
docker cp "$ROSC:/workspace/renders/deploy/flight_fpv.npy" renders/deploy/flight_fpv.npy 2>/dev/null || echo "  (no flight_fpv.npy)"

say "Rendering video (FPV + top-down over matched world) ..."
python3 scripts/render_flight_video.py --gt renders/deploy/flight_gt.csv \
  --fpv renders/deploy/flight_fpv.npy --world "${SRC_PREFIX}.world.json" \
  --inference "${INFERENCE}" --actors "${SRC_PREFIX}_actors.npy" --drone-mock "${SRC_PREFIX}_traj.npy" \
  --out renders/deploy/gazebo_flight 2>&1 | grep -v FutureWarning || true

say "done. (path <- Mock best zigzag policy; world <- matched zigzag_deploy.sdf)"

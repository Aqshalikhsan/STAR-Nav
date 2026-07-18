# STAR-Nav — novelty perception & Gazebo deploy (part-by-part record)

End-to-end record of the novelty modules and the Mock→Gazebo deploy, so nothing
is lost. Three helper scripts wrap the flow: `scripts/train_mock.sh`,
`scripts/monitor.sh`, `scripts/deploy_best.sh`.

Pipeline: **SACR** (structure-aware corridor rep) → **CAMR** (consistency-aware
memory) → **AGSS-PPO** (PPO + adaptive geometric safety shield). Perception is
pretrained supervised and **frozen**; the policy is a domain-agnostic MLP over
the 256-d belief `h_t`, which is what lets a Mock-trained policy run in Gazebo.

---

## 0. Novelty components (what makes it novel)

| Module | Novelty head | Loss | Feeds |
|---|---|---|---|
| SACR | aleatoric **depth uncertainty** (per-region log-variance) | heteroscedastic NLL `L_unc` | shield `σ` term |
| CAMR | **anticipatory occupancy** (future [left,right] actor) | BCE `L_occ` | shield `occ` term |
| AGSS | **asymmetric adaptive shield** | — | `d_safe_side = d0 + α·c + β·σ + γ·occ` |

Dims: `z_struct_aug = 134` (struct 128 + pooled depth 3 + depth-logvar 3);
CAMR input `= 147` (134 + pose 7 + imu 6); belief `= 256` (2×128, fwd+rev LSTM).
Config flags in `configs/default.yaml`: `sacr.depth_uncertainty`,
`camr.predict_occupancy`, `agss_ppo.beta_unc`, `agss_ppo.gamma_occ`.
(Gotcha, already fixed: `agss_ppo.gamma` is the **PPO discount**; the shield
weight is `gamma_occ` — do not collide them.)

---

## 1. Mock training  → `scripts/train_mock.sh`

`MockCorridorEnv` is a dependency-free 2-D ray-cast corridor with per-episode
random tree layout and moving CLASS_ACTOR workers. One command runs both phases:

- **Phase 1 (perception)** — if `checkpoints/mock/{sacr,camr}.pt` are not cached,
  `train_ppo.py` auto-collects Mock episodes and trains+freezes the novelty SACR
  and CAMR on them.
- **Phase 2 (policy)** — AGSS-PPO **curriculum**, 12 stages (10 m → 48 m, then up
  to 5 workers). A stage advances after `--advance-streak` iters at/above
  `--advance-at` success. Writes `checkpoints/mock/`: `ppo.pt` (deploy this),
  `ppo_best.pt`, `ppo_last.pt` (resume).

```bash
./scripts/train_mock.sh                       # fresh curriculum (2000 iters)
ITERS=800 ./scripts/train_mock.sh             # shorter
RESUME=checkpoints/mock/ppo_last.pt ./scripts/train_mock.sh
START_STAGE=7 ./scripts/train_mock.sh         # jump into the curriculum
EXTRA="--no-occupancy" ./scripts/train_mock.sh   # ablation
```

---

## 2. Gazebo perception recapture  (novelty needs GT the old datasets lack)

The novelty heads need supervision the pre-novelty Gazebo datasets never
recorded: **full metric depth maps** (SACR `L_depth` + `L_unc`) and **actor
pixels** (CAMR occupancy). So recapture with the glide-rig collector, which now
saves `depth` too:

```bash
# generate the labeled + worker capture world (host)
python3 ros_gazebo_bridge/perception_capture/capture_world_gen.py \
  --base-world ros_gazebo_bridge/worlds/scenario_a.sdf \
  --out ros_gazebo_bridge/perception_capture/capture_scenario_a.sdf \
  --worker-xs "10,16,22,28,34,40"

# standalone gz render container (person_worker lives in the repo, so MOUNT px4_models)
docker run -d --name caprig --network host --gpus all \
  -e GZ_SIM_RESOURCE_PATH=/opt/PX4-Autopilot/Tools/simulation/gz/models:/px4_models \
  -v $PWD/ros_gazebo_bridge/perception_capture/capture_scenario_a.sdf:/w.sdf:ro \
  -v $PWD/ros_gazebo_bridge/px4_models:/px4_models:ro \
  --entrypoint bash docker-px4-gazebo:latest -c 'gz sim -s -r /w.sdf'

# collector container (start FRESH after caprig is up -- gz-transport discovery is stale otherwise)
docker run -d --name capcli --network host --gpus all \
  -v $PWD/ros_gazebo_bridge/perception_capture:/pc -v $PWD/data:/out \
  --entrypoint bash docker-ros-bridge:latest -c 'sleep infinity'
docker exec capcli bash -lc 'source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash; \
  python3 /pc/collect_dataset.py --out /out/gazebo_novelty_p1.npz --num-workers 6 \
    --capture-hz 6 --max-x 44'
```

- One glide = one pass; the rig does NOT auto-reset to x=2, so **`docker restart
  caprig` between passes** for a fresh spawn. Concatenate passes → one npz (each
  pass restarts at x=2, so `split_episodes` cleanly separates them).
- Result used here: `data/sacr_gazebo_novelty.npz` — 779 frames, 611 with
  actors, depth finite [0, 20]. (`data/*.npz` is gitignored — regenerable.)

---

## 3. Retrain Gazebo perception with novelty heads

`scripts/train_sacr.py` and `scripts/train_camr.py` now build the novelty heads
from config and consume the new GT:

- **SACR** loads `depth` and passes `depth_target` → real `L_depth` (metric depth
  supervision; previously the Gazebo depth net was unsupervised/random, so the
  shield's `d_left/d_right` were garbage) + `L_unc`. `L_unc` can go **negative** —
  that is correct (confident, accurate heteroscedastic NLL), not a bug.
- **CAMR** derives the anticipatory occupancy target **from the seg CLASS_ACTOR
  pixels** (left/right presence over the next `OCC_HORIZON` frames) — no
  world-frame actor coordinates needed — and trains the BCE `L_occ` head.

```bash
python3 scripts/train_sacr.py --data data/sacr_gazebo_novelty.npz --epochs 40 --out-dir checkpoints/gazebo
python3 scripts/train_camr.py --data data/sacr_gazebo_novelty.npz \
  --sacr-ckpt checkpoints/gazebo/sacr.pt --epochs 60 --out-dir checkpoints/gazebo
```

Result: `checkpoints/gazebo/{sacr,camr}.pt` now **134/147-dim** — matching the
policy's belief space. Pre-novelty ckpts saved in
`checkpoints/gazebo/prenovelty_backup/`. (SACR val acc ~96.5%, `L_depth`
8.8→1.3; CAMR `L_occ` 0.69→0.36.)

---

## 4. Deploy the policy in Gazebo  → `scripts/deploy_best.sh`

```bash
./scripts/deploy_best.sh                       # 1 episode, 150 steps
EPISODES=3 MAXSTEPS=200 ./scripts/deploy_best.sh
```

Chain: Gazebo-novelty SACR+CAMR → `mock/ppo.pt` policy → adaptive AGSS shield →
`GazeboROSEnv` (arm → takeoff → altitude-hold → fly). The script encodes every
gotcha below so it is one command.

### Deploy gotchas (all real — the script handles them)

1. **Mock policy is fixed-altitude.** `MockCorridorEnv.step()` unpacks `v_z`
   (action[2]) but never applies it, so the policy gets zero gradient there and
   emits untrained noise — applied raw it rockets the drone to ~80 m. Fix in
   `ros_gazebo_bridge/env.py`: `reset()` now `_takeoff()`s to 2.2 m (the
   capture-rig height, so the view is in-distribution) before policy control, and
   `step()` **ignores action[2]** and holds altitude with a P-loop.
2. **Arming denied after a previous landing.** PX4's EKF doesn't self-recover
   after flight+land (`Preflight Fail: position estimate error`). Fix: fresh PX4
   (`docker restart star_nav_px4_gazebo`, a restart NOT a recreate so docker-cp'd
   files survive), then restart ros-bridge for MAVROS to reconnect.
3. **gt_odom model-name mismatch.** The baked bridge publishes
   `/model/x500_depth_0/odometry`, but PX4 spawns `fpv5_0`; the env waits on
   `/model/fpv5_0/odometry`. Fix: run one more `parameter_bridge` for it.
4. **Stale baked container.** The ros-bridge image's `/workspace/star_nav` is
   pre-novelty and lacks `scripts/`; the installed package lacks
   `worlds/scenario_a.world.json`. Fix: `docker cp` the live code + world json in.
5. `PX4_GZ_WORLD` must be the **in-container** path `/worlds/scenario_a.sdf`
   (compose bind-mounts `../worlds:/worlds`), not the host path.

---

## 5. Monitoring  → `scripts/monitor.sh`

```bash
./scripts/monitor.sh mock     # Mock PPO curriculum (success rate / stage)
./scripts/monitor.sh sacr     # Gazebo SACR retrain (L_SACR / acc)
./scripts/monitor.sh camr     # Gazebo CAMR retrain (L_CAMR / L_occ)
./scripts/monitor.sh deploy   # Gazebo deploy (reward / agss interventions)
```

Prints the last key metric lines then follows the log live.

---

## 6. Status

The full deploy pipeline runs end-to-end in real Gazebo: novelty perception on
real imagery, policy + adaptive shield, arm / takeoff / altitude-hold (z pinned
~2.2 m) and flight. Checkpoints load with matching dims.

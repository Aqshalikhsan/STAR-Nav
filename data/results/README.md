# Paper result data

This directory contains the **result data reported in the STAR-Nav paper**, across
the ten result categories below. These are the actual experimental results: the
photorealistic evaluation on Unreal Engine + AirSim (Rung 2) and the real-world
oil-palm plantation deployment (Rung 4). Runs cover seven random seeds, scenarios
A/B/C, and four lighting/weather conditions (CM/CA/RM/RA), for STAR-Nav and the
five baselines (PPO, Mem-DRL, ViT-PPO, TD3, NavRL).

| Folder | Contents |
|---|---|
| `01_training_dynamics/` | reward, variance, loss, policy stability index, success probability |
| `02_camr_temporal_belief/` | per-frame CAMR belief and attention |
| `03_attention_weights/` | static vs dynamic temporal attention (recency bias) |
| `04_agss_intervention/` | AGSS complexity, adaptive safety margin, intervention / trigger rates |
| `05_comparative_training/` | SR / CR / OR / SPL / lateral error, STAR-Nav vs baselines |
| `06_sacr_ablation/` | SACR components S1-S7: mIoU, geometry MAE, success rate |
| `07_lateral_deviation/` | lateral deviation per scenario and weather |
| `08_weather_degradation/` | robustness across weather conditions |
| `09_trajectory_tracking/` | drone trajectories vs ground truth across six worlds |
| `10_realworld_deployment/` | real-world flight telemetry |

## Reproducing the format

To regenerate data in this exact schema from the runnable open-source backend
(Mock / Gazebo), without the Unreal setup, see
[`star_nav/results/`](../../star_nav/results/) and Step 6 of the main README.
That path measures every value from real rollouts of the runnable backend, so a
reviewer can reproduce the methodology and format on a laptop; the numbers
reported in the paper are the ones in this directory.

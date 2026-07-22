# Step 6: Generate the paper's result set

After deploying (Steps 0–5), this layer reproduces the paper's result set in the
exact parameter and metric schema, by **running the real pipeline** on the
runnable backend (Mock / Gazebo). Every value is measured from a rollout or a
perception pass; nothing is synthesized. It lets a reviewer reproduce the format
and methodology of every table and figure without the Unreal setup.

## The 10 categories

| Folder | Metric | Driven by |
|---|---|---|
| `01_training_dynamics` | reward / variance / loss / PSI / success | `train_baseline.py` (logged during training) |
| `02_camr_temporal_belief` | per-frame belief + attention | STAR-Nav |
| `03_attention_weights` | static vs dynamic recency bias | STAR-Nav (occlusion saliency over the CAMR window) |
| `04_agss_intervention` | complexity / adaptive margin / trigger | STAR-Nav (`AGSSShield`) |
| `05_comparative_training` | SR / CR / OR / SPL / lateral | STAR-Nav + baselines |
| `06_sacr_ablation` | mIoU / Geo. MAE / SR | SACR variants S1–S7 |
| `07_lateral_deviation` | lateral deviation | STAR-Nav + baselines |
| `08_weather_degradation` | degradation vs clear | STAR-Nav + baselines |
| `09_trajectory_tracking` | x, y paths | STAR-Nav + baselines |
| `10_realworld_deployment` | flight telemetry | `hardware/telemetry_logger.py` (data stays real flights) |

## Run

```bash
# STAR-Nav categories (uses your trained checkpoints/):
python scripts/generate_results.py --config configs/default.yaml \
    --checkpoint-dir checkpoints --out results_out \
    --categories 02,03,04,06,07,08,09 --episodes 20

# Train a baseline, then use it in the comparison categories:
python scripts/train_baseline.py --method TD3 --seed 1 --iterations 5000 \
    --ckpt-out ckpts/TD3_seed1.pt --curve-out results_out
python scripts/generate_results.py --method TD3 --baseline-ckpt ckpts/TD3_seed1.pt \
    --categories 05,07,08,09 --out results_out
```

Baselines: `PPO`, `ViTPPO`, `MemDRL`, `NavRL`, `TD3` (`star_nav/baselines/`).

## Notes

- Numbers are measured, so they depend on your trained checkpoints and training
  runs; the per-variant SACR ablation SR, the comparison baselines, and the
  10 000-episode training curves require the real training campaigns.
- `10_realworld_deployment` data comes from real hardware flights; the code
  provides the logger schema only.

# Baseline methods

The comparison baselines (PPO, Mem-DRL, ViT-PPO, TD3, NavRL) reported in the
paper were run using each method's **original author implementation**. The
reported numbers in [`data/results/`](../../data/results/) come from those
original runs.

The code in this folder is a **simplified, contextually-equivalent
reimplementation** of each method, provided so the comparison is runnable on the
open-source Mock / Gazebo backend (same monocular RGB + pose + IMU observation,
same 4-DoF action, comparable network capacity) without setting up each original
codebase. It is not a bit-exact reproduction of the source works.

## Sources (for rechecking)

| Method | Paper | Original code |
|---|---|---|
| PPO | Schulman et al., *Proximal Policy Optimization Algorithms*, arXiv:1707.06347 (2017). [DOI](https://doi.org/10.48550/arXiv.1707.06347) | https://github.com/openai/baselines |
| TD3 | Fujimoto et al., *Addressing Function Approximation Error in Actor-Critic Methods*, ICML 2018, arXiv:1802.09477. [DOI](https://doi.org/10.48550/arXiv.1802.09477) | https://github.com/sfujim/TD3 |
| Mem-DRL | Haddad & Khudher, *Dual-Memory Architecture for Robust UAV Navigation Integrating LSTM and Transformer within a PPO Framework*, IJRCS 5(5) 2025. [DOI](https://doi.org/10.31763/ijrcs.v5i5.2208) | (add the repository you ran) |
| ViT-PPO (DTPPO) | Wei et al., *DTPPO: Dual-Transformer Encoder-Based PPO for Multi-UAV Navigation*, Drones 8(12) 2024, 720. [DOI](https://doi.org/10.3390/drones8120720) | (add the repository you ran) |
| NavRL | Xu, Han, Shen, Jin, Shimada, *NavRL: Learning Safe Flight in Dynamic Environments*, IEEE 2024. | https://github.com/Zhefan-Xu/NavRL |

The DOI links resolve to the exact papers cited in the manuscript; a reviewer can
follow them to the source works and their code.

## What is simplified here (contextually the same method)

- **PPO** (`ppo.py`) clipped-surrogate PPO with GAE and a CNN encoder. Faithful
  to the algorithm.
- **TD3** (`td3.py`) twin critics, target policy smoothing, delayed actor, target
  networks. Faithful to the algorithm.
- **Mem-DRL** (`mem_drl.py`) dual-memory recurrent policy (fast working + slow
  contextual LSTM) trained with PPO. The source work also integrates a
  Transformer inside the memory; this version is LSTM-only.
- **ViT-PPO / DTPPO** (`vit_ppo.py`) dual-transformer policy (spatial patch
  transformer + temporal window transformer) trained with PPO. A single-agent,
  reduced-depth version of the multi-UAV DTPPO.
- **NavRL** (`navrl.py`) PPO with a velocity-obstacle-inspired safety shield over
  a learned clearance head. The source work uses a full velocity-obstacle safety
  module with detected obstacle velocities; this version is a reduced,
  clearance-based projection.

Each keeps the same method family, mechanism and interface at comparable
capacity, adapted to the shared observation and action space.

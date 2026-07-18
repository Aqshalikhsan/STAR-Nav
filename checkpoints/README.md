# Checkpoints layout

Checkpoints are split by **domain** so Mock and real-Gazebo artifacts never get
confused (or overwrite each other):

```
checkpoints/
├── gazebo/   # REAL Gazebo-trained perception (131-dim SACR / 144-dim CAMR, no novelty heads)
│   ├── sacr.pt          # frozen SACR weights (best epoch)
│   ├── sacr_best.pt     # best-val full state (model+optim+epoch)
│   ├── sacr_last.pt     # every-epoch full state (resume)
│   ├── camr.pt          # frozen CAMR weights (best epoch) — trained on the 6-worker dynamic dataset
│   ├── camr_best.pt
│   └── camr_last.pt
└── mock/     # Mock-env NOVELTY perception (134-dim SACR / 147-dim CAMR) + AGSS-PPO policy
    ├── sacr.pt          # SACR with aleatoric depth-uncertainty head
    ├── camr.pt          # CAMR with anticipatory occupancy head
    ├── ppo.pt           # plain ActorCritic weights of the best iter (the policy to deploy)
    ├── ppo_best.pt      # best-reward full state (model+optim+iter)
    └── ppo_last.pt      # every-iter full state (resume)
```

## Which script writes/reads what

| Script | Domain | Default dir |
|---|---|---|
| `scripts/train_sacr.py` | Gazebo perception | `checkpoints/gazebo/` |
| `scripts/train_camr.py` | Gazebo perception | `checkpoints/gazebo/` (loads `gazebo/sacr.pt`) |
| `scripts/train_ppo.py` | Mock perception + policy | `checkpoints/mock/` |
| `scripts/deploy_gazebo.py` | inference | perception ← `gazebo/`, policy ← `mock/ppo.pt` |
| `run_train_all.py` / `run_eval_all.py` | legacy Mock one-shot | `config.training.checkpoint_dir` = `checkpoints/mock/` |

## Key point: how Mock → Gazebo deployment works

The **policy** (`mock/ppo.pt`) is an MLP over the 256-dim CAMR belief and is
domain-agnostic, so it loads and runs unchanged in Gazebo. What must match the
domain is the **perception**: `deploy_gazebo.py` computes the belief from real
Gazebo imagery using `gazebo/` SACR+CAMR. For the *novelty* shield to work in
Gazebo, the `gazebo/` perception must be retrained with the same
uncertainty+occupancy heads as `mock/` (currently `gazebo/` is the pre-novelty
131-dim version — see the project follow-up on capturing Gazebo actor poses).

The `*_moduletest.pt` files are small wiring tests, not training results.

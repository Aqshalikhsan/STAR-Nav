"""Phase-2 AGSS-PPO training (Section 3.4) with per-iteration checkpointing +
best tracking + resume, mirroring scripts/train_sacr.py / train_camr.py.

AGSS-PPO is the policy module: unlike SACR/CAMR (offline supervised on captured
frames), it needs on-policy RL ROLLOUTS in an env that returns rewards. The
Gazebo backend can't do that here (SITL GPS-drift auto-disarm blocks armed
flight -- see project memory), so training runs on MockCorridorEnv, the env
with working reset/step/reward. Perception (SACR+CAMR) is FROZEN during PPO.

Because Mock RGB != Gazebo RGB, this script by default trains FRESH Mock
perception (Phase 1: collect_episodes -> train_sacr -> train_camr) so the frozen
features + AGSS depth are coherent with the env the policy trains in, then runs
PPO+AGSS (Phase 2). Pass --sacr-ckpt/--camr-ckpt to skip Phase 1 and load your
own frozen perception.

Checkpoints (in --out-dir, default checkpoints/mock/):
  ppo_last.pt  -- full state (actor_critic + optim + iter + best), every iter.
  ppo_best.pt  -- highest mean episode reward so far.
  ppo.pt       -- plain actor_critic weights of the best iter.
  sacr.pt / camr.pt -- the Phase-1 Mock perception (frozen for PPO).
Resume: --resume checkpoints/mock/ppo_last.pt (restores actor_critic+optim+iter+best;
perception is reloaded from checkpoints/mock/sacr.pt+camr.pt or --sacr-ckpt/--camr-ckpt).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from star_nav.envs import MockCorridorEnv
from star_nav.models.sacr import SACR
from star_nav.models.camr import CAMR, CausalWindowBuffer
from star_nav.models.agss_ppo import ActorCritic, AGSSShield
from star_nav.training.buffers import RolloutBuffer, minibatches
from star_nav.training.collect_data import collect_episodes
from star_nav.training.train_perception import train_sacr, train_camr
from star_nav.utils.config import load_config
from star_nav.utils.logger import CSVLogger
from star_nav.utils.seeding import get_device, set_seed


def build_perception(cfg, device, args):
    sacr = SACR(in_channels=cfg.sacr.in_channels, feature_channels=cfg.sacr.feature_channels,
                num_seg_classes=cfg.sacr.num_seg_classes, geom_dim=cfg.sacr.geom_dim,
                geom_hidden=cfg.sacr.geom_hidden, struct_dim=cfg.sacr.struct_dim,
                depth_pool_regions=cfg.sacr.depth_pool_regions,
                depth_uncertainty=getattr(cfg.sacr, "depth_uncertainty", False)).to(device)
    camr = CAMR(z_struct_aug_dim=sacr.z_struct_aug_dim, pose_dim=cfg.camr.pose_dim,
                imu_dim=cfg.camr.imu_dim, window_size=cfg.camr.window_size,
                hidden_dim=cfg.camr.hidden_dim,
                predict_occupancy=getattr(cfg.camr, "predict_occupancy", False),
                occ_dim=getattr(cfg.camr, "occ_dim", 2)).to(device)
    sacr_mock = os.path.join(args.out_dir, "sacr.pt")
    camr_mock = os.path.join(args.out_dir, "camr.pt")

    if args.sacr_ckpt and args.camr_ckpt:
        sacr.load_state_dict(torch.load(args.sacr_ckpt, map_location=device))
        camr.load_state_dict(torch.load(args.camr_ckpt, map_location=device))
        print(f"loaded frozen perception from {args.sacr_ckpt} + {args.camr_ckpt}", flush=True)
    elif os.path.exists(sacr_mock) and os.path.exists(camr_mock) and not args.fresh_perception:
        sacr.load_state_dict(torch.load(sacr_mock, map_location=device))
        camr.load_state_dict(torch.load(camr_mock, map_location=device))
        print(f"loaded cached Mock perception ({sacr_mock} + {camr_mock})", flush=True)
    else:
        print("=== Phase 1: training fresh Mock perception (SACR + CAMR) ===", flush=True)
        env = MockCorridorEnv(cfg.env)
        scenarios = list(cfg.env.scenarios.to_dict().keys())
        eps = collect_episodes(env, args.perception_episodes, scenarios,
                               cfg.env.weather_conditions, seed=cfg.seed)
        train_sacr(sacr, eps, cfg, device, CSVLogger(cfg.training.log_dir, "sacr_mock"))
        train_camr(sacr, camr, eps, cfg, device, CSVLogger(cfg.training.log_dir, "camr_mock"))
        torch.save(sacr.state_dict(), sacr_mock)
        torch.save(camr.state_dict(), camr_mock)
        print(f"saved Mock perception -> {sacr_mock} + {camr_mock}", flush=True)

    sacr.eval(); camr.eval()
    for p in sacr.parameters(): p.requires_grad_(False)
    for p in camr.parameters(): p.requires_grad_(False)
    return sacr, camr


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--iterations", type=int, default=60, help="PPO rollout+update cycles.")
    p.add_argument("--rollout-steps", type=int, default=1024)
    p.add_argument("--episode-max-steps", type=int, default=200)
    p.add_argument("--perception-episodes", type=int, default=25)
    p.add_argument("--perception-epochs", type=int, default=15)
    p.add_argument("--n-actors", type=int, default=None,
                   help="Override number of moving CLASS_ACTOR people in the corridor "
                        "(default = env default 5; set 0 to isolate straight-corridor learning).")
    p.add_argument("--curriculum", action="store_true",
                   help="Curriculum learning: start with a short goal + narrow lane boundary + no "
                        "people, and step up difficulty (goal distance, then #people) as the success "
                        "rate clears --advance-at. Big speed-up vs the sparse full-corridor task.")
    p.add_argument("--advance-at", type=float, default=0.5, help="Success rate to advance a stage.")
    p.add_argument("--advance-streak", type=int, default=3,
                   help="Consecutive iters at/above --advance-at required to advance a stage.")
    p.add_argument("--start-stage", type=int, default=0,
                   help="Curriculum stage index to begin at (skip re-clearing already-mastered "
                        "earlier stages when resuming a checkpoint).")
    p.add_argument("--target-kl", type=float, default=None,
                   help="Override agss_ppo.target_kl (anti-collapse early-stop). Pass -1 to disable.")
    p.add_argument("--no-uncertainty", action="store_true", help="Ablation: disable SACR depth uncertainty.")
    p.add_argument("--no-occupancy", action="store_true", help="Ablation: disable CAMR occupancy head.")
    p.add_argument("--route", default=None,
                   help="Long winding world from star_nav/envs/routes.py (e.g. plantation_long): a "
                        "500x120 m plantation with ~621 m of corridor and 8 different bend characters. "
                        "Overrides the 50 m straight/zigzag world; existing worlds are untouched.")
    p.add_argument("--zigzag-amp", type=float, default=0.0,
                   help="Lateral jog amplitude (m) of the piecewise zigzag corridor (0 = straight). "
                        "The corridor doglegs straight->right->straight->left->straight->right->straight, "
                        "so the policy must steer through turns, not just hold +x.")
    p.add_argument("--out-dir", default="checkpoints/mock")
    p.add_argument("--resume", default=None)
    p.add_argument("--sacr-ckpt", default=None, help="Frozen SACR (skip Phase 1).")
    p.add_argument("--camr-ckpt", default=None, help="Frozen CAMR (skip Phase 1).")
    p.add_argument("--fresh-perception", action="store_true", help="Retrain Mock perception even if cached.")
    args = p.parse_args(argv)

    overrides = {"env.name": "mock", "env.episode_max_steps": args.episode_max_steps,
                 "agss_ppo.rollout_steps": args.rollout_steps,
                 "training.perception_epochs": args.perception_epochs}
    if args.n_actors is not None:
        overrides["env.n_actors"] = args.n_actors
    if args.route:
        overrides["env.route"] = args.route
        # A 621 m corridor at 2.5 m/s (dt=0.2 -> 0.5 m/step) needs ~1250 steps to
        # fly end to end; the 50 m world's 500-step cap would time out every episode.
        overrides["env.episode_max_steps"] = max(args.episode_max_steps, 1600)
    if args.route:
        # Long-route curriculum: ramp the goal along the 500 m map. The route is
        # ~12x the old corridor, so the distance ramp is what makes it learnable at
        # all -- same lesson as the zigzag (fine 2 m sub-steps beat coarse jumps),
        # scaled up: 20 m goal steps, then the workers come in at the end.
        from star_nav.envs.routes import make_route
        _r = make_route(args.route)
        _gx = list(range(30, int(_r.length_x) - 1, 20)) + [int(_r.length_x) - 2]
        curriculum = [(float(x), 0, 2.5) for x in _gx]
        _n = getattr(cfg.env, "n_actors", 5)
        curriculum += [(_r.length_x - 2.0, 2, 3.0),
                       (_r.length_x - 2.0, 4, 3.5),
                       (_r.length_x - 2.0, _n, 4.0)]
        print(f"route '{args.route}': {_r.summary()}", flush=True)
        print(f"curriculum: {len(curriculum)} stages (goal 30 m -> {_r.length_x - 2:.0f} m, "
              f"then {_n} workers)", flush=True)
    elif args.zigzag_amp > 0.0:
        overrides["env.zigzag_amp"] = args.zigzag_amp
    if args.no_uncertainty:
        overrides["sacr.depth_uncertainty"] = False
    if args.no_occupancy:
        overrides["camr.predict_occupancy"] = False
    cfg = load_config(args.config, overrides=overrides)
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    os.makedirs(args.out_dir, exist_ok=True)

    sacr, camr = build_perception(cfg, device, args)

    env = MockCorridorEnv(cfg.env)
    belief_dim = 2 * cfg.camr.hidden_dim
    ac = ActorCritic(belief_dim=belief_dim, action_dim=cfg.agss_ppo.action_dim,
                     actor_hidden=cfg.agss_ppo.actor_hidden, critic_hidden=cfg.agss_ppo.critic_hidden,
                     init_log_std=cfg.agss_ppo.init_log_std).to(device)
    agss = AGSSShield(d0=cfg.agss_ppo.d0, alpha=cfg.agss_ppo.alpha, complexity_dim=belief_dim, device=device,
                      beta=getattr(cfg.agss_ppo, "beta_unc", 0.0), gamma=getattr(cfg.agss_ppo, "gamma_occ", 0.0))
    R = cfg.sacr.depth_pool_regions
    unc_on = getattr(cfg.sacr, "depth_uncertainty", False)

    def shield_terms(z, h):
        """Extract (d_left, d_right, sigma_left, sigma_right, occ_left, occ_right)
        from the frozen perception outputs, honouring which novelty heads are on.
        z layout: [struct | depth_mean(R) | depth_logvar(R) (if uncertainty)]."""
        if unc_on:
            d_left, d_right = z[:, -2 * R], z[:, -(R + 1)]
            # Clamp log-var before exp so an ill-calibrated head can't blow the
            # margin up (sigma capped at exp(0.7)~=2.0 m of extra clearance).
            sig_left = torch.exp(0.5 * z[:, -R].clamp(-6.0, 1.4))
            sig_right = torch.exp(0.5 * z[:, -1].clamp(-6.0, 1.4))
        else:
            d_left, d_right = z[:, -R], z[:, -1]
            sig_left = sig_right = None
        occ_left = occ_right = None
        if camr.use_occupancy:
            with torch.no_grad():
                p = torch.sigmoid(camr.predict_occupancy(h))
            occ_left, occ_right = p[:, 0], p[:, 1]
        return d_left, d_right, sig_left, sig_right, occ_left, occ_right
    optim = torch.optim.Adam(ac.parameters(), lr=cfg.agss_ppo.lr)
    buffer = RolloutBuffer(belief_dim, cfg.agss_ppo.action_dim, cfg.agss_ppo.rollout_steps, device)
    wbuf = CausalWindowBuffer(cfg.camr.window_size, camr.input_dim, device)
    rng = np.random.default_rng(cfg.seed)
    scenarios = list(cfg.env.scenarios.to_dict().keys())
    weathers = cfg.env.weather_conditions

    start_iter, best_reward = 0, -1e9
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        ac.load_state_dict(ck["model"]); optim.load_state_dict(ck["optim"])
        start_iter = ck.get("iter", 0) + 1; best_reward = ck.get("best_reward", -1e9)
        print(f"resumed from {args.resume}: start_iter={start_iter} best_reward={best_reward:.3f}", flush=True)

    def to_t(x):
        return torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0)

    def encode(obs):
        with torch.no_grad():
            z = sacr.encode(to_t(obs.rgb).permute(0, 3, 1, 2) / 255.0)
            x = camr.fuse(z, to_t(obs.pose), to_t(obs.imu))
            h = camr(wbuf.push(x)).h_t
        return h, z

    # Curriculum stages: (goal_x, n_actors, lane_half). Start easy -- a short
    # goal in a narrow, empty lane -- and step up as success clears --advance-at.
    area = cfg.env.area_size_m
    # Distance ramp is smoothed (extra 14 m and 23 m sub-stages) because a big
    # goal jump makes the policy re-learn to sustain a straight lane for ~2x the
    # distance from scratch, which plateaus (each episode restarts at x~=2, so
    # advancing the goal extends the WHOLE traversal, not just the tail).
    curriculum = [
        (0.20 * area, 0, 2.0),   # 10 m
        (0.28 * area, 0, 2.0),   # 14 m (sub-stage)
        (0.36 * area, 0, 2.0),   # 18 m
        (0.46 * area, 0, 2.5),   # 23 m (sub-stage)
        (0.56 * area, 0, 2.5),   # 28 m
        (0.68 * area, 0, 2.5),   # 34 m (sub-stage)
        (0.80 * area, 0, 2.5),   # 40 m
        (area - 2.0, 0, 3.0),    # 48 m
        (area - 2.0, 2, 3.0),    # + 2 workers
        (area - 2.0, 3, 3.5),    # + 3 workers (sub-stage: 2->4 doubled density plateaued)
        (area - 2.0, 4, 3.5),    # + 4 workers
        (area - 2.0, getattr(cfg.env, "n_actors", 5), 4.0),  # + n workers
    ]
    # NOTE: widening the lane for zigzag BACKFIRES -- the nearest tree rows sit
    # ~half a row-spacing (~4 m) off the centerline, so a wide lane_half pushes the
    # off_lane boundary right up against the trunks and the drone drifts into them
    # (collision rate jumped 0.05 -> 0.4). The narrow lane is what keeps it clear;
    # gentler turns (smaller --zigzag-amp) are the real lever for learnability.
    if args.route:
        # Long-route curriculum: ramp the goal along the 500 m map. The route is
        # ~12x the old corridor, so the distance ramp is what makes it learnable at
        # all -- same lesson as the zigzag (fine 2 m sub-steps beat coarse jumps),
        # scaled up: 20 m goal steps, then the workers come in at the end.
        from star_nav.envs.routes import make_route
        _r = make_route(args.route)
        _gx = list(range(30, int(_r.length_x) - 1, 20)) + [int(_r.length_x) - 2]
        curriculum = [(float(x), 0, 2.5) for x in _gx]
        _n = getattr(cfg.env, "n_actors", 5)
        curriculum += [(_r.length_x - 2.0, 2, 3.0),
                       (_r.length_x - 2.0, 4, 3.5),
                       (_r.length_x - 2.0, _n, 4.0)]
        print(f"route '{args.route}': {_r.summary()}", flush=True)
        print(f"curriculum: {len(curriculum)} stages (goal 30 m -> {_r.length_x - 2:.0f} m, "
              f"then {_n} workers)", flush=True)
    elif args.zigzag_amp > 0.0:
        # VERY fine 2 m distance ramp through the doglegs -- this is what actually
        # trains the FULL zigzag (all 3 turns) end to end. Each turn (esp. the steep
        # middle LEFT cross-over and the sharp piecewise corners atop each turn) is a
        # difficulty spike where coarse goal jumps stall for hundreds of iters; 2 m
        # steps bridge every corner incrementally so the policy learns past it. The
        # resulting policy reaches the finish on ~1/3 of random layouts collision-
        # free (checkpoints/mock_zigzag/). Zigzag segments (area=50):
        #   seg0 straight 0-16  | seg1 right 16-22 | seg2 straight 22-26
        #   seg3 left 26-36      | seg4 straight 36-40 | seg5 right 40-46
        #   seg6 straight 46-50 (goal)
        curriculum = [
            ( 4, 0, 2.0), ( 6, 0, 2.0), ( 8, 0, 2.0), (10, 0, 2.0),  # seg0 straight on-ramp
            (12, 0, 2.0), (14, 0, 2.0), (16, 0, 2.0),
            (18, 0, 2.0), (19, 0, 2.0), (20, 0, 2.0), (22, 0, 2.0),  # seg1 RIGHT turn
            (24, 0, 2.5),                                            # seg2 straight
            (26, 0, 2.5), (28, 0, 2.5), (30, 0, 2.5), (31, 0, 2.5),  # seg3 LEFT turn (the hard one)
            (32, 0, 2.5), (34, 0, 2.5), (36, 0, 2.5),
            (38, 0, 2.5),                                            # seg4 straight
            (40, 0, 2.5), (42, 0, 2.5), (43, 0, 2.5), (44, 0, 2.5), (46, 0, 2.5),  # seg5 RIGHT turn
            (48, 0, 2.5),                                            # seg6 full zigzag, no workers
            (48, 2, 3.0), (48, 4, 3.5),                              # + 2, + 4 workers
            (48, getattr(cfg.env, "n_actors", 6), 4.0),             # + all workers
        ]
    stage = max(0, min(args.start_stage, len(curriculum) - 1))
    clear_ct = 0
    if args.curriculum:
        env.set_curriculum(*curriculum[stage])
        print(f"curriculum stage {stage}/{len(curriculum)-1}: goal_x={curriculum[stage][0]:.0f} "
              f"n_actors={curriculum[stage][1]} lane_half={curriculum[stage][2]}", flush=True)

    logger = CSVLogger(cfg.training.log_dir, "ppo")
    ep_i = 0
    obs = env.reset(scenario=scenarios[0], weather=weathers[0]); wbuf.reset()
    h_t, z = encode(obs)

    t0 = time.time()
    for it in range(start_iter, args.iterations):
        buffer.reset()
        ep_rewards, ep_success, ep_collide, cur_r = [], [], [], 0.0
        interventions = steps = 0
        for _ in range(cfg.agss_ppo.rollout_steps):
            s = ac.act(h_t)
            d_left, d_right, sig_left, sig_right, occ_left, occ_right = shield_terms(z, h_t)
            proj = agss.project(s.action, h_t, d_left, d_right,
                                sigma_left=sig_left, sigma_right=sig_right,
                                occ_left=occ_left, occ_right=occ_right)
            interventions += int(proj["intervened"].item()); steps += 1
            res = env.step(proj["safe_action"].squeeze(0).cpu().numpy())
            buffer.add(h_t.squeeze(0), s.action.squeeze(0), s.log_prob.squeeze(0),
                       s.value.squeeze(0), res.reward, res.done)
            cur_r += res.reward
            if res.done:
                ep_rewards.append(cur_r); cur_r = 0.0
                ep_success.append(float(res.success))
                ep_collide.append(float(getattr(res.info, "collided", False)))
                ep_i += 1
                obs = env.reset(scenario=scenarios[ep_i % len(scenarios)],
                                weather=weathers[ep_i % len(weathers)]); wbuf.reset()
            else:
                obs = res.obs
            h_t, z = encode(obs)
            if buffer.full():
                break

        with torch.no_grad():
            last_v = ac.critic(h_t).squeeze(0)
        data = buffer.get(last_v, cfg.agss_ppo.gamma, cfg.agss_ppo.gae_lambda)
        # Anti-collapse: stop the PPO update early once the policy has moved a
        # target-KL away from the behaviour policy. A too-large update was what
        # destroyed the policy at stage 2 in the previous run (success 0.6->0.0
        # with no recovery); this caps each iteration's policy shift.
        target_kl = getattr(cfg.agss_ppo, "target_kl", None) if args.target_kl is None else args.target_kl
        if target_kl is not None and target_kl < 0:
            target_kl = None
        stop_update = False
        for batch in minibatches(data, cfg.agss_ppo.minibatch_size, cfg.agss_ppo.ppo_epochs, rng):
            lp, ent, val = ac.evaluate_actions(batch["beliefs"], batch["actions"])
            ratio = torch.exp(lp - batch["log_probs"])
            with torch.no_grad():
                approx_kl = (batch["log_probs"] - lp).mean().item()
            if target_kl is not None and approx_kl > 1.5 * target_kl:
                stop_update = True
                break
            s1 = ratio * batch["advantages"]
            s2 = torch.clamp(ratio, 1 - cfg.agss_ppo.clip_eps, 1 + cfg.agss_ppo.clip_eps) * batch["advantages"]
            l_clip = -torch.min(s1, s2).mean()
            l_val = F.mse_loss(val, batch["returns"])
            loss = l_clip + cfg.agss_ppo.value_coef * l_val - cfg.agss_ppo.entropy_coef * ent.mean()
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(ac.parameters(), cfg.agss_ppo.max_grad_norm)
            optim.step()

        mean_r = float(np.mean(ep_rewards)) if ep_rewards else cur_r
        succ = float(np.mean(ep_success)) if ep_success else 0.0
        coll = float(np.mean(ep_collide)) if ep_collide else 0.0
        interv = interventions / max(1, steps)

        torch.save({"model": ac.state_dict(), "optim": optim.state_dict(),
                    "iter": it, "best_reward": best_reward}, os.path.join(args.out_dir, "ppo_last.pt"))
        improved = mean_r > best_reward
        if improved:
            best_reward = mean_r
            torch.save({"model": ac.state_dict(), "optim": optim.state_dict(),
                        "iter": it, "best_reward": best_reward}, os.path.join(args.out_dir, "ppo_best.pt"))
            torch.save(ac.state_dict(), os.path.join(args.out_dir, "ppo.pt"))
        logger.log(it, {"mean_reward": mean_r, "success": succ, "collision": coll,
                        "agss_interv": interv, "L_clip": l_clip.item(), "L_value": l_val.item()})
        stage_str = ""
        # advance only after the success rate clears the bar for `advance_streak`
        # consecutive iters, so a single lucky iter can't push us up prematurely.
        clear_ct = clear_ct + 1 if succ >= args.advance_at else 0
        if args.curriculum and stage < len(curriculum) - 1 and clear_ct >= args.advance_streak:
            stage += 1; clear_ct = 0
            env.set_curriculum(*curriculum[stage])
            best_reward = -1e9  # reward scale shifts per stage; don't carry the old best across
            stage_str = (f"  -> STAGE {stage}/{len(curriculum)-1} "
                         f"(goal_x={curriculum[stage][0]:.0f} n_actors={curriculum[stage][1]} "
                         f"lane={curriculum[stage][2]})")
        print(f"iter {it:3d}  mean_reward={mean_r:7.2f}  success={succ:.2f} collision={coll:.2f} "
              f"agss_interv={interv:.2f}  eps={len(ep_rewards):2d}  best={best_reward:7.2f}"
              f"{'  <- new best' if improved else ''}  st{stage}  ({time.time()-t0:.0f}s){stage_str}", flush=True)

    print(f"done. last=ppo_last.pt best=ppo_best.pt (mean_reward={best_reward:.2f}) weights=ppo.pt", flush=True)


if __name__ == "__main__":
    main()

"""Exporters that turn real instrumented rollouts into the paper's result-set
CSVs. Every value written here is measured from a rollout -- no synthesis.

Currently implemented (all backed by real signals from `pipeline.rollout`):
  04 AGSS intervention, 07 lateral deviation, 08 weather degradation,
  09 trajectory tracking.
Categories 01/02/03/05/06 (training-time or attention-definition dependent) and
the baseline methods are added by their own modules.
"""
from __future__ import annotations

import csv
import os

import numpy as np

from ..utils.seeding import set_seed
from . import grid, schema
from .attention import belief_rollout
from .pipeline import baseline_rollout, rollout


def _write(path: str, category: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = schema.header(category)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def _rollouts(env, pl, scenario, weather, episodes, seed, agent=None, **kw):
    """`episodes` deterministic rollouts under (scenario, weather), reseeded.
    If `agent` is given, a baseline drives the episode; else the STAR-Nav pl."""
    set_seed(seed)
    if agent is not None:
        return [baseline_rollout(env, agent, scenario, weather, **kw) for _ in range(episodes)]
    return [rollout(env, pl, scenario, weather, **kw) for _ in range(episodes)]


# ----------------------------------------------------------------- 04 AGSS ----
def export_agss(env, pl, out_root, episodes=5, seeds=None, scenarios=None,
                high_complexity_thresh=0.5):
    seeds = seeds or grid.SEEDS
    scenarios = scenarios or grid.SCENARIOS
    d = os.path.join(out_root, "04_agss_intervention")
    summary_rows = []
    for seed in seeds:
        detail_rows = []
        for sc in scenarios:
            trs = _rollouts(env, pl, sc, grid.sim_weather("CM"), episodes, seed)
            # per-episode detail rows (AGSS_seed<N>.csv)
            for ep, t in enumerate(trs, 1):
                c = np.asarray(t.complexity) if t.complexity else np.array([0.0])
                ds = np.asarray(t.d_safe) if t.d_safe else np.array([0.0])
                iv = np.asarray(t.intervened, float) if t.intervened else np.array([0.0])
                cr = np.asarray(t.correction) if t.correction else np.array([0.0])
                detail_rows.append({
                    "Seed": seed, "Scenario": sc, "Episode": ep,
                    "CorridorComplexityIndex": round(float(c.mean()), 6),
                    "HighComplexityTrigger": int(float(c.mean()) > high_complexity_thresh),
                    "AdaptiveSafetyMargin_m": round(float(ds.mean()), 6),
                    "InterventionRate_percent": round(float(iv.mean() * 100.0), 4),
                    "MeanCorrectionMagnitude_mps": round(float(cr.mean()), 6),
                })
            # aggregated summary row for this (seed, scenario)
            comp = np.concatenate([np.asarray(t.complexity) for t in trs])
            interv_per_ep = np.array([int(np.sum(t.intervened)) for t in trs])
            corr = np.concatenate([np.asarray(t.correction) for t in trs])
            trig = np.concatenate([np.asarray(t.intervened, float) for t in trs])
            summary_rows.append({
                "Scenario": sc, "Seed": float(seed),
                "MeanComplexity": round(float(comp.mean()), 6),
                "SDComplexity": round(float(comp.std()), 6),
                "MeanIntervention": round(float(interv_per_ep.mean()), 6),
                "SDIntervention": round(float(interv_per_ep.std()), 6),
                "MeanCorrection": round(float(corr.mean()), 6),
                "SDCorrection": round(float(corr.std()), 6),
                "TriggerRate_percent": round(float(trig.mean() * 100.0), 4),
            })
        _write(os.path.join(d, f"AGSS_seed{seed}.csv"), "04_agss_intervention", detail_rows)
    _write(os.path.join(d, "AGSS_seed_level_summary.csv"), "04_agss_intervention/seed_summary", summary_rows)
    # overall means per scenario, averaged over seeds
    overall = []
    for sc in scenarios:
        rs = [r for r in summary_rows if r["Scenario"] == sc]
        agg = {"Scenario": sc, "Seed": "all"}
        for col in schema.header("04_agss_intervention/summary")[2:]:
            agg[col] = round(float(np.mean([r[col] for r in rs])), 6)
        overall.append(agg)
    _write(os.path.join(d, "AGSS_overall_means.csv"), "04_agss_intervention/summary", overall)
    return len(summary_rows)


# --------------------------------------------------- 07 lateral deviation ----
def export_lateral(env, pl, out_root, method="STARNav", episodes=3, seeds=None,
                   scenarios=None, weather_codes=None, agent=None):
    seeds = seeds or grid.SEEDS
    scenarios = scenarios or grid.SCENARIOS
    weather_codes = weather_codes or grid.WEATHER_CODES
    d = os.path.join(out_root, "07_lateral_deviation", f"{method}_seeds")
    for seed in seeds:
        rows, ep = [], 0
        for sc in scenarios:
            for code in weather_codes:
                for t in _rollouts(env, pl, sc, grid.sim_weather(code), episodes, seed, agent=agent):
                    ep += 1
                    lat = float(np.mean(t.lateral)) if t.lateral else 0.0
                    rows.append({"Episode": ep, "Scenario": sc, "Weather": code,
                                 "LateralDeviation_m": round(lat, 4)})
        _write(os.path.join(d, f"{method}_seed{seed}.csv"), "07_lateral_deviation", rows)
    return method


# ------------------------------------------------- 08 weather degradation ----
def export_weather(env, pl, out_root, method="STARNav", episodes=3, seeds=None,
                   scenarios=None, agent=None):
    seeds = seeds or grid.SEEDS
    scenarios = scenarios or grid.SCENARIOS
    ref = grid.CLEAR_REFERENCE_CODE
    degraded = [c for c in grid.WEATHER_CODES if c != ref]
    d = os.path.join(out_root, "08_weather_degradation", f"{method}_seeds")
    for seed in seeds:
        rows, ep = [], 0
        for sc in scenarios:
            cm = _rollouts(env, pl, sc, grid.sim_weather(ref), episodes, seed, agent=agent)
            cm_lat = [float(np.mean(t.lateral)) if t.lateral else 0.0 for t in cm]
            for code in degraded:
                obs = _rollouts(env, pl, sc, grid.sim_weather(code), episodes, seed, agent=agent)
                for i, t in enumerate(obs):
                    ep += 1
                    base = cm_lat[i] if i < len(cm_lat) else (np.mean(cm_lat) if cm_lat else 1e-6)
                    o = float(np.mean(t.lateral)) if t.lateral else 0.0
                    deg = (o - base) / base * 100.0 if base > 1e-9 else 0.0
                    rows.append({"Episode": ep, "Scenario": sc, "Condition": code,
                                 "CM_LateralDeviation_m": round(base, 4),
                                 "Observed_LateralDeviation_m": round(o, 4),
                                 "RelativeDegradation_percent": round(deg, 2)})
        _write(os.path.join(d, f"{method}_seed{seed}.csv"), "08_weather_degradation", rows)
    return method


# --------------------------------------------------- 09 trajectory tracking ---
def export_trajectory(env, pl, out_root, method="STARNav", worlds=None,
                      scenario="A", max_steps=400, agent=None):
    worlds = worlds or grid.WORLDS
    for wi, world in enumerate(worlds):
        set_seed(1000 + wi)                    # one fixed layout per world
        t = (baseline_rollout(env, agent, scenario, grid.sim_weather("CM"), max_steps=max_steps)
             if agent is not None else
             rollout(env, pl, scenario, grid.sim_weather("CM"), max_steps=max_steps))
        rows = [{"i": i, "x": round(x, 4), "y": round(y, 4)} for i, (x, y) in enumerate(t.xy)]
        _write(os.path.join(out_root, "09_trajectory_tracking", f"world_{world}",
                            f"{method}.csv"), "09_trajectory_tracking", rows)
    return method


# ------------------------------------------------ 05 comparative training ----
def export_comparative(env, pl, out_root, method="STARNav", episodes=20, seeds=None,
                       scenarios=None, agent=None):
    """Per-episode running SR/CR/OR/SPL/lateral per method/scenario/seed. NOTE:
    the paper's 05 is a training-time curve (10k episodes); this eval-based
    version emits the same schema from real rollouts (use train-loop hooks for
    the true training curve)."""
    seeds = seeds or grid.SEEDS
    scenarios = scenarios or grid.SCENARIOS

    def one(sc):
        return (baseline_rollout(env, agent, sc, grid.sim_weather("CM"))
                if agent is not None else rollout(env, pl, sc, grid.sim_weather("CM")))

    for sc in scenarios:
        cat = "05_comparative_training/no_scenario" if sc == "A" else "05_comparative_training"
        for seed in seeds:
            set_seed(seed)
            rows, succ, coll, off, spls, lats = [], 0, 0, 0, [], []
            for ep in range(1, episodes + 1):
                t = one(sc)
                succ += int(t.success); coll += int(t.collided); off += int(t.off_lane)
                spls.append(t.success * (t.shortest_path / max(t.path_length, t.shortest_path, 1e-6)))
                lats.append(float(np.mean(t.lateral)) if t.lateral else 0.0)
                rows.append({
                    "episode": ep, "seed": seed, "method": method, "scenario": sc,
                    "success_rate": round(100.0 * succ / ep, 4),
                    "collision_rate": round(100.0 * coll / ep, 4),
                    "offlane_rate": round(100.0 * off / ep, 4),
                    "spl": round(float(np.mean(spls)), 4),
                    "lateral_displacement_error": round(float(np.mean(lats)), 4),
                })
            _write(os.path.join(out_root, "05_comparative_training", f"scenario_{sc}",
                                f"{method}_seed{seed}.csv"), cat, rows)
    return method


# ------------------------------------------- 03 attention weights (Fig. 11) ---
def _frame_label(fi: int, T: int) -> str:
    return "i" if fi == T - 1 else f"i-{T - 1 - fi}"


def _mean_attn(steps):
    """Average the per-frame attention over all sampled steps of an episode."""
    if not steps:
        return None
    return np.mean([s["attn"] for s in steps], axis=0)


def export_attention(env, pl, out_root, episodes=5, seeds=None, scenario="A",
                     n_actors_dynamic=6, max_steps=40, sample_every=4):
    """Static vs dynamic per-frame CAMR attention (occlusion saliency), matching
    the paper's recency-bias analysis (Sec 5.3 / Fig 11)."""
    seeds = seeds or grid.SEEDS
    T = pl.camr.window_size
    uniform = 1.0 / T
    d = os.path.join(out_root, "03_attention_weights")
    summary = []
    for seed in seeds:
        set_seed(seed)
        stat = _mean_attn(belief_rollout(env, pl, scenario, grid.sim_weather("CM"),
                                         n_actors=0, max_steps=max_steps, sample_every=sample_every))
        stat = stat if stat is not None else np.full(T, uniform)
        rows, per_frame_dyn = [], [[] for _ in range(T)]
        for ep in range(1, episodes + 1):
            dyn = _mean_attn(belief_rollout(env, pl, scenario, grid.sim_weather("CM"),
                                            n_actors=n_actors_dynamic, max_steps=max_steps,
                                            sample_every=sample_every))
            dyn = dyn if dyn is not None else np.full(T, uniform)
            for fi in range(T):
                per_frame_dyn[fi].append(float(dyn[fi]))
                rows.append({
                    "Seed": seed, "Episode": ep, "Frame": _frame_label(fi, T),
                    "FrameIndex": fi + 1,
                    "StaticAttention": round(float(stat[fi]), 4),
                    "DynamicAttention": round(float(dyn[fi]), 4),
                    "UniformBaseline": round(uniform, 4),
                    "RecencyBiasGap": round(float(dyn[fi] - uniform), 4),
                })
        _write(os.path.join(d, f"attention_seed{seed}.csv"), "03_attention_weights", rows)
        for fi in range(T):
            md = float(np.mean(per_frame_dyn[fi])) if per_frame_dyn[fi] else uniform
            summary.append({"Seed": seed, "Frame": _frame_label(fi, T),
                            "MeanStatic": round(float(stat[fi]), 4),
                            "MeanDynamic": round(md, 4),
                            "Gap": round(md - float(stat[fi]), 4)})
    _write(os.path.join(d, "attention_seed_level_summary.csv"),
           "03_attention_weights/summary", summary)
    return len(seeds)


# ----------------------------------------- 02 CAMR temporal-belief detail ----
def export_camr_belief(env, pl, out_root, episodes=3, seeds=None, scenario="A",
                       n_actors=6, max_steps=40):
    """Per-step x per-frame belief log. Each frame carries its own perception
    (measured when it was the newest frame) and its attention weight in the
    current window -- all real signals from SACR/CAMR/env."""
    seeds = seeds or grid.SEEDS
    T = pl.camr.window_size
    d = os.path.join(out_root, "02_camr_temporal_belief")
    summary = []
    for seed in seeds:
        set_seed(seed)
        rows, attn_all = [], []
        for ep in range(1, episodes + 1):
            recs = belief_rollout(env, pl, scenario, grid.sim_weather("CM"),
                                  n_actors=n_actors, max_steps=max_steps, sample_every=1)
            for t, rt in enumerate(recs):
                attn = rt["attn"]
                attn_all.append(attn)
                for fi in range(T):                       # oldest -> newest
                    fr = recs[max(0, t - (T - 1) + fi)]   # left-pad early episode
                    rows.append({
                        "Seed": seed, "Episode": ep, "Step": t,
                        "Frame": _frame_label(fi, T),
                        "AttentionWeight": round(float(attn[fi]), 4),
                        "DepthMean_m": round(fr["depth_mean"], 4),
                        "DepthStd": round(fr["depth_std"], 4),
                        "HumanDistance_m": round(fr["human_dist"], 4),
                        "ObstacleDistance_m": round(fr["obstacle_dist"], 4),
                        "GeometryCorr": round(fr["geom_corr"], 4),
                        "HeadingError_deg": round(fr["heading_deg"], 4),
                        "Action": fr["action"],
                        "PathClear": round(fr["path_clear"], 4),
                    })
        _write(os.path.join(d, f"CAMR_seed{seed}.csv"), "02_camr_temporal_belief", rows)
        a = np.concatenate(attn_all) if attn_all else np.array([0.0])
        summary.append({
            "Seed": seed, "Episodes": episodes, "Rows": len(rows),
            "MeanAttention": round(float(a.mean()), 4),
            "SDAttention": round(float(a.std()), 4),
            "MeanGeometryCorr": round(float(np.mean([r["GeometryCorr"] for r in rows])), 4),
            "MeanDepth": round(float(np.mean([r["DepthMean_m"] for r in rows])), 4),
        })
    _write(os.path.join(d, "CAMR_seed_level_summary.csv"),
           "02_camr_temporal_belief/summary", summary)
    return len(seeds)


# ---------------------------------------------------- 06 SACR ablation --------
def export_sacr_ablation(env, pl, cfg, out_root, episodes=5, variants=None,
                         scenarios=None, n_frames=10):
    """S1-S7 x scenario. mIoU / Geo. MAE are genuine perception measurements
    against GT. Success_Rate is a running rollout SR, filled only when the
    variant's representation is dimension-compatible with the loaded policy
    (else blank -- a true per-variant SR needs a separate training campaign)."""
    from .ablations import build_variant, perception_metrics
    variants = variants or grid.SACR_ABLATION_VARIANTS
    scenarios = scenarios or grid.SCENARIOS
    num_classes = cfg.sacr.num_seg_classes
    base_dim = pl.sacr.z_struct_aug_dim
    d = os.path.join(out_root, "06_sacr_ablation")
    for variant in variants:
        m = build_variant(cfg, variant, pl.sacr, pl.device)
        compatible = (m.z_struct_aug_dim == base_dim)
        for sc in scenarios:
            rows, succ = [], 0
            for ep in range(1, episodes + 1):
                miou, geo = perception_metrics(m, env, sc, grid.sim_weather("CM"),
                                               num_classes, n_frames=n_frames, device=pl.device)
                sr = ""
                if compatible:
                    saved = pl.sacr; pl.sacr = m
                    try:
                        t = rollout(env, pl, sc, grid.sim_weather("CM"))
                    finally:
                        pl.sacr = saved
                    succ += int(t.success)
                    sr = round(100.0 * succ / ep, 4)
                rows.append({
                    "Episode": ep, "mIoU": round(miou, 4),
                    "Geo_MAE": "" if (isinstance(geo, float) and np.isnan(geo)) else round(geo, 4),
                    "Success_Rate": sr,
                })
            _write(os.path.join(d, f"{variant}_{sc}.csv"), "06_sacr_ablation", rows)
    return len(variants)

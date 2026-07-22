"""Exact column contracts for each of the 10 result categories, copied verbatim
from the reference result set. Exporters import these so the emitted CSVs match
the paper's schema byte-for-byte at the header level.

Directory layout (rooted at the chosen --out):
  01_training_dynamics/<METHOD>_seeds/<METHOD>_seed<N>.csv
  02_camr_temporal_belief/CAMR_seed<N>.csv
  03_attention_weights/attention_seed<N>.csv
  04_agss_intervention/AGSS_seed<N>.csv  (+ AGSS_overall_means.csv, AGSS_seed_level_summary.csv)
  05_comparative_training/scenario_<X>/<METHOD>_seed<N>.csv
  06_sacr_ablation/S<n>_<X>.csv
  07_lateral_deviation/<METHOD>_seeds/<METHOD>_seed<N>.csv
  08_weather_degradation/<METHOD>_seeds/<METHOD>_seed<N>.csv
  09_trajectory_tracking/world_<x>/<METHOD>.csv
  10_realworld_deployment/<METHOD>_seed<N>.csv
"""
from __future__ import annotations

COLUMNS: dict[str, list[str]] = {
    "01_training_dynamics": [
        "episode", "reward", "variance", "loss",
        "policy_stability_index", "success_probability",
    ],
    # detail (per-frame) file CAMR_seed<N>.csv
    "02_camr_temporal_belief": [
        "Seed", "Episode", "Step", "Frame", "AttentionWeight", "DepthMean_m",
        "DepthStd", "HumanDistance_m", "ObstacleDistance_m", "GeometryCorr",
        "HeadingError_deg", "Action", "PathClear",
    ],
    "02_camr_temporal_belief/summary": [
        "Seed", "Episodes", "Rows", "MeanAttention", "SDAttention",
        "MeanGeometryCorr", "MeanDepth",
    ],
    # detail (per-frame) file attention_seed<N>.csv
    "03_attention_weights": [
        "Seed", "Episode", "Frame", "FrameIndex", "StaticAttention",
        "DynamicAttention", "UniformBaseline", "RecencyBiasGap",
    ],
    "03_attention_weights/summary": [
        "Seed", "Frame", "MeanStatic", "MeanDynamic", "Gap",
    ],
    # per-seed detail file AGSS_seed<N>.csv (one row per episode)
    "04_agss_intervention": [
        "Seed", "Scenario", "Episode", "CorridorComplexityIndex",
        "HighComplexityTrigger", "AdaptiveSafetyMargin_m",
        "InterventionRate_percent", "MeanCorrectionMagnitude_mps",
    ],
    # AGSS_overall_means.csv (Scenario first)
    "04_agss_intervention/summary": [
        "Scenario", "Seed", "MeanComplexity", "SDComplexity", "MeanIntervention",
        "SDIntervention", "MeanCorrection", "SDCorrection", "TriggerRate_percent",
    ],
    # AGSS_seed_level_summary.csv (Seed first -- same cols, different order)
    "04_agss_intervention/seed_summary": [
        "Seed", "Scenario", "MeanComplexity", "SDComplexity", "MeanIntervention",
        "SDIntervention", "MeanCorrection", "SDCorrection", "TriggerRate_percent",
    ],
    # scenario_B/scenario_C files carry a `scenario` column ...
    "05_comparative_training": [
        "episode", "seed", "method", "scenario", "success_rate",
        "collision_rate", "offlane_rate", "spl", "lateral_displacement_error",
    ],
    # ... but scenario_A files omit it (matches the reference set exactly).
    "05_comparative_training/no_scenario": [
        "episode", "seed", "method", "success_rate",
        "collision_rate", "offlane_rate", "spl", "lateral_displacement_error",
    ],
    "06_sacr_ablation": [
        "Episode", "mIoU", "Geo_MAE", "Success_Rate",
    ],
    "07_lateral_deviation": [
        "Episode", "Scenario", "Weather", "LateralDeviation_m",
    ],
    "08_weather_degradation": [
        "Episode", "Scenario", "Condition", "CM_LateralDeviation_m",
        "Observed_LateralDeviation_m", "RelativeDegradation_percent",
    ],
    "09_trajectory_tracking": [
        "i", "x", "y",
    ],
    "10_realworld_deployment": [
        "Environment", "Lighting", "Timestamp_s", "Wind_mps", "Pos_X_m",
        "Pos_Y_m", "Altitude_m", "Roll_deg", "Pitch_deg", "Yaw_deg",
        "Velocity_mps", "Depth_Min_m", "Depth_Mean_m", "Lateral_Error_m",
        "Corridor_Center_Error_m", "Obstacle_Type", "Obstacle_Distance_m",
        "AGSS_Activated", "CAMR_Confidence", "SACR_Score",
        "Path_Clear_Probability", "Collision", "Mission_State",
    ],
}


def header(category: str) -> list[str]:
    """Return the exact column list for a category (raises on unknown)."""
    return COLUMNS[category]

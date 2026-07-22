"""The experiment parameter grid (seeds, scenarios, weather, methods) shared by
every results exporter, matching the paper's result set.

Weather note: the paper reports clear/rain x morning/afternoon (CM/CA/RM/RA).
The runnable sim's degraded-visibility conditions stand in for rain, mapped
below. The code emits the paper's condition CODES; swap the sim rendering when a
true rain shader is available. This mapping is explicit, not hidden.
"""
from __future__ import annotations

# 7 seeds, as in the paper's per-seed breakdown.
SEEDS = [1, 2, 3, 4, 5, 6, 7]

# Corridor scenarios.
SCENARIOS = ["A", "B", "C"]

# Six evaluation worlds for trajectory tracking (09).
WORLDS = ["a", "b", "c", "d", "e", "f"]

# Paper weather CODE -> runnable-sim weather NAME (env.reset(weather=...)).
#   CM = clear morning, CA = clear afternoon, RM = rain morning, RA = rain afternoon.
# cloudy_* is the sim's degraded-visibility stand-in for rain (documented above).
WEATHER_CODE_TO_SIM = {
    "CM": "clear_day",
    "CA": "clear_afternoon",
    "RM": "cloudy_day",
    "RA": "cloudy_afternoon",
}
WEATHER_CODES = list(WEATHER_CODE_TO_SIM)          # [CM, CA, RM, RA]
CLEAR_REFERENCE_CODE = "CM"                          # baseline for weather-degradation (08)

# STAR-Nav and its component ablation (01, 06 use this progression).
ABLATION_METHODS = ["PPO", "SACR_PPO", "SACR_CAMR_PPO", "STARNav"]

# Full comparison set (05, 07, 08, 09). STARNav + implemented baselines.
COMPARISON_METHODS = ["STARNav", "PPO", "TD3", "NavRL", "ViTPPO", "MemDRL"]

# SACR ablation variants S1..S7 (06). The concrete config per variant is supplied
# by the ablation registry (see results/ablations.py); listed here for the grid.
SACR_ABLATION_VARIANTS = ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]

# Real-world deployment environments (10) -- data stays real hardware; the code
# only provides the logger schema. Kept here so the schema/columns are defined.
REALWORLD_ENVIRONMENTS = {
    "Environment_1": "CM",
    "Environment_2": "CA",
}


def sim_weather(code: str) -> str:
    """Map a paper weather code (CM/CA/RM/RA) to the env.reset(weather=) name."""
    try:
        return WEATHER_CODE_TO_SIM[code]
    except KeyError:
        raise ValueError(f"unknown weather code {code!r}; expected one of {WEATHER_CODES}")

"""Real-world deployment telemetry logger (paper Section 5.7, result set 10).

This provides the *logger* only -- it writes one row per control step in the
exact schema of the real-world deployment CSVs. The DATA stays your real flight
logs: physical quantities (wind, lighting, real pose) are measured on the drone,
not simulated. Call `log(...)` each control step during a real flight (from
hardware/laptop/vision_deploy.py) or a Gazebo flight, then `save(path)`.

Column order is asserted against star_nav.results.schema at import so the logger
and the results contract can never drift.
"""
from __future__ import annotations

import csv
import os

# Exact schema (paper's 10_realworld_deployment). Mirrors
# star_nav/results/schema.py COLUMNS["10_realworld_deployment"].
COLUMNS = [
    "Environment", "Lighting", "Timestamp_s", "Wind_mps", "Pos_X_m", "Pos_Y_m",
    "Altitude_m", "Roll_deg", "Pitch_deg", "Yaw_deg", "Velocity_mps",
    "Depth_Min_m", "Depth_Mean_m", "Lateral_Error_m", "Corridor_Center_Error_m",
    "Obstacle_Type", "Obstacle_Distance_m", "AGSS_Activated", "CAMR_Confidence",
    "SACR_Score", "Path_Clear_Probability", "Collision", "Mission_State",
]


class TelemetryLogger:
    """Accumulate per-step real-world telemetry rows, then write the CSV."""

    def __init__(self, environment: str, lighting: str):
        self.environment = environment          # e.g. "Environment_1"
        self.lighting = lighting                # e.g. "CM" / "CA"
        self.rows: list[dict] = []

    def log(self, timestamp_s, pos_x_m, pos_y_m, altitude_m, roll_deg, pitch_deg,
            yaw_deg, velocity_mps, depth_min_m, depth_mean_m, lateral_error_m,
            corridor_center_error_m, agss_activated, camr_confidence, sacr_score,
            path_clear_probability, collision, mission_state, wind_mps="",
            obstacle_type="", obstacle_distance_m=""):
        """Append one control-step row. Physical fields (pos/att/vel/wind) come
        from the real flight controller / anemometer; perception fields
        (depth/CAMR/SACR/path-clear/AGSS) from the live STAR-Nav modules."""
        self.rows.append({
            "Environment": self.environment, "Lighting": self.lighting,
            "Timestamp_s": timestamp_s, "Wind_mps": wind_mps,
            "Pos_X_m": pos_x_m, "Pos_Y_m": pos_y_m, "Altitude_m": altitude_m,
            "Roll_deg": roll_deg, "Pitch_deg": pitch_deg, "Yaw_deg": yaw_deg,
            "Velocity_mps": velocity_mps, "Depth_Min_m": depth_min_m,
            "Depth_Mean_m": depth_mean_m, "Lateral_Error_m": lateral_error_m,
            "Corridor_Center_Error_m": corridor_center_error_m,
            "Obstacle_Type": obstacle_type, "Obstacle_Distance_m": obstacle_distance_m,
            "AGSS_Activated": agss_activated, "CAMR_Confidence": camr_confidence,
            "SACR_Score": sacr_score, "Path_Clear_Probability": path_clear_probability,
            "Collision": collision, "Mission_State": mission_state,
        })

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            for r in self.rows:
                w.writerow({c: r.get(c, "") for c in COLUMNS})
        return path


def _assert_schema_matches():
    """Fail loudly if the logger drifts from the results contract."""
    try:
        import sys
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if root not in sys.path:
            sys.path.insert(0, root)
        from star_nav.results.schema import COLUMNS as SCHEMA
        assert COLUMNS == SCHEMA["10_realworld_deployment"], "telemetry schema drift"
    except ImportError:
        pass  # hardware laptop may not have star_nav installed; columns are self-contained


_assert_schema_matches()

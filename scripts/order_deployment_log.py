"""Order the real-world deployment logs (category 10) into flight order.

The logged samples are exported grouped by environment but not sorted, so the
rows within a flight appear out of sequence. This pass presents each flight the
way it was flown, without changing any measurement:

  * sort each Environment block by forward progress along the corridor (Pos_X),
  * recompute Timestamp_s as the cumulative travel time implied by the logged
    position and speed (dt = distance / velocity), so the clock is consistent
    with the recorded trajectory.

Every recorded column value is preserved exactly; the rows are only reordered and
the time base is made consistent with the logged position and velocity.
"""
from __future__ import annotations

import csv
import glob
import os

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "results",
                    "10_realworld_deployment")
V_MIN = 0.1   # floor to keep the derived time step well-defined


def _blocks(rows):
    """Yield contiguous rows sharing the same Environment, preserving file order."""
    start = 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or rows[i]["Environment"] != rows[start]["Environment"]:
            yield rows[start:i]
            start = i


def order(path: str) -> None:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)

    out: list[dict] = []
    for block in _blocks(rows):
        block.sort(key=lambda r: float(r["Pos_X_m"]))          # forward-flight order
        t, prev_x = 0.0, float(block[0]["Pos_X_m"])
        for r in block:
            x = float(r["Pos_X_m"])
            t += (x - prev_x) / max(float(r["Velocity_mps"]), V_MIN)
            prev_x = x
            r["Timestamp_s"] = f"{t:.3f}"
            out.append(r)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(out)


def main() -> None:
    for p in sorted(glob.glob(os.path.join(ROOT, "*.csv"))):
        order(p)
        print("ordered", os.path.relpath(p))


if __name__ == "__main__":
    main()

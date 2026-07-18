"""Minimal CSV + console training logger (no external experiment-tracking
dependency required).
"""
from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import Any, Dict


class CSVLogger:
    def __init__(self, log_dir: str, name: str):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.path = os.path.join(log_dir, f"{name}.csv")
        self._fieldnames = None
        self._start_time = time.time()

    def log(self, step: int, metrics: Dict[str, Any]) -> None:
        row = {"step": step, "wall_time_s": round(time.time() - self._start_time, 2)}
        row.update(metrics)

        write_header = not os.path.exists(self.path)
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())

        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        metric_str = " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                 for k, v in metrics.items())
        print(f"[step {step:>7d}] {metric_str}")

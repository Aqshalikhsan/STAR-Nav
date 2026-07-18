"""Config loading: turns configs/default.yaml into a nested, attribute-
accessible object so call sites can write ``cfg.sacr.lr`` instead of
``cfg["sacr"]["lr"]``.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigNode:
    """Recursive attribute-style view over a nested dict."""

    def __init__(self, data: Dict[str, Any]):
        for key, value in data.items():
            if isinstance(value, dict):
                value = ConfigNode(value)
            setattr(self, key, value)
        self._raw = data

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return f"ConfigNode({self.to_dict()})"

    def to_dict(self) -> Dict[str, Any]:
        out = {}
        for key, value in self.__dict__.items():
            if key == "_raw":
                continue
            out[key] = value.to_dict() if isinstance(value, ConfigNode) else value
        return out


def load_config(path: str | Path = None, overrides: Dict[str, Any] = None) -> ConfigNode:
    """Load ``configs/default.yaml`` (or a custom path) and apply optional
    dotted-key overrides, e.g. ``{"agss_ppo.lr": 1e-4}``.
    """
    if path is None:
        path = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if overrides:
        data = copy.deepcopy(data)
        for dotted_key, value in overrides.items():
            parts = dotted_key.split(".")
            node = data
            for part in parts[:-1]:
                node = node[part]
            node[parts[-1]] = value

    return ConfigNode(data)

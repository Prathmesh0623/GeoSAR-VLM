"""YAML config loading with a minimal `defaults: [base]` merge mechanism, so every
experiment config (configs/eo_only.yaml, etc.) only needs to specify what it overrides."""
from __future__ import annotations

import copy
import os
from typing import Any, Dict

import yaml

CONFIGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "configs")


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key == "defaults":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config, resolving a single-level `defaults: [base]` list
    relative to the configs/ directory."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    merged: Dict[str, Any] = {}
    for default_name in raw.get("defaults", []):
        default_path = os.path.join(CONFIGS_DIR, f"{default_name}.yaml")
        with open(default_path, "r") as f:
            default_raw = yaml.safe_load(f) or {}
        merged = _deep_merge(merged, default_raw)

    merged = _deep_merge(merged, raw)
    merged.setdefault("experiment_name", os.path.splitext(os.path.basename(path))[0])
    return merged


class ConfigDict(dict):
    """Dict subclass that also supports attribute access, e.g. cfg.model.lr"""

    def __getattr__(self, item):
        try:
            value = self[item]
        except KeyError as e:
            raise AttributeError(item) from e
        if isinstance(value, dict):
            return ConfigDict(value)
        return value

    __setattr__ = dict.__setitem__


def load_config_obj(path: str) -> ConfigDict:
    return ConfigDict(load_config(path))

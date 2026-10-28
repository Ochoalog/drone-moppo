"""YAML config loading with single-level inheritance (`base:` key) and deep merge."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"
FIGS = RESULTS / "figs"


def deep_merge(a: dict, b: dict) -> dict:
    """Return a copy of `a` updated recursively with `b`."""
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path, overrides: dict | None = None) -> dict[str, Any]:
    """Load a YAML config; `base: other.yaml` is resolved relative to the file."""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = ROOT / path
    cfg = yaml.safe_load(path.read_text()) or {}
    base = cfg.pop("base", None)
    if base:
        cfg = deep_merge(load_config(path.parent / base), cfg)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def default_config() -> dict[str, Any]:
    return load_config(ROOT / "configs" / "default.yaml")

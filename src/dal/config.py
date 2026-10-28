"""YAML config loading with single-level inheritance (`base:` key) and deep merge."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
# Overridable by environment variable (read at import time); defaults are the Taquari PoC folders.
DATA = Path(os.environ.get("DAL_DATA_DIR", ROOT / "data"))
RUNS = Path(os.environ.get("DAL_RUNS_DIR", ROOT / "runs"))
RESULTS = Path(os.environ.get("DAL_RESULTS_DIR", ROOT / "results"))
FIGS = RESULTS / "figs"


def deep_merge(a: dict, b: dict) -> dict:
    """Return a copy of `a` updated recursively with `b`. A dict in `b` carrying `__replace__: true`
    replaces the inherited dict instead of being merged into it."""
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and v.get("__replace__"):
            out[k] = {kk: copy.deepcopy(vv) for kk, vv in v.items() if kk != "__replace__"}
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
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


def paths(cfg: dict) -> dict[str, Path]:
    """Output folders of a config (`paths:` block, relative to ROOT); defaults are DATA/RUNS/RESULTS."""
    p = cfg.get("paths") or {}

    def get(key: str, default: Path) -> Path:
        v = p.get(key)
        return default if v is None else (Path(v) if Path(v).is_absolute() else ROOT / v)

    out = dict(data=get("data", DATA), runs=get("runs", RUNS), results=get("results", RESULTS))
    out["figs"] = out["results"] / "figs"
    return out

"""Fumaça do scripts/verdicts_v3.py com episódios sintéticos no formato do evaluate_job."""
import importlib.util

import numpy as np
import pandas as pd

from dal.config import ROOT


def _load_script():
    spec = importlib.util.spec_from_file_location("verdicts_v3", ROOT / "scripts" / "verdicts_v3.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frame(method, seed, region, scenario, split, mode, success, n=120, rng=None):
    rng = rng or np.random.default_rng(0)
    return pd.DataFrame({"method": method, "model_seed": seed, "region": region, "scenario": scenario,
                         "split": split, "mode": mode, "episode": np.arange(n), "window_r0": np.arange(n) % 12,
                         "window_c0": 0, "success": rng.random(n) < success, "false_loc": rng.random(n) < 0.02,
                         "flight_time_s": 100 + rng.normal(0, 5, n), "mean_altitude_m": 100.0,
                         "window_change_prior": rng.random(n), "gate_original_passed": seed % 2 == 0})


def test_verdict_script_runs_all_modes_and_gate_criteria():
    script = _load_script()
    rng = np.random.default_rng(1)
    frames = []
    places = [("taquari", "S2", "test"), ("taquari", "S2", "train"), ("poa_ilhas", "P1", "test"),
              ("poa_centro", "P1", "test"), ("canoas", "C1", "test")]
    for region, scenario, split in places:
        for name in ("straight", "straight_ca", "lawnmower_ca", "random_ca", "greedy_map_ca", "straight_150_ca"):
            frames.append(_frame(name, -1, region, scenario, split, "—", 0.9, rng=rng))
        for mode in ("argmax", "sampled"):
            for method in ("ppo", "ppo_ca", "ppo_ca_alt", "ppo_ca_stdfilter"):
                for seed in range(4):
                    frames.append(_frame(method, seed, region, scenario, split, mode, 0.92, rng=rng))
    episodes = pd.concat(frames, ignore_index=True)
    for mode in ("argmax", "sampled"):
        for gate in ("emended", "original"):
            result, table = script.run(episodes, mode, gate)
            assert {"H2", "H3", "H4", "H5"} <= set(result)
            assert len(table) == 4
            assert result["H5"]["verdict"] in ("suportada", "não suportada", "não avaliável")

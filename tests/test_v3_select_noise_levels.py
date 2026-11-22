"""Regra de escolha dos níveis do E-synth (V46)."""
import importlib.util

import pandas as pd
import pytest

from dal.config import ROOT

spec = importlib.util.spec_from_file_location("select_noise_levels_v3", ROOT / "scripts" / "select_noise_levels_v3.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _scan(successes):
    sigmas = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
    return pd.DataFrame({"sigma": sigmas[:len(successes)], "random_success": successes,
                         "kappa": 0.04, "calibration_file": "x"})


def test_picks_nearest_to_each_target_in_order():
    result = module.select_levels(_scan([0.92, 0.80, 0.70, 0.56, 0.40, 0.30, 0.10, 0.02]))
    assert {k: v["sigma"] for k, v in result["levels"].items()} == {"L1": 0.5, "L2": 1.0, "L3": 1.25}
    assert result["invalid_sigmas"] == [3.0]


def test_tie_goes_to_smaller_sigma_and_sigmas_are_not_repeated():
    result = module.select_levels(_scan([0.80, 0.70, 0.60, 0.60, 0.60]))
    levels = {k: v["sigma"] for k, v in result["levels"].items()}
    assert levels["L1"] == 0.25  # |0,80 − 0,75| = |0,70 − 0,75|: empate → menor σ
    assert len(set(levels.values())) == 3


def test_invalid_sigma_is_skipped_for_the_nearest_valid_one():
    result = module.select_levels(_scan([0.90, 0.76, 0.56, 0.36, 0.03]))
    assert result["levels"]["L3"]["sigma"] == 1.0
    result = module.select_levels(_scan([0.90, 0.76, 0.56, 0.20, 0.04]))  # 0,04 < 5%: fora; L3 = 0,20
    assert result["levels"]["L3"]["sigma"] == 1.0


def test_stops_when_no_valid_level_is_below_75_percent():
    with pytest.raises(module.NoValidLevelError):
        module.select_levels(_scan([0.95, 0.90, 0.80, 0.03, 0.01]))

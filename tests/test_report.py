"""report.build on synthetic episode tables covering the full overnight layout."""
import numpy as np
import pandas as pd

from dal import report
from helpers import small_cfg


def _episodes(split: str, n: int, rng) -> pd.DataFrame:
    rows = []
    p = {"S0": dict(straight=1.0, lawnmower=1.0, random=1.0, greedy=1.0, greedy_map=1.0, ppo=1.0),
         "S1": dict(straight=0.8, lawnmower=0.75, random=0.6, greedy=0.85, greedy_map=0.8, ppo=0.9),
         "S2": dict(straight=0.5, lawnmower=0.45, random=0.4, greedy=0.6, greedy_map=0.5, ppo=0.75)}
    for scen, d in p.items():
        for m, ps in d.items():
            for ms in ([0, 1, 2] if m == "ppo" else [-1]):
                for i in range(n):
                    ok = rng.random() < ps
                    fl = (not ok) and rng.random() < 0.3
                    st = 1 if scen == "S0" else int(rng.integers(3, 40)) if ok else int(rng.integers(5, 81))
                    rows.append(dict(method=m, model_seed=ms, scenario=scen, descriptor="hand", split=split, episode=i,
                                     seed=i, success=ok, false_loc=fl, steps=st, steps_to_success=st if ok else np.nan,
                                     final_error_m=0 if ok else 300.0, distance_m=50.0 * st, n_blocked=0,
                                     final_entropy=1.0, decision_ms=100.0 if m.startswith("greedy") else 1.0))
    return pd.DataFrame(rows)


def test_report_builds_with_all_sections(tmp_path):
    rng = np.random.default_rng(0)
    res = tmp_path / "results"
    res.mkdir()
    _episodes("test", 120, rng).to_csv(res / "episodes_test.csv", index=False)
    _episodes("train", 60, rng).to_csv(res / "episodes_train.csv", index=False)
    cfg = small_cfg()
    cfg["eval"]["bootstrap"] = 200
    text = report.build(cfg, res, [tmp_path / "no_runs"], tmp_path / "REPORT.md")
    for h in ("### H1", "### H2", "### H3"):
        assert h in text
    assert "degenerate" in text and "S0" in text
    assert (res / "summary_test.csv").exists() and (res / "paired_tests_test.csv").exists()
    assert (res / "figs" / "success_vs_steps.png").exists()
    t = pd.read_csv(res / "paired_tests_test.csv")
    assert set(t.vs) == {"straight", "lawnmower", "random", "greedy", "greedy_map"}

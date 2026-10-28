import pandas as pd

from dal import evaluate as ev
from helpers import small_cfg, synthetic_data

EXPECTED = {"method", "scenario", "episode", "seed", "success", "false_loc", "steps", "steps_to_success",
            "final_error_m", "distance_m", "decision_ms"}


def test_evaluate_writes_expected_columns(tmp_path):
    cfg = small_cfg()
    cfg["env"]["descriptor"] = "synthetic"
    ev._CACHE[("synthetic", "SYN")] = synthetic_data()
    out = tmp_path / "ep.csv"
    df = ev.evaluate(cfg, ["SYN"], ["straight", "lawnmower", "random", "greedy"], None, 4, "test", out,
                     workers=1, save_traj=0)
    disk = pd.read_csv(out)
    assert EXPECTED <= set(disk.columns)
    assert len(disk) == 16 and set(disk.method) == {"straight", "lawnmower", "random", "greedy"}
    # paired design: every method sees the same seeds
    assert disk.groupby("method").seed.apply(tuple).nunique() == 1
    assert (disk.steps >= 1).all() and (disk.steps <= cfg["filter"]["max_steps"]).all()
    assert not (df.success & df.false_loc).any()

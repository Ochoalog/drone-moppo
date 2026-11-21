"""Fluxo completo dos jobs (calibração → treino → avaliação) num local sintético pequeno, na CPU."""
import numpy as np
import pandas as pd
import pytest

from dal.v3 import jobs, pipeline
from dal.v3.sensor import specs

SITE = specs.Site(name="taquari", aoi={"lon0": -51.95, "lon1": -51.915, "lat0": -29.33, "lat1": -29.235},
                  obs_date="post", map_dates=("pre_oct23", "post"), sources={"pre_oct23": "a", "post": "b"},
                  offsets={})
DIM, K = 12, 4


@pytest.fixture
def out_root(tmp_path, monkeypatch):
    monkeypatch.setattr(specs, "load_site", lambda name: SITE)
    jobs._TABLES_CACHE.clear()
    grid = specs.site_grid(SITE)
    n = grid.nrows * grid.ncols
    rng = np.random.default_rng(0)
    base = {h: rng.normal(size=(n, DIM)) for h in specs.ALTITUDES_M}
    for altitude in specs.ALTITUDES_M:
        for date, kind, data in (("pre_oct23", "map", base[altitude]), ("post", "map", base[altitude]),
                                 ("post", "obs", np.concatenate([base[altitude] + 0.3 * rng.normal(size=(n, DIM))
                                                                 for _ in range(K)]))):
            directory = pipeline.spec_dir(tmp_path / "features" / "v3", "fake", "taquari", date, kind, altitude)
            directory.mkdir(parents=True)
            np.save(directory / "gem.npy", data.astype(np.float16))
            np.save(directory / "valid_frac.npy", np.ones((len(data), 1), np.float16))
            np.save(directory / "cloud_frac.npy", np.zeros((len(data), 1), np.float16))
    return tmp_path


def test_calibrate_train_and_evaluate(out_root):
    job = {"descriptor": "fake", "fold": "A", "episodes": 8}
    calib = jobs.calibrate_job(job, out_root, "cpu")
    assert set(calib["beta1"]) == set(specs.ALTITUDES_M) and calib["alpha"] == 0.0  # sem prior → α = 0
    train_job = {**job, "method": "ppo", "seed": 0, "total_timesteps": 256, "n_envs": 8, "gate_min_advantage": -1.0}
    trained = jobs.train_job(train_job, out_root, "cpu", "ppo_s0")
    assert trained["global_step"] >= 256 and trained["steps_per_s_median"] > 0 and not trained["invalid_gate"]
    # Gate reprovado (no local sintético a aleatória já acerta tudo): a execução vira INVALID_GATE.
    failed = jobs.train_job({**train_job, "seed": 1, "gate_min_advantage": 0.05}, out_root, "cpu", "ppo_s1")
    assert failed["invalid_gate"]
    eval_job = {**job, "episodes": 6, "targets": [{"site": "taquari", "scenario": "S2"}],
                "baselines": ["straight", "lawnmower"],
                "models": [{"method": "ppo", "seed": 0, "run": "ppo_s0"},
                           {"method": "ppo", "seed": 1, "run": "ppo_s1"}]}
    episodes = jobs.evaluate_job(eval_job, out_root, "cpu", "eval_test")
    assert set(episodes.method) == {"straight", "straight_ca", "lawnmower", "lawnmower_ca", "ppo"}
    assert (episodes.groupby("method").size() == 6).all()
    assert set(episodes[episodes.method == "ppo"].model_seed) == {0}  # a seed 1 (INVALID_GATE) ficou de fora
    # Pareamento: as mesmas janelas para todos os métodos.
    windows = episodes.pivot_table(index="episode", columns="method", values="window_r0")
    assert (windows.nunique(axis=1) == 1).all()
    for column in ("source_file", "protocol", "n_episodes", "fold", "git_commit", "config_hash"):
        assert column in episodes.columns
    assert isinstance(episodes, pd.DataFrame)


def test_restrict_to_region_keeps_windows_inside_poa_subregion():
    import numpy as np

    from dal.v3.envs.realistic_env import PerceptionTables, valid_windows
    from dal.v3.jobs import restrict_to_region
    from dal.v3.sensor import specs

    grid = specs.site_grid(specs.load_site("poa"))
    shape = (grid.nrows, grid.ncols)
    ok = np.ones(shape, bool)
    tables = PerceptionTables({}, {}, ok, ok, {}, shape, np.arange(grid.nrows))
    full = valid_windows(tables, 128, 0.9)
    for region in ("ilhas", "centro"):
        restricted = restrict_to_region(tables, "poa", region)
        windows = valid_windows(restricted, 128, 0.9)
        assert 0 < len(windows) < len(full)
        cols = set(restricted.cols_allowed.tolist())
        assert all(c0 in cols and c0 + 127 in cols for _, c0 in windows)
    assert restrict_to_region(tables, "poa", None) is tables

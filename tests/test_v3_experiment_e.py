"""Experimento E (DECISIONS_V3.md V46): divisão de Canoas sem vazamento, ruído na observação, seleção dos níveis."""
import json

import numpy as np
import pytest
import torch

from dal.v3 import jobs, pipeline
from dal.v3.descriptors.obs_noise import add_observation_noise, noisy_observations
from dal.v3.envs.realistic_env import PerceptionTables, valid_windows
from dal.v3.evaluation import site_split
from dal.v3.sensor import specs

DIM, K = 12, 4
TAQUARI = specs.Site(name="taquari", aoi={"lon0": -51.95, "lon1": -51.915, "lat0": -29.33, "lat1": -29.235},
                     obs_date="post", map_dates=("pre_oct23", "post"), sources={"pre_oct23": "a", "post": "b"},
                     offsets={})
# Canoas sintética: 192 × 480 células (5 × 4 = 20 janelas na faixa de teste de 180 colunas).
CANOAS = specs.Site(name="canoas", aoi={"x0": 474000.0, "x1": 474000.0 + 480 * 25, "y0": 6687000.0,
                                        "y1": 6687000.0 + 192 * 25},
                    obs_date="poa_post", map_dates=("poa_pre_feb24", "poa_post"), sources={}, offsets={})


def _write_site(root, site, dates):
    grid = specs.site_grid(site)
    n = grid.nrows * grid.ncols
    rng = np.random.default_rng(hash(site.name) % 2**32)
    base = {h: rng.normal(size=(n, DIM)) for h in specs.ALTITUDES_M}
    for altitude in specs.ALTITUDES_M:
        items = [(date, "map", base[altitude]) for date in dates] + [
            (site.obs_date, "obs", np.concatenate([base[altitude] + 0.3 * rng.normal(size=(n, DIM))
                                                   for _ in range(K)]))]
        for date, kind, data in items:
            directory = pipeline.spec_dir(root / "features" / "v3", "fake", site.name, date, kind, altitude)
            directory.mkdir(parents=True, exist_ok=True)
            np.save(directory / "gem.npy", data.astype(np.float16))
            np.save(directory / "valid_frac.npy", np.ones((len(data), 1), np.float16))
            np.save(directory / "cloud_frac.npy", np.zeros((len(data), 1), np.float16))


@pytest.fixture
def out_root(tmp_path, monkeypatch):
    monkeypatch.setattr(specs, "load_site", lambda name: {"taquari": TAQUARI, "canoas": CANOAS}[name])
    jobs._TABLES_CACHE.clear()
    _write_site(tmp_path, TAQUARI, ("pre_oct23", "post"))
    _write_site(tmp_path, CANOAS, ("poa_pre_feb24", "poa_post"))
    return tmp_path


def _tables(shape, ok=None):
    ok = np.ones(shape, bool) if ok is None else ok
    return PerceptionTables({}, {}, ok, ok, {}, shape, np.arange(shape[0]))


# ------------------------------------------------------------------ divisão por colunas
def test_column_split_on_full_canoas_box_uses_minimum_width():
    split = site_split.choose_column_split(_tables((320, 480)))
    assert split.test_width == 180 and split.margin == 8 and split.train_end == 292
    assert split.n_test_windows >= 20
    train = valid_windows(site_split.with_columns(_tables((320, 480)), split.cols("train")), 128, 0.7)
    test = valid_windows(site_split.with_columns(_tables((320, 480)), split.cols("test")), 128, 0.7)
    assert train[:, 1].max() + 128 <= split.train_end  # nenhuma janela de treino toca a margem ou o teste
    assert test[:, 1].min() >= split.test_start


def test_column_split_widens_when_east_is_invalid_and_gives_up_beyond_two_windows():
    ok = np.ones((320, 480), bool)
    ok[:, 400:] = False  # as últimas 80 colunas inválidas: a faixa precisa crescer
    split = site_split.choose_column_split(_tables((320, 480), ok))
    assert split is not None and split.test_width > 180 and split.n_test_windows >= 20
    ok[:, 250:] = False
    assert site_split.choose_column_split(_tables((320, 480), ok)) is None  # parar e perguntar (V46)


def test_split_diagnostic_counts_windows_per_side_only_from_masks():
    ok = np.ones((320, 480), bool)
    ok[:, 250:] = False  # leste inválido: o oeste e as linhas ainda têm janelas
    rows = {(r["side"], r["width"], r["min_valid"]): r["n_windows"]
            for r in site_split.split_diagnostic(_tables((320, 480), ok), min_valids=(0.7,))}
    assert rows[("east", 256, 0.7)] == 0 and rows[("west", 256, 0.7)] > 20
    assert rows[("all", None, 0.7)] == len(valid_windows(_tables((320, 480), ok), 128, 0.7))
    assert rows[("south", 320, 0.7)] == rows[("all", None, 0.7)]


def test_canoas_calibration_and_training_never_touch_test_or_margin_columns(out_root, monkeypatch):
    base = {"descriptor": "fake", "fold": "A", "episodes": 8}
    jobs.calibrate_job(base, out_root, "cpu")  # PCA da dobra A do Taquari (aplicada em Canoas)
    seen = []
    original_kappa = jobs.calibration.calibrate_kappa

    def spy_kappa(tables, *args, **kwargs):
        seen.append(valid_windows(tables, 128, 0.7))
        return original_kappa(tables, *args, **kwargs)

    monkeypatch.setattr(jobs.calibration, "calibrate_kappa", spy_kappa)
    calib = jobs.calibrate_job({**base, "site": "canoas", "column_split": True, "kappas": [0.5, 0.1],
                                "calibration_file": "calibration_canoas.json"}, out_root, "cpu")
    split = site_split.ColumnSplit.from_dict(calib["column_split"])
    assert split.test_width == 180 and split.train_end == 480 - 180 - 8
    assert seen and seen[0][:, 1].max() + 128 <= split.train_end
    directory = jobs.work_dir(out_root, "fake", "A")
    assert json.loads((directory / "calibration.json").read_text())["site"] == "taquari"  # Taquari intacta
    job = {**base, "site": "canoas", "site_calibration": {"canoas": "calibration_canoas.json"}}
    env = jobs.make_env(job, out_root, "canoas", "C1", "train", False, False, "cpu")
    assert env.windows[:, 1].max() + 128 <= split.train_end
    test_env = jobs.make_env(job, out_root, "canoas", "C1", "test", False, False, "cpu")
    assert test_env.windows[:, 1].min() >= split.test_start
    trained = jobs.train_job({**job, "method": "ppo", "seed": 0, "total_timesteps": 256, "n_envs": 8,
                              "gate_min_advantage": 0.05}, out_root, "cpu", "E_canoas_ppo_s0")
    assert trained["global_step"] >= 256  # o treino vai até o fim mesmo com o gate reprovado
    eval_job = {**job, "episodes": 4, "include_invalid_gate": True, "baseline_filters": ["standard"],
                "baselines": ["random"], "modes": ["argmax", "sampled"],
                "targets": [{"site": "canoas", "scenario": "C1", "split": "test"}],
                "models": [{"method": "ppo", "seed": 0, "run": "E_canoas_ppo_s0"}]}
    episodes = jobs.evaluate_job(eval_job, out_root, "cpu", "E_eval_test")
    assert set(episodes.method) == {"random", "ppo"}  # sem as variantes CA
    ppo = episodes[episodes.method == "ppo"]
    assert set(ppo["mode"]) == {"argmax", "sampled"}
    assert ppo.invalid_gate.all() == trained["invalid_gate"]
    assert (episodes.window_c0 >= split.test_start).all()


def test_invalid_gate_runs_only_enter_with_explicit_flag(tmp_path):
    (tmp_path / "runs" / "x_INVALID_GATE").mkdir(parents=True)
    assert jobs.resolve_run_dir(tmp_path, "x", include_invalid_gate=False) is None
    assert jobs.resolve_run_dir(tmp_path, "x", include_invalid_gate=True).name == "x_INVALID_GATE"


# ------------------------------------------------------------------ ruído na observação
def _noise_tables(seed=0):
    rng = np.random.default_rng(seed)
    obs = rng.normal(size=(K, 50, 16)).astype(np.float32)
    obs /= np.linalg.norm(obs, axis=-1, keepdims=True)
    maps = rng.normal(size=(50, 16)).astype(np.float32)
    ok = np.ones((5, 10), bool)
    return PerceptionTables({100: maps}, {100: obs}, ok, ok, {100: 1.0}, (5, 10), np.arange(5))


def test_noise_zero_is_identity_and_noise_is_deterministic_unit_norm_and_leaves_map():
    tables = _noise_tables()
    assert add_observation_noise(tables, 0.0) is tables
    first = add_observation_noise(tables, 1.0, seed=0)
    second = add_observation_noise(tables, 1.0, seed=0)
    other = add_observation_noise(tables, 1.0, seed=1)
    np.testing.assert_array_equal(first.obs_desc[100], second.obs_desc[100])
    assert not np.allclose(first.obs_desc[100], other.obs_desc[100])
    np.testing.assert_allclose(np.linalg.norm(first.obs_desc[100], axis=-1), 1.0, atol=1e-5)
    assert first.map_desc[100] is tables.map_desc[100]
    assert not np.allclose(first.obs_desc[100], tables.obs_desc[100])


def test_noise_is_independent_of_block_size_per_block_seed_and_grows_with_sigma():
    tables = _noise_tables()
    obs = tables.obs_desc[100]
    small = noisy_observations(obs, 0.25, 0, 100)
    large = noisy_observations(obs, 2.0, 0, 100)
    cos_small = (small * obs).sum(-1).mean()
    cos_large = (large * obs).sum(-1).mean()
    assert 1.0 > cos_small > cos_large  # mais σ, menos parecida com a observação original
    # O ruído por bloco é fixo: a mesma semente e o mesmo bloco dão o mesmo sorteio.
    torch.manual_seed(123)  # o gerador global não interfere
    np.testing.assert_array_equal(small, noisy_observations(obs, 0.25, 0, 100))


def test_all_methods_of_a_level_share_the_same_noisy_table(out_root):
    jobs.calibrate_job({"descriptor": "fake", "fold": "A", "episodes": 8}, out_root, "cpu")
    scan = jobs.scan_noise_job({"descriptor": "fake", "fold": "A", "episodes": 8, "sigmas": [0.5],
                                "kappas": [0.5, 0.1]}, out_root, "cpu")
    assert list(scan.sigma) == [0.5] and {"random_success", "top1_100m", "kappa"} <= set(scan.columns)
    job = {"descriptor": "fake", "fold": "A", "calibration_file": "calibration_noise_0.5.json",
           "obs_noise_sigma": 0.5, "obs_noise_seed": 0}
    first = jobs.make_env(job, out_root, "taquari", "S2", "test", False, False, "cpu")
    second = jobs.make_env(job, out_root, "taquari", "S2", "test", False, True, "cpu")
    assert torch.equal(first.obs_desc[100], second.obs_desc[100])
    clean = jobs.make_env({"descriptor": "fake", "fold": "A"}, out_root, "taquari", "S2", "test", False, False, "cpu")
    assert not torch.equal(first.obs_desc[100], clean.obs_desc[100])
    assert torch.equal(first.map_desc[100], clean.map_desc[100])
    with pytest.raises(ValueError):  # calibração de outro σ
        jobs.make_env({**job, "obs_noise_sigma": 1.0}, out_root, "taquari", "S2", "test", False, False, "cpu")


def test_failed_canoas_split_carries_masks_only_diagnostic(out_root, monkeypatch):
    base = {"descriptor": "fake", "fold": "A", "episodes": 8}
    jobs.calibrate_job(base, out_root, "cpu")
    monkeypatch.setattr(jobs, "choose_column_split", lambda *a, **k: None)
    with pytest.raises(jobs.SplitUnavailableError) as failure:
        jobs.calibrate_job({**base, "site": "canoas", "column_split": True, "kappas": [0.5],
                            "calibration_file": "calibration_canoas.json"}, out_root, "cpu")
    assert {r["side"] for r in failure.value.diagnostic} == {"all", "east", "west", "south", "north"}
    assert len(failure.value.column_valid) == 480
    assert not (jobs.work_dir(out_root, "fake", "A") / "calibration_canoas.json").exists()

"""Pipeline: tabelas a partir do layout de descritores do notebook 01 (local sintético pequeno)."""
import numpy as np
import pytest

from dal.v3 import pipeline
from dal.v3.sensor import specs

SITE = specs.Site(name="taquari", aoi={"lon0": -51.95, "lon1": -51.915, "lat0": -29.33, "lat1": -29.235},
                  obs_date="post", map_dates=("pre_oct23", "post"), sources={"pre_oct23": "a", "post": "b"},
                  offsets={})
DIM, K = 16, 4


@pytest.fixture
def features(tmp_path, monkeypatch):
    monkeypatch.setattr(specs, "load_site", lambda name: SITE)
    grid = specs.site_grid(SITE)
    n = grid.nrows * grid.ncols
    rng = np.random.default_rng(0)
    for altitude in (100,):
        for date, kind, rows in (("pre_oct23", "map", n), ("post", "map", n), ("post", "obs", K * n)):
            directory = pipeline.spec_dir(tmp_path, "fake", "taquari", date, kind, altitude)
            directory.mkdir(parents=True)
            np.save(directory / "gem.npy", rng.normal(size=(rows, DIM)).astype(np.float16))
            np.save(directory / "valid_frac.npy", np.ones((rows, 1), np.float16))
            np.save(directory / "cloud_frac.npy", np.zeros((rows, 1), np.float16))
    return tmp_path, grid


def test_fold_tables_use_training_band_and_project_to_pca(features):
    root, grid = features
    request = pipeline.DataRequest("taquari", "S2", "A", descriptor="fake", altitudes=(100,), split="test")
    tables, fits = pipeline.load_tables(root, request, beta={100: 10.0})
    assert tables.map_desc[100].shape == (grid.nrows * grid.ncols, min(DIM, 128))
    assert tables.obs_desc[100].shape[0] == K
    assert tables.rows_allowed.min() > grid.nrows // 2  # dobra A, divisão de teste = faixa sul
    assert "distinct" in tables.extra_channels and 100 in fits


def test_zero_shot_requires_taquari_pca(features):
    root, _ = features
    request = pipeline.DataRequest("taquari", "S2", None, descriptor="fake", altitudes=(100,))
    with pytest.raises(ValueError):
        pipeline.load_tables(root, request, beta={100: 10.0})
    _, fits = pipeline.load_tables(root, pipeline.DataRequest("taquari", "S2", "A", descriptor="fake",
                                                              altitudes=(100,)), beta={100: 10.0})
    tables, used = pipeline.load_tables(root, request, beta={100: 10.0}, pca_fits=fits)
    assert used[100] is fits[100]

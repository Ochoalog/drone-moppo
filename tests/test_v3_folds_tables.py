"""Dobras espaciais e montagem das tabelas: faixas disjuntas e PCA sem vazamento da faixa de teste."""
import numpy as np
import pytest

from dal.v3.descriptors.tables import build_tables
from dal.v3.evaluation.folds import make_fold


@pytest.mark.parametrize("name", ["A", "B", "C"])
def test_folds_have_disjoint_train_and_test_with_margin(name):
    fold = make_fold(name, n_rows=926, cell_m=25.0, window_cells=128)
    assert not set(fold.train_rows).intersection(fold.test_rows)
    gap = min(abs(int(t) - int(r)) for t in (fold.test_rows.min(), fold.test_rows.max()) for r in fold.train_rows)
    assert gap >= 8  # 200 m de margem = 8 células
    assert len(fold.test_rows) >= 128


def test_fold_a_tests_the_southern_band():
    fold = make_fold("A", n_rows=926, cell_m=25.0, window_cells=128)
    assert fold.test_rows.min() > fold.train_rows.max()


def test_pca_ignores_cells_outside_the_training_band():
    rng = np.random.default_rng(0)
    rows, cols, dim = 30, 10, 24
    map_raw = {100: rng.normal(size=(rows * cols, dim)).astype(np.float32)}
    obs_raw = {100: rng.normal(size=(2, rows * cols, dim)).astype(np.float32)}
    ok = np.ones((rows, cols), bool)
    train_rows = np.arange(0, 15)
    first, _ = build_tables(map_raw, obs_raw, ok, ok, np.arange(rows), train_rows, {100: 10.0}, dim=8)
    # Muda só as células de teste: a projeção das células de treino não pode mudar.
    test_cells = np.arange(15 * cols, rows * cols)
    map_raw[100][test_cells] += 100.0
    obs_raw[100][:, test_cells] -= 50.0
    second, _ = build_tables(map_raw, obs_raw, ok, ok, np.arange(rows), train_rows, {100: 10.0}, dim=8)
    train_cells = np.arange(0, 15 * cols)
    np.testing.assert_allclose(first.map_desc[100][train_cells], second.map_desc[100][train_cells], atol=1e-6)


def test_fit_pca_covariance_matches_svd_subspace():
    """Patch da IA-2 (d398eb0): PCA pela covariância dá os mesmos componentes do SVD, a menos do sinal."""
    from dal.v3.descriptors.tables import fit_pca

    rng = np.random.default_rng(0)
    x = (rng.normal(size=(2000, 32)) @ rng.normal(size=(32, 32))).astype(np.float32)
    fit = fit_pca(x, 8)
    _, _, vt = np.linalg.svd(x - x.mean(0), full_matrices=False)
    np.testing.assert_allclose(np.abs(np.sum(fit.components * vt[:8], axis=1)), 1.0, atol=1e-4)
    np.testing.assert_allclose(fit.mean, x.mean(0), atol=1e-5)

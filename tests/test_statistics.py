"""Testes do bootstrap hierárquico e da reconstrução de janelas (P5 do plano v2)."""
import numpy as np
import pytest

from dal.evaluation.statistics import (
    MethodMatrix,
    hierarchical_bootstrap,
    percentile_ci,
    reconstruct_episode_window,
    summarize_with_ci,
)


def test_window_reconstruction_matches_episode_reset():
    from dal.env import Episode
    from helpers import small_cfg, synthetic_data

    config = small_cfg()
    episode = Episode(synthetic_data(), config["filter"], "train", config["data"]["cell_m"])
    for seed in range(2024, 2054):
        episode.reset(seed=seed)
        assert reconstruct_episode_window(episode.windows, seed) == (episode.r0, episode.c0)


def test_identical_methods_have_zero_paired_difference():
    values = np.random.default_rng(0).random((1, 50))
    clusters = np.repeat(np.arange(10), 5)
    table = summarize_with_ci({"a": MethodMatrix(values), "b": MethodMatrix(values.copy())}, clusters,
                              n_boot=200, reference="a")
    row = table.set_index("method").loc["b"]
    assert row.diff_ci_low == pytest.approx(0.0) and row.diff_ci_high == pytest.approx(0.0)


def test_clustering_widens_interval_when_windows_are_correlated():
    # 10 janelas, 20 episódios cada, todos com o mesmo resultado dentro da janela.
    rng = np.random.default_rng(1)
    window_outcome = rng.integers(0, 2, size=10).astype(float)
    values = np.repeat(window_outcome, 20)[None, :]
    by_window = hierarchical_bootstrap({"m": MethodMatrix(values)}, np.repeat(np.arange(10), 20), n_boot=500)["m"]
    by_episode = hierarchical_bootstrap({"m": MethodMatrix(values)}, np.arange(200), n_boot=500)["m"]
    width = lambda r: np.subtract(*percentile_ci(r)[::-1])  # noqa: E731
    assert width(by_window) > 2 * width(by_episode)


def test_seed_level_resampling_reflects_seed_variation():
    # 3 seeds com taxas muito diferentes: o IC precisa cobrir essa variação.
    values = np.stack([np.full(100, p) for p in (0.6, 0.7, 0.8)])
    replicates = hierarchical_bootstrap({"m": MethodMatrix(values)}, np.arange(100), n_boot=500)["m"]
    low, high = percentile_ci(replicates)
    assert low <= 0.65 and high >= 0.75

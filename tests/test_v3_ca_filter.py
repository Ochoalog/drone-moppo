"""Filtro ciente da mudança (CA): α = 0 é idêntico ao padrão; propriedades da mistura."""
import numpy as np
import pytest
import torch

from dal.v3.envs.batched_env import BatchedLocalizationEnv
from dal.v3.localization.ca_filter import change_aware_likelihood
from helpers import small_cfg, synthetic_data


def test_alpha_zero_returns_the_standard_likelihood():
    likelihood = torch.rand(3, 8, 8, dtype=torch.float64)
    prior = torch.rand(3, 8, 8, dtype=torch.float64)
    assert torch.equal(change_aware_likelihood(likelihood, prior, 0.0), likelihood)


def test_fully_changed_cells_get_the_mean_likelihood():
    likelihood = torch.rand(2, 5, 5, dtype=torch.float64)
    out = change_aware_likelihood(likelihood, torch.ones(2, 5, 5), 1.0)
    expected = likelihood.reshape(2, -1).mean(1).view(2, 1, 1).expand(2, 5, 5)
    torch.testing.assert_close(out, expected)


def test_unchanged_cells_keep_their_likelihood():
    likelihood = torch.rand(1, 4, 4, dtype=torch.float64)
    prior = torch.zeros(1, 4, 4)
    prior[0, 0, 0] = 1.0
    out = change_aware_likelihood(likelihood, prior, 0.5)
    assert torch.equal(out[0, 1:], likelihood[0, 1:])
    assert out[0, 0, 0] != likelihood[0, 0, 0]


def test_alpha_out_of_range_is_rejected():
    with pytest.raises(ValueError):
        change_aware_likelihood(torch.ones(1, 2, 2), torch.ones(1, 2, 2), 1.5)


def test_batched_env_with_alpha_zero_matches_env_without_prior():
    config = small_cfg()
    data = synthetic_data()
    seeds = list(range(10))
    prior = np.random.default_rng(0).random(data.map_ok.shape)
    plain = BatchedLocalizationEnv(data, config["filter"], "train")
    ca_zero = BatchedLocalizationEnv(data, config["filter"], "train", change_prior=prior, alpha=0.0)
    for env in (plain, ca_zero):
        env.reset(seeds)
    actions = torch.tensor(np.random.default_rng(1).integers(8, size=(15, len(seeds))))
    for step in actions:
        plain.step(step)
        ca_zero.step(step)
    assert torch.equal(plain.belief, ca_zero.belief)
    assert plain.trajectories == ca_zero.trajectories


def test_batched_env_alpha_changes_the_belief_where_the_prior_is_high():
    config = small_cfg()
    data = synthetic_data()
    prior = np.ones(data.map_ok.shape)
    plain = BatchedLocalizationEnv(data, config["filter"], "train")
    full_ca = BatchedLocalizationEnv(data, config["filter"], "train", change_prior=prior, alpha=1.0)
    plain.reset([3])
    full_ca.reset([3])
    # Com m = 1 e α = 1 a verossimilhança é constante: a crença inicial fica uniforme nas células válidas.
    valid = full_ca.map_ok[0].double()
    torch.testing.assert_close(full_ca.belief[0], valid / valid.sum())
    assert not torch.allclose(plain.belief[0], full_ca.belief[0])

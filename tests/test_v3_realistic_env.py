"""Ambiente realista: deriva com a média e o desvio configurados, pareamento do ruído, massa, altitude e tempo."""
import numpy as np
import pytest
import torch

from dal.v3.envs.realistic_env import (
    ACTION_UP,
    PerceptionTables,
    RealisticBatchedEnv,
    RealisticConfig,
    apply_drift,
    gaussian_blur_batched,
    sample_drift,
)

ROWS, COLS, DIM, K = 48, 40, 16, 4


def _tables(same_map_and_obs: bool = True, seed: int = 0) -> PerceptionTables:
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(ROWS * COLS, DIM)).astype(np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    obs = np.stack([base + (0 if same_map_and_obs else 0.3) * rng.normal(size=base.shape) for _ in range(K)])
    obs = (obs / np.linalg.norm(obs, axis=-1, keepdims=True)).astype(np.float32)
    altitudes = (60, 100, 150)
    return PerceptionTables(map_desc={h: base for h in altitudes}, obs_desc={h: obs for h in altitudes},
                            map_ok=np.ones((ROWS, COLS), bool), obs_ok=np.ones((ROWS, COLS), bool),
                            beta={h: 40.0 for h in altitudes}, grid_shape=(ROWS, COLS), rows_allowed=np.arange(ROWS),
                            change_prior=rng.uniform(size=(ROWS, COLS)))


def _config(**overrides) -> RealisticConfig:
    return RealisticConfig(window_cells=32, max_steps=30, **overrides)


@pytest.mark.parametrize("fraction", [0.0, 0.02, 0.05])
def test_drift_has_the_configured_mean_and_spread(fraction):
    eps, dtheta = sample_drift(np.random.default_rng(0), 200_000, fraction)
    command = np.tile([[0.0, 50.0]], (len(eps), 1))
    actual = apply_drift(command, eps, dtheta)
    error = actual - command
    # Média: zero no eixo transversal; ao longo do rumo, o giro encurta em E[cos δθ] − 1 ≈ −σ²/2 (2ª ordem).
    expected_along = 50.0 * (np.exp(-fraction**2 / 2) - 1)
    np.testing.assert_allclose(error.mean(0), [0.0, expected_along], atol=0.01)
    np.testing.assert_allclose(error.std(0), [50 * fraction, 50 * fraction], rtol=0.02, atol=1e-9)


def test_episode_noise_is_paired_across_policies():
    env_a, env_b = RealisticBatchedEnv(_tables(), _config()), RealisticBatchedEnv(_tables(), _config())
    seeds = list(range(2024, 2034))
    env_a.reset(seeds)
    env_b.reset(seeds)
    for name in ("origin", "pos_m", "scale_error", "heading_error", "variants"):
        assert torch.equal(getattr(env_a, name), getattr(env_b, name)), name
    # Ações diferentes não mudam o ruído sorteado no reset.
    env_a.step(torch.zeros(10, dtype=torch.long))
    env_b.step(torch.full((10,), 4, dtype=torch.long))
    assert torch.equal(env_a.variants, env_b.variants) and torch.equal(env_a.scale_error, env_b.scale_error)


def test_blur_preserves_mass():
    belief = torch.rand(3, 20, 20, dtype=torch.float64)
    belief /= belief.sum(dim=(1, 2), keepdim=True)
    out = gaussian_blur_batched(belief, torch.tensor([0.5, 1.2, 2.0], dtype=torch.float64))
    torch.testing.assert_close(out.sum(dim=(1, 2)), torch.ones(3, dtype=torch.float64))


def test_altitude_action_changes_level_and_costs_more_time():
    env = RealisticBatchedEnv(_tables(), _config(altitude_actions=True))
    env.reset([1, 2])
    out = env.step(torch.tensor([ACTION_UP, 2]))
    assert env.altitude_index.tolist() == [2, 1]
    assert out["step_time_s"].tolist() == [16.0, 6.0]


def test_control_with_identical_map_and_observation_localizes():
    # Controle (análogo ao G3): mapa = observação e β alto → a crença concentra na posição real.
    env = RealisticBatchedEnv(_tables(same_map_and_obs=True), _config())
    env.reset(list(range(20)))
    while not bool(env.done.all()):
        env.step(torch.full((env.n,), 2, dtype=torch.long))
    assert env.success.float().mean() >= 0.9


def test_policy_observation_shapes():
    tables = _tables()
    tables.extra_channels = {"distinct": np.zeros((ROWS, COLS), np.float32)}
    env = RealisticBatchedEnv(tables, _config(altitude_actions=True))
    env.reset([0, 1, 2])
    obs = env.policy_obs(include_change_prior=True)
    assert obs["map"].shape == (3, 4, 32, 32) and obs["vec"].shape == (3, 6)


def test_reset_rows_matches_a_fresh_reset_for_those_episodes():
    env, fresh = RealisticBatchedEnv(_tables(False), _config()), RealisticBatchedEnv(_tables(False), _config())
    env.reset([10, 11, 12])
    env.step(torch.tensor([0, 1, 2]))
    env.reset_rows(torch.tensor([1]), [99])
    fresh.reset([99])
    for key in ("pos_m", "belief", "variants", "window_cells"):
        assert torch.equal(getattr(env, key)[1], getattr(fresh, key)[0]), key
    assert int(env.t[1]) == 0 and int(env.t[0]) == 1

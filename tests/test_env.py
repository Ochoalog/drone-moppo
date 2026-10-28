import numpy as np
from gymnasium.utils.env_checker import check_env

from dal.baselines import make_baselines
from dal.env import DroneLocEnv, Episode
from helpers import small_cfg, synthetic_data


def test_check_env_passes():
    cfg = small_cfg()
    env = DroneLocEnv(synthetic_data(), cfg)
    check_env(env, skip_render_check=True)


def test_same_seed_same_trajectory():
    cfg = small_cfg()
    env = DroneLocEnv(synthetic_data(), cfg)
    runs = []
    for _ in range(2):
        obs, _ = env.reset(seed=123)
        acts = np.random.default_rng(5).integers(8, size=cfg["filter"]["max_steps"])
        traj, rews = [], []
        for a in acts:
            obs, r, term, trunc, _ = env.step(int(a))
            rews.append(r)
            if term or trunc:
                break
        runs.append((list(env.ep.traj), rews, obs["map"].copy()))
    assert runs[0][0] == runs[1][0]
    assert np.allclose(runs[0][1], runs[1][1])
    assert np.array_equal(runs[0][2], runs[1][2])


def test_episode_stays_in_window_and_terminates():
    cfg = small_cfg()
    ep = Episode(synthetic_data(), cfg["filter"])
    rng = np.random.default_rng(0)
    ep.reset(seed=1)
    while not ep.done:
        ep.step(int(rng.integers(8)))
        assert 0 <= ep.pos[0] < ep.W and 0 <= ep.pos[1] < ep.W
    assert ep.t <= cfg["filter"]["max_steps"]


def test_border_penalty_applied():
    cfg = small_cfg()
    e = cfg["env"]
    env = DroneLocEnv(synthetic_data(), cfg)
    env.reset(seed=0, options=dict(start=(0, 5)))
    h0 = env.ep.filter.entropy()
    _, r, term, *_ = env.step(0)  # north from row 0 -> blocked
    assert env.ep.pos == (0, 5) and env.ep.last.blocked
    assert env.ep.blocked_axes == (True, False)
    expected = (h0 - env.ep.filter.entropy()) - e["step_cost"] - e["border_penalty"]
    if term:
        expected += e["success_reward"] if env.ep.last.success else -e["false_loc_penalty"]
    assert np.isclose(r, expected)


def test_each_baseline_runs_an_episode():
    cfg = small_cfg()
    data = synthetic_data()
    ep = Episode(data, cfg["filter"])
    for name, pol in make_baselines(cfg).items():
        ep.reset(seed=7)
        pol.reset(ep, np.random.default_rng(0))
        while not ep.done:
            a = pol.act(ep)
            assert 0 <= a < 8, name
            ep.step(a)
        assert ep.t >= 1


def test_greedy_localizes_on_easy_synthetic_world():
    cfg = small_cfg(window=24, max_steps=40)
    data = synthetic_data(noise=0.1, beta=30.0)
    ep = Episode(data, cfg["filter"])
    pol = make_baselines(cfg)["greedy"]
    wins = 0
    for s in range(5):
        ep.reset(seed=s)
        pol.reset(ep, np.random.default_rng(s))
        while not ep.done:
            ep.step(pol.act(ep))
        wins += ep.last.success
    assert wins >= 4


def test_no_train_test_leakage_synthetic():
    data = synthetic_data()
    for split in ("train", "test"):
        rows = set(data.rows(split).tolist())
        for r0, _ in data.windows(split, 32, 0.7):
            assert set(range(r0, r0 + 32)) <= rows

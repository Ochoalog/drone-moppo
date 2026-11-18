"""Baselines em lote no ambiente realista: rodam até o fim, são pareadas e a reta funciona no controle."""
import torch

from dal.v3.agents.baselines import BASELINES, run_policy, shift_belief
from dal.v3.envs.realistic_env import RealisticBatchedEnv
from test_v3_realistic_env import _config, _tables


def test_every_baseline_runs_to_completion_and_is_deterministic():
    seeds = list(range(8))
    for name, cls in BASELINES.items():
        config = _config(altitude_actions=(name == "straight_150"))
        kwargs = {"samples": 4} if name == "greedy_map" else {}
        first = run_policy(RealisticBatchedEnv(_tables(False), config), cls(**kwargs), seeds)
        second = run_policy(RealisticBatchedEnv(_tables(False), config), cls(**kwargs), seeds)
        assert bool(first.done.all()), name
        assert torch.equal(first.pos_m, second.pos_m) and torch.equal(first.success, second.success), name


def test_straight_policy_localizes_in_the_control():
    env = run_policy(RealisticBatchedEnv(_tables(True), _config()), BASELINES["straight"](), list(range(30)))
    assert env.success.float().mean() >= 0.9


def test_straight_150_climbs_before_flying():
    env = RealisticBatchedEnv(_tables(True), _config(altitude_actions=True))
    policy = BASELINES["straight_150"]()
    env.reset([0, 1])
    policy.reset(env, [0, 1])
    env.step(policy.act(env))
    assert env.altitude_index.tolist() == [2, 2]


def test_shift_keeps_mass_and_blocks_at_the_border():
    belief = torch.rand(2, 6, 6, dtype=torch.float64)
    belief /= belief.sum(dim=(1, 2), keepdim=True)
    torch.testing.assert_close(shift_belief(belief, 2, -2).sum(dim=(1, 2)), torch.ones(2, dtype=torch.float64))

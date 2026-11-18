"""Gate G2 (paridade do ambiente em lote com o de CPU), versão do pytest.

Por padrão roda 100 episódios de teste de S2 (rápido). O gate oficial roda os 1000 episódios:
`DAL_G2_FULL=1 pytest tests/test_v3_parity.py` ou `scripts/gate_g2_parity_v3.py` (grava o JSON do gate).
"""
import os

import numpy as np
import pandas as pd
import pytest

from dal.config import ROOT, load_config
from dal.v3.envs.parity import batched_baseline, batched_ppo, compare_baseline, run_cpu_baseline

N_EPISODES = 1000 if os.environ.get("DAL_G2_FULL") == "1" else 100
SEEDS = [2024 + i for i in range(N_EPISODES)]
pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def scenario():
    from dal.env import ScenarioData

    config = load_config("configs/default.yaml")
    return config, ScenarioData.load("hand", "S2", config)


@pytest.mark.parametrize("name", ["straight", "lawnmower"])
def test_baselines_reproduce_cpu_trajectories_and_beliefs(scenario, name):
    config, data = scenario
    result = compare_baseline(run_cpu_baseline(data, config, name, SEEDS),
                              batched_baseline(data, config, name, SEEDS))
    assert result["identical_trajectories"] == N_EPISODES, result
    assert result["max_belief_abs_diff"] <= 1e-4, result


@pytest.mark.skipif(not (ROOT / "runs/overnight/S2_hand_s0/final_model.zip").exists(), reason="modelo da PoC ausente")
def test_poc_ppo_reproduces_recorded_episodes(scenario):
    config, data = scenario
    env = batched_ppo(data, config, str(ROOT / "runs/overnight/S2_hand_s0/final_model.zip"), SEEDS)
    recorded = pd.read_csv(ROOT / "results/episodes_test.csv")
    recorded = recorded[(recorded.scenario == "S2") & (recorded.method == "ppo") & (recorded.model_seed == 0)
                        & (recorded.episode < N_EPISODES)].sort_values("episode")
    np.testing.assert_array_equal(env.last.success.numpy(), recorded.success.to_numpy())
    np.testing.assert_array_equal(env.t.numpy(), recorded.steps.to_numpy())

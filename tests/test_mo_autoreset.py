"""Coleta vetorizada do MOPPO (decisão A7 do DECISIONS_MO_V2.md).

Com SAME_STEP, cada passo do buffer é uma transição real, e a recompensa do passo terminal vem de
`final_info`. Com NEXT_STEP (v1), o passo depois do fim do episódio é um reset que ignora a ação.
"""
import gymnasium as gym
import numpy as np

from dal.env import DroneLocEnv
from dal.mo_env import DroneLocMOEnv
from dal.mo_ppo import extract_reward_vectors
from helpers import small_cfg, synthetic_data

N_STEPS = 120


def _make_env():
    return DroneLocMOEnv(DroneLocEnv(synthetic_data(), small_cfg(max_steps=12)), memory=True)


def _collect(mode):
    """Roda o vetor de 1 ambiente com ações fixas; devolve recompensas, fins e ações."""
    envs = gym.vector.SyncVectorEnv([_make_env], autoreset_mode=mode)
    envs.reset(seed=7)
    actions = np.random.default_rng(0).integers(8, size=N_STEPS)
    rewards, dones = [], []
    for action in actions:
        _, _, terminated, truncated, infos = envs.step(np.array([action]))
        rewards.append(extract_reward_vectors(infos)[0])
        dones.append(bool(terminated[0] or truncated[0]))
    envs.close()
    return np.array(rewards), np.array(dones), actions


def _replay(actions, skip_after_done: bool):
    """Reexecução sequencial do mesmo ambiente; `skip_after_done` imita o passo de reset do NEXT_STEP."""
    env = _make_env()
    env.reset(seed=7)
    rewards, pending_reset = [], False
    for action in actions:
        if pending_reset:
            env.reset()
            pending_reset = False
            if skip_after_done:
                rewards.append(np.zeros(4, np.float32))  # transição espúria: a ação é ignorada
                continue
        _, _, terminated, truncated, info = env.step(int(action))
        rewards.append(info["r_vec"])
        pending_reset = terminated or truncated
    return np.array(rewards)


def test_same_step_matches_sequential_replay():
    rewards, dones, actions = _collect(gym.vector.AutoresetMode.SAME_STEP)
    assert dones.sum() >= 3, "o teste precisa de vários fins de episódio"
    np.testing.assert_allclose(rewards, _replay(actions, skip_after_done=False), atol=1e-6)


def test_next_step_has_one_spurious_transition_per_episode():
    rewards, dones, actions = _collect(gym.vector.AutoresetMode.NEXT_STEP)
    np.testing.assert_allclose(rewards, _replay(actions, skip_after_done=True), atol=1e-6)

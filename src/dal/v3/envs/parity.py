"""Gate G2 (paridade): o ambiente em lote, no modo legado, reproduz o ambiente de CPU da v1.

Critérios (§3.2 do plano v3): reta e zigue-zague produzem **as mesmas trajetórias** que o ambiente de
CPU nos episódios de teste de S2, com crenças finais iguais até 1e-4; a PPO da PoC (argmax),
carregada no ambiente em lote, reproduz o sucesso do REPORT.md dentro do IC.
"""
from __future__ import annotations

import numpy as np
import torch

from dal.baselines import LawnmowerPolicy, StraightPolicy
from dal.env import Episode
from dal.v3.envs.batched_env import BatchedLocalizationEnv

BELIEF_TOLERANCE = 1e-4
BASELINES = {"straight": StraightPolicy, "lawnmower": LawnmowerPolicy}


class EpisodeView:
    """Visão de um episódio do lote com a interface que as baselines da v1 leem (`blocked_axes`, `t`)."""

    def __init__(self, env: BatchedLocalizationEnv, index: int):
        self.env, self.index = env, index

    @property
    def blocked_axes(self) -> tuple[bool, bool]:
        return tuple(bool(v) for v in self.env.blocked_axes[self.index].tolist())

    @property
    def t(self) -> int:
        return int(self.env.t[self.index])


def run_cpu_baseline(data, config: dict, name: str, seeds: list[int]) -> list[dict]:
    """Referência: o simulador da v1, episódio a episódio."""
    episode = Episode(data, config["filter"], "test", config["data"]["cell_m"])
    rows = []
    for seed in seeds:
        episode.reset(seed=seed)
        policy = BASELINES[name]()
        policy.reset(episode, np.random.default_rng([seed, 7]))
        while not episode.done:
            episode.step(policy.act(episode))
        rows.append({"trajectory": list(episode.traj), "belief": episode.filter.b.copy(),
                     "success": bool(episode.last.success), "false_loc": bool(episode.last.false_loc),
                     "steps": int(episode.t)})
    return rows


def run_batched_policy(data, config: dict, seeds: list[int], act_fn, reset_fn=None,
                       device: str = "cpu") -> BatchedLocalizationEnv:
    """Roda todos os episódios em lote; `act_fn(env) -> LongTensor(N)` decide as ações."""
    env = BatchedLocalizationEnv(data, config["filter"], "test", config["data"]["cell_m"], device=device)
    env.reset(seeds)
    if reset_fn is not None:
        reset_fn(env)
    while not bool(env.done.all()):
        env.step(act_fn(env))
    return env


def batched_baseline(data, config: dict, name: str, seeds: list[int], device: str = "cpu") -> BatchedLocalizationEnv:
    policies: list = []

    def reset(env):
        for i, seed in enumerate(seeds):
            policy = BASELINES[name]()
            policy.reset(EpisodeView(env, i), np.random.default_rng([seed, 7]))
            policies.append(policy)

    def act(env):
        return torch.tensor([policies[i].act(EpisodeView(env, i)) if not bool(env.done[i]) else 0
                             for i in range(env.n)])

    return run_batched_policy(data, config, seeds, act, reset, device)


def batched_ppo(data, config: dict, model_path: str, seeds: list[int], device: str = "cpu") -> BatchedLocalizationEnv:
    """PPO da PoC (SB3, argmax) decidindo para o lote inteiro a cada passo."""
    from stable_baselines3 import PPO

    model = PPO.load(model_path, device="cpu")

    def act(env):
        actions, _ = model.predict(env.policy_obs(), deterministic=True)
        return torch.as_tensor(actions, dtype=torch.long)

    return run_batched_policy(data, config, seeds, act, None, device)


def compare_baseline(cpu_rows: list[dict], env: BatchedLocalizationEnv) -> dict:
    """Resumo da comparação: trajetórias iguais, maior diferença de crença e taxas de sucesso."""
    same = [cpu_rows[i]["trajectory"] == env.trajectories[i] for i in range(env.n)]
    belief = env.belief.cpu().numpy()
    max_diff = max(float(np.abs(cpu_rows[i]["belief"] - belief[i]).max()) for i in range(env.n))
    return {"n": env.n, "identical_trajectories": int(sum(same)), "max_belief_abs_diff": max_diff,
            "success_cpu": float(np.mean([r["success"] for r in cpu_rows])),
            "success_batched": float(env.last.success.float().mean()),
            "passed": bool(all(same) and max_diff <= BELIEF_TOLERANCE)}

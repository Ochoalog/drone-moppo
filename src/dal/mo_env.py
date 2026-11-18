"""Ambiente multiobjetivo sobre o DroneLocEnv (AMOR, Alegre et al., SIGGRAPH 2025)."""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from .env import DroneLocEnv, ScenarioData, make_env


class DroneLocMOEnv(gym.Env):
    """POMDP multiobjetivo para a localização ativa do drone.

    Objetivos r_t em R^4:
        0: info:     H_{t-1} - H_t (redução da entropia da crença)
        1: custo:    -step_cost - border_penalty * bloqueado
        2: sucesso:  +success_reward no passo terminal, se houve sucesso; senão 0
        3: risco:    -false_loc_penalty no passo terminal, se houve localização falsa; senão 0

    Observação:
        map: (C, W, W) float32 em [-1, 1]
        vec: [w (4), histórico de ações (64, se memory=True), 1 - t/max_steps (1)]
             Tamanho total: 69 (com memória) ou 5 (sem memória).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        base_env: DroneLocEnv,
        memory: bool = True,
        memory_len: int = 8,
        scale_factors: np.ndarray | list[float] | None = None,
        default_w: np.ndarray | list[float] | None = None,
    ):
        super().__init__()
        self.base_env = base_env
        self.memory = bool(memory)
        self.memory_len = int(memory_len)
        self.num_objectives = 4
        self.action_dim = int(base_env.action_space.n)
        self.action_space = base_env.action_space

        self.scale_factors = (
            np.array(scale_factors, dtype=np.float32)
            if scale_factors is not None
            else np.ones(self.num_objectives, dtype=np.float32)
        )
        self.default_w = (
            np.array(default_w, dtype=np.float32) if default_w is not None else None
        )
        self.w = (
            self.default_w.copy()
            if self.default_w is not None
            else np.full(self.num_objectives, 1.0 / self.num_objectives, dtype=np.float32)
        )
        self.action_history = np.zeros((self.memory_len, self.action_dim), dtype=np.float32)

        nv = self.num_objectives + 1 + (self.memory_len * self.action_dim if self.memory else 0)
        self.observation_space = gym.spaces.Dict({
            "map": base_env.observation_space["map"],
            "vec": gym.spaces.Box(low=-1.0, high=1.0, shape=(nv,), dtype=np.float32),
        })

    # Repassa propriedades do ambiente base, para o wrapper ser transparente
    @property
    def ep(self):
        return self.base_env.ep

    @property
    def ecfg(self):
        return self.base_env.ecfg

    @property
    def cfg(self):
        return self.base_env.cfg

    @property
    def H(self):
        return self.base_env.H

    def _get_vec_obs(self) -> np.ndarray:
        t_frac = np.float32(1.0 - self.base_env.ep.t / self.base_env.ep.max_steps)
        if self.memory:
            return np.concatenate(
                [self.w, self.action_history.flatten(), np.array([t_frac], dtype=np.float32)],
                dtype=np.float32,
            )
        return np.concatenate([self.w, np.array([t_frac], dtype=np.float32)], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        obs, info = self.base_env.reset(seed=seed, options=options)

        # Zera o histórico de ações (especificação §2 e §3c)
        self.action_history.fill(0.0)

        # Vetor de pesos: dado nas opções, fixo (default_w) ou sorteado
        if "w" in options and options["w"] is not None:
            self.w = np.array(options["w"], dtype=np.float32)
        elif self.default_w is not None:
            self.w = self.default_w.copy()
        else:
            # Sorteia w ~ Dirichlet(alpha=1) uma vez por episódio (AMOR §5)
            self.w = self.np_random.dirichlet(np.ones(self.num_objectives, dtype=np.float64)).astype(
                np.float32
            )

        mo_obs = {"map": obs["map"], "vec": self._get_vec_obs()}
        info["w"] = self.w.copy()
        info["r_vec"] = np.zeros(self.num_objectives, dtype=np.float32)
        info["r_vec_raw"] = np.zeros(self.num_objectives, dtype=np.float32)
        return mo_obs, info

    def step(self, action: int):
        h_prev = float(self.base_env.H)
        obs, _, terminated, truncated, info = self.base_env.step(int(action))

        # Decomposição da recompensa conforme a §2:
        e = self.base_env.ecfg
        res = self.base_env.ep.last
        r0 = float(h_prev - res.entropy)
        r1 = float(-e["step_cost"] - e["border_penalty"] * res.blocked)
        r2 = float(e["success_reward"] * res.success)
        r3 = float(-e["false_loc_penalty"] * res.false_loc)
        r_vec_raw = np.array([r0, r1, r2, r3], dtype=np.float32)

        # Aplica os fatores de escala por objetivo, se configurados (padrão: 1)
        r_vec = r_vec_raw * self.scale_factors
        scalar_reward = float(np.dot(r_vec, self.w))

        # Atualiza o histórico de ações (fila FIFO de one-hots)
        if self.memory:
            self.action_history[:-1] = self.action_history[1:]
            one_hot = np.zeros(self.action_dim, dtype=np.float32)
            one_hot[int(action)] = 1.0
            self.action_history[-1] = one_hot

        mo_obs = {"map": obs["map"], "vec": self._get_vec_obs()}
        info["w"] = self.w.copy()
        info["r_vec"] = r_vec
        info["r_vec_raw"] = r_vec_raw

        return mo_obs, scalar_reward, terminated, truncated, info

    def close(self):
        self.base_env.close()


def make_mo_env(
    descriptor: str,
    scenario: str,
    cfg: dict,
    split: str = "train",
    memory: bool = True,
    default_w: np.ndarray | list[float] | None = None,
    scale_factors: np.ndarray | list[float] | None = None,
) -> DroneLocMOEnv:
    """Cria um DroneLocMOEnv envolvendo o DroneLocEnv padrão."""
    base_env = make_env(descriptor, scenario, cfg, split=split)
    return DroneLocMOEnv(
        base_env,
        memory=memory,
        scale_factors=scale_factors,
        default_w=default_w,
    )

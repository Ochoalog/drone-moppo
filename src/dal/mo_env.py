"""Multi-objective environment for DroneLocEnv (AMOR / Alegre et al., SIGGRAPH 2025)."""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from .env import DroneLocEnv, ScenarioData, make_env


class DroneLocMOEnv(gym.Env):
    """Multi-objective POMDP environment for active drone localization.

    Objectives r_t in R^4:
        0: info:      H_{t-1} - H_t (belief entropy reduction)
        1: cost:      -step_cost - border_penalty * blocked
        2: success:   +success_reward at terminal step if success else 0
        3: risk:      -false_loc_penalty at terminal step if false_loc else 0

    Observation:
        map: (C, W, W) float32 in [-1, 1]
        vec: [w (4), action_history (64, if memory=True), 1 - t/max_steps (1)]
             Total size: 69 (with memory) or 5 (without memory).
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

    # Forward properties to base_env for transparency
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

        # Reset action history to zeros (spec §2 & §3c)
        self.action_history.fill(0.0)

        # Weight vector sampling or specification
        if "w" in options and options["w"] is not None:
            self.w = np.array(options["w"], dtype=np.float32)
        elif self.default_w is not None:
            self.w = self.default_w.copy()
        else:
            # Sample w ~ Dirichlet(alpha=1) once per episode (AMOR §5)
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

        # Reward decomposition according to §2:
        e = self.base_env.ecfg
        res = self.base_env.ep.last
        r0 = float(h_prev - res.entropy)
        r1 = float(-e["step_cost"] - e["border_penalty"] * res.blocked)
        r2 = float(e["success_reward"] * res.success)
        r3 = float(-e["false_loc_penalty"] * res.false_loc)
        r_vec_raw = np.array([r0, r1, r2, r3], dtype=np.float32)

        # Apply scaling factors if configured (default ones)
        r_vec = r_vec_raw * self.scale_factors
        scalar_reward = float(np.dot(r_vec, self.w))

        # Update action history FIFO
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
    """Creates a DroneLocMOEnv wrapping the standard DroneLocEnv."""
    base_env = make_env(descriptor, scenario, cfg, split=split)
    return DroneLocMOEnv(
        base_env,
        memory=memory,
        scale_factors=scale_factors,
        default_w=default_w,
    )

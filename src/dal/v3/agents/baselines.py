"""Baselines em lote para o ambiente realista (mesmas regras da v1, adaptadas a N episódios).

Cada política sorteia o que precisa no `reset`, a partir da semente do episódio (`[seed, 7]`, como na
v1), e decide de forma determinística depois. Assim o pareamento entre métodos vale também para elas.
"""
from __future__ import annotations

import numpy as np
import torch

from dal.v3.envs.realistic_env import ACTION_UP, HORIZONTAL_DIRECTIONS, RealisticBatchedEnv

_DIRECTION_INDEX = {d: i for i, d in enumerate(HORIZONTAL_DIRECTIONS)}


class BatchedPolicy:
    name = "policy"

    def reset(self, env: RealisticBatchedEnv, seeds: list[int]) -> None:
        self.rngs = [np.random.default_rng([seed, 7]) for seed in seeds]

    def act(self, env: RealisticBatchedEnv) -> torch.Tensor:
        raise NotImplementedError


class RandomPolicy(BatchedPolicy):
    """Ação horizontal aleatória a cada passo (sorteios pré-calculados no reset)."""

    name = "random"

    def reset(self, env, seeds):
        super().reset(env, seeds)
        self.plan = torch.tensor(np.stack([rng.integers(8, size=env.cfg.max_steps) for rng in self.rngs]))

    def act(self, env):
        return self.plan.gather(1, env.t.cpu().clamp(max=env.cfg.max_steps - 1).view(-1, 1)).squeeze(1)


class StraightPolicy(BatchedPolicy):
    """Rumo fixo sorteado; reflete a(s) componente(s) bloqueada(s) na borda (como a v1)."""

    name = "straight"

    def reset(self, env, seeds):
        super().reset(env, seeds)
        self.direction = np.array([HORIZONTAL_DIRECTIONS[rng.integers(8)] for rng in self.rngs])

    def act(self, env):
        blocked = env.blocked_axes.cpu().numpy()
        self.direction = np.where(blocked, -self.direction, self.direction)
        return torch.tensor([_DIRECTION_INDEX[tuple(d)] for d in self.direction])


class StraightHighPolicy(StraightPolicy):
    """"Sobe para 150 m e voa reto": sobe até o nível mais alto e depois segue a reta."""

    name = "straight_150"

    def act(self, env):
        horizontal = super().act(env)
        top = len(env.cfg.altitudes_m) - 1
        return torch.where(env.altitude_index.cpu() < top, torch.full_like(horizontal, ACTION_UP), horizontal)


class LawnmowerPolicy(BatchedPolicy):
    """Zigue-zague (boustrofédon) com as regras da v1: faixas de `leg` passos e `strip` passos entre faixas."""

    name = "lawnmower"

    def __init__(self, strip: int = 2, leg: int = 16):
        # Passo de 50 m: 2 passos entre faixas = 100 m (≈ footprint a 60 m); 16 passos = 800 m por faixa.
        self.strip, self.leg = strip, leg

    def reset(self, env, seeds):
        super().reset(env, seeds)
        self.along, self.across, self.phase, self.k = [], [], [], []
        for rng in self.rngs:
            horizontal = rng.random() < 0.5
            s1, s2 = rng.choice([-1, 1], 2)
            self.along.append((0, int(s1)) if horizontal else (int(s1), 0))
            self.across.append((int(s2), 0) if horizontal else (0, int(s2)))
            self.phase.append("leg")
            self.k.append(0)

    def act(self, env):
        blocked = env.blocked_axes.cpu().numpy().any(1)
        steps = env.t.cpu().numpy()
        actions = []
        for i in range(env.n):
            if blocked[i] and steps[i] > 0:
                if self.phase[i] == "leg":
                    self.phase[i], self.k[i] = "across", 0
                else:
                    self.across[i] = (-self.across[i][0], -self.across[i][1])
            if self.phase[i] == "leg" and self.k[i] >= self.leg:
                self.phase[i], self.k[i] = "across", 0
            if self.phase[i] == "across" and self.k[i] >= self.strip:
                self.phase[i], self.k[i] = "leg", 0
                self.along[i] = (-self.along[i][0], -self.along[i][1])
            self.k[i] += 1
            actions.append(_DIRECTION_INDEX[self.along[i] if self.phase[i] == "leg" else self.across[i]])
        return torch.tensor(actions)


class GreedyMapPolicy(BatchedPolicy):
    """Gulosa de ganho de informação com o **mapa** (justa: só o que o drone sabe), como a `greedy_map` da v1.

    Para cada ação horizontal, sorteia M células da crença predita e usa o descritor do **mapa** da célula
    de destino como observação prevista; escolhe a ação de menor entropia posterior esperada. Usa a mesma
    verossimilhança do ambiente (inclusive a CA, se o ambiente a usar).
    """

    name = "greedy_map"

    def __init__(self, samples: int = 16):
        self.samples = samples

    def act(self, env):
        n, W = env.n, env.W
        step_cells = int(round(env.cfg.step_m / env.cfg.cell_m))
        expected = torch.full((n, 8), float("inf"), dtype=torch.float64)
        draws = torch.stack([torch.as_tensor(rng.random(self.samples)) for rng in self.rngs])  # (N, M)
        for action, (d_row, d_col) in enumerate(HORIZONTAL_DIRECTIONS):
            # Como na v1: crença predita para a ação e células sorteadas dela.
            predicted = shift_belief(env.belief, d_row * step_cells, d_col * step_cells)
            cdf = predicted.view(n, -1).double().cumsum(1).cpu()
            sampled = torch.searchsorted(cdf, draws * cdf[:, -1:]).clamp(max=W * W - 1).to(env.device)
            entropies = []
            for m in range(self.samples):
                cell = env.window_cells.gather(1, sampled[:, m:m + 1]).squeeze(1)
                entropies.append(_posterior_entropy(env, predicted, cell))
            expected[:, action] = torch.stack(entropies, 1).mean(1).cpu()
        return expected.argmin(1)


def shift_belief(belief: torch.Tensor, d_row: int, d_col: int) -> torch.Tensor:
    """Desloca a crença; a massa que sairia da janela fica parada (bloqueio), como no ambiente."""
    W = belief.shape[-1]
    out = torch.zeros_like(belief)
    src_r = slice(max(0, -d_row), min(W, W - d_row))
    src_c = slice(max(0, -d_col), min(W, W - d_col))
    out[:, src_r.start + d_row:src_r.stop + d_row, src_c.start + d_col:src_c.stop + d_col] += belief[:, src_r, src_c]
    stays = torch.ones(W, W, dtype=torch.bool, device=belief.device)
    stays[src_r, src_c] = False
    out[:, stays] += belief[:, stays]
    return out


def _posterior_entropy(env: RealisticBatchedEnv, belief: torch.Tensor, predicted_cell: torch.Tensor) -> torch.Tensor:
    """Entropia da crença depois de observar o descritor do mapa da célula prevista (na altitude atual)."""
    n, W = env.n, env.W
    sims, betas = torch.zeros(n, W * W, device=env.device), torch.ones(n, 1, dtype=env.dtype, device=env.device)
    for index, altitude in enumerate(env.cfg.altitudes_m):
        rows = torch.nonzero(env.altitude_index == index).flatten()
        if rows.numel() == 0:
            continue
        z = env.map_desc[altitude][predicted_cell[rows]]
        sims[rows] = (z @ env.map_desc[altitude].T).gather(1, env.window_cells[rows])
        betas[rows] = env.tables.beta[altitude]
    valid = env.win_map_ok.view(n, -1)
    sims = sims.to(env.dtype).masked_fill(~valid, float("-inf"))
    likelihood = torch.exp(betas * (sims - sims.max(1, keepdim=True).values))
    mean_valid = (likelihood * valid).sum(1, keepdim=True) / valid.sum(1, keepdim=True).clamp(min=1)
    likelihood = torch.where(valid, likelihood, mean_valid.expand_as(likelihood))
    if env.cfg.alpha > 0:
        from dal.v3.localization.ca_filter import change_aware_likelihood

        likelihood = change_aware_likelihood(likelihood, env.prior[env.window_cells], env.cfg.alpha, valid)
    posterior = belief.view(n, -1) * likelihood
    posterior = posterior / posterior.sum(1, keepdim=True).clamp(min=1e-300)
    return -(posterior * torch.log(posterior.clamp(min=1e-30))).sum(1)


BASELINES = {cls.name: cls for cls in (RandomPolicy, StraightPolicy, StraightHighPolicy, LawnmowerPolicy,
                                        GreedyMapPolicy)}


def run_policy(env: RealisticBatchedEnv, policy: BatchedPolicy, seeds: list[int]) -> RealisticBatchedEnv:
    """Roda os episódios até o fim (todas as políticas, inclusive a PPO, usam esta mesma função)."""
    env.reset(seeds)
    policy.reset(env, seeds)
    while not bool(env.done.all()):
        env.step(policy.act(env))
    return env

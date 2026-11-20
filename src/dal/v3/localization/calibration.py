"""Calibração de β (por altitude e descritor), κ e α, só com janelas da faixa de treino da dobra.

Mesma regra da PoC (`dal.calibrate`):
- β₁: maximiza a log-verossimilhança média da célula real (consultas = variantes de observação);
- κ (β = κ·β₁): maximiza sucesso − localização falsa de uma política aleatória; empate → maior κ;
- α (filtro CA): mesma utilidade, com β já temperado; empate → menor α (o filtro padrão é preferido).
"""
from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import torch

from dal.v3.agents.baselines import RandomPolicy, run_policy
from dal.v3.envs.realistic_env import PerceptionTables, RealisticBatchedEnv, RealisticConfig, valid_windows
from dal.v3.localization.ca_filter import ALPHA_GRID

BETAS = np.round(np.geomspace(1, 300, 40), 3)
KAPPAS = (1.0, 0.7, 0.5, 0.35, 0.25, 0.18, 0.125, 0.09, 0.06, 0.04)


def calibrate_beta1(tables: PerceptionTables, altitude: int, window: int, n_windows: int = 20, seed: int = 0,
                    min_valid: float = 0.7, max_queries: int = 200) -> dict:
    """β₁ de máxima log-verossimilhança da célula real em janelas de treino (só linhas permitidas)."""
    rng = np.random.default_rng(seed)
    windows = valid_windows(tables, window, min_valid)
    chosen = windows[rng.choice(len(windows), size=min(n_windows, len(windows)), replace=False)]
    cols_total = tables.grid_shape[1]
    map_desc = torch.as_tensor(tables.map_desc[altitude])
    obs_desc = torch.as_tensor(tables.obs_desc[altitude])
    ok = torch.as_tensor(tables.map_ok & tables.obs_ok)
    betas = torch.as_tensor(BETAS, dtype=torch.float64)
    loglik, top1 = [], []
    for r0, c0 in chosen:
        rows = torch.arange(r0, r0 + window).view(-1, 1)
        cols = torch.arange(c0, c0 + window).view(1, -1)
        cells = (rows * cols_total + cols).reshape(-1)
        valid = ok.reshape(-1)[cells]
        cells = cells[valid]
        queries = cells[torch.as_tensor(rng.choice(len(cells), size=min(max_queries, len(cells)), replace=False))]
        positions = torch.searchsorted(cells, queries)
        for k in range(obs_desc.shape[0]):
            sims = (obs_desc[k, queries] @ map_desc[cells].T).double()  # (consultas, células válidas)
            true = sims.gather(1, positions.view(-1, 1))
            top1.append((sims.argmax(1) == positions).double())
            scaled = betas.view(1, -1, 1) * sims.unsqueeze(1)  # (consultas, β, células)
            loglik.append(betas.view(1, -1) * true - torch.logsumexp(scaled, dim=2))
    mean_loglik = torch.cat(loglik).mean(0)
    best = int(mean_loglik.argmax())
    return {"altitude": altitude, "beta1": float(BETAS[best]), "loglik_true": float(mean_loglik[best]),
            "top1": float(torch.cat(top1).mean()), "n_windows": len(chosen)}


def _random_policy_utility(tables: PerceptionTables, config: RealisticConfig, seeds: list[int],
                           device: str) -> dict:
    env = run_policy(RealisticBatchedEnv(tables, config, device), RandomPolicy(), seeds)
    success, false_loc = float(env.success.float().mean()), float(env.false_loc.float().mean())
    return {"success": success, "false_loc": false_loc, "utility": success - false_loc}


def calibrate_kappa(tables: PerceptionTables, config: RealisticConfig, beta1: dict[int, float], episodes: int = 200,
                    seed: int = 0, device: str = "cpu", kappas: tuple[float, ...] = KAPPAS) -> tuple[float, list[dict]]:
    """κ comum às altitudes (β_h = κ·β₁_h); utilidade da política aleatória em janelas de treino."""
    seeds = [seed * 100_000 + i for i in range(episodes)]
    rows = []
    for kappa in kappas:
        tempered = copy.copy(tables)
        tempered.beta = {h: kappa * b for h, b in beta1.items()}
        rows.append({"kappa": kappa, **_random_policy_utility(tempered, replace(config, alpha=0.0), seeds, device)})
    best = max(r["utility"] for r in rows)
    return max(r["kappa"] for r in rows if r["utility"] >= best - 1e-9), rows


def calibrate_alpha(tables: PerceptionTables, config: RealisticConfig, episodes: int = 200, seed: int = 0,
                    device: str = "cpu") -> tuple[float, list[dict]]:
    """α do filtro CA com β já temperado; empate → menor α."""
    seeds = [seed * 100_000 + i for i in range(episodes)]
    rows = [{"alpha": a, **_random_policy_utility(tables, replace(config, alpha=a), seeds, device)} for a in ALPHA_GRID]
    best = max(r["utility"] for r in rows)
    return min(r["alpha"] for r in rows if r["utility"] >= best - 1e-9), rows

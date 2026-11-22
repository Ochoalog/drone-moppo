"""Ruído controlado no descritor da observação (experimento E-synth; DECISIONS_V3.md V46).

z′ = normalizar(z + σ·g), g ~ N(0, I/D), aplicado **uma vez** às tabelas de observação, e nunca ao mapa.
O ruído é sorteado na CPU, com uma semente por (semente, altitude, bloco), e por isso a tabela não depende
do dispositivo. A soma e a normalização rodam no dispositivo pedido (GPU no Colab), em blocos de células.
σ = 0 devolve as tabelas sem cópia nem mudança.
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from dal.v3.envs.realistic_env import PerceptionTables

BLOCK_CELLS = 65_536


def _block_generator(seed: int, altitude: int, block: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed) * 1_000_003 + int(altitude) * 10_007 + int(block))


def noisy_observations(obs: np.ndarray, sigma: float, seed: int, altitude: int, device: str = "cpu",
                       block_cells: int = BLOCK_CELLS) -> np.ndarray:
    """(K, células, D) → mesma forma, com ruído e normalização L2 por linha (float32)."""
    k, n, dim = obs.shape
    out = np.empty((k, n, dim), np.float32)
    for block, start in enumerate(range(0, n, block_cells)):
        stop = min(start + block_cells, n)
        noise = torch.randn((k, stop - start, dim), generator=_block_generator(seed, altitude, block))
        z = torch.as_tensor(np.asarray(obs[:, start:stop], np.float32), device=device)
        z = z + sigma * noise.to(device) / np.sqrt(dim)
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
        out[:, start:stop] = z.cpu().numpy()
    return out


def add_observation_noise(tables: PerceptionTables, sigma: float, seed: int = 0,
                          altitudes: tuple[int, ...] | None = None, device: str = "cpu") -> PerceptionTables:
    """Cópia das tabelas com ruído nas observações das altitudes dadas (todas, por padrão)."""
    if sigma == 0:
        return tables
    noisy = copy.copy(tables)
    noisy.obs_desc = dict(tables.obs_desc)
    for altitude in altitudes or tuple(tables.obs_desc):
        noisy.obs_desc[altitude] = noisy_observations(tables.obs_desc[altitude], sigma, seed, altitude, device)
    return noisy

"""Monta as `PerceptionTables` do ambiente realista a partir dos descritores extraídos (Colab).

Para cada altitude: PCA (128 dimensões) ajustada **só** nas células da faixa de treino da dobra (mapa e
variantes de observação), aplicada a todas as células, seguida de normalização L2. O teste de vazamento
confere que células fora da faixa de treino não mudam a PCA.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dal.v3.envs.realistic_env import PerceptionTables

PCA_DIM = 128


@dataclass
class PCAFit:
    mean: np.ndarray
    components: np.ndarray  # (D_out, D_in)

    def transform(self, x: np.ndarray) -> np.ndarray:
        projected = (x.astype(np.float32) - self.mean) @ self.components.T
        return (projected / (np.linalg.norm(projected, axis=-1, keepdims=True) + 1e-8)).astype(np.float32)


def fit_pca(samples: np.ndarray, dim: int = PCA_DIM) -> PCAFit:
    """PCA por SVD sobre as amostras (já restritas à faixa de treino)."""
    x = samples.astype(np.float64)
    mean = x.mean(0)
    _, _, vt = np.linalg.svd(x - mean, full_matrices=False)
    return PCAFit(mean.astype(np.float32), vt[:dim].astype(np.float32))


def cells_in_rows(rows: np.ndarray, n_cols: int) -> np.ndarray:
    return (rows[:, None] * n_cols + np.arange(n_cols)[None]).ravel()


def load_spec_descriptors(features_dir: Path, spec: pd.DataFrame, name: str = "gem") -> np.ndarray:
    """Descritores (N, D) na ordem da spec, gravados pelo `dal.v3.descriptors.extract`."""
    return np.load(features_dir / f"{name}.npy", mmap_mode="r")


def project_altitude(map_raw: np.ndarray, obs_raw: np.ndarray, train_cells: np.ndarray,
                     dim: int = PCA_DIM) -> tuple[np.ndarray, np.ndarray, PCAFit]:
    """Ajusta a PCA nas células de treino (mapa + variantes) e projeta tudo."""
    samples = np.concatenate([map_raw[train_cells], obs_raw[:, train_cells].reshape(-1, obs_raw.shape[-1])])
    pca = fit_pca(samples, dim)
    obs = np.stack([pca.transform(o) for o in obs_raw])
    return pca.transform(map_raw), obs, pca


def build_tables(map_raw: dict[int, np.ndarray], obs_raw: dict[int, np.ndarray], map_ok: np.ndarray,
                 obs_ok: np.ndarray, rows_allowed: np.ndarray, train_rows: np.ndarray | None, beta: dict[int, float],
                 change_prior: np.ndarray | None = None, extra_channels: dict | None = None,
                 dim: int = PCA_DIM, pca_fits: dict[int, PCAFit] | None = None
                 ) -> tuple[PerceptionTables, dict[int, PCAFit]]:
    """`rows_allowed` = linhas onde as janelas podem ficar (treino ou teste).

    A PCA é ajustada só em `train_rows`; ou, num local zero-shot, vem pronta em `pca_fits` (ajustada no
    treino do Taquari) e é só aplicada, sem olhar o local de teste.
    """
    n_rows, n_cols = map_ok.shape
    maps, obs, fits = {}, {}, {}
    if pca_fits is None:
        train_cells = cells_in_rows(train_rows, n_cols)
        train_cells = train_cells[(map_ok & obs_ok).ravel()[train_cells]]
    for altitude in map_raw:
        if pca_fits is None:
            maps[altitude], obs[altitude], fits[altitude] = project_altitude(map_raw[altitude], obs_raw[altitude],
                                                                              train_cells, dim)
        else:
            fits[altitude] = pca_fits[altitude]
            maps[altitude] = fits[altitude].transform(map_raw[altitude])
            obs[altitude] = np.stack([fits[altitude].transform(o) for o in obs_raw[altitude]])
    tables = PerceptionTables(map_desc=maps, obs_desc=obs, map_ok=map_ok, obs_ok=obs_ok, beta=beta,
                              grid_shape=(n_rows, n_cols), rows_allowed=rows_allowed, change_prior=change_prior,
                              extra_channels=extra_channels or {})
    return tables, fits

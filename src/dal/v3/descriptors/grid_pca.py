"""Descritor em grade (GeM espacial k × k dos tokens do DINOv2) comprimido por PCA ajustada **por dobra**.

O GeM 4 × 4 tem 6144 dimensões: guardar isso para ~9,4 M de recortes daria ~115 GB. Então, por local e
altitude:
1. **ajuste:** para cada dobra do Taquari (A, B, C), sorteia células **só da faixa de treino** da dobra,
   renderiza os recortes de mapa (todas as datas de mapa) e de observação dessas células e ajusta uma PCA
   6144 → 128; nos locais zero-shot, usa a PCA da dobra A do Taquari (já gravada);
2. **extração:** grava `pca_<dobra>.npy` (float16, 128 dim) para cada dobra pedida.

Assim cada dobra só vê projeções aprendidas no seu próprio treino (sem vazamento entre dobras).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from dal.v3.descriptors.extract import DinoV2Extractor
from dal.v3.descriptors.finetune import training_cells
from dal.v3.sensor import specs
from dal.v3.sensor.render import batch_from_spec, render_crops

FIT_CELLS_PER_FOLD = 3000


def pca_path(root: Path, model: str, fold: str, altitude: int) -> Path:
    return Path(root) / "features" / "v3" / model / "pca" / f"fold_{fold}_{altitude}m.npz"


def fit_cells(site: specs.Site, fold: str, n_cells: int, seed: int = 0) -> np.ndarray:
    """Células sorteadas da faixa de treino da dobra (o único conjunto que a PCA da dobra pode ver)."""
    cells = training_cells(site, fold)
    return np.sort(np.random.default_rng([seed, ord(fold)]).choice(cells, min(n_cells, len(cells)), replace=False))


def fit_pca_torch(samples: torch.Tensor, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """PCA por SVD (na GPU, se houver): devolve média e componentes (dim × D)."""
    x = samples.double()
    mean = x.mean(0)
    _, _, vh = torch.linalg.svd(x - mean, full_matrices=False)
    return mean.float().cpu().numpy(), vh[:dim].float().cpu().numpy()


class GridPCAExtractor:
    """Calcula o GeM em grade e projeta com as PCAs das dobras: saídas `pca_<dobra>`."""

    def __init__(self, base: DinoV2Extractor, grid: int, pcas: dict[str, tuple[np.ndarray, np.ndarray]]):
        self.base, self.key = base, f"grid{grid}"
        self.pcas = {fold: (torch.as_tensor(mean), torch.as_tensor(components)) for fold, (mean, components)
                     in pcas.items()}

    def __call__(self, crops: torch.Tensor) -> dict[str, np.ndarray]:
        grid = torch.as_tensor(self.base(crops)[self.key], dtype=torch.float32)
        out = {}
        for fold, (mean, components) in self.pcas.items():
            projected = (grid - mean) @ components.T
            out[f"pca_{fold}"] = projected.numpy().astype(np.float16)
        return out


def render_fit_batch(site: specs.Site, altitude: int, cells: np.ndarray, images: dict,
                     device: str) -> list[torch.Tensor]:
    """Recortes de ajuste: mapa (cada data de mapa) e uma variante de observação por célula."""
    batches = []
    for date in site.map_dates:
        rows = specs.map_spec(site, date, altitude).iloc[cells]
        batches.append((rows, date))
    obs = specs.obs_spec(site, altitude)
    n = len(obs) // specs.N_OBS_VARIANTS
    batches.append((obs.iloc[cells + n * (cells % specs.N_OBS_VARIANTS)], site.obs_date))
    crops = []
    for rows, date in batches:
        image, georef = images[date]
        rows = rows.assign(crop_id=np.arange(len(rows)))
        for start in range(0, len(rows), 256):
            crops.append(render_crops(image, georef, batch_from_spec(rows.iloc[start:start + 256], device)))
    return crops


def fit_fold_pcas(site: specs.Site, altitude: int, folds: list[str], base: DinoV2Extractor, grid: int,
                  images: dict, device: str, dim: int = 128, n_cells: int = FIT_CELLS_PER_FOLD
                  ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Ajusta uma PCA por dobra com células só da faixa de treino dela."""
    pcas = {}
    for fold in folds:
        crops = render_fit_batch(site, altitude, fit_cells(site, fold, n_cells), images, device)
        features = torch.cat([torch.as_tensor(base(c)[f"grid{grid}"], dtype=torch.float32) for c in crops])
        pcas[fold] = fit_pca_torch(features.to(device), dim)
    return pcas

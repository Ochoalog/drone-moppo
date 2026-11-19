"""Montagem dos dados do ambiente realista a partir dos descritores extraídos (notebook 01).

Layout dos descritores (gravado por `dal.v3.descriptors.extract_site`):
`<features_root>/<modelo>/v3_<local>_<data>_<map|obs>_<h>m/{gem,valid_frac,cloud_frac}.npy`,
com as linhas na ordem da spec: mapa = uma linha por célula; observação = K blocos de células (k maior).

Locais zero-shot (Porto Alegre, terceira área): a PCA vem da faixa de treino do Taquari (dobra A) e é só
aplicada; nada do local de teste entra no ajuste (evita a padronização transdutiva do estudo v1).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dal.env import distinctiveness
from dal.v3.descriptors.tables import PCAFit, build_tables
from dal.v3.envs.realistic_env import PerceptionTables
from dal.v3.evaluation.folds import make_fold
from dal.v3.sensor import specs

VALID_MIN = 0.9
CLOUD_MAX = 0.1
MASK_ALTITUDE = 100  # as máscaras de validade e nuvem usam o footprint da altitude intermediária
DISTINCT_RADII = (3, 5)  # anel (em células) da distintividade, como na v1
WINDOW_CELLS = 128
SCENARIO_MAP_DATE = {"taquari": {"S0": "post", "S1": "pre_mar23", "S2": "pre_oct23"},
                     "poa": {"P0": "poa_post", "P1": "poa_pre_feb24", "P2": "poa_pre_dec22"},
                     "canoas": {"C0": "poa_post", "C1": "poa_pre_feb24"}}


@dataclass(frozen=True)
class DataRequest:
    site: str
    scenario: str
    fold: str | None  # dobra do Taquari; None = local inteiro (zero-shot)
    descriptor: str = "dinov2_vits14"
    altitudes: tuple[int, ...] = specs.ALTITUDES_M
    split: str = "train"  # onde as janelas podem ficar: "train" ou "test"
    # Arquivo do descritor: "gem" (GeM global) ou "pca_fold" (GeM em grade com a PCA da própria dobra;
    # no zero-shot, a da dobra A do Taquari).
    feature: str = "gem"

    def feature_file(self) -> str:
        if self.feature != "pca_fold":
            return self.feature
        return f"pca_{self.fold or 'A'}"


def spec_dir(features_root: Path, descriptor: str, site: str, date: str, kind: str, altitude: int) -> Path:
    return Path(features_root) / descriptor / f"v3_{site}_{date}_{kind}_{altitude}m"


def _load(directory: Path, name: str) -> np.ndarray:
    return np.load(directory / f"{name}.npy", mmap_mode="r")


def load_raw(features_root: Path, request: DataRequest, n_cells: int):
    """Descritores brutos por altitude (mapa (n, D), observação (K, n, D)) e a qualidade dos recortes."""
    site = specs.load_site(request.site)
    map_date = SCENARIO_MAP_DATE[request.site][request.scenario]
    map_raw, obs_raw = {}, {}
    for altitude in request.altitudes:
        map_raw[altitude] = np.asarray(_load(spec_dir(features_root, request.descriptor, request.site, map_date,
                                                      "map", altitude), request.feature_file()), np.float32)
        obs = np.asarray(_load(spec_dir(features_root, request.descriptor, request.site, site.obs_date, "obs",
                                        altitude), request.feature_file()), np.float32)
        obs_raw[altitude] = obs.reshape(-1, n_cells, obs.shape[-1])
    mask_altitude = MASK_ALTITUDE if MASK_ALTITUDE in request.altitudes else request.altitudes[0]
    quality = {}
    for prefix, date in (("map_", map_date), ("obs_", site.obs_date)):
        directory = spec_dir(features_root, request.descriptor, request.site, date, "map", mask_altitude)
        for name in ("valid_frac", "cloud_frac"):
            quality[prefix + name] = np.asarray(_load(directory, name), np.float32).ravel()
    return map_raw, obs_raw, quality


def masks_from_quality(quality: dict[str, np.ndarray], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Célula válida = recorte com ≥ 90% de pixels válidos e ≤ 10% de nuvem (limiares da v1)."""
    map_ok = (quality["map_valid_frac"] >= VALID_MIN) & (quality["map_cloud_frac"] <= CLOUD_MAX)
    obs_ok = (quality["obs_valid_frac"] >= VALID_MIN) & (quality["obs_cloud_frac"] <= CLOUD_MAX)
    return map_ok.reshape(shape), obs_ok.reshape(shape)


def load_tables(features_root: Path, request: DataRequest, beta: dict[int, float],
                change_prior: np.ndarray | None = None,
                pca_fits: dict[int, PCAFit] | None = None) -> tuple[PerceptionTables, dict[int, PCAFit]]:
    """Tabelas do ambiente para (local, cenário, dobra) e os ajustes de PCA usados.

    Taquari: PCA ajustada só na faixa de treino da dobra. Zero-shot (`fold=None`): exige `pca_fits`.
    """
    grid = specs.site_grid(specs.load_site(request.site))
    shape = (grid.nrows, grid.ncols)
    map_raw, obs_raw, quality = load_raw(features_root, request, grid.nrows * grid.ncols)
    map_ok, obs_ok = masks_from_quality(quality, shape)
    if request.fold is None:
        if pca_fits is None:
            raise ValueError("local zero-shot precisa da PCA ajustada no treino do Taquari (pca_fits)")
        train_rows, rows_allowed = None, np.arange(grid.nrows)
    else:
        fold = make_fold(request.fold, grid.nrows, specs.CELL_M, WINDOW_CELLS)
        train_rows = fold.train_rows
        rows_allowed = fold.train_rows if request.split == "train" else fold.test_rows
    tables, fits = build_tables(map_raw, obs_raw, map_ok, obs_ok, rows_allowed, train_rows, beta, change_prior,
                                pca_fits=pca_fits)
    reference = MASK_ALTITUDE if MASK_ALTITUDE in tables.map_desc else next(iter(tables.map_desc))
    tables.extra_channels = {"distinct": distinctiveness(tables.map_desc[reference].reshape(*shape, -1), map_ok,
                                                         *DISTINCT_RADII)}
    return tables, fits

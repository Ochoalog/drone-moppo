"""Descritor em grade com PCA por dobra: células de ajuste só do treino e projeção com a forma certa."""
import numpy as np
import torch

from dal.v3.descriptors import grid_pca
from dal.v3.descriptors.extract import grid_gem
from dal.v3.evaluation.folds import make_fold
from dal.v3.sensor import specs

SITE = specs.Site(name="fake", aoi={"lon0": -51.95, "lon1": -51.93, "lat0": -29.33, "lat1": -29.235},
                  obs_date="post", map_dates=("pre", "post"), sources={"pre": "a", "post": "b"}, offsets={})


def test_fit_cells_come_only_from_each_fold_training_band():
    grid = specs.site_grid(SITE)
    for fold in ("A", "B", "C"):
        rows = grid_pca.fit_cells(SITE, fold, 500) // grid.ncols
        split = make_fold(fold, grid.nrows, specs.CELL_M, 128)
        assert set(rows.tolist()) <= set(split.train_rows.tolist())
        assert not set(rows.tolist()) & set(split.test_rows.tolist())


def test_grid_gem_keeps_spatial_layout():
    patches = torch.zeros(1, 256, 4)
    patches[0, :8] = 1.0  # topo da imagem (primeiras linhas de tokens)
    out = grid_gem(patches, 2).view(4, 4)
    assert out[0].sum() > out[3].sum()  # o quadrante superior-esquerdo carrega o sinal


def test_grid_pca_extractor_projects_each_fold():
    class Base:
        def __call__(self, crops):
            return {"grid4": np.random.default_rng(0).normal(size=(len(crops), 64)).astype(np.float16)}

    rng = np.random.default_rng(1)
    pcas = {f: (rng.normal(size=64).astype(np.float32), rng.normal(size=(8, 64)).astype(np.float32)) for f in "AB"}
    out = grid_pca.GridPCAExtractor(Base(), 4, pcas)(torch.zeros(5, 16, 16, 3, dtype=torch.uint8))
    assert set(out) == {"pca_A", "pca_B"} and out["pca_A"].shape == (5, 8)

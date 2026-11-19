"""Rasteriza o prior de mudança na grade de 25 m de cada local e gera a figura de conferência visual.

Entradas: `data_v3/prior/*.tif` (baixados de `MyDrive/drone_v3/prior/`, fontes independentes da câmera).
Saídas: `results_v3/prior/<local>_prior_25m.npy` (float16, linhas × colunas) e `<local>_prior_check.png`,
com o prior sobre a miniatura **pré-evento** (nunca sobre a pós-evento). O código do prior não abre a imagem
pós-evento (`dal.v3.localization.change_prior`).

Uso: .venv/bin/python scripts/rasterize_prior_v3.py
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from dal.config import ROOT  # noqa: E402
from dal.v3.localization.change_prior import default_prior_dir, rasterize_prior  # noqa: E402
from dal.v3.sensor import specs  # noqa: E402

SOURCES = {  # local → (raster do prior, miniatura pré-evento para a figura, rótulo da fonte)
    "taquari": ("taquari_prior_emsr720_le0605_10m.tif", "data/raw/pre_mar23_thumb.tif",
                "Copernicus EMS EMSR720 (fontes ≤ 06/05/2024)"),
    "poa": ("porto_alegre_prior_emsn194_10m.tif", "data_poa/raw/poa_pre_feb24_thumb.tif",
            "Copernicus EMS EMSN194 (fontes 08/05/2024; prior 'otimista')"),
    "canoas": ("canoas_prior_emsn194_10m.tif", "data_poa/raw/poa_pre_feb24_thumb.tif",
               "Copernicus EMS EMSN194 recortado (fontes 08/05/2024; prior 'otimista')"),
}
OUT = ROOT / "results_v3" / "prior"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for site_name, (prior_file, thumb_file, label) in SOURCES.items():
        site = specs.load_site(site_name)
        grid = specs.site_grid(site)
        prior = rasterize_prior(default_prior_dir() / prior_file, grid.x0, grid.y1, grid.nrows, grid.ncols,
                                specs.CELL_M)
        np.save(OUT / f"{site_name}_prior_25m.npy", prior.astype(np.float16))
        summary[site_name] = {"source": label, "file": prior_file, "grid": [grid.nrows, grid.ncols],
                              "mean_flooded_fraction": float(prior.mean()),
                              "cells_ge_0_5": int((prior >= 0.5).sum()),
                              "flooded_km2_equivalent": float(prior.sum() * specs.CELL_M**2 / 1e6)}
        figure, axis = plt.subplots(figsize=(4, 7))
        extent = (grid.x0, grid.x0 + grid.ncols * grid.cell, grid.y1 - grid.nrows * grid.cell, grid.y1)
        thumb_path = ROOT / thumb_file
        if thumb_path.exists():
            import rasterio

            with rasterio.open(thumb_path) as dataset:  # imagem PRÉ-evento (mapa), nunca a pós-evento
                image = np.moveaxis(dataset.read([1, 2, 3]), 0, -1)
                bounds = dataset.bounds
            axis.imshow(image, extent=(bounds.left, bounds.right, bounds.bottom, bounds.top))
        masked = np.ma.masked_where(prior < 0.05, prior)
        shown = axis.imshow(masked, extent=extent, cmap="Blues", vmin=0, vmax=1, alpha=0.65)
        axis.set_xlim(extent[0], extent[1])
        axis.set_ylim(extent[2], extent[3])
        axis.set_title(f"{site_name}: prior de mudança (25 m)\n{label}", fontsize=8)
        axis.tick_params(labelsize=6)
        figure.colorbar(shown, ax=axis, fraction=0.04, label="fração inundada da célula")
        figure.savefig(OUT / f"{site_name}_prior_check.png", dpi=150, bbox_inches="tight")
        plt.close(figure)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

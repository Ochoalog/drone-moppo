"""Prior de mudança m_x ∈ [0, 1]: fração da célula de 25 m marcada como inundada por uma fonte
independente da imagem observada (Copernicus EMS ou Sentinel-1; princípio 5 do plano v3).

Proveniência: este módulo **recusa** abrir qualquer fonte da imagem pós-evento (a que a câmera vê).
O teste `tests/test_v3_change_prior.py` intercepta o `rasterio.open` e confere isso.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from dal.config import ROOT, load_config

POST_EVENT_KEYS = {"configs/default.yaml": "post", "configs_poa/poa.yaml": "poa_post"}


def post_event_sources() -> set[str]:
    """URLs (e nomes de arquivo locais) das imagens pós-evento de cada local."""
    sources = set()
    for config_path, key in POST_EVENT_KEYS.items():
        url = load_config(config_path)["data"]["sources"][key]
        sources.update({url, Path(url).name, f"{key}.tif", f"{key}_thumb.tif"})
    return sources


def assert_independent_source(path: str | Path) -> None:
    """Falha se o caminho apontar para a imagem pós-evento (ou um derivado local dela)."""
    text, name = str(path), Path(str(path)).name
    for forbidden in post_event_sources():
        if forbidden in text or name == forbidden:
            raise PermissionError(f"prior de mudança não pode ler a imagem pós-evento: {path}")


def rasterize_prior(prior_tif: str | Path, x0: float, y1: float, n_rows: int, n_cols: int,
                    cell_m: float = 25.0) -> np.ndarray:
    """Fração inundada por célula da grade (linha 0 = norte), por média da máscara (0/1) da fonte."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import Affine
    from rasterio.warp import reproject

    assert_independent_source(prior_tif)
    destination = np.zeros((n_rows, n_cols), np.float32)
    with rasterio.open(prior_tif) as source:
        mask = (source.read(1) > 0).astype(np.float32)
        nodata_fill = np.isnan(mask)
        mask[nodata_fill] = 0.0
        reproject(source=mask, destination=destination, src_transform=source.transform, src_crs=source.crs,
                  dst_transform=Affine(cell_m, 0, x0, 0, -cell_m, y1), dst_crs="EPSG:32722",
                  resampling=Resampling.average)
    return np.clip(destination, 0.0, 1.0)


def default_prior_dir() -> Path:
    """Pasta local onde ficam os rasters do prior baixados do Drive (`MyDrive/drone_v3/prior/`)."""
    return ROOT / "data_v3" / "prior"

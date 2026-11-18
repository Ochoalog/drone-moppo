"""Diagnóstico (não é o G1 oficial): o pooling espacial dos tokens do DINOv2 separa células vizinhas?

Numa janela de treino do Taquari (40 × 40 células de 50 m, recortes exatos da spec `legacy_50m`, raster local
da v1), compara o top-1 de recuperação de: GeM global, GeM em grade 2 × 2 e 4 × 4 dos tokens de patch, CLS e
o hand, em S0 (mapa = observação) e S1. Consultas: 300 células, variante de observação 0.
Saída: results_v3/g1/pooling_probe.json.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch

from dal.config import ROOT
from dal.data import load_date
from dal.features import hand_descriptor
from dal.v3.descriptors.extract import GEM_P, grid_gem, preprocess_for_dino
from dal.v3.sensor.render import RasterGeoref, batch_from_spec, render_crops

WINDOW = 40
N_QUERIES = 300


def gem(tokens: torch.Tensor) -> torch.Tensor:
    return tokens.clamp(min=1e-6).pow(GEM_P).mean(dim=1).pow(1.0 / GEM_P)


def descriptors(model, crops: torch.Tensor) -> dict[str, np.ndarray]:
    with torch.inference_mode():
        features = model.forward_features(preprocess_for_dino(crops, 224))
    patches = features["x_norm_patchtokens"]  # (B, 256, 384): grade 16 × 16
    out = {"gem": gem(patches), "cls": features["x_norm_clstoken"]}
    for k in (2, 4):
        out[f"grid{k}"] = grid_gem(patches, k)
    result = {name: value.numpy() for name, value in out.items()}
    result["hand"] = hand_descriptor(crops.numpy())
    return result


def top1(map_desc: np.ndarray, query_desc: np.ndarray, query_positions: np.ndarray) -> float:
    def normalize(x):
        x = (x - map_desc.mean(0)) / (map_desc.std(0) + 1e-6)
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

    sims = normalize(query_desc) @ normalize(map_desc).T
    return float((sims.argmax(1) == query_positions).mean())


def main() -> None:
    torch.set_num_threads(8)
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval()
    spec = pd.read_parquet(ROOT / "colab/specs/legacy_50m.parquet")
    train = spec[(spec.split == "train")]
    r0, c0 = int(train.cell_row.min()) + 40, int(train.cell_col.min()) + 40
    in_window = train[(train.cell_row >= r0) & (train.cell_row < r0 + WINDOW) & (train.cell_col >= c0)
                      & (train.cell_col < c0 + WINDOW)]
    cells = np.sort(in_window.cell_index.unique())
    rng = np.random.default_rng(0)
    queries = np.sort(rng.choice(cells, N_QUERIES, replace=False))
    results, started = {}, time.time()
    rasters = {}

    def render(rows: pd.DataFrame, date: str) -> torch.Tensor:
        if date not in rasters:
            raster = load_date(date)
            rasters[date] = (torch.as_tensor(raster.img), RasterGeoref(raster.x_origin, raster.y_origin, raster.res))
        image, georef = rasters[date]
        parts = [render_crops(image, georef, batch_from_spec(rows.iloc[i:i + 64])) for i in range(0, len(rows), 64)]
        return torch.cat(parts)

    def describe(rows: pd.DataFrame, date: str) -> dict[str, np.ndarray]:
        crops = render(rows, date)
        parts = [descriptors(model, crops[i:i + 32]) for i in range(0, len(crops), 32)]
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

    obs_rows = in_window[(in_window.kind == "obs") & (in_window.variant == 0)].set_index("cell_index").loc[queries]
    query_desc = describe(obs_rows.reset_index(), "post")
    for scenario, date in (("S0", "post"), ("S1", "pre_mar23")):
        map_rows = in_window[(in_window.kind == "map") & (in_window.date_id == date)].set_index("cell_index").loc[cells]
        map_desc = describe(map_rows.reset_index(), date)
        positions = np.searchsorted(cells, queries)
        results[scenario] = {name: top1(map_desc[name], query_desc[name], positions) for name in map_desc}
        print(scenario, {k: round(v, 3) for k, v in results[scenario].items()},
              f"({(time.time() - started) / 60:.1f} min)", flush=True)
    results["setup"] = {"window_cells": WINDOW, "queries": N_QUERIES, "window_origin": [r0, c0],
                        "candidates": len(cells),
                        "note": "diagnóstico local (raster JPEG da v1, normalização na própria janela); não é o G1"}
    (ROOT / "results_v3/g1/pooling_probe.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()

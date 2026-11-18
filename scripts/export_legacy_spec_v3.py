"""Exporta colab/specs/legacy_50m.parquet: os recortes exatos do pipeline v1 (G1 preliminar).

Recria a grade de 50 m, os deslocamentos de alinhamento e a sequência de sorteios de
`dal.features.precompute` (mesmo gerador, mesma ordem: datas na ordem da config; por data, o mapa
sem aumento e, para `post`, K variantes com deslocamento, rotação, ganho, viés e ruído por recorte).
Só entram as linhas da faixa de treino e da de teste (a faixa de separação fica de fora).

Conferência (gravada em colab/specs/legacy_50m_check.json): para 20 recortes, renderiza com
`dal.v3.sensor.render` a partir do raster local (overview 4×, o mesmo da v1), injetando o ruído que
a v1 sorteou, e compara (a) os pixels com o recorte da v1 e (b) o descritor hand com o gravado em
data/features/. Os dados da v1 são só lidos.

Uso: .venv/bin/python scripts/export_legacy_spec_v3.py
"""
from __future__ import annotations

import json
import subprocess

import numpy as np
import pandas as pd
import torch

from dal.align import load_offsets
from dal.config import DATA, ROOT, load_config
from dal.data import load_date
from dal.features import crop, hand_descriptor, photometric
from dal.grid import Grid
from dal.v3.sensor.render import (
    SOURCE_RES_COLUMN,
    RasterGeoref,
    apply_photometric,
    batch_from_spec,
    sample_geometry,
)

OUT = ROOT / "colab" / "specs"
SITE = "taquari"
N_CHECK = 20


def replay_v1_draws(config: dict, n_cells: int, keep_noise_for: set[tuple[int, int]]):
    """Refaz os sorteios da v1: por variante k, deslocamentos e rotação de todas as células e, por recorte,
    ganho, viés e o ruído (guardado só para os recortes da conferência)."""
    features = config["features"]
    rng = np.random.default_rng(features["seed"])
    px = features["crop_px"]
    draws, kept_noise = [], {}
    for date in config["data"]["sources"]:
        if date != "post":
            continue  # mapas: sem aumento, não consomem o gerador
        for k in range(features["obs_variants"]):
            jx = rng.uniform(-features["jitter_m"], features["jitter_m"], n_cells)
            jy = rng.uniform(-features["jitter_m"], features["jitter_m"], n_cells)
            theta = rng.uniform(-features["rot_deg"], features["rot_deg"], n_cells)
            gain = np.empty(n_cells)
            bias = np.empty(n_cells)
            for i in range(n_cells):
                # Mesma ordem de `photometric`: ganho, viés e o ruído do recorte inteiro.
                gain[i] = 1.0 + rng.uniform(-features["photo_gain"], features["photo_gain"])
                bias[i] = rng.uniform(-features["photo_gain"], features["photo_gain"]) * 128
                noise = rng.normal(0, features["photo_noise"], (px, px, 3))
                if (k, i) in keep_noise_for:
                    kept_noise[(k, i)] = noise
            draws.append({"jx": jx, "jy": jy, "theta": theta, "gain": gain, "bias": bias})
    return draws, kept_noise


def build_spec(config: dict, grid: Grid, rows_by_split: dict[str, np.ndarray], draws: list[dict],
               source_res: dict[str, float]) -> pd.DataFrame:
    """Uma linha por recorte: mapas (3 datas) e as K variantes de observação de `post`."""
    features, sources = config["features"], config["data"]["sources"]
    offsets = load_offsets()
    rr, cc = np.mgrid[0:grid.nrows, 0:grid.ncols]
    rr, cc = rr.ravel(), cc.ravel()
    cx, cy = grid.center(rr, cc)
    split = np.full(len(rr), "", dtype=object)
    for name, rows in rows_by_split.items():
        split[np.isin(rr, rows)] = name
    keep = split != ""
    frames = []

    def frame(date: str, kind: str, variant: int, x, y, yaw, contrast, brightness, noise_std) -> pd.DataFrame:
        return pd.DataFrame({
            "site": SITE, "date_id": date, "cog_url": sources[date], "x": x[keep], "y": y[keep],
            "footprint_m": float(features["footprint_m"]), "yaw_deg": yaw[keep], "roll_deg": 0.0, "pitch_deg": 0.0,
            "brightness": brightness[keep], "contrast": contrast[keep], "blur_sigma": 0.0, "noise_std": noise_std,
            "out_px": int(features["crop_px"]), SOURCE_RES_COLUMN: source_res[date],
            "kind": kind, "variant": variant, "cell_index": np.flatnonzero(keep), "cell_row": rr[keep],
            "cell_col": cc[keep], "split": split[keep],
        })

    zeros, ones = np.zeros(len(rr)), np.ones(len(rr))
    for date in sources:
        dx, dy = offsets[date]
        frames.append(frame(date, "map", -1, cx + dx, cy + dy, zeros, ones, zeros, 0.0))
    dx, dy = offsets["post"]
    for k, d in enumerate(draws):
        frames.append(frame("post", "obs", k, cx + dx + d["jx"], cy + dy + d["jy"], d["theta"], d["gain"],
                            d["bias"] / 128, float(features["photo_noise"])))
    spec = pd.concat(frames, ignore_index=True)
    spec.insert(0, "crop_id", np.arange(len(spec), dtype=np.int64))
    return spec


def check_against_v1(spec: pd.DataFrame, config: dict, kept_noise: dict, check_rows: pd.DataFrame) -> dict:
    """Renderiza os recortes de conferência e compara com a v1 (pixels e descritor hand gravado)."""
    features = config["features"]
    results = []
    for date, rows in check_rows.groupby("date_id"):
        raster = load_date(date)
        georef = RasterGeoref(raster.x_origin, raster.y_origin, raster.res)
        image = torch.as_tensor(raster.img)
        batch = batch_from_spec(rows)
        geometry = sample_geometry(image, georef, batch)
        for j, (_, row) in enumerate(rows.iterrows()):
            v1 = crop(raster, row.x, row.y, row.footprint_m, int(row.out_px), row.yaw_deg)
            ours = geometry[j:j + 1]
            if row.kind == "obs":
                noise = torch.as_tensor(kept_noise[(int(row.variant), int(row.cell_index))]).unsqueeze(0)
                single = batch_from_spec(rows.iloc[[j]])
                ours = apply_photometric(ours, single, noise=noise)
                v1 = (v1.astype(np.float32) * row.contrast + row.brightness * 128
                      + kept_noise[(int(row.variant), int(row.cell_index))]).clip(0, 255).astype(np.uint8)
            ours_np = ours[0].numpy()
            stored = (np.load(DATA / "features" / f"hand_{date}_{row.kind}.npy", mmap_mode="r"))
            stored_vec = (stored[int(row.variant), int(row.cell_index)] if row.kind == "obs"
                          else stored[int(row.cell_index)]).astype(np.float32)
            ours_vec = hand_descriptor(ours_np[None])[0]
            diff = np.abs(ours_np.astype(int) - v1.astype(int))
            results.append({
                "crop_id": int(row.crop_id), "date_id": date, "kind": row.kind, "variant": int(row.variant),
                "pixels_equal_fraction": float((diff == 0).mean()), "pixel_max_abs_diff": int(diff.max()),
                "hand_max_abs_diff_vs_stored": float(np.abs(ours_vec - stored_vec).max()),
                "hand_cosine_vs_stored": float(ours_vec @ stored_vec
                                               / (np.linalg.norm(ours_vec) * np.linalg.norm(stored_vec))),
            })
        del raster
    _ = photometric, features  # a v1 usa `photometric`; aqui a fórmula é refeita com o ruído injetado
    return {"n_checked": len(results), "min_pixels_equal_fraction": min(r["pixels_equal_fraction"] for r in results),
            "max_pixel_abs_diff": max(r["pixel_max_abs_diff"] for r in results),
            "min_hand_cosine_vs_stored": min(r["hand_cosine_vs_stored"] for r in results), "crops": results}


def main() -> None:
    config = load_config("configs/default.yaml")
    data_cfg = config["data"]
    grid = Grid.from_aoi(**data_cfg["aoi"], cell=data_cfg["cell_m"])
    train_rows, test_rows = grid.split_rows(data_cfg["split_lat"], data_cfg["split_margin_m"])
    n_cells = grid.nrows * grid.ncols

    rng = np.random.default_rng(0)
    candidate_cells = np.flatnonzero(np.isin(np.arange(n_cells) // grid.ncols, np.concatenate([train_rows, test_rows])))
    obs_check = {(int(rng.integers(2)), int(i)) for i in rng.choice(candidate_cells, 10, replace=False)}
    draws, kept_noise = replay_v1_draws(config, n_cells, obs_check)

    source_res = {date: float(load_date(date, thumb=False).res) for date in data_cfg["sources"]}
    spec = build_spec(config, grid, {"train": train_rows, "test": test_rows}, draws, source_res)
    OUT.mkdir(parents=True, exist_ok=True)
    spec.to_parquet(OUT / "legacy_50m.parquet", index=False)

    obs_rows = spec[(spec.kind == "obs") & spec.apply(lambda r: (r.variant, r.cell_index) in obs_check, axis=1)]
    map_rows = spec[spec.kind == "map"].sample(10, random_state=0)
    check = check_against_v1(spec, config, kept_noise, pd.concat([map_rows, obs_rows]))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    summary = {"rows": len(spec), "by_kind_date": spec.groupby(["date_id", "kind"]).size().to_dict(),
               "by_split_cells": spec[spec.kind == "map"].drop_duplicates("cell_index").split.value_counts().to_dict(),
               "source_res_m": source_res, "git_commit": commit, **check}
    summary["by_kind_date"] = {f"{d}/{k}": int(v) for (d, k), v in summary["by_kind_date"].items()}
    (OUT / "legacy_50m_check.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "crops"}, indent=1))


if __name__ == "__main__":
    main()

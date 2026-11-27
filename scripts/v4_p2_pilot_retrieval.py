"""Piloto rápido P2-retrieval: mede retrieval real UAV Pós-Enchente (CBPMESP Mathias Velho)
contra Satélite Pré-Enchente (Maxar 02/02/2024) usando DINOv2 ViT-S/14 zero-shot (GeM e grid4)
sobre 400 células amostradas da área de sobreposição (5,17 km²), com correção do deslocamento
residual (~3,5 m, -4,3 m) e máscara de bordas pretas e brancas (nodata).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_v4" / "uav_canoas"
OUT_RES = ROOT / "results_v4"

RES_M = 0.6103515625
FOOTPRINT_M = 180.0  # altitude 100 m (v3: 2 * 100 * tan(42°))
FOOTPRINT_PX = int(round(FOOTPRINT_M / RES_M))  # ~295 px
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)


@torch.inference_mode()
def extract_dino(model, crops_uint8: np.ndarray, batch_size: int = 32) -> dict[str, np.ndarray]:
    gems, g4s = [], []
    for i in range(0, len(crops_uint8), batch_size):
        b = torch.from_numpy(crops_uint8[i : i + batch_size]).permute(0, 3, 1, 2).float() / 255.0
        b = F.interpolate(b, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
        b = (b - IMAGENET_MEAN) / IMAGENET_STD
        patches = model.forward_features(b)["x_norm_patchtokens"].float()
        gem = patches.clamp(min=1e-6).pow(3.0).mean(dim=1).pow(1.0 / 3.0)
        n, _, dim = patches.shape
        cells = patches.view(n, 4, 4, 4, 4, dim).permute(0, 1, 3, 2, 4, 5).reshape(n, 16, 16, dim)
        g4 = cells.clamp(min=1e-6).pow(3.0).mean(dim=2).pow(1.0 / 3.0).reshape(n, -1)
        gems.append(F.normalize(gem, dim=-1).numpy())
        g4s.append(F.normalize(g4, dim=-1).numpy())
    return {"gem": np.concatenate(gems), "grid4": np.concatenate(g4s)}


def main() -> None:
    with rasterio.open(DATA / "canoas_uav_0.61m.tif") as du, rasterio.open(DATA / "canoas_pre_feb24_0.61m.tif") as ds:
        uav = np.transpose(du.read(), (1, 2, 0))  # (H, W, 3)
        sat = np.transpose(ds.read(), (1, 2, 0))

    H, W, _ = uav.shape
    # Máscara válida: exclui borda preta (<=3) e borda branca (>=252) do nodata dos TIFs de 15 cm
    uav_valid = (uav.max(axis=-1) > 3) & (uav.min(axis=-1) < 252)
    sat_valid = sat.max(axis=-1) > 3
    both = uav_valid & sat_valid

    # Grade de 75 m (3 células de 25 m) para piloto rápido (~400–600 células bem espaçadas)
    step_px = int(round(75.0 / RES_M))
    half = FOOTPRINT_PX // 2
    # Correção média de alinhamento medida no P2-G0: dx=+3.5m (~6px), dy=-4.3m (~+7px em linha)
    dx_px, dy_px = 6, 7

    centers = []
    uav_crops, sat_crops = [], []
    for r in range(half + 10, H - half - 10, step_px):
        for c in range(half + 10, W - half - 10, step_px):
            if both[r - half : r + half, c - half : c + half].mean() >= 0.95:
                u_crop = uav[r - half : r + half, c - half : c + half]
                s_crop = sat[r + dy_px - half : r + dy_px + half, c + dx_px - half : c + dx_px + half]
                if s_crop.shape == u_crop.shape:
                    centers.append((r * RES_M, c * RES_M))
                    uav_crops.append(u_crop)
                    sat_crops.append(s_crop)

    centers = np.array(centers)
    uav_crops = np.stack(uav_crops)
    sat_crops = np.stack(sat_crops)
    print(f"Canoas UAV->Sat pilot: {len(centers)} valid 180m-footprint cells (75m spacing, >=95% valid)")

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False).eval()
    f_uav = extract_dino(model, uav_crops)
    f_sat = extract_dino(model, sat_crops)

    dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    summary = {"n_cells": int(len(centers)), "spacing_m": 75.0, "footprint_m": FOOTPRINT_M, "descriptors": {}}
    for name in ("gem", "grid4"):
        sim = f_uav[name] @ f_sat[name].T
        top1 = sim.argmax(axis=1)
        top5 = np.argsort(-sim, axis=1)[:, :5]
        idx = np.arange(len(centers))
        top1_exact = float((top1 == idx).mean() * 100)
        top1_100m = float((dists[idx, top1] <= 100.0).mean() * 100)
        top5_100m = float(np.mean([(dists[i, top5[i]] <= 100.0).any() for i in range(len(centers))]) * 100)
        true_cos = sim[idx, idx]
        hard_neg = np.where(dists > 150.0, sim, -1.0).max(axis=1)
        summary["descriptors"][name] = {
            "top1_exact_pct": round(top1_exact, 1),
            "top1_within_100m_pct": round(top1_100m, 1),
            "top5_within_100m_pct": round(top5_100m, 1),
            "mean_true_cos": round(float(true_cos.mean()), 4),
            "mean_hard_neg_cos": round(float(hard_neg.mean()), 4),
            "margin_cos": round(float((true_cos - hard_neg).mean()), 4),
        }

    (OUT_RES / "p2_pilot_retrieval.json").write_text(json.dumps(summary, indent=2))
    print("\n=== P2 CANOAS REAL UAV -> SAT RETRIEVAL PILOT ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

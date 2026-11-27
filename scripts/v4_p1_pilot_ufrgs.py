"""Gate P1-G0 (corrigido com traj_truth_sync.txt): valida alinhamento visual e mede
retrieval real Drone -> Satélite no voo real UFRGS (DJI Matrice 100 + Parrot Sequoia RGB,
05/05/2018) contra mapas de satélite recentes (2018) e desatualizados (2012 e 2002),
usando DINOv2 ViT-S/14 (GeM e grid4) + teste de filtro bayesiano passivo ao longo do voo.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
OUT_RES = ROOT / "results_v4"
OUT_RES.mkdir(parents=True, exist_ok=True)

UFRGS_DIR = Path("/storage/UAVs Localization/Datasets/UFRGS VET/voo_100m")
MAPS = {
    "2018_recent": UFRGS_DIR / "Mapa" / "UFRGS_VET_V3_21_03_2018.jpg",
    "2012_6y_old": UFRGS_DIR / "Mapa" / "UFRGS_VET_V3_12_10_2012.jpg",
    "2002_16y_old": UFRGS_DIR / "Mapa" / "UFRGS_VET_V3_17_12_2002.jpg",
}
M_PER_PX = 0.10  # UFRGS_VET_V3 (voos_e_mapas_do_artigo.md)
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)


def load_ufrgs_frames(stride: int = 8):
    traj = [Path(l.strip()).name for l in (UFRGS_DIR / "traj.txt").read_text().splitlines() if l.strip()]
    sync = [l.strip().split() for l in (UFRGS_DIR / "traj_truth_sync.txt").read_text().splitlines() if l.strip()]
    # Fatia avaliada canônica: frames 0390..0613 -> índices 204..427 (224 frames)
    samples = []
    for idx in range(204, 428, stride):
        x, y, z, yaw = map(float, sync[idx])
        img_path = UFRGS_DIR / "Images_RGB" / traj[idx]
        samples.append((idx, img_path, x, y, z, yaw))
    return samples


def crop_rotated_square(img_rgb: np.ndarray, yaw_deg: float, out_px: int = 224) -> tuple[np.ndarray, int]:
    """Rotaciona a imagem do drone para norte-para-cima (convenção verificada: -90 - yaw)."""
    h, w = img_rgb.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), -90.0 - yaw_deg, 1.0)
    rot = cv2.warpAffine(img_rgb, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    side = int(min(h, w) * 0.70)
    x0, y0 = int(cx - side / 2), int(cy - side / 2)
    crop = rot[y0 : y0 + side, x0 : x0 + side]
    return cv2.resize(crop, (out_px, out_px), interpolation=cv2.INTER_AREA), side


def crop_map_at(sat_rgb: np.ndarray, x: float, y: float, side_px: int, out_px: int = 224) -> np.ndarray:
    h, w = sat_rgb.shape[:2]
    half = side_px // 2
    x0, y0 = int(round(x)) - half, int(round(y)) - half
    x0 = max(0, min(w - side_px, x0))
    y0 = max(0, min(h - side_px, y0))
    crop = sat_rgb[y0 : y0 + side_px, x0 : x0 + side_px]
    return cv2.resize(crop, (out_px, out_px), interpolation=cv2.INTER_AREA)


@torch.inference_mode()
def extract_dino(model, crops_uint8: np.ndarray, batch_size: int = 32) -> dict[str, np.ndarray]:
    gems, g4s = [], []
    for i in range(0, len(crops_uint8), batch_size):
        b = torch.from_numpy(crops_uint8[i : i + batch_size]).permute(0, 3, 1, 2).float() / 255.0
        b = (b - IMAGENET_MEAN) / IMAGENET_STD
        patches = model.forward_features(b)["x_norm_patchtokens"].float()
        gem = patches.clamp(min=1e-6).pow(3.0).mean(dim=1).pow(1.0 / 3.0)
        n, _, dim = patches.shape
        cells = patches.view(n, 4, 4, 4, 4, dim).permute(0, 1, 3, 2, 4, 5).reshape(n, 16, 16, dim)
        g4 = cells.clamp(min=1e-6).pow(3.0).mean(dim=2).pow(1.0 / 3.0).reshape(n, -1)
        gems.append(F.normalize(gem, dim=-1).numpy())
        g4s.append(F.normalize(g4, dim=-1).numpy())
    return {"gem": np.concatenate(gems), "grid4": np.concatenate(g4s)}


def run_passive_histogram_filter(
    q_feats: np.ndarray,
    m_feats: np.ndarray,
    q_xy: np.ndarray,
    grid_xs: np.ndarray,
    grid_ys: np.ndarray,
    beta: float = 20.0,
) -> dict:
    """Roda o histogram filter bayesiano ao longo da trajetória real (usando apenas deslocamento relativo)."""
    ny, nx = len(grid_ys), len(grid_xs)
    step_px = float(grid_xs[1] - grid_xs[0])
    belief = np.full((ny, nx), 1.0 / (ny * nx), dtype=np.float64)
    errors_m = []
    for t in range(len(q_xy)):
        if t > 0:
            # Deslocamento relativo (odometria em células da grade + blur gaussiano)
            dx_cells = int(round((q_xy[t, 0] - q_xy[t - 1, 0]) / step_px))
            dy_cells = int(round((q_xy[t, 1] - q_xy[t - 1, 1]) / step_px))
            shifted = np.zeros_like(belief)
            src_y0 = max(0, -dy_cells)
            src_y1 = min(ny, ny - dy_cells)
            src_x0 = max(0, -dx_cells)
            src_x1 = min(nx, nx - dx_cells)
            dst_y0 = max(0, dy_cells)
            dst_y1 = min(ny, ny + dy_cells)
            dst_x0 = max(0, dx_cells)
            dst_x1 = min(nx, nx + dx_cells)
            shifted[dst_y0:dst_y1, dst_x0:dst_x1] = belief[src_y0:src_y1, src_x0:src_x1]
            belief = cv2.GaussianBlur(shifted, (3, 3), 0.7)
            belief = np.maximum(belief, 1e-12)
        sim = (m_feats @ q_feats[t]).reshape(ny, nx)
        lik = np.exp(beta * (sim - sim.max()))
        belief = belief * lik
        belief /= belief.sum()
        # Posição estimada (moda)
        iy, ix = np.unravel_index(np.argmax(belief), belief.shape)
        est_x, est_y = grid_xs[ix], grid_ys[iy]
        err_m = float(np.hypot(est_x - q_xy[t, 0], est_y - q_xy[t, 1]) * M_PER_PX)
        errors_m.append(err_m)
    return {
        "final_err_m": round(errors_m[-1], 1),
        "mean_last5_err_m": round(float(np.mean(errors_m[-5:])), 1),
        "median_err_m": round(float(np.median(errors_m)), 1),
        "converged_within_35m": bool(errors_m[-1] <= 35.0),
    }


def main() -> None:
    samples = load_ufrgs_frames(stride=8)
    print(f"Loaded {len(samples)} UFRGS real drone frames (stride=8, slice 204..427, traj_truth_sync.txt)")

    drone_crops, map_sides = [], []
    for _, img_path, x, y, z, yaw in samples:
        rgb = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        crop, side = crop_rotated_square(rgb, yaw, out_px=224)
        drone_crops.append(crop)
        map_sides.append(int(round(side * z)))
    drone_crops = np.stack(drone_crops)
    median_side = int(np.median(map_sides))
    print(f"Map footprint side: median={median_side} px ({median_side * M_PER_PX:.1f} m)")

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False).eval()
    q_feats = extract_dino(model, drone_crops)

    # Grade de 250 px (= 25 m!) cobrindo todo o mapa 4800x4800 (~17x17 = 289 células)
    grid_step = 250
    margin = median_side // 2 + 10
    xs = np.arange(margin, 4800 - margin, grid_step)
    ys = np.arange(margin, 4800 - margin, grid_step)
    gx, gy = np.meshgrid(xs, ys)
    cand_xy = np.stack([gx.ravel(), gy.ravel()], axis=1)
    q_xy = np.array([[s[2], s[3]] for s in samples])

    results = {
        "flight": "UFRGS_VET_voo_100m",
        "n_queries": len(samples),
        "grid_step_m": grid_step * M_PER_PX,
        "map_footprint_m": round(median_side * M_PER_PX, 1),
        "n_map_cells": len(cand_xy),
        "maps": {},
    }

    vis_sat = None
    for map_name, map_path in MAPS.items():
        sat_rgb = cv2.cvtColor(cv2.imread(str(map_path)), cv2.COLOR_BGR2RGB)
        if map_name == "2018_recent":
            vis_sat = sat_rgb
        cand_crops = np.stack([crop_map_at(sat_rgb, cx, cy, median_side, 224) for cx, cy in cand_xy])
        m_feats = extract_dino(model, cand_crops)

        map_res = {}
        dists_m = np.linalg.norm(q_xy[:, None, :] - cand_xy[None, :, :], axis=-1) * M_PER_PX
        nearest = dists_m.argmin(axis=1)
        idx = np.arange(len(samples))
        for desc_name in ("gem", "grid4"):
            sim = q_feats[desc_name] @ m_feats[desc_name].T
            top1 = sim.argmax(axis=1)
            top5 = np.argsort(-sim, axis=1)[:, :5]
            top1_ok = float((dists_m[idx, top1] <= 35.0).mean() * 100)
            top5_ok = float(np.mean([(dists_m[i, top5[i]] <= 35.0).any() for i in range(len(samples))]) * 100)
            true_cos = sim[idx, nearest]
            hard_neg = np.where(dists_m > 50.0, sim, -1.0).max(axis=1)
            filt = run_passive_histogram_filter(q_feats[desc_name], m_feats[desc_name], q_xy, xs, ys, beta=25.0)
            map_res[desc_name] = {
                "top1_35m_pct": round(top1_ok, 1),
                "top5_35m_pct": round(top5_ok, 1),
                "mean_true_cos": round(float(true_cos.mean()), 4),
                "mean_hard_neg_cos": round(float(hard_neg.mean()), 4),
                "margin_cos": round(float((true_cos - hard_neg).mean()), 4),
                "passive_filter": filt,
            }
        results["maps"][map_name] = map_res
        print(f"Map {map_name}: {json.dumps(map_res)}")

    (OUT_RES / "p1_g0_ufrgs_pilot.json").write_text(json.dumps(results, indent=2))

    # Figura visual de 4 pares verificados
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), dpi=130)
    picks = np.linspace(0, len(samples) - 1, 4, dtype=int)
    for col, k in enumerate(picks):
        idx_f, _, x, y, z, yaw = samples[k]
        axes[0, col].imshow(drone_crops[k])
        axes[0, col].set_title(f"Drone #{idx_f} (yaw={yaw:.0f}°)")
        axes[1, col].imshow(crop_map_at(vis_sat, x, y, map_sides[k], 224))
        axes[1, col].set_title(f"Sat 2018 ({x:.0f},{y:.0f})")
        axes[0, col].axis("off")
        axes[1, col].axis("off")
    plt.tight_layout()
    fig.savefig(OUT_RES / "p1_g0_ufrgs_pairs.png")
    plt.close(fig)
    print(f"Saved {OUT_RES / 'p1_g0_ufrgs_pairs.png'}")


if __name__ == "__main__":
    main()

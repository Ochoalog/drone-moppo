"""Plano 1 (ICUAS 2027) — Avaliação cross-view real (Drone -> Satélite) e filtro bayesiano passivo
nos voos reais ADM1 (0035) e ADM2 (0036) em Arroio do Meio, RS (Vale do Taquari, 18/04/2019),
usando o GT canônico sincronizado (`traj_truth_sync.txt`), convenção angular verificada (`-90 - yaw`)
e DINOv2 ViT-S/14 (GeM e grid4) sobre os mapas canônicos ADM_35 (2018, 2016 e 2014).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
OUT_RES = ROOT / "results_v4"
OUT_RES.mkdir(parents=True, exist_ok=True)

ADM_BASE = Path("/storage/UAVs Localization/Datasets/ADM")
MAPS = {
    "2018_recent": ADM_BASE / "0039" / "Mapas" / "ADM_35_17_12_2018.jpg",
    "2016_3y_old": ADM_BASE / "0039" / "Mapas" / "ADM_35_24_11_2016.jpg",
    "2014_5y_old": ADM_BASE / "0039" / "Mapas" / "ADM_35_27_01_2014.jpg",
}
M_PER_PX = 0.1887  # ADM_35 (voos_e_mapas_do_artigo.md)
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)


def find_img(folder: Path, name: str) -> Path:
    p = folder / name
    if p.exists():
        return p
    stem = Path(name).stem
    for cand in sorted(folder.glob(stem + "*")):
        return cand
    raise FileNotFoundError(f"Imagem não encontrada: {folder / name}")


def load_flight(flight_dir: Path, start_idx: int, end_idx: int, stride: int = 8):
    traj = [Path(l.strip()).name for l in (flight_dir / "traj.txt").read_text().splitlines() if l.strip()]
    sync = [l.strip().split() for l in (flight_dir / "traj_truth_sync.txt").read_text().splitlines() if l.strip()]
    samples = []
    for idx in range(start_idx, min(end_idx, len(traj), len(sync)), stride):
        x, y, z, yaw = map(float, sync[idx])
        img_path = find_img(flight_dir / "Images_RGB", traj[idx])
        samples.append((idx, img_path, x, y, z, yaw))
    return samples


def crop_rotated_square(img_rgb: np.ndarray, yaw_deg: float, out_px: int = 224) -> tuple[np.ndarray, int]:
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
    x0 = max(0, min(w - side_px, int(round(x)) - half))
    y0 = max(0, min(h - side_px, int(round(y)) - half))
    return cv2.resize(sat_rgb[y0 : y0 + side_px, x0 : x0 + side_px], (out_px, out_px), interpolation=cv2.INTER_AREA)


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


def run_passive_filter(q_feats, m_feats, q_xy, grid_xs, grid_ys, beta: float = 25.0) -> dict:
    ny, nx = len(grid_ys), len(grid_xs)
    step_px = float(grid_xs[1] - grid_xs[0])
    belief = np.full((ny, nx), 1.0 / (ny * nx), dtype=np.float64)
    errors_m = []
    for t in range(len(q_xy)):
        if t > 0:
            dx = int(round((q_xy[t, 0] - q_xy[t - 1, 0]) / step_px))
            dy = int(round((q_xy[t, 1] - q_xy[t - 1, 1]) / step_px))
            shifted = np.zeros_like(belief)
            sy0, sy1 = max(0, -dy), min(ny, ny - dy)
            sx0, sx1 = max(0, -dx), min(nx, nx - dx)
            dy0, dy1 = max(0, dy), min(ny, ny + dy)
            dx0, dx1 = max(0, dx), min(nx, nx + dx)
            if sy1 > sy0 and sx1 > sx0:
                shifted[dy0:dy1, dx0:dx1] = belief[sy0:sy1, sx0:sx1]
            belief = cv2.GaussianBlur(shifted, (3, 3), 0.7)
            belief = np.maximum(belief, 1e-12)
        sim = (m_feats @ q_feats[t]).reshape(ny, nx)
        belief = belief * np.exp(beta * (sim - sim.max()))
        belief /= belief.sum()
        iy, ix = np.unravel_index(np.argmax(belief), belief.shape)
        errors_m.append(float(np.hypot(grid_xs[ix] - q_xy[t, 0], grid_ys[iy] - q_xy[t, 1]) * M_PER_PX))
    return {
        "final_err_m": round(errors_m[-1], 1),
        "mean_last5_err_m": round(float(np.mean(errors_m[-5:])), 1),
        "median_err_m": round(float(np.median(errors_m)), 1),
        "converged_within_50m": bool(errors_m[-1] <= 50.0),
    }


def main() -> None:
    # Fatias canônicas (voos_e_mapas_do_artigo.md): ADM1 0103-0402 (idx ~18..318), ADM2 0199-0394 (idx ~84..280)
    flights = {
        "ADM1_0035": load_flight(ADM_BASE / "0035", 18, 318, stride=10),
        "ADM2_0036": load_flight(ADM_BASE / "0036", 84, 280, stride=7),
    }
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False).eval()

    q_data = {}
    all_sides = []
    for fname, samples in flights.items():
        crops, sides = [], []
        for _, p, x, y, z, yaw in samples:
            rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            c, side = crop_rotated_square(rgb, yaw, 224)
            crops.append(c)
            sides.append(int(round(side * z)))
        q_feats = extract_dino(model, np.stack(crops))
        q_xy = np.array([[s[2], s[3]] for s in samples])
        q_data[fname] = {"n": len(samples), "feats": q_feats, "xy": q_xy, "median_side_px": int(np.median(sides))}
        all_sides.extend(sides)
        print(f"{fname}: {len(samples)} frames, median footprint={int(np.median(sides))*M_PER_PX:.1f} m", flush=True)

    # Grade compartilhada de 220 px (~41.5 m) sobre o mapa 4800x4800 (~19x19 = 361 células)
    median_side = int(np.median(all_sides))
    grid_step = 220
    margin = median_side // 2 + 10
    xs = np.arange(margin, 4800 - margin, grid_step)
    ys = np.arange(margin, 4800 - margin, grid_step)
    gx, gy = np.meshgrid(xs, ys)
    cand_xy = np.stack([gx.ravel(), gy.ravel()], axis=1)

    results = {
        "grid_step_m": round(grid_step * M_PER_PX, 1),
        "map_footprint_m": round(median_side * M_PER_PX, 1),
        "n_map_cells": len(cand_xy),
        "flights": {k: {} for k in flights},
    }

    for map_name, map_path in MAPS.items():
        sat_rgb = cv2.cvtColor(cv2.imread(str(map_path)), cv2.COLOR_BGR2RGB)
        cand_crops = np.stack([crop_map_at(sat_rgb, cx, cy, median_side, 224) for cx, cy in cand_xy])
        m_feats = extract_dino(model, cand_crops)
        for fname, fd in q_data.items():
            dists_m = np.linalg.norm(fd["xy"][:, None, :] - cand_xy[None, :, :], axis=-1) * M_PER_PX
            nearest = dists_m.argmin(axis=1)
            idx = np.arange(fd["n"])
            res_map = {}
            for dname in ("gem", "grid4"):
                sim = fd["feats"][dname] @ m_feats[dname].T
                top1 = sim.argmax(axis=1)
                top5 = np.argsort(-sim, axis=1)[:, :5]
                t1 = float((dists_m[idx, top1] <= 50.0).mean() * 100)
                t5 = float(np.mean([(dists_m[i, top5[i]] <= 50.0).any() for i in range(fd["n"])]) * 100)
                filt = run_passive_filter(fd["feats"][dname], m_feats[dname], fd["xy"], xs, ys, beta=25.0)
                res_map[dname] = {"top1_50m_pct": round(t1, 1), "top5_50m_pct": round(t5, 1), "passive_filter": filt}
            results["flights"][fname][map_name] = res_map
            print(f"  {fname} | {map_name}: grid4 top1={res_map['grid4']['top1_50m_pct']}% | filter={res_map['grid4']['passive_filter']}", flush=True)

    (OUT_RES / "p1_adm_multiflight.json").write_text(json.dumps(results, indent=2))
    print(f"Saved {OUT_RES / 'p1_adm_multiflight.json'}", flush=True)


if __name__ == "__main__":
    main()

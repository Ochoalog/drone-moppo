"""Camera footprints (crops) and descriptors, precomputed once for every cell.

Map: 1 descriptor per cell per map date (crop centered on the cell, no augmentation).
Observation (`post`): K variants per cell with +-jitter, +-rotation and photometric noise.
Raw descriptors are stored as float16; `load_descriptors` standardizes them per date with
train-region statistics and L2-normalizes them.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from .config import DATA, load_config
from .data import Raster, load_date, load_masks
from .grid import Grid

FEAT = DATA / "features"


# ----------------------------------------------------------------------------- crops
def crop(r: Raster, x: float, y: float, footprint_m: float, out_px: int, theta_deg: float = 0.0) -> np.ndarray:
    """Square north-up footprint centered at UTM (x, y), rotated by theta, resampled to out_px."""
    s = footprint_m / out_px / r.res  # input px per output px
    t = math.radians(theta_deg)
    a, b = s * math.cos(t), s * math.sin(t)
    cu, cv = r.world_to_px(x, y)
    h = (out_px - 1) / 2
    # dst (u, v) -> src: [a -b; b a] @ ([u, v] - h) + [cu, cv]
    M = np.array([[a, -b, cu - a * h + b * h], [b, a, cv - b * h - a * h]], np.float64)
    return cv2.warpAffine(r.img, M, (out_px, out_px), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def photometric(img: np.ndarray, rng: np.random.Generator, gain: float, noise: float) -> np.ndarray:
    g = 1.0 + rng.uniform(-gain, gain)
    bias = rng.uniform(-gain, gain) * 128
    out = img.astype(np.float32) * g + bias + rng.normal(0, noise, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------- hand descriptor
HAND_BINS = 8
HAND_GRID = 4


def hand_descriptor(batch: np.ndarray) -> np.ndarray:
    """(B, H, W, 3) uint8 -> (B, 4*4*8 + 4*4*6) gradient-orientation histograms + Lab mean/std per cell."""
    B, H, W, _ = batch.shape
    g = HAND_GRID
    gray = batch.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, :, 1:-1] = gray[:, :, 2:] - gray[:, :, :-2]
    gy[:, 1:-1, :] = gray[:, 2:, :] - gray[:, :-2, :]
    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.mod(np.arctan2(gy, gx), np.pi)  # unsigned orientation
    bins = np.minimum((ang / np.pi * HAND_BINS).astype(np.int64), HAND_BINS - 1)
    # group index = ((b * g + gy) * g + gx) * bins + bin, histogram via one weighted bincount
    rows = (np.arange(H) // (H // g))[:, None]
    cols = (np.arange(W) // (W // g))[None, :]
    cell = (rows * g + cols)[None]
    idx = ((np.arange(B)[:, None, None] * g * g + cell) * HAND_BINS + bins).ravel()
    hog = np.bincount(idx, weights=mag.ravel(), minlength=B * g * g * HAND_BINS).astype(np.float32)
    hog = np.sqrt(hog.reshape(B, -1) / (H * W / g / g))  # Hellinger-like compression
    lab = cv2.cvtColor(batch.reshape(B * H, W, 3), cv2.COLOR_RGB2LAB).astype(np.float32)  # one call
    lab = lab.reshape(B, g, H // g, g, W // g, 3).transpose(0, 1, 3, 2, 4, 5).reshape(B, g, g, -1, 3)
    mu = lab.mean(axis=3)
    sd = np.sqrt(np.maximum((lab * lab).mean(axis=3) - mu * mu, 0))
    col = np.concatenate([mu, sd], axis=-1).reshape(B, -1)
    return np.concatenate([hog, col], axis=1)


# ----------------------------------------------------------------------------- CNN descriptor
class CNNDescriptor:
    """ResNet18 (ImageNet) global-average-pooled 512-d features."""

    def __init__(self, device: str = "cpu"):
        import torch
        import torchvision

        self.torch = torch
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = torch.nn.Identity()
        self.model = m.eval().to(device)
        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        t = self.torch
        with t.inference_mode():
            x = t.from_numpy(batch).to(self.device).permute(0, 3, 1, 2).float() / 255.0
            x = (x - self.mean) / self.std
            return self.model(x).cpu().numpy()


# ----------------------------------------------------------------------------- precompute
def _cells(grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    rr, cc = np.mgrid[0 : grid.nrows, 0 : grid.ncols]
    return rr.ravel(), cc.ravel()


def compute_set(r: Raster, grid: Grid, fcfg: dict, offset: tuple[float, float], extractors: dict,
                augment: bool, rng: np.random.Generator, batch: int = 256) -> dict[str, np.ndarray]:
    """Descriptors for every cell (row-major) of one image, one variant."""
    rr, cc = _cells(grid)
    x, y = grid.center(rr, cc)
    x = x + offset[0]
    y = y + offset[1]
    n = len(rr)
    if augment:
        jx = rng.uniform(-fcfg["jitter_m"], fcfg["jitter_m"], n)
        jy = rng.uniform(-fcfg["jitter_m"], fcfg["jitter_m"], n)
        th = rng.uniform(-fcfg["rot_deg"], fcfg["rot_deg"], n)
    else:
        jx = jy = th = np.zeros(n)
    out: dict[str, list] = {k: [] for k in extractors}
    px = fcfg["crop_px"]
    buf = np.empty((batch, px, px, 3), np.uint8)
    for s in range(0, n, batch):
        e = min(s + batch, n)
        for i in range(s, e):
            im = crop(r, x[i] + jx[i], y[i] + jy[i], fcfg["footprint_m"], px, th[i])
            if augment:
                im = photometric(im, rng, fcfg["photo_gain"], fcfg["photo_noise"])
            buf[i - s] = im
        for k, f in extractors.items():
            out[k].append(f(buf[: e - s]).astype(np.float16))
    return {k: np.concatenate(v) for k, v in out.items()}


def precompute(cfg: dict, descriptors: list[str] | None = None, limit_rows: int | None = None,
               out_dir: Path = FEAT) -> dict:
    """Compute and save map descriptors (all map dates) and K observation variants (post)."""
    import torch

    torch.set_num_threads(int(__import__("os").environ.get("DAL_TORCH_THREADS", __import__("os").cpu_count() or 1)))
    d, fcfg = cfg["data"], cfg["features"]
    grid = Grid.from_aoi(**d["aoi"], cell=d["cell_m"])
    if limit_rows:
        grid = Grid(grid.x0, grid.y1, limit_rows, grid.ncols, grid.cell)
    from .align import load_offsets

    offsets = load_offsets()
    descriptors = descriptors or fcfg["descriptors"]
    ex = {}
    if "hand" in descriptors:
        ex["hand"] = hand_descriptor
    if "cnn" in descriptors:
        ex["cnn"] = CNNDescriptor()
    rng = np.random.default_rng(fcfg["seed"])
    out_dir.mkdir(parents=True, exist_ok=True)
    timing = {}
    n = grid.nrows * grid.ncols
    for date in d["sources"]:
        r = load_date(date)
        jobs = [("map", False)] + ([("obs", True)] if date == "post" else [])
        for kind, aug in jobs:
            reps = fcfg["obs_variants"] if kind == "obs" else 1
            res = {k: [] for k in ex}
            t = time.time()
            for _ in range(reps):
                one = compute_set(r, grid, fcfg, offsets[date], ex, aug, rng)
                for k in ex:
                    res[k].append(one[k])
            dt = time.time() - t
            timing[f"{date}_{kind}"] = dict(seconds=dt, crops=n * reps, crops_per_s=n * reps / dt)
            print(f"[features] {date} {kind}: {n * reps} crops in {dt:.0f}s ({n * reps / dt:.0f}/s)", flush=True)
            for k in ex:
                arr = np.stack(res[k]) if kind == "obs" else res[k][0]
                np.save(out_dir / f"{k}_{date}_{kind}.npy", arr)
        del r
    meta = dict(grid=grid.to_dict(), descriptors=list(ex), timing=timing, cfg=fcfg,
                total_seconds=sum(v["seconds"] for v in timing.values()))
    (out_dir / f"meta_{'_'.join(ex)}.json").write_text(json.dumps(meta, indent=1))
    return meta


# ----------------------------------------------------------------------------- loading
def _standardize(x: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    ref = x[..., train_idx, :] if x.ndim == 3 else x[train_idx]
    ref = ref.reshape(-1, x.shape[-1])
    mu = ref.mean(axis=0)
    sd = ref.std(axis=0) + 1e-3 * (ref.std() + 1e-6)
    x = (x - mu) / sd
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def load_descriptors(desc: str, map_date: str, grid: Grid, train_rows: np.ndarray, feat_dir: Path = FEAT
                     ) -> tuple[np.ndarray, np.ndarray]:
    """(map (R, C, D), obs (K, R, C, D)) float32, standardized per date with train cells, L2-normed."""
    train_idx = (train_rows[:, None] * grid.ncols + np.arange(grid.ncols)[None]).ravel()
    m = _standardize(np.load(feat_dir / f"{desc}_{map_date}_map.npy"), train_idx)
    o = _standardize(np.load(feat_dir / f"{desc}_post_obs.npy"), train_idx)
    D = m.shape[-1]
    return m.reshape(grid.nrows, grid.ncols, D), o.reshape(o.shape[0], grid.nrows, grid.ncols, D)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--descriptors", nargs="*")
    ap.add_argument("--limit-rows", type=int)
    ap.add_argument("--out", default=str(FEAT))
    a = ap.parse_args()
    precompute(load_config(a.config), a.descriptors, a.limit_rows, Path(a.out))


if __name__ == "__main__":
    main()

"""Download AOI windows from remote COGs (overview levels only), build cell masks.

Only windows are read through `/vsicurl/`; reads are clipped to the dataset bounds and the
rest is zero-filled (boundless reads are very slow on these huge rasters).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
    GDAL_HTTP_MULTIRANGE="YES",
    GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
    GDAL_HTTP_MAX_RETRY="5",
    GDAL_HTTP_RETRY_DELAY="2",
    VSI_CACHE="TRUE",
)
for _k, _v in GDAL_ENV.items():
    os.environ.setdefault(_k, _v)

import rasterio  # noqa: E402  (after GDAL env)
from rasterio.transform import Affine  # noqa: E402
from rasterio.windows import Window  # noqa: E402

from .config import DATA, load_config  # noqa: E402
from .grid import CRS, Grid  # noqa: E402

RAW = DATA / "raw"


@dataclass
class Raster:
    """In-memory RGB image with an axis-aligned UTM georeference."""

    img: np.ndarray  # (H, W, 3) uint8
    x_origin: float  # UTM x of the left edge
    y_origin: float  # UTM y of the top edge
    res: float  # m / px

    def world_to_px(self, x, y):
        """Continuous pixel coordinates (pixel centers at integers, OpenCV convention)."""
        return (np.asarray(x) - self.x_origin) / self.res - 0.5, (self.y_origin - np.asarray(y)) / self.res - 0.5

    def window(self, xmin: float, ymin: float, xmax: float, ymax: float) -> np.ndarray:
        """Nearest-pixel sub-array covering a UTM box (clipped to the array)."""
        c0 = int(math.floor((xmin - self.x_origin) / self.res))
        c1 = int(math.ceil((xmax - self.x_origin) / self.res))
        r0 = int(math.floor((self.y_origin - ymax) / self.res))
        r1 = int(math.ceil((self.y_origin - ymin) / self.res))
        H, W = self.img.shape[:2]
        return self.img[max(r0, 0) : min(r1, H), max(c0, 0) : min(c1, W)]


def _open(url: str, factor: int):
    level = int(round(math.log2(factor))) - 1  # overview_level 0 -> factor 2
    path = url if url.startswith("/") or url.startswith("/vsi") else "/vsicurl/" + url
    return rasterio.open(path, overview_level=level) if level >= 0 else rasterio.open(path)


def read_box(url: str, bounds: tuple[float, float, float, float], factor: int, workers: int = 8,
             chunk: int = 2048) -> Raster:
    """Read a UTM box at an overview level, snapped to the dataset pixel grid.

    Chunks are read in parallel threads, each with its own dataset handle.
    """
    xmin, ymin, xmax, ymax = bounds
    with _open(url, factor) as ds:
        assert str(ds.crs) == CRS, ds.crs
        t = ds.transform
        res = t.a
        W, H = ds.width, ds.height
    c0 = int(math.floor((xmin - t.c) / res))
    c1 = int(math.ceil((xmax - t.c) / res))
    r0 = int(math.floor((t.f - ymax) / res))
    r1 = int(math.ceil((t.f - ymin) / res))
    out = np.zeros((r1 - r0, c1 - c0, 3), np.uint8)
    tiles = [(r, c) for r in range(r0, r1, chunk) for c in range(c0, c1, chunk)]
    local = threading.local()

    def job(rc):
        r, c = rc
        rr0, rr1 = max(r, 0), min(r + chunk, r1, H)
        cc0, cc1 = max(c, 0), min(c + chunk, c1, W)
        if rr1 <= rr0 or cc1 <= cc0:
            return
        if not hasattr(local, "ds"):
            local.ds = _open(url, factor)
        a = local.ds.read([1, 2, 3], window=Window(cc0, rr0, cc1 - cc0, rr1 - rr0))
        out[rr0 - r0 : rr1 - r0, cc0 - c0 : cc1 - c0] = np.moveaxis(a, 0, -1)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(job, tiles))
    return Raster(out, t.c + c0 * res, t.f - r0 * res, res)


def save_raster(r: Raster, path: Path, jpeg: bool = True) -> None:
    """Tiled GeoTIFF (JPEG q95 for imagery to save disk, DEFLATE otherwise)."""
    prof = dict(driver="GTiff", width=r.img.shape[1], height=r.img.shape[0], count=3, dtype="uint8",
                crs=CRS, transform=Affine(r.res, 0, r.x_origin, 0, -r.res, r.y_origin), tiled=True,
                blockxsize=512, blockysize=512)
    prof.update(dict(compress="JPEG", jpeg_quality=95, photometric="YCBCR") if jpeg else dict(compress="DEFLATE"))
    with rasterio.open(path, "w", **prof) as ds:
        ds.write(np.moveaxis(r.img, -1, 0))


def load_raster(path: Path) -> Raster:
    with rasterio.open(path) as ds:
        img = np.moveaxis(ds.read([1, 2, 3]), 0, -1).copy()
        t = ds.transform
    return Raster(img, t.c, t.f, t.a)


def load_date(date: str, thumb: bool = False) -> Raster:
    return load_raster(RAW / f"{date}{'_thumb' if thumb else ''}.tif")


def valid_mask(img: np.ndarray) -> np.ndarray:
    """Pixel has data. JPEG ringing leaves tiny non-zero values in nodata, hence > 3."""
    return img.max(axis=2) > 3


def _disk(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def cloud_mask(img: np.ndarray, v_min: int, s_max: int, dilate_px: int, open_px: int = 0) -> np.ndarray:
    """Bright & unsaturated pixels (HSV: V > v_min, S < s_max), optionally opened (removes
    small bright objects such as white roofs, which are not clouds) and dilated (soft edges)."""
    m = np.zeros(img.shape[:2], np.uint8)
    step = 2048  # row blocks to bound memory
    for r in range(0, img.shape[0], step):
        hsv = cv2.cvtColor(img[r : r + step], cv2.COLOR_RGB2HSV)
        m[r : r + step] = ((hsv[..., 2] > v_min) & (hsv[..., 1] < s_max)).astype(np.uint8)
    if open_px > 1:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _disk(open_px))
    if dilate_px > 1:
        m = cv2.dilate(m, _disk(dilate_px))
    return m.astype(bool)


def cell_fractions(mask: np.ndarray, r: Raster, grid: Grid, size_m: float) -> np.ndarray:
    """Fraction of True pixels in a size_m box centered on every cell (pixels outside count False)."""
    ii = cv2.integral(mask.astype(np.uint8), sdepth=cv2.CV_32S)  # (H+1, W+1)
    H, W = mask.shape
    x, y = grid.centers
    half = size_m / 2
    c0 = np.clip(np.round((x - half - r.x_origin) / r.res).astype(int), 0, W)
    c1 = np.clip(np.round((x + half - r.x_origin) / r.res).astype(int), 0, W)
    r0 = np.clip(np.round((r.y_origin - (y + half)) / r.res).astype(int), 0, H)
    r1 = np.clip(np.round((r.y_origin - (y - half)) / r.res).astype(int), 0, H)
    s = ii[r1, c1].astype(np.int64) - ii[r0, c1] - ii[r1, c0] + ii[r0, c0]
    n = np.round(size_m / r.res) ** 2
    return (s / n).astype(np.float32)


def download(cfg: dict, force: bool = False) -> None:
    """Fetch AOI (+margin) at the working overview and the thumbnail overview for all dates."""
    d = cfg["data"]
    grid = Grid.from_aoi(**d["aoi"], cell=d["cell_m"])
    m = d["margin_m"]
    box = (grid.x0 - m, grid.y0 - m, grid.x1 + m, grid.y1 + m)
    RAW.mkdir(parents=True, exist_ok=True)
    for date, url in d["sources"].items():
        for factor, suffix in ((d["overview_factor"], ""), (d["thumb_factor"], "_thumb")):
            path = RAW / f"{date}{suffix}.tif"
            if path.exists() and not force:
                print(f"[data] {path.name} exists, skipping")
                continue
            t = time.time()
            r = read_box(url, box, factor)
            save_raster(r, path, jpeg=True)
            print(f"[data] {date}{suffix}: {r.img.shape} @ {r.res:.3f} m/px in {time.time() - t:.0f}s "
                  f"-> {path.stat().st_size / 1e6:.0f} MB")


def build_masks(cfg: dict) -> dict:
    """Per-date, per-cell valid/cloud fractions (150 m footprint) and AOI-level stats."""
    d = cfg["data"]
    grid = Grid.from_aoi(**d["aoi"], cell=d["cell_m"])
    fp = cfg["features"]["footprint_m"]
    out, stats = {}, {}
    for date in d["sources"]:
        r = load_date(date)
        val = valid_mask(r.img)
        cld = cloud_mask(r.img, d["cloud_v"], d["cloud_s"], d["cloud_dilate_px"], d["cloud_open_px"]) & val
        vf = cell_fractions(val, r, grid, fp)
        cf = cell_fractions(cld, r, grid, fp)
        aoi = r.window(grid.x0, grid.y0, grid.x1, grid.y1)
        a_val = valid_mask(aoi)
        a_raw = cloud_mask(aoi, d["cloud_v"], d["cloud_s"], d["cloud_raw_dilate_px"]) & a_val
        a_cld = cloud_mask(aoi, d["cloud_v"], d["cloud_s"], d["cloud_dilate_px"], d["cloud_open_px"]) & a_val
        ok = (vf >= d["valid_min"]) & (cf <= d["cloud_max"])
        out[f"{date}_valid_frac"] = vf
        out[f"{date}_cloud_frac"] = cf
        out[f"{date}_ok"] = ok
        stats[date] = dict(aoi_valid_frac=float(a_val.mean()),
                           aoi_cloud_raw_frac=float(a_raw.sum() / max(a_val.sum(), 1)),
                           aoi_cloud_frac_of_valid=float(a_cld.sum() / max(a_val.sum(), 1)),
                           cells_ok_frac=float(ok.mean()))
        print(f"[masks] {date}: {stats[date]}")
        del r, val, cld, aoi, a_val, a_raw, a_cld
    np.savez_compressed(DATA / "masks.npz", **out, grid=json.dumps(grid.to_dict()))
    (DATA / "masks_stats.json").write_text(json.dumps(stats, indent=1))
    return stats


def load_masks() -> dict[str, np.ndarray]:
    z = np.load(DATA / "masks.npz")
    return {k: z[k] for k in z.files if k != "grid"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--step", choices=["download", "masks", "stats", "all"], default="all")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.step in ("download", "all"):
        download(cfg, a.force)
    if a.step in ("masks", "all"):
        build_masks(cfg)
    if a.step == "stats":
        from .config import RESULTS

        export_stats(cfg, RESULTS / "data_stats.json")


if __name__ == "__main__":
    main()


def export_stats(cfg: dict, out: Path) -> None:
    """Data facts used by the report (copied into results/ so REPORT numbers come from there)."""
    d = cfg["data"]
    grid = Grid.from_aoi(**d["aoi"], cell=d["cell_m"])
    train, test = grid.split_rows(d["split_lat"], d["split_margin_m"])
    masks = json.loads((DATA / "masks_stats.json").read_text())
    align = json.loads((DATA / "align.json").read_text())
    s = dict(grid=grid.to_dict(), train_rows=[int(train.min()), int(train.max())],
             test_rows=[int(test.min()), int(test.max())],
             dates={k: dict(masks=masks[k], align=align[k]) for k in d["sources"]})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(s, indent=1))

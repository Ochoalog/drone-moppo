"""Porto Alegre data products: overview windows, cell masks (valid / cloud / water), alignment,
data-gate statistics and the overview figure. Everything is written under the POA data/results
folders of the config (`paths:`); the Taquari folders are never touched.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .config import load_config, paths
from .data import Raster, cell_fractions, cloud_mask, load_raster, read_box, save_raster, valid_mask
from .grid import Grid, lonlat_to_utm


# ============================================================================ grid and regions
def poa_grid(cfg: dict) -> Grid:
    d = cfg["data"]
    return Grid.from_aoi(**d["aoi"], cell=d["cell_m"])


def region_slices(cfg: dict, grid: Grid | None = None) -> dict[str, tuple[slice, slice]]:
    """(rows, cols) slices of every region inside the POA grid (cells fully inside the lon/lat box)."""
    grid = grid or poa_grid(cfg)
    out = {}
    for name, b in cfg["regions"].items():
        g = Grid.from_aoi(**b, cell=grid.cell)
        r0 = int(round((grid.y1 - g.y1) / grid.cell))
        c0 = int(round((g.x0 - grid.x0) / grid.cell))
        r0c, c0c = max(r0, 0), max(c0, 0)
        r1, c1 = min(r0 + g.nrows, grid.nrows), min(c0 + g.ncols, grid.ncols)
        out[name] = (slice(r0c, r1), slice(c0c, c1))
    return out


def region_overlap(cfg: dict) -> dict:
    sl = region_slices(cfg)
    (a, sa), (b, sb) = list(sl.items())[:2]
    rows = max(0, min(sa[0].stop, sb[0].stop) - max(sa[0].start, sb[0].start))
    cols = max(0, min(sa[1].stop, sb[1].stop) - max(sa[1].start, sb[1].start))
    cell_km2 = (cfg["data"]["cell_m"] / 1000) ** 2
    area = lambda s: (s[0].stop - s[0].start) * (s[1].stop - s[1].start) * cell_km2  # noqa: E731
    return dict(regions=[a, b], overlap_cells=rows * cols, overlap_km2=rows * cols * cell_km2,
                area_km2={a: area(sa), b: area(sb)},
                slices={k: dict(rows=[s[0].start, s[0].stop], cols=[s[1].start, s[1].stop]) for k, s in sl.items()})


# ============================================================================ download
def raw_dir(cfg: dict) -> Path:
    return paths(cfg)["data"] / "raw"


def load_date(cfg: dict, date: str, thumb: bool = False) -> Raster:
    return load_raster(raw_dir(cfg) / f"{date}{'_thumb' if thumb else ''}.tif")


def download(cfg: dict, force: bool = False) -> None:
    d = cfg["data"]
    grid = poa_grid(cfg)
    m = d["margin_m"]
    box = (grid.x0 - m, grid.y0 - m, grid.x1 + m, grid.y1 + m)
    out = raw_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    for date, url in d["sources"].items():
        for factor, suffix in ((d["overview_factor"], ""), (d["thumb_factor"], "_thumb")):
            p = out / f"{date}{suffix}.tif"
            if p.exists() and not force:
                print(f"[poa-data] {p.name} exists, skipping")
                continue
            t = time.time()
            r = read_box(url, box, factor)
            save_raster(r, p, jpeg=True)
            print(f"[poa-data] {date}{suffix}: {r.img.shape} @ {r.res:.3f} m/px in {time.time() - t:.0f}s "
                  f"-> {p.stat().st_size / 1e6:.0f} MB", flush=True)


# ============================================================================ water
def water_mask(img: np.ndarray, w: dict) -> np.ndarray:
    """Water by colour + texture: homogeneous (low local std) and either dark or brown/reddish
    (sediment-laden flood water), then a morphological opening to drop small dark objects."""
    m = np.zeros(img.shape[:2], np.uint8)
    step = 2048
    k = int(w["texture_win"])
    for r in range(0, img.shape[0], step):
        lo, hi = max(r - k, 0), min(r + step + k, img.shape[0])
        blk = img[lo:hi]
        g = cv2.cvtColor(blk, cv2.COLOR_RGB2GRAY).astype(np.float32)
        mu = cv2.blur(g, (k, k))
        sd = np.sqrt(np.maximum(cv2.blur(g * g, (k, k)) - mu * mu, 0))
        hsv = cv2.cvtColor(blk, cv2.COLOR_RGB2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        dark = v < w["dark_v"]
        brown = (h >= w["brown_h"][0]) & (h <= w["brown_h"][1]) & (s >= w["brown_s_min"]) & (v <= w["brown_v_max"])
        wm = (sd < w["texture_max"]) & (dark | brown) & (v > 3)
        m[r : min(r + step, img.shape[0])] = wm[r - lo : r - lo + min(step, img.shape[0] - r)]
    if w.get("open_px", 0) > 1:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w["open_px"],) * 2))
    return m.astype(bool)


# ============================================================================ clouds (POA variant)
def cloud_mask_poa(img: np.ndarray, c: dict) -> np.ndarray:
    """PoC detector (bright & unsaturated) OR smooth haze (moderately bright, unsaturated, low local
    std: thin grey clouds), opened with a larger disk (urban white roofs are bigger than in the
    Taquari) and dilated with a ~120 m halo for the soft edges."""
    m = np.zeros(img.shape[:2], np.uint8)
    step, k = 2048, int(c["std_win"])
    for r in range(0, img.shape[0], step):
        lo, hi = max(r - k, 0), min(r + step + k, img.shape[0])
        blk = img[lo:hi]
        g = cv2.cvtColor(blk, cv2.COLOR_RGB2GRAY).astype(np.float32)
        mu = cv2.blur(g, (k, k))
        sd = np.sqrt(np.maximum(cv2.blur(g * g, (k, k)) - mu * mu, 0))
        hsv = cv2.cvtColor(blk, cv2.COLOR_RGB2HSV)
        s_, v_ = hsv[..., 1], hsv[..., 2]
        cm = ((v_ > c["v_bright"]) & (s_ < c["s_bright"])) | ((v_ > c["v_haze"]) & (s_ < c["s_haze"]) & (sd < c["haze_std_max"]))
        n = min(step, img.shape[0] - r)
        m[r : r + n] = cm[r - lo : r - lo + n]
    disk = lambda n: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))  # noqa: E731
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, disk(c["open_px"]))
    m = cv2.dilate(m, disk(c["dilate_px"]))
    return m.astype(bool)


# ============================================================================ masks + stats
def build_masks(cfg: dict) -> dict:
    d = cfg["data"]
    grid = poa_grid(cfg)
    fp = cfg["features"]["footprint_m"]
    P = paths(cfg)
    out, stats = {}, {}
    vals = {}
    for date in d["sources"]:
        r = load_date(cfg, date)
        aoi = r.window(grid.x0, grid.y0, grid.x1, grid.y1)
        val = valid_mask(r.img)
        cld = cloud_mask_poa(r.img, cfg["cloud_poa"]) & val
        wat = water_mask(r.img, cfg["water"]) & val
        vf, cf, wf = (cell_fractions(x, r, grid, fp) for x in (val, cld, wat))
        ok = (vf >= d["valid_min"]) & (cf <= d["cloud_max"])
        a_val = valid_mask(aoi)
        a_raw = cloud_mask(aoi, d["cloud_v"], d["cloud_s"], d["cloud_raw_dilate_px"]) & a_val
        a_cld = cloud_mask_poa(aoi, cfg["cloud_poa"]) & a_val
        a_poc = cloud_mask(aoi, d["cloud_v"], d["cloud_s"], d["cloud_dilate_px"], d["cloud_open_px"]) & a_val
        vals[date] = a_val
        out.update({f"{date}_valid_frac": vf, f"{date}_cloud_frac": cf, f"{date}_water_frac": wf, f"{date}_ok": ok})
        px_km2 = (r.res / 1000) ** 2
        stats[date] = dict(valid_km2=float(a_val.sum() * px_km2), aoi_valid_frac=float(a_val.mean()),
                           cloud_raw_frac=float(a_raw.sum() / max(a_val.sum(), 1)),
                           cloud_frac_of_valid=float(a_cld.sum() / max(a_val.sum(), 1)),
                           cloud_poc_detector_frac=float(a_poc.sum() / max(a_val.sum(), 1)),
                           water_frac_of_valid=float(wf[vf >= d["valid_min"]].mean()) if (vf >= d["valid_min"]).any() else 0.0,
                           cells_ok_frac=float(ok.mean()), res_m=r.res)
        print(f"[poa-masks] {date}: {stats[date]}", flush=True)
        del r, val, cld, wat, aoi, a_raw, a_cld, a_poc
    obs = d["obs_date"]
    for date in d["sources"]:
        if date == obs:
            continue
        a, b = vals[obs], vals[date]
        h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])  # same snapped grid, same res
        stats[date]["overlap_km2"] = float((a[:h, :w] & b[:h, :w]).sum() * (stats[date]["res_m"] / 1000) ** 2)
    pd_ = P["data"]
    pd_.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pd_ / "masks.npz", **out, grid=json.dumps(grid.to_dict()))
    (pd_ / "masks_stats.json").write_text(json.dumps(stats, indent=1))
    return stats


def load_masks(cfg: dict) -> dict[str, np.ndarray]:
    z = np.load(paths(cfg)["data"] / "masks.npz")
    return {k: z[k] for k in z.files if k != "grid"}


# ============================================================================ alignment
def align(cfg: dict) -> dict:
    from .align import estimate_offset

    d = cfg["data"]
    grid = poa_grid(cfg)
    masks = load_masks(cfg)
    obs = d["obs_date"]
    post = load_date(cfg, obs)
    out = {obs: dict(dx_m=0.0, dy_m=0.0, norm_m=0.0, applied=False)}
    P = paths(cfg)
    for date in d["sources"]:
        if date == obs:
            continue
        other = load_date(cfg, date)
        est = estimate_offset(post, other, masks[f"{obs}_ok"] & masks[f"{date}_ok"], grid, n_tiles=d["align_tiles"])
        est["applied"] = bool(est["norm_m"] > d["align_apply_min_m"])
        tiles = est.pop("tiles")
        resp = np.array([t["resp"] for t in tiles])
        dd = np.array([[t["dx_m"], t["dy_m"]] for t in tiles])
        strong = resp >= np.median(resp)
        est["norm_m_strong_half"] = float(np.hypot(*np.median(dd[strong], 0))) if strong.any() else np.nan
        out[date] = est
        print(f"[poa-align] {date}: dx={est['dx_m']:.2f} dy={est['dy_m']:.2f} |d|={est['norm_m']:.2f} m "
              f"MAD={np.round(est['mad_m'], 2).tolist()} used {est['n_used']}/{est['n_tiles']} "
              f"(strong half |d|={est['norm_m_strong_half']:.2f}) applied={est['applied']}", flush=True)
        (P["data"] / f"align_tiles_{date}.json").write_text(json.dumps(tiles, indent=1))
        del other
    (P["data"] / "align.json").write_text(json.dumps(out, indent=1))
    return out


def load_offsets(cfg: dict) -> dict[str, tuple[float, float]]:
    a = json.loads((paths(cfg)["data"] / "align.json").read_text())
    return {k: ((v["dx_m"], v["dy_m"]) if v.get("applied") else (0.0, 0.0)) for k, v in a.items()}


def data_gate(cfg: dict) -> dict:
    """Measured vs hand-checked facts (+-30 %) and alignment <= align_max_m; written to results."""
    d = cfg["data"]
    P = paths(cfg)
    st = json.loads((P["data"] / "masks_stats.json").read_text())
    al = json.loads((P["data"] / "align.json").read_text())
    rows, ok = [], True
    for kind, ref in d["expected"].items():
        for date, v in ref.items():
            key = "valid_km2" if kind == "valid_km2" else "overlap_km2"
            m = st[date][key]
            rel = m / v - 1
            good = abs(rel) <= 0.3
            ok &= good
            rows.append(dict(check=f"{kind} {date}", expected=v, measured=round(m, 2), rel_diff=round(rel, 3), ok=good))
    for date, a in al.items():
        if date == d["obs_date"]:
            continue
        good = a["norm_m"] <= d["align_max_m"]
        ok &= good
        rows.append(dict(check=f"align {date} (m)", expected=f"<= {d['align_max_m']}", measured=round(a["norm_m"], 2),
                         rel_diff=None, ok=good))
    res = dict(passed=bool(ok), rows=rows, masks=st, align={k: {kk: vv for kk, vv in v.items()} for k, v in al.items()},
               regions=region_overlap(cfg), grid=poa_grid(cfg).to_dict())
    P["results"].mkdir(parents=True, exist_ok=True)
    (P["results"] / "data_gate.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"[poa-gate] data gate {'PASSED' if ok else 'FAILED'}")
    for r in rows:
        print("   ", r)
    return res


# ============================================================================ figures
DATE_LABEL = {"poa_post": "poa_post (2024-05-09, flood)", "poa_pre_feb24": "poa_pre_feb24 (2024-02-02)",
              "poa_pre_dec22": "poa_pre_dec22 (2022-12-12)"}


def _extent(t: Raster, grid: Grid) -> list[float]:
    H, W = t.img.shape[:2]
    x0 = (t.x_origin - grid.x0) / 1000
    y0 = (grid.y1 - t.y_origin) / 1000
    return [x0, x0 + W * t.res / 1000, y0 + H * t.res / 1000, y0]


def _region_boxes(ax, cfg: dict, grid: Grid) -> None:
    from matplotlib.patches import Rectangle

    col = {"ilhas": "cyan", "centro": "orange"}
    for name, (rs, cs) in region_slices(cfg, grid).items():
        c = grid.cell / 1000
        ax.add_patch(Rectangle((cs.start * c, rs.start * c), (cs.stop - cs.start) * c, (rs.stop - rs.start) * c,
                               fill=False, ec=col.get(name, "white"), lw=1.6))
        ax.text(cs.start * c + 0.1, rs.start * c + 0.35, name, color=col.get(name, "white"), fontsize=8,
                fontweight="bold")


def fig_overview(cfg: dict) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = poa_grid(cfg)
    masks = load_masks(cfg)
    P = paths(cfg)
    st = json.loads((P["data"] / "masks_stats.json").read_text())
    al = json.loads((P["data"] / "align.json").read_text())
    dates = list(cfg["data"]["sources"])
    Wk, Hk = grid.ncols * grid.cell / 1000, grid.nrows * grid.cell / 1000
    fig, axes = plt.subplots(3, len(dates), figsize=(4.2 * len(dates), 15), constrained_layout=True)
    for i, date in enumerate(dates):
        t = load_date(cfg, date, thumb=True)
        ax = axes[0, i]
        ax.imshow(t.img, extent=_extent(t, grid))
        ax.set_xlim(0, Wk)
        ax.set_ylim(Hk, 0)
        _region_boxes(ax, cfg, grid)
        ax.set_title(f"{DATE_LABEL.get(date, date)}\nvalid {st[date]['valid_km2']:.1f} km², cloud "
                     f"{100 * st[date]['cloud_frac_of_valid']:.1f}%, offset {al[date]['norm_m']:.1f} m", fontsize=9)
        ax = axes[1, i]
        ok = masks[f"{date}_ok"].astype(float)
        cf = masks[f"{date}_cloud_frac"]
        ax.imshow(ok, extent=[0, Wk, Hk, 0], cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
        ax.contour(np.linspace(0, Wk, grid.ncols), np.linspace(0, Hk, grid.nrows), cf, levels=[0.1], colors="blue",
                   linewidths=0.6)
        _region_boxes(ax, cfg, grid)
        ax.set_title(f"usable cells {100 * ok.mean():.0f}% (green); blue: cloud frac 0.1", fontsize=9)
        ax = axes[2, i]
        im = ax.imshow(masks[f"{date}_water_frac"], extent=[0, Wk, Hk, 0], cmap="Blues", vmin=0, vmax=1,
                       interpolation="nearest")
        _region_boxes(ax, cfg, grid)
        ax.set_title(f"water fraction (colour+texture), mean {100 * st[date]['water_frac_of_valid']:.0f}%", fontsize=9)
        for a in axes[:, i]:
            a.set_xlabel("km east")
    for a in axes[:, 0]:
        a.set_ylabel("km south of box top")
    fig.colorbar(im, ax=axes[2, -1], shrink=0.6)
    fig.suptitle(f"Porto Alegre box {grid.ncols}x{grid.nrows} cells of {grid.cell:.0f} m (lon "
                 f"{cfg['data']['aoi']['lon0']}..{cfg['data']['aoi']['lon1']}, lat {cfg['data']['aoi']['lat0']}.."
                 f"{cfg['data']['aoi']['lat1']})")
    P["figs"].mkdir(parents=True, exist_ok=True)
    out = P["figs"] / "data_overview.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out


def fig_mask_check(cfg: dict, date: str, box_lonlat: tuple[float, float, float, float], name: str) -> Path:
    """Zoom on the working resolution image with cloud (red) and water (blue) contours: visual QA."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = cfg["data"]
    r = load_date(cfg, date)
    x0, y0 = lonlat_to_utm(box_lonlat[0], box_lonlat[2])
    x1, y1 = lonlat_to_utm(box_lonlat[1], box_lonlat[3])
    img = r.window(x0, y0, x1, y1).copy()
    cld = cloud_mask_poa(img, cfg["cloud_poa"])
    wat = water_mask(img, cfg["water"])
    s = max(1, int(np.ceil(max(img.shape[:2]) / 1600)))
    img, cld, wat = img[::s, ::s], cld[::s, ::s], wat[::s, ::s]
    fig, axes = plt.subplots(1, 2, figsize=(16, 8 * img.shape[0] / img.shape[1] + 0.8), constrained_layout=True)
    axes[0].imshow(img)
    axes[0].set_title(f"{date} @ {r.res * s:.1f} m/px")
    axes[1].imshow(img)
    axes[1].contour(cld, levels=[0.5], colors="red", linewidths=0.8)
    over = np.zeros(img.shape[:2] + (4,), np.float32)
    over[wat] = (0, 0.3, 1, 0.45)
    axes[1].imshow(over)
    axes[1].set_title(f"red: cloud ({100 * cld.mean():.1f}%), blue: water ({100 * wat.mean():.1f}%)")
    for a in axes:
        a.axis("off")
    P = paths(cfg)
    P["figs"].mkdir(parents=True, exist_ok=True)
    out = P["figs"] / f"maskcheck_{name}.png"
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs_poa/poa.yaml")
    ap.add_argument("--step", choices=["download", "masks", "align", "gate", "figs", "all"], default="all")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.step in ("download", "all"):
        download(cfg, a.force)
    if a.step in ("masks", "all"):
        build_masks(cfg)
    if a.step in ("align", "all"):
        align(cfg)
    if a.step in ("gate", "all"):
        data_gate(cfg)
    if a.step in ("figs", "all"):
        print(fig_overview(cfg))


if __name__ == "__main__":
    main()

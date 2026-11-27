"""Gate P2-G0: prepara e valida ortomosaicos reais de UAV do CBPMESP em Canoas (Mathias Velho)
contra o mapa de satélite pré-enchente Maxar (poa_pre_feb24, 02/02/2024).

Reprojeta os 3 ortomosaicos UAV (Leste 25/05, Leste 26/05 e Oeste 25/05) para EPSG:32722 (0,61035 m/px),
lê a mesma caixa do satélite pré-evento Maxar, mede a cobertura válida em grade de 25 m, verifica o
deslocamento de alinhamento (correlação de fase) e gera figura de inspeção visual.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

ROOT = Path(__file__).resolve().parents[1]
OUT_DATA = ROOT / "data_v4" / "uav_canoas"
OUT_RES = ROOT / "results_v4"
OUT_DATA.mkdir(parents=True, exist_ok=True)
OUT_RES.mkdir(parents=True, exist_ok=True)

TARGET_CRS = "EPSG:32722"
RES_M = 0.6103515625  # exatamente a mesma resolução da v3 (SOURCE_RES_M)

# Caixa UTM 22S que cobre Setor Oeste + Setor Leste (Mathias Velho, Canoas) com folga de 150 m
X0, Y0, X1, Y1 = 476500.0, 6690900.0, 482700.0, 6693050.0  # 6200 m x 2150 m

UAV_SOURCES = [
    ("leste_20240525_15cm", str(OUT_DATA / "mathias_leste_20240525_15cm.tif")),
    ("oeste_20240525_15cm", str(OUT_DATA / "mathias_oeste_20240525_15cm.tif")),
    (
        "leste_20240526_4cm",
        "/vsicurl/https://oin-hotosm-temp.s3.us-east-1.amazonaws.com/6653dbc349dd860001fae221/0/6653dbc349dd860001fae222.tif",
    ),
]
SAT_PRE_URL = "/vsicurl/https://oin-hotosm-temp.s3.us-east-1.amazonaws.com/66426b996049ef00013b8598/1/66426b996049ef00013b859a.tif"


def warp_to_grid(src_path: str, dst_shape: tuple[int, int], dst_transform) -> np.ndarray:
    h, w = dst_shape
    dst = np.zeros((3, h, w), dtype=np.uint8)
    # Para COGs remotos de altíssima resolução (4 cm ou 30 cm), seleciona overview adequado (~0.3–0.65 m/px)
    open_kwargs = {}
    if src_path.startswith("/vsicurl/"):
        with rasterio.open(src_path) as probe:
            ovs = probe.overviews(1)
            # Estima resolução nativa em metros
            nat_m = probe.res[0] * (111000.0 if probe.crs and probe.crs.is_geographic else 1.0)
            chosen = None
            for idx, ov in enumerate(ovs):
                if nat_m * ov <= RES_M * 1.15:
                    chosen = idx
            if chosen is not None:
                open_kwargs["overview_level"] = chosen
                print(f"    (usando overview_level={chosen}, fator={ovs[chosen]}x, ~{nat_m * ovs[chosen]:.2f} m/px)")
    with rasterio.open(src_path, **open_kwargs) as src:
        reproject(
            source=rasterio.band(src, [1, 2, 3]),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=TARGET_CRS,
            resampling=Resampling.bilinear,
            src_nodata=0,
            dst_nodata=0,
        )
    return dst


def save_geotiff(path: Path, arr: np.ndarray, transform) -> None:
    profile = {
        "driver": "GTiff",
        "height": arr.shape[1],
        "width": arr.shape[2],
        "count": 3,
        "dtype": "uint8",
        "crs": TARGET_CRS,
        "transform": transform,
        "compress": "deflate",
        "tiled": True,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr)


def estimate_shift_m(uav_rgb: np.ndarray, sat_rgb: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    """Correlação de fase em escala cinza (suavizada) para estimar deslocamento residual (dx, dy) em metros."""
    u_gray = cv2.cvtColor(np.transpose(uav_rgb, (1, 2, 0)), cv2.COLOR_RGB2GRAY).astype(np.float32)
    s_gray = cv2.cvtColor(np.transpose(sat_rgb, (1, 2, 0)), cv2.COLOR_RGB2GRAY).astype(np.float32)
    # Normaliza apenas dentro da máscara válida
    u_gray = np.where(mask, u_gray - u_gray[mask].mean(), 0.0)
    s_gray = np.where(mask, s_gray - s_gray[mask].mean(), 0.0)
    # Borda suave para reduzir vazamento espectral
    u_edges = cv2.Sobel(u_gray, cv2.CV_32F, 1, 1, ksize=3) * mask.astype(np.float32)
    s_edges = cv2.Sobel(s_gray, cv2.CV_32F, 1, 1, ksize=3) * mask.astype(np.float32)
    (dx_px, dy_px), resp = cv2.phaseCorrelate(s_edges, u_edges)
    return float(dx_px * RES_M), float(-dy_px * RES_M), float(resp)


def main() -> None:
    w = int(round((X1 - X0) / RES_M))
    h = int(round((Y1 - Y0) / RES_M))
    transform = from_origin(X0, Y1, RES_M, RES_M)
    print(f"Target UTM grid: {w} x {h} px ({(X1 - X0):.0f} m x {(Y1 - Y0):.0f} m) @ {RES_M:.4f} m/px")

    mosaic = np.zeros((3, h, w), dtype=np.uint8)
    per_source = {}
    for name, src_path in UAV_SOURCES:
        print(f"Warping {name}...")
        arr = warp_to_grid(src_path, (h, w), transform)
        valid = arr.max(axis=0) > 3
        per_source[name] = {
            "valid_px": int(valid.sum()),
            "valid_km2": round(float(valid.sum()) * (RES_M ** 2) / 1e6, 3),
        }
        print(f"  -> {name}: {per_source[name]['valid_km2']:.3f} km^2 válidos")
        # Sobrescreve onde o novo mosaico tem pixel válido (26/05 4cm tem prioridade sobre 25/05 15cm na borda)
        mosaic[:, valid] = arr[:, valid]

    uav_path = OUT_DATA / "canoas_uav_0.61m.tif"
    save_geotiff(uav_path, mosaic, transform)

    print("Reading pre-event Maxar satellite (poa_pre_feb24) for the same bbox...")
    sat = warp_to_grid(SAT_PRE_URL, (h, w), transform)
    sat_path = OUT_DATA / "canoas_pre_feb24_0.61m.tif"
    save_geotiff(sat_path, sat, transform)

    uav_valid = mosaic.max(axis=0) > 3
    sat_valid = sat.max(axis=0) > 3
    both_valid = uav_valid & sat_valid
    area_both_km2 = float(both_valid.sum()) * (RES_M ** 2) / 1e6

    # Checar alinhamento em um recorte central do Setor Leste (26/05) e Setor Oeste (25/05)
    # Setor Leste: x in [480700, 482200], y in [6691500, 6692500]
    def slice_utm(x0, y0, x1, y1):
        c0, c1 = int((x0 - X0) / RES_M), int((x1 - X0) / RES_M)
        r0, r1 = int((Y1 - y1) / RES_M), int((Y1 - y0) / RES_M)
        return slice(r0, r1), slice(c0, c1)

    rs_e, cs_e = slice_utm(480700, 6691500, 482200, 6692500)
    dx_e, dy_e, resp_e = estimate_shift_m(mosaic[:, rs_e, cs_e], sat[:, rs_e, cs_e], both_valid[rs_e, cs_e])
    rs_w, cs_w = slice_utm(476800, 6691300, 477700, 6692600)
    dx_w, dy_w, resp_w = estimate_shift_m(mosaic[:, rs_w, cs_w], sat[:, rs_w, cs_w], both_valid[rs_w, cs_w])

    # Contagem de células 25 m válidas (>= 90% pixels válidos no footprint 100 m = 180 m)
    cell_px = int(round(25.0 / RES_M))
    nrows, ncols = h // cell_px, w // cell_px
    cell_valid = np.zeros((nrows, ncols), dtype=bool)
    for r in range(nrows):
        for c in range(ncols):
            patch = both_valid[r * cell_px : (r + 1) * cell_px, c * cell_px : (c + 1) * cell_px]
            cell_valid[r, c] = patch.mean() >= 0.90

    # Contar janelas válidas de 40x40 (1.0 km) e 48x48 (1.2 km) com >= 50% células válidas
    win_counts = {}
    for w_cells in (32, 40, 48):
        cnt = 0
        for r in range(0, nrows - w_cells + 1, 4):
            for c in range(0, ncols - w_cells + 1, 4):
                if cell_valid[r : r + w_cells, c : c + w_cells].mean() >= 0.50:
                    cnt += 1
        win_counts[f"{w_cells}x{w_cells} ({w_cells*25}m)"] = cnt

    summary = {
        "per_source": per_source,
        "total_overlap_km2": round(area_both_km2, 3),
        "valid_25m_cells": int(cell_valid.sum()),
        "grid_shape_25m": [nrows, ncols],
        "valid_windows_stride4": win_counts,
        "shift_leste_m": {"dx_m": round(dx_e, 2), "dy_m": round(dy_e, 2), "response": round(resp_e, 4)},
        "shift_oeste_m": {"dx_m": round(dx_w, 2), "dy_m": round(dy_w, 2), "response": round(resp_w, 4)},
    }
    (OUT_RES / "p2_g0_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== GATE P2-G0 SUMMARY ===")
    print(json.dumps(summary, indent=2))

    # Figura de inspeção visual: visão geral + zoom 500 m no Setor Leste (UAV pós vs Satélite pré)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=150)
    ds = 8
    axes[0, 0].imshow(np.transpose(mosaic[:, ::ds, ::ds], (1, 2, 0)))
    axes[0, 0].set_title(f"CBPMESP UAV Post-Flood (25-26/05/2024) — {area_both_km2:.2f} km²")
    axes[0, 1].imshow(np.transpose(sat[:, ::ds, ::ds], (1, 2, 0)))
    axes[0, 1].set_title("Maxar Pre-Flood Satellite (02/02/2024)")
    rz, cz = slice_utm(481000, 6691700, 481600, 6692300)
    axes[1, 0].imshow(np.transpose(mosaic[:, rz, cz], (1, 2, 0)))
    axes[1, 0].set_title("Zoom 600 m (Mathias Velho Leste) — UAV Real")
    axes[1, 1].imshow(np.transpose(sat[:, rz, cz], (1, 2, 0)))
    axes[1, 1].set_title("Zoom 600 m (Mathias Velho Leste) — Satélite Pré")
    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(OUT_RES / "p2_g0_uav_vs_sat.png")
    plt.close(fig)
    print(f"Saved inspection figure to {OUT_RES / 'p2_g0_uav_vs_sat.png'}")


if __name__ == "__main__":
    main()

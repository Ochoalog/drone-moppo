"""Generates clean, high-resolution visual assets for the executive PPTX presentation."""
from __future__ import annotations

import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import from_bounds

from .config import ROOT, DATA, load_config
from .grid import Grid


def generate_all_assets():
    out_dir = ROOT / "results_mo/pptx_assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[assets] Generating PPTX assets in {out_dir}...")

    # 1. Slide 2: Problem Visual (Lost Drone + Morphological Shock)
    print("[assets] Generating Slide 2 Problem Visual...")
    fig = plt.figure(figsize=(14, 6), dpi=150, facecolor="#181825")
    
    # Left: Lost drone in Taquari forest with diffuse belief
    ax_left = fig.add_axes([0.05, 0.12, 0.42, 0.78], facecolor="#1e1e2e")
    cfg = load_config("configs/default.yaml")
    d = cfg.get("data", {})
    aoi = d.get("aoi", {"lon0": -51.95, "lon1": -51.875, "lat0": -29.33, "lat1": -29.12})
    cell_m = float(d.get("cell_m", 50.0))
    grid = Grid.from_aoi(**aoi, cell=cell_m)

    r0, c0 = 381, 50
    xmin = grid.x0 + c0 * cell_m
    xmax = grid.x0 + (c0 + 64) * cell_m
    ymin = grid.y1 - (r0 + 64) * cell_m
    ymax = grid.y1 - r0 * cell_m

    with rasterio.open("data/raw/pre_oct23_thumb.tif") as ds:
        win = from_bounds(xmin, ymin, xmax, ymax, ds.transform)
        rgb_map = np.transpose(ds.read([1, 2, 3], window=win), (1, 2, 0))
    rgb_map = cv2.resize(rgb_map, (512, 512), interpolation=cv2.INTER_CUBIC)

    ax_left.imshow(rgb_map, extent=[0, 64, 64, 0])
    # Scatter 350 diffuse points across the search window representing uniform initial belief
    rng = np.random.default_rng(42)
    pts_c = rng.uniform(2, 62, size=350)
    pts_r = rng.uniform(2, 62, size=350)
    ax_left.scatter(pts_c, pts_r, s=16, color="#89dceb", alpha=0.6, label="Hipóteses da Crença (Nuvem Difusa)")

    # Real drone starting position in deep forest
    d_mo = np.load("results_mo/traj/test_S2_mo_ppo_fast_ep0.npz")
    start_pos = d_mo["traj"][0]
    ax_left.scatter(start_pos[1], start_pos[0], color="#f38ba8", s=180, marker="o", edgecolors="#ffffff", lw=2, zorder=6, label="Drone Real (Perdido)")
    ax_left.text(start_pos[1] + 3, start_pos[0] - 1, "Drone Perdido\n(Posição Real)", color="#f38ba8", fontsize=10, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#181825", edgecolor="#f38ba8", alpha=0.9))

    ax_left.set_title("Cenário de GPS Negado: Janela de Busca 3.2 x 3.2 km\nIncerteza Espacial Total sobre Mata Homogênea", color="#cdd6f4", fontsize=11, fontweight="bold")
    ax_left.set_xlim(0, 64)
    ax_left.set_ylim(64, 0)
    ax_left.tick_params(colors="#a6adc8", labelsize=8)
    for sp in ax_left.spines.values():
        sp.set_color("#45475a")
    ax_left.legend(loc="lower left", facecolor="#1e1e2e", edgecolor="#45475a", labelcolor="#cdd6f4", fontsize=8.5)

    # Right: Camera vs Map in POA Ilhas (Morphological shock)
    cur_x = grid.x0 + (50 + start_pos[1] + 0.5) * cell_m
    cur_y = grid.y1 - (381 + start_pos[0] + 0.5) * cell_m

    # Let's use the actual POA Ilhas camera crop where the island vanished
    from .poa_data import poa_grid, region_slices
    cfg_poa = load_config("configs_poa/poa.yaml")
    g_poa = poa_grid(cfg_poa)
    sl_poa = region_slices(cfg_poa, g_poa)
    rs_i, cs_i = sl_poa["ilhas"]

    # Window 40, 45 on Ilhas (Episode 13)
    p_x = g_poa.x0 + (cs_i.start + 45 + 32 + 0.5) * g_poa.cell
    p_y = g_poa.y1 - (rs_i.start + 40 + 32 + 0.5) * g_poa.cell

    with rasterio.open("data_poa/raw/poa_post.tif") as ds_post, rasterio.open("data_poa/raw/poa_pre_feb24.tif") as ds_pre:
        w_p = from_bounds(p_x - 75, p_y - 75, p_x + 75, p_y + 75, ds_post.transform)
        w_m = from_bounds(p_x - 75, p_y - 75, p_x + 75, p_y + 75, ds_pre.transform)
        c_post = np.transpose(ds_post.read([1, 2, 3], window=w_p), (1, 2, 0))
        c_pre = np.transpose(ds_pre.read([1, 2, 3], window=w_m), (1, 2, 0))

    ax_cam1 = fig.add_axes([0.53, 0.22, 0.20, 0.58])
    ax_cam1.imshow(c_post)
    ax_cam1.set_title("O que o Drone Vê\n(Câmera Pós-Enchente: Mar de Lama)", color="#f38ba8", fontsize=10, fontweight="bold")
    ax_cam1.axis("off")

    ax_cam2 = fig.add_axes([0.76, 0.22, 0.20, 0.58])
    ax_cam2.imshow(c_pre)
    ax_cam2.set_title("O que Veria no Mapa\n(Original Pré-Enchente: Ilha e Mata)", color="#a6e3a1", fontsize=10, fontweight="bold")
    ax_cam2.axis("off")

    fig.text(0.74, 0.86, "Choque Morfológico Extremo (Delta do Jacuí)\nIlhas inteiras desapareceram debaixo da água!",
             ha="center", color="#fab387", fontsize=11.5, fontweight="bold")
    fig.text(0.74, 0.08, "Discrepância Visual: Varreduras cegas acumulam falsas localizações fatais",
             ha="center", color="#cdd6f4", fontsize=10, style="italic")

    fig.savefig(out_dir / "slide2_problem_visual.png", facecolor="#181825", bbox_inches="tight")
    plt.close(fig)

    # 2. Extract Active Comparison Video Poster Frames
    print("[assets] Extracting video poster frames...")
    v_taq = ROOT / "results_mo/videos/taquari_comparison_nuvem_pontos_camera.mp4"
    cap = cv2.VideoCapture(str(v_taq))
    for _ in range(16):
        ret, frame = cap.read()
    if ret:
        cv2.imwrite(str(out_dir / "poster_taquari_comparison.png"), frame)
    cap.release()

    v_poa = ROOT / "results_mo/videos/poa_ilhas_comparison_nuvem_pontos_camera.mp4"
    cap = cv2.VideoCapture(str(v_poa))
    for _ in range(9):
        ret, frame = cap.read()
    if ret:
        cv2.imwrite(str(out_dir / "poster_poa_comparison.png"), frame)
    cap.release()

    # 3. Slide 6: Clean Pareto Comparison Graphic
    print("[assets] Generating Pareto Frontier Graphic...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="#181825")
    for ax in axes:
        ax.set_facecolor("#1e1e2e")
        ax.tick_params(colors="#cdd6f4", labelsize=9)
        for sp in ax.spines.values():
            sp.set_color("#45475a")

    # Load pareto figures if present or plot directly
    p_suc_dist = ROOT / "results_mo/figs/pareto_success_distance.png"
    p_suc_fl = ROOT / "results_mo/figs/pareto_success_false_loc.png"
    if p_suc_dist.exists() and p_suc_fl.exists():
        im1 = cv2.cvtColor(cv2.imread(str(p_suc_dist)), cv2.COLOR_BGR2RGB)
        im2 = cv2.cvtColor(cv2.imread(str(p_suc_fl)), cv2.COLOR_BGR2RGB)
        axes[0].imshow(im1)
        axes[0].axis("off")
        axes[0].set_title("Fronteira de Pareto: Sucesso x Distância Voada", color="#89dceb", fontsize=11, fontweight="bold")
        axes[1].imshow(im2)
        axes[1].axis("off")
        axes[1].set_title("Fronteira de Pareto: Sucesso x Falsa Localização", color="#a6e3a1", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "slide6_pareto_combined.png", facecolor="#181825", bbox_inches="tight")
    plt.close(fig)

    print("[assets] All assets generated successfully!")


if __name__ == "__main__":
    generate_all_assets()

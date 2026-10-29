"""Episode video renderer and policy comparison generator for Active Drone Localization.

Renders high-resolution animated videos and GIFs showing:
1. Spatial belief convergence (posterior probability heatmap b_t)
2. Trajectory over distinctive map features (visual landmarks)
3. Synchronized comparison between MOPPO, Straight and Zigzag baselines
4. Dynamic Pareto frontier exploration
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .config import DATA, ROOT, load_config
from .env import ScenarioData
from .filter import entropy
from .grid import Grid
import rasterio
from rasterio.windows import from_bounds


def load_satellite_background(
    r0: int,
    c0: int,
    cfg: dict,
    date: str = "pre_oct23",
    window_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Crops the true-color satellite orthophoto for the given cell window (r0, c0).
    Returns (rgb_bg, gray_bg) resized to (512, 512, 3) for clean rendering."""
    d = cfg.get("data", {})
    aoi = d.get("aoi", {"lon0": -51.95, "lon1": -51.875, "lat0": -29.33, "lat1": -29.12})
    cell_m = float(d.get("cell_m", 50.0))
    grid = Grid.from_aoi(**aoi, cell=cell_m)

    xmin = grid.x0 + c0 * cell_m
    xmax = grid.x0 + (c0 + window_size) * cell_m
    ymin = grid.y1 - (r0 + window_size) * cell_m
    ymax = grid.y1 - r0 * cell_m

    tif_path = DATA / "raw" / f"{date}_thumb.tif"
    if not tif_path.exists():
        tif_path = DATA / "raw" / "post_thumb.tif"

    if tif_path.exists():
        with rasterio.open(tif_path) as ds:
            win = from_bounds(xmin, ymin, xmax, ymax, ds.transform)
            rgb = ds.read([1, 2, 3], window=win)
            rgb = np.transpose(rgb, (1, 2, 0))
    else:
        rgb = np.full((512, 512, 3), 40, dtype=np.uint8)

    rgb_bg = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(rgb_bg, cv2.COLOR_RGB2GRAY)
    gray_bg = np.stack([gray // 3] * 3, axis=-1)
    return rgb_bg, gray_bg


def compute_frame(
    t: int,
    dist_win: np.ndarray,
    traj: np.ndarray,
    map_est: np.ndarray,
    beliefs: np.ndarray,
    method_name: str,
    total_steps: int,
    success: bool,
    false_loc: bool,
    rgb_bg: np.ndarray | None = None,
    gray_bg: np.ndarray | None = None,
) -> np.ndarray:
    """Renders a single frame containing map, belief heatmap, and telemetry curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#181825")
    for ax in axes:
        ax.set_facecolor("#1e1e2e")
        for spine in ax.spines.values():
            spine.set_color("#45475a")
        ax.tick_params(colors="#cdd6f4", labelsize=9)

    cur_traj = traj[: t + 1]
    cur_est = map_est[: t + 1]
    cur_b = beliefs[t]
    max_mass = float(cur_b.max()) * 100.0

    # -------------------------------------------------------------
    # Panel 1: Satellite Map & Trajectory
    # -------------------------------------------------------------
    if rgb_bg is not None:
        axes[0].imshow(rgb_bg, extent=[0, 64, 64, 0], aspect="equal")
    else:
        axes[0].imshow(dist_win, cmap="cividis", origin="upper", vmin=0.0, vmax=max(0.6, dist_win.max()))

    # Landmark contours (high distinctiveness areas)
    Y, X = np.mgrid[0:64, 0:64]
    axes[0].contour(X, Y, dist_win, levels=[0.35], colors=["#f9e2af"], linewidths=[1.3], alpha=0.7)

    # Path
    if len(cur_traj) > 1:
        axes[0].plot(cur_traj[:, 1], cur_traj[:, 0], color="#a6e3a1", lw=2.5, alpha=0.9, label="Trajetória Real")
    # Start point
    axes[0].scatter(traj[0, 1], traj[0, 0], color="#89b4fa", s=60, marker="o", edgecolors="#181825", zorder=5, label="Início")
    # Current drone position
    axes[0].scatter(cur_traj[-1, 1], cur_traj[-1, 0], color="#a6e3a1", s=100, marker="o", edgecolors="#181825", zorder=5, label="Drone (Real)")
    # Filter estimated position
    axes[0].scatter(cur_est[-1, 1], cur_est[-1, 0], color="#f38ba8", s=100, marker="x", linewidths=2.5, zorder=5, label=r"Estimativa $\hat{x}_t$")

    axes[0].set_xlim(0, 64)
    axes[0].set_ylim(64, 0)
    axes[0].set_title(f"Mapa Satélite Real & Trajetória ({method_name})", color="#cdd6f4", fontsize=11, fontweight="bold")
    axes[0].legend(loc="upper left", facecolor="#1e1e2e", edgecolor="#45475a", labelcolor="#cdd6f4", fontsize=8)

    # -------------------------------------------------------------
    # Panel 2: Spatial Belief Distribution b_t
    # -------------------------------------------------------------
    if gray_bg is not None:
        axes[1].imshow(gray_bg, extent=[0, 64, 64, 0], aspect="equal")
        axes[1].imshow(cur_b, cmap="magma", extent=[0, 64, 64, 0], alpha=np.clip(cur_b / max(0.005, cur_b.max() * 0.7), 0.15, 0.95), origin="upper")
    else:
        axes[1].imshow(cur_b, cmap="magma", origin="upper", vmin=0.0, vmax=max(0.01, cur_b.max()))

    # Crosshair on true position
    axes[1].scatter(cur_traj[-1, 1], cur_traj[-1, 0], color="#a6e3a1", s=80, marker="+", linewidths=2, zorder=5, label="Posição Real")
    # Circle on mode
    mode_r, mode_c = cur_est[-1]
    circle = plt.Circle((mode_c, mode_r), 2.0, color="#f38ba8", fill=False, lw=1.5, linestyle="--", label="Raio Sucesso (100m)")
    axes[1].add_patch(circle)
    axes[1].set_xlim(0, 64)
    axes[1].set_ylim(64, 0)

    status_txt = "Buscando referências..."
    if t == total_steps - 1:
        if success:
            status_txt = "LOCALIZADO COM SUCESSO!"
        elif false_loc:
            status_txt = "LOCALIZAÇÃO FALSA!"
        else:
            status_txt = "LIMITE DE PASSOS ATINGIDO"
    elif max_mass > 5.0:
        status_txt = "Convergindo crença..."

    axes[1].set_title(
        f"Crença Espacial $b_t$ sobre Satélite\nPasso {t+1}/{total_steps} | Pico: {max_mass:.1f}%\nStatus: {status_txt}",
        color="#cdd6f4",
        fontsize=10,
        fontweight="bold",
    )
    axes[1].legend(loc="upper left", facecolor="#1e1e2e", edgecolor="#45475a", labelcolor="#cdd6f4", fontsize=8)

    # -------------------------------------------------------------
    # Panel 3: Convergence Telemetry (Entropy & Error)
    # -------------------------------------------------------------
    time_steps = np.arange(1, t + 2)
    entropies = [entropy(beliefs[i]) for i in range(t + 1)]
    errors_m = [np.hypot(traj[i, 0] - map_est[i, 0], traj[i, 1] - map_est[i, 1]) * 50.0 for i in range(t + 1)]

    ax3_top = axes[2]
    ax3_top.plot(time_steps, entropies, color="#f9e2af", lw=2, marker="o", markersize=3, label=r"Entropia $H_t$ (nats)")
    ax3_top.set_ylabel("Entropia (nats)", color="#f9e2af", fontsize=10)
    ax3_top.set_ylim(-0.2, 8.5)
    ax3_top.grid(True, linestyle="--", alpha=0.2, color="#45475a")

    ax3_bot = ax3_top.twinx()
    ax3_bot.plot(time_steps, errors_m, color="#89dceb", lw=2, linestyle="--", marker="s", markersize=3, label="Erro (m)")
    ax3_bot.set_ylabel("Erro de Posição (m)", color="#89dceb", fontsize=10)
    ax3_bot.tick_params(colors="#89dceb", labelsize=9)
    ax3_bot.set_ylim(-20, max(800.0, max(errors_m) + 50.0))

    # Combined legends
    lines1, labels1 = ax3_top.get_legend_handles_labels()
    lines2, labels2 = ax3_bot.get_legend_handles_labels()
    ax3_top.legend(lines1 + lines2, labels1 + labels2, loc="upper right", facecolor="#1e1e2e", edgecolor="#45475a", labelcolor="#cdd6f4", fontsize=8)

    dist_cum = float(np.sum([np.hypot(traj[i, 0] - traj[i-1, 0], traj[i, 1] - traj[i-1, 1]) * 50.0 for i in range(1, t + 1)])) if t > 0 else 0.0
    axes[2].set_title(f"Telemetria de Convergência\nDistância voada: {dist_cum:.1f} m", color="#cdd6f4", fontsize=10, fontweight="bold")
    axes[2].set_xlabel("Passos de Voo", color="#cdd6f4", fontsize=10)
    axes[2].set_xlim(0.8, max(total_steps, 15) + 0.2)

    plt.tight_layout()

    # Render figure to canvas and convert to numpy RGB
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
    plt.close(fig)
    return rgb


def render_episode_video(
    traj_path: Path,
    data: ScenarioData,
    cfg: dict,
    out_mp4: Path,
    out_gif: Path | None = None,
    method_name: str = "MOPPO",
    fps: int = 4,
) -> tuple[Path, Path | None]:
    """Renders a single episode trajectory file to MP4 and animated GIF."""
    npz = np.load(traj_path)
    r0, c0 = npz["window"]
    dist_win = data.distinct[r0 : r0 + 64, c0 : c0 + 64]
    traj = npz["traj"]
    map_est = npz["map_est"]
    beliefs = npz["beliefs"]
    success = bool(npz["success"])
    false_loc = bool(npz["false_loc"])
    T = len(traj)

    rgb_bg, gray_bg = load_satellite_background(r0, c0, cfg, date="pre_oct23")

    frames_rgb = []
    for t in range(T):
        frame = compute_frame(
            t, dist_win, traj, map_est, beliefs, method_name, T, success, false_loc,
            rgb_bg=rgb_bg, gray_bg=gray_bg,
        )
        frames_rgb.append(frame)

    # Pause on the final frame for 1.5 seconds (6 frames at 4 FPS)
    for _ in range(int(fps * 1.5)):
        frames_rgb.append(frames_rgb[-1])

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    h, w, _ = frames_rgb[0].shape

    # Write MP4 using OpenCV
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_mp4), fourcc, fps, (w, h))
    for f in frames_rgb:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()

    # Write GIF using PIL
    if out_gif:
        pil_frames = [Image.fromarray(f).resize((w // 2, h // 2), Image.Resampling.BILINEAR) for f in frames_rgb]
        duration_ms = int(1000 / fps)
        pil_frames[0].save(
            str(out_gif),
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
        )

    return out_mp4, out_gif


def render_comparison_video(
    npz_mo: Path,
    npz_str: Path,
    npz_zig: Path,
    data: ScenarioData,
    cfg: dict,
    out_mp4: Path,
    out_gif: Path | None = None,
    fps: int = 4,
) -> tuple[Path, Path | None]:
    """Renders a synchronized side-by-side comparison of MOPPO vs Straight vs Zigzag."""
    d_mo = np.load(npz_mo)
    d_str = np.load(npz_str)
    d_zig = np.load(npz_zig)

    r0, c0 = d_mo["window"]
    dist_win = data.distinct[r0 : r0 + 64, c0 : c0 + 64]
    rgb_bg, gray_bg = load_satellite_background(r0, c0, cfg, date="pre_oct23")

    t_mo, t_str, t_zig = len(d_mo["traj"]), len(d_str["traj"]), len(d_zig["traj"])
    max_t = max(t_mo, t_str, t_zig)

    Y, X = np.mgrid[0:64, 0:64]
    frames_rgb = []

    for t in range(max_t):
        idx_mo = min(t, t_mo - 1)
        idx_str = min(t, t_str - 1)
        idx_zig = min(t, t_zig - 1)

        fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor="#181825")
        for row in axes:
            for ax in row:
                ax.set_facecolor("#1e1e2e")
                for spine in ax.spines.values():
                    spine.set_color("#45475a")
                ax.tick_params(colors="#cdd6f4", labelsize=8)

        methods_data = [
            ("MOPPO (com memória)", d_mo, idx_mo, t_mo, "#a6e3a1", 0),
            ("Linha Reta (Straight)", d_str, idx_str, t_str, "#cba6f7", 1),
            ("Zigue-Zague (Lawnmower)", d_zig, idx_zig, t_zig, "#fab387", 2),
        ]

        for name, d, cur_idx, total_T, col, c_idx in methods_data:
            tr = d["traj"][: cur_idx + 1]
            est = d["map_est"][: cur_idx + 1]
            b = d["beliefs"][cur_idx]
            succ = bool(d["success"])
            floc = bool(d["false_loc"])
            dist = float(np.sum([np.hypot(tr[i, 0] - tr[i-1, 0], tr[i, 1] - tr[i-1, 1]) * 50.0 for i in range(1, cur_idx + 1)])) if cur_idx > 0 else 0.0

            # Row 0: Real Satellite Map & Trajectory
            axes[0, c_idx].imshow(rgb_bg, extent=[0, 64, 64, 0], aspect="equal")
            axes[0, c_idx].contour(X, Y, dist_win, levels=[0.35], colors=["#f9e2af"], linewidths=[1.2], alpha=0.6)
            if len(tr) > 1:
                axes[0, c_idx].plot(tr[:, 1], tr[:, 0], color=col, lw=2.5, alpha=0.9)
            axes[0, c_idx].scatter(tr[0, 1], tr[0, 0], color="#89b4fa", s=50, marker="o", zorder=5)
            axes[0, c_idx].scatter(tr[-1, 1], tr[-1, 0], color=col, s=80, marker="o", edgecolors="#181825", zorder=5)
            axes[0, c_idx].scatter(est[-1, 1], est[-1, 0], color="#f38ba8", s=80, marker="x", linewidths=2.5, zorder=5)
            axes[0, c_idx].set_xlim(0, 64)
            axes[0, c_idx].set_ylim(64, 0)

            st_str = "Em voo..."
            if cur_idx == total_T - 1:
                st_str = "SUCESSO!" if succ else ("FALSA LOC!" if floc else "LIMITE")

            axes[0, c_idx].set_title(
                f"{name}\nPasso {cur_idx+1}/{total_T} | Dist: {dist:.0f}m | {st_str}",
                color=col,
                fontsize=10,
                fontweight="bold",
            )

            # Row 1: Spatial Belief Heatmap over dimmed satellite
            axes[1, c_idx].imshow(gray_bg, extent=[0, 64, 64, 0], aspect="equal")
            axes[1, c_idx].imshow(b, cmap="magma", extent=[0, 64, 64, 0], alpha=np.clip(b / max(0.005, b.max() * 0.7), 0.15, 0.95), origin="upper")
            axes[1, c_idx].scatter(tr[-1, 1], tr[-1, 0], color=col, s=60, marker="+", linewidths=2, zorder=5)
            mode_r, mode_c = est[-1]
            circle = plt.Circle((mode_c, mode_r), 2.0, color="#f38ba8", fill=False, lw=1.5, linestyle="--")
            axes[1, c_idx].add_patch(circle)
            axes[1, c_idx].set_xlim(0, 64)
            axes[1, c_idx].set_ylim(64, 0)

            H = entropy(b)
            axes[1, c_idx].set_title(
                f"Crença Espacial $b_t$\nEntropia: {H:.2f} nats | Pico: {b.max()*100:.1f}%",
                color="#cdd6f4",
                fontsize=9,
            )

        plt.suptitle("Comparação Sincronizada com Imagem Satélite Real: MOPPO vs Linha Reta vs Zigue-Zague", color="#cdd6f4", fontsize=13, fontweight="bold")
        plt.tight_layout()

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
        plt.close(fig)
        frames_rgb.append(rgb)

    # Pause on end
    for _ in range(int(fps * 2.0)):
        frames_rgb.append(frames_rgb[-1])

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    h, w, _ = frames_rgb[0].shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_mp4), fourcc, fps, (w, h))
    for f in frames_rgb:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()

    if out_gif:
        pil_frames = [Image.fromarray(f).resize((w // 2, h // 2), Image.Resampling.BILINEAR) for f in frames_rgb]
        duration_ms = int(1000 / fps)
        pil_frames[0].save(
            str(out_gif),
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
        )

    return out_mp4, out_gif


def render_pareto_animation(
    summary_csv: Path,
    out_mp4: Path,
    out_gif: Path | None = None,
    fps: int = 4,
) -> tuple[Path, Path | None]:
    """Generates an animated video explaining how Pareto tradeoffs operate across weights."""
    import pandas as pd
    df = pd.read_csv(summary_csv)
    df_mo = df[df["method"] == "mo_ppo"].sort_values("w_cost")
    df_nomem = df[df["method"] == "mo_ppo_nomem"]
    df_fix = df[df["method"].str.startswith("ppo_fixed")]
    df_str = df[df["method"] == "straight"].iloc[0]

    frames_rgb = []
    weights_sequence = list(df_mo.itertuples())

    for idx, row in enumerate(weights_sequence):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#181825")
        for ax in axes:
            ax.set_facecolor("#1e1e2e")
            for spine in ax.spines.values():
                spine.set_color("#45475a")
            ax.tick_params(colors="#cdd6f4", labelsize=9)
            ax.grid(True, linestyle="--", alpha=0.25, color="#45475a")

        # Plot 1: Success vs Distance
        # Baselines
        axes[0].scatter(df_str["distance_m"] / 1000.0, df_str["success"] * 100, color="#cba6f7", s=140, marker="D", label="Linha Reta")
        axes[0].scatter(df_nomem["distance_m"] / 1000.0, df_nomem["success"] * 100, color="#fab387", s=40, marker="^", alpha=0.6, label="MO sem memória (Dominada)")
        axes[0].scatter(df_fix["distance_m"] / 1000.0, df_fix["success"] * 100, color="#f38ba8", s=100, marker="s", label="PPO Fixo")

        # Full MO points
        axes[0].scatter(df_mo["distance_m"] / 1000.0, df_mo["success"] * 100, color="#585b70", s=60, alpha=0.5)
        # Traversed path on frontier
        axes[0].plot(df_mo["distance_m"][: idx + 1] / 1000.0, df_mo["success"][: idx + 1] * 100, color="#a6e3a1", lw=2, linestyle="--")
        # Active point
        axes[0].scatter(row.distance_m / 1000.0, row.success * 100, color="#a6e3a1", s=180, edgecolors="#ffffff", lw=2, label="Ponto Ativo $w$")

        axes[0].set_title("Fronteira: Sucesso $\\uparrow$ vs Distância $\\downarrow$", color="#cdd6f4", fontsize=11, fontweight="bold")
        axes[0].set_xlabel("Distância Média (km) $\\downarrow$", color="#cdd6f4")
        axes[0].set_ylabel("Taxa de Sucesso (%) $\\uparrow$", color="#cdd6f4")
        axes[0].legend(loc="lower right", facecolor="#1e1e2e", edgecolor="#45475a", labelcolor="#cdd6f4", fontsize=8)

        # Plot 2: Preference Vector w Breakdown
        w_vals = [row.w_info, row.w_cost, row.w_success, row.w_risk]
        w_labels = ["Info\n($r_0$)", "Custo\n($r_1$)", "Sucesso\n($r_2$)", "Risco\n($r_3$)"]
        colors = ["#89b4fa", "#fab387", "#a6e3a1", "#f38ba8"]

        bars = axes[1].bar(w_labels, w_vals, color=colors, edgecolor="#ffffff", alpha=0.85, width=0.55)
        axes[1].set_ylim(0.0, 1.05)
        axes[1].set_title(f"Vetor de Pesos Condicionado: $w = [{row.w_info:.2f}, {row.w_cost:.2f}, {row.w_success:.2f}, {row.w_risk:.2f}]$\nSucesso: {row.success*100:.1f}% | Distância: {row.distance_m:.0f}m", color="#cdd6f4", fontsize=11, fontweight="bold")
        axes[1].set_ylabel("Peso Normalizado", color="#cdd6f4")

        for bar in bars:
            yval = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width() / 2.0, yval + 0.02, f"{yval:.2f}", ha="center", va="bottom", color="#cdd6f4", fontsize=9, fontweight="bold")

        plt.suptitle("Como Funciona a Fronteira de Pareto: Navegação Post-Hoc pelo Espaço de Pesos", color="#cdd6f4", fontsize=13, fontweight="bold")
        plt.tight_layout()

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
        plt.close(fig)
        frames_rgb.append(rgb)

    # Pause on end
    for _ in range(int(fps * 2.0)):
        frames_rgb.append(frames_rgb[-1])

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    h, w, _ = frames_rgb[0].shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_mp4), fourcc, fps, (w, h))
    for f in frames_rgb:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()

    if out_gif:
        pil_frames = [Image.fromarray(f).resize((w // 2, h // 2), Image.Resampling.BILINEAR) for f in frames_rgb]
        duration_ms = int(1000 / fps)
        pil_frames[0].save(
            str(out_gif),
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
        )

    return out_mp4, out_gif


def main():
    print("=" * 60)
    print("Generating Active Localization Videos and Animations")
    print("=" * 60)

    cfg = load_config("configs_mo/default.yaml")
    data = ScenarioData.load("hand", "S2", cfg)

    traj_dir = ROOT / "results_mo/traj"
    video_dir = ROOT / "results_mo/videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    # 1. Single episode video: MOPPO fast ep 0
    ep0_fast = traj_dir / "test_S2_mo_ppo_fast_ep0.npz"
    if ep0_fast.exists():
        print("[video] Rendering Episode 0 (MOPPO Fast)...")
        render_episode_video(
            ep0_fast, data, cfg,
            out_mp4=video_dir / "episode_0_mo_fast.mp4",
            out_gif=video_dir / "episode_0_mo_fast.gif",
            method_name="MOPPO (Rápido)",
            fps=4,
        )
        print("  -> Saved episode_0_mo_fast.mp4 and .gif")

    # 2. Single episode video: MOPPO safe ep 2
    ep2_safe = traj_dir / "test_S2_mo_ppo_safe_ep2.npz"
    if ep2_safe.exists():
        print("[video] Rendering Episode 2 (MOPPO Safe)...")
        render_episode_video(
            ep2_safe, data, cfg,
            out_mp4=video_dir / "episode_2_mo_safe.mp4",
            out_gif=video_dir / "episode_2_mo_safe.gif",
            method_name="MOPPO (Seguro)",
            fps=4,
        )
        print("  -> Saved episode_2_mo_safe.mp4 and .gif")

    # 3. Synchronized comparison: Ep 0 (MOPPO vs Straight vs Zigzag)
    ep0_str = traj_dir / "test_S2_straight_ep0.npz"
    ep0_zig = traj_dir / "test_S2_zigzag_ep0.npz"
    if ep0_fast.exists() and ep0_str.exists() and ep0_zig.exists():
        print("[video] Rendering Synchronized Comparison (Episode 0: MOPPO vs Straight vs Zigzag)...")
        render_comparison_video(
            ep0_fast, ep0_str, ep0_zig, data, cfg,
            out_mp4=video_dir / "comparison_ep0_mo_vs_straight_vs_zigzag.mp4",
            out_gif=video_dir / "comparison_ep0_mo_vs_straight_vs_zigzag.gif",
            fps=4,
        )
        print("  -> Saved comparison_ep0_mo_vs_straight_vs_zigzag.mp4 and .gif")

    # 4. Synchronized comparison: Ep 2 (MOPPO vs Straight vs Zigzag)
    ep2_str = traj_dir / "test_S2_straight_ep2.npz"
    ep2_zig = traj_dir / "test_S2_zigzag_ep2.npz"
    if ep2_safe.exists() and ep2_str.exists() and ep2_zig.exists():
        print("[video] Rendering Synchronized Comparison (Episode 2: MOPPO vs Straight vs Zigzag)...")
        render_comparison_video(
            ep2_safe, ep2_str, ep2_zig, data, cfg,
            out_mp4=video_dir / "comparison_ep2_mo_vs_straight_vs_zigzag.mp4",
            out_gif=video_dir / "comparison_ep2_mo_vs_straight_vs_zigzag.gif",
            fps=4,
        )
        print("  -> Saved comparison_ep2_mo_vs_straight_vs_zigzag.mp4 and .gif")

    # 5. Pareto Frontier Explainer Animation
    summary_csv = ROOT / "results_mo/summary_test.csv"
    if summary_csv.exists():
        print("[video] Rendering Pareto Frontier Explainer Animation...")
        render_pareto_animation(
            summary_csv,
            out_mp4=video_dir / "pareto_frontier_explainer.mp4",
            out_gif=video_dir / "pareto_frontier_explainer.gif",
            fps=3,
        )
        print("  -> Saved pareto_frontier_explainer.mp4 and .gif")

    print("\n[video] All videos and animations generated successfully!")


if __name__ == "__main__":
    main()

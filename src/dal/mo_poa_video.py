"""Video generator for Porto Alegre zero-shot evaluation on real satellite maps.

Generates high-resolution animations for:
1. Porto Alegre Centro: MOPPO Safe over Parque da Redenção / Cais Mauá / Guaíba
2. Porto Alegre Ilhas: MOPPO Safe over Jacuí Delta river channels and BR-290 bridge
3. Synchronized Comparisons (MOPPO vs Straight vs Zigzag) for both regions.
"""
from __future__ import annotations

from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import rasterio
from rasterio.windows import from_bounds

from .config import ROOT, load_config
from .poa_data import poa_grid, region_slices
from .poa import beta_for, region_data
from .filter import entropy
from .mo_video import compute_frame


def load_poa_satellite_background(
    region: str,
    r0: int,
    c0: int,
    cfg_poa: dict,
    date: str = "poa_pre_feb24",
    window_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Crops the true-color satellite orthophoto of Porto Alegre for the given cell window (r0, c0)."""
    grid = poa_grid(cfg_poa)
    slices = region_slices(cfg_poa, grid)
    rs, cs = slices[region]
    r_glob = rs.start + r0
    c_glob = cs.start + c0
    cell = grid.cell

    xmin = grid.x0 + c_glob * cell
    xmax = grid.x0 + (c_glob + window_size) * cell
    ymin = grid.y1 - (r_glob + window_size) * cell
    ymax = grid.y1 - r_glob * cell

    tif_path = ROOT / "data_poa/raw" / f"{date}_thumb.tif"
    if not tif_path.exists():
        tif_path = ROOT / "data_poa/raw/poa_post_thumb.tif"

    with rasterio.open(tif_path) as ds:
        win = from_bounds(xmin, ymin, xmax, ymax, ds.transform)
        rgb = ds.read([1, 2, 3], window=win)
        rgb = np.transpose(rgb, (1, 2, 0))

    rgb_bg = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(rgb_bg, cv2.COLOR_RGB2GRAY)
    gray_bg = np.stack([gray // 3] * 3, axis=-1)
    return rgb_bg, gray_bg


def render_poa_episode_video(
    npz_path: Path,
    distinct_map: np.ndarray,
    region: str,
    cfg_poa: dict,
    out_mp4: Path,
    out_gif: Path | None = None,
    method_name: str = "MOPPO POA",
    fps: int = 4,
) -> tuple[Path, Path | None]:
    """Renders a single episode video on Porto Alegre terrain."""
    npz = np.load(npz_path)
    r0, c0 = npz["window"]
    dist_win = distinct_map[r0 : r0 + 64, c0 : c0 + 64]
    traj = npz["traj"]
    map_est = npz["map_est"]
    beliefs = npz["beliefs"]
    success = bool(npz["success"])
    false_loc = bool(npz["false_loc"])
    T = len(traj)

    rgb_bg, gray_bg = load_poa_satellite_background(region, r0, c0, cfg_poa)

    frames_rgb = []
    for t in range(T):
        frame = compute_frame(
            t, dist_win, traj, map_est, beliefs, method_name, T, success, false_loc,
            rgb_bg=rgb_bg, gray_bg=gray_bg,
        )
        frames_rgb.append(frame)

    for _ in range(int(fps * 1.5)):
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


def render_poa_comparison_video(
    npz_mo: Path,
    npz_str: Path,
    npz_zig: Path,
    distinct_map: np.ndarray,
    region: str,
    cfg_poa: dict,
    out_mp4: Path,
    out_gif: Path | None = None,
    region_title: str = "Centro",
    fps: int = 4,
) -> tuple[Path, Path | None]:
    """Renders synchronized side-by-side comparison for Porto Alegre."""
    d_mo = np.load(npz_mo)
    d_str = np.load(npz_str)
    d_zig = np.load(npz_zig)

    r0, c0 = d_mo["window"]
    dist_win = distinct_map[r0 : r0 + 64, c0 : c0 + 64]
    rgb_bg, gray_bg = load_poa_satellite_background(region, r0, c0, cfg_poa)

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
            ("MOPPO (Zero-Shot)", d_mo, idx_mo, t_mo, "#a6e3a1", 0),
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

        plt.suptitle(f"Porto Alegre ({region_title}): Comparação Sincronizada com Satélite Real", color="#cdd6f4", fontsize=13, fontweight="bold")
        plt.tight_layout()

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
        plt.close(fig)
        frames_rgb.append(rgb)

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
    print("Generating Porto Alegre Zero-Shot Videos and Comparisons")
    print("=" * 60)

    cfg_poa = load_config("configs_poa/poa.yaml")
    traj_dir = ROOT / "results_mo/traj_poa"
    video_dir = ROOT / "results_mo/videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    # 1. Porto Alegre Centro: MOPPO Safe Ep 1
    beta_c = beta_for(cfg_poa, "centro", "P1", "zeroshot")
    data_c = region_data(cfg_poa, "centro", "P1", beta_c, pca_from="S2")
    ep1_c_mo = traj_dir / "poa_centro_mo_safe_ep1.npz"
    ep1_c_str = traj_dir / "poa_centro_straight_ep1.npz"
    ep1_c_zig = traj_dir / "poa_centro_zigzag_ep1.npz"

    if ep1_c_mo.exists():
        print("[poa-video] Rendering Porto Alegre Centro (MOPPO Safe Ep 1)...")
        render_poa_episode_video(
            ep1_c_mo, data_c.distinct, "centro", cfg_poa,
            out_mp4=video_dir / "poa_centro_mo_safe.mp4",
            out_gif=video_dir / "poa_centro_mo_safe.gif",
            method_name="POA Centro (MOPPO)",
            fps=4,
        )
        print("  -> Saved poa_centro_mo_safe.mp4 and .gif")

    if ep1_c_mo.exists() and ep1_c_str.exists() and ep1_c_zig.exists():
        print("[poa-video] Rendering Porto Alegre Centro Comparison...")
        render_poa_comparison_video(
            ep1_c_mo, ep1_c_str, ep1_c_zig, data_c.distinct, "centro", cfg_poa,
            out_mp4=video_dir / "poa_comparison_centro_mo_vs_straight_vs_zigzag.mp4",
            out_gif=video_dir / "poa_comparison_centro_mo_vs_straight_vs_zigzag.gif",
            region_title="Centro Histórico / Redenção",
            fps=4,
        )
        print("  -> Saved poa_comparison_centro_mo_vs_straight_vs_zigzag.mp4 and .gif")

    # 2. Porto Alegre Ilhas: MOPPO Safe Ep 4
    beta_i = beta_for(cfg_poa, "ilhas", "P1", "zeroshot")
    data_i = region_data(cfg_poa, "ilhas", "P1", beta_i, pca_from="S2")
    ep4_i_mo = traj_dir / "poa_ilhas_mo_safe_ep4.npz"
    ep4_i_str = traj_dir / "poa_ilhas_straight_ep4.npz"
    ep4_i_zig = traj_dir / "poa_ilhas_zigzag_ep4.npz"

    if ep4_i_mo.exists():
        print("[poa-video] Rendering Porto Alegre Ilhas (MOPPO Safe Ep 4)...")
        render_poa_episode_video(
            ep4_i_mo, data_i.distinct, "ilhas", cfg_poa,
            out_mp4=video_dir / "poa_ilhas_mo_safe.mp4",
            out_gif=video_dir / "poa_ilhas_mo_safe.gif",
            method_name="POA Ilhas (MOPPO)",
            fps=4,
        )
        print("  -> Saved poa_ilhas_mo_safe.mp4 and .gif")

    if ep4_i_mo.exists() and ep4_i_str.exists() and ep4_i_zig.exists():
        print("[poa-video] Rendering Porto Alegre Ilhas Comparison...")
        render_poa_comparison_video(
            ep4_i_mo, ep4_i_str, ep4_i_zig, data_i.distinct, "ilhas", cfg_poa,
            out_mp4=video_dir / "poa_comparison_ilhas_mo_vs_straight_vs_zigzag.mp4",
            out_gif=video_dir / "poa_comparison_ilhas_mo_vs_straight_vs_zigzag.gif",
            region_title="Delta do Jacuí / Ilhas",
            fps=4,
        )
        print("  -> Saved poa_comparison_ilhas_mo_vs_straight_vs_zigzag.mp4 and .gif")

    print("\n[poa-video] All Porto Alegre videos generated successfully!")


if __name__ == "__main__":
    main()

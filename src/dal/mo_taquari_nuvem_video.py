"""Renders high-resolution videos for Vale do Taquari focusing on:
1. Training environment (Ambiente onde o MOPPO foi treinado).
2. Point cloud of hypotheses (Nuvem de pontos / partículas da crença b_t).
3. Synchronized camera view: "O que o drone vê" (Pós-Enchente) vs "O que veria no mapa original" (Pré-Enchente).
4. Synchronized comparison of:
   - Linha Reta (Straight)
   - Primeiro Treino (PPO PoC S2)
   - Treino MO (MOPPO com Memória)
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import rasterio
from rasterio.windows import from_bounds

from .config import ROOT, load_config
from .grid import Grid
from .filter import entropy


def generate_taquari_nuvem_comparison_video(
    episode_idx: int = 0,
    out_mp4: Path = ROOT / "results_mo/videos/taquari_comparison_nuvem_pontos_camera.mp4",
    out_gif: Path = ROOT / "results_mo/videos/taquari_comparison_nuvem_pontos_camera.gif",
    fps: int = 4,
):
    print("=" * 60)
    print(f"Generating Taquari Nuvem de Pontos & Câmera Video (Episode {episode_idx})")
    print("=" * 60)

    cfg = load_config("configs/default.yaml")
    d = cfg.get("data", {})
    aoi = d.get("aoi", {"lon0": -51.95, "lon1": -51.875, "lat0": -29.33, "lat1": -29.12})
    cell_m = float(d.get("cell_m", 50.0))
    grid = Grid.from_aoi(**aoi, cell=cell_m)

    # Load pre-saved trajectories with beliefs
    f_str = ROOT / f"results_mo/traj/test_S2_straight_ep{episode_idx}.npz"
    f_ppo = ROOT / f"results/traj/test_S2_ppo2_ep{episode_idx}.npz"
    f_mo  = ROOT / f"results_mo/traj/test_S2_mo_ppo_fast_ep{episode_idx}.npz"

    d_str = np.load(f_str)
    d_ppo = np.load(f_ppo)
    d_mo  = np.load(f_mo)

    runs = {
        "straight": {
            "label": "Linha Reta (Straight)",
            "traj": d_str["traj"],
            "beliefs": d_str["beliefs"],
            "map_est": d_str["map_est"],
            "success": bool(d_str["success"]),
            "false_loc": bool(d_str["false_loc"]),
            "window": d_str["window"],
        },
        "ppo_poc": {
            "label": "Primeiro Treino (PPO PoC S2)",
            "traj": d_ppo["traj"],
            "beliefs": d_ppo["beliefs"],
            "map_est": d_ppo["map_est"],
            "success": bool(d_ppo["success"]),
            "false_loc": bool(d_ppo["false_loc"]),
            "window": d_ppo["window"],
        },
        "moppo": {
            "label": "Treino MO (MOPPO com Memória)",
            "traj": d_mo["traj"],
            "beliefs": d_mo["beliefs"],
            "map_est": d_mo["map_est"],
            "success": bool(d_mo["success"]),
            "false_loc": bool(d_mo["false_loc"]),
            "window": d_mo["window"],
        },
    }

    r0, c0 = d_mo["window"]
    print(f"[video] Taquari Window: r0={r0}, c0={c0}")
    for k, v in runs.items():
        print(f"  {v['label']}: steps={len(v['traj'])}, succ={v['success']}, false_loc={v['false_loc']}")

    # Window bounds
    xmin = grid.x0 + c0 * cell_m
    xmax = grid.x0 + (c0 + 64) * cell_m
    ymin = grid.y1 - (r0 + 64) * cell_m
    ymax = grid.y1 - r0 * cell_m

    # Background satellite image
    with rasterio.open("data/raw/pre_oct23_thumb.tif") as ds:
        win = from_bounds(xmin, ymin, xmax, ymax, ds.transform)
        rgb_map = np.transpose(ds.read([1, 2, 3], window=win), (1, 2, 0))
    rgb_map = cv2.resize(rgb_map, (512, 512), interpolation=cv2.INTER_CUBIC)

    # Downward camera rasters
    ds_post = rasterio.open("data/raw/post.tif")
    ds_pre = rasterio.open("data/raw/pre_oct23.tif")

    max_t = max(len(runs["straight"]["traj"]), len(runs["ppo_poc"]["traj"]), len(runs["moppo"]["traj"]))
    print(f"[video] Rendering {max_t} synchronized comparison frames for Taquari...")

    cols_config = [
        ("straight", "#cba6f7", 0),
        ("ppo_poc", "#fab387", 1),
        ("moppo", "#a6e3a1", 2),
    ]

    frames_rgb = []
    rng_pts = np.random.default_rng(2026)

    for t in range(max_t):
        fig = plt.figure(figsize=(18, 10.5), facecolor="#181825")

        for key, col, c_idx in cols_config:
            rdata = runs[key]
            total_T = len(rdata["traj"])
            cur_idx = min(t, total_T - 1)
            tr = rdata["traj"][: cur_idx + 1]
            est = rdata["map_est"][: cur_idx + 1]
            b = rdata["beliefs"][cur_idx]

            # -------------------------------------------------------------
            # Top Panel: Satellite Map + Point Cloud + Trajectory
            # -------------------------------------------------------------
            ax_map = plt.subplot2grid((5, 6), (0, c_idx * 2), rowspan=3, colspan=2, facecolor="#1e1e2e")
            ax_map.imshow(rgb_map, extent=[0, 64, 64, 0], aspect="equal")

            # Sample 400 points from spatial belief distribution b_t
            flat_b = b.ravel()
            pts_idx = rng_pts.choice(len(flat_b), size=400, p=flat_b)
            pts_r = pts_idx // 64 + rng_pts.uniform(-0.4, 0.4, size=400)
            pts_c = pts_idx % 64 + rng_pts.uniform(-0.4, 0.4, size=400)
            ax_map.scatter(pts_c, pts_r, s=14, color="#89dceb", alpha=0.5, label="Nuvem de Pontos (Crença)")

            # Trajectory
            if len(tr) > 1:
                ax_map.plot(tr[:, 1], tr[:, 0], color=col, lw=2.5, alpha=0.9, label="Trajetória")
            ax_map.scatter(tr[0, 1], tr[0, 0], color="#89b4fa", s=50, marker="o", zorder=5, label="Início")
            ax_map.scatter(tr[-1, 1], tr[-1, 0], color=col, s=90, marker="o", edgecolors="#181825", zorder=6, label="Drone Real")
            ax_map.scatter(est[-1, 1], est[-1, 0], color="#f38ba8", s=90, marker="x", linewidths=2.5, zorder=6, label=r"Estimativa $\hat{x}_t$")

            # 100m success circle
            circle = plt.Circle((est[-1, 1], est[-1, 0]), 2.0, color="#f38ba8", fill=False, lw=1.5, linestyle="--")
            ax_map.add_patch(circle)
            ax_map.set_xlim(0, 64)
            ax_map.set_ylim(64, 0)
            for spine in ax_map.spines.values():
                spine.set_color("#45475a")
            ax_map.tick_params(colors="#cdd6f4", labelsize=8)

            dist_cum = float(np.sum([np.hypot(tr[i, 0] - tr[i-1, 0], tr[i, 1] - tr[i-1, 1]) * 50.0 for i in range(1, cur_idx + 1)])) if cur_idx > 0 else 0.0
            st = "Em voo..."
            if cur_idx == total_T - 1:
                st = "LOCALIZADO!" if rdata["success"] else ("FALSA LOC!" if rdata["false_loc"] else "LIMITE")
            H = entropy(b)
            pico = b.max() * 100.0

            ax_map.set_title(
                f"{rdata['label']}\nPasso {cur_idx+1}/{total_T} | Dist: {dist_cum:.0f}m | Entropia: {H:.2f} nats\nPico: {pico:.1f}% | {st}",
                color=col,
                fontsize=9.5,
                fontweight="bold",
            )
            if c_idx == 0:
                ax_map.legend(loc="lower left", facecolor="#1e1e2e", edgecolor="#45475a", labelcolor="#cdd6f4", fontsize=7)

            # -------------------------------------------------------------
            # Bottom Panel: Drone Camera (Post-flood) vs Original Map (Pre-flood)
            # -------------------------------------------------------------
            cur_pos = tr[-1]
            cur_rg = r0 + cur_pos[0]
            cur_cg = c0 + cur_pos[1]
            cur_x = grid.x0 + (cur_cg + 0.5) * cell_m
            cur_y = grid.y1 - (cur_rg + 0.5) * cell_m

            win_p = from_bounds(cur_x - 75, cur_y - 75, cur_x + 75, cur_y + 75, ds_post.transform)
            win_m = from_bounds(cur_x - 75, cur_y - 75, cur_x + 75, cur_y + 75, ds_pre.transform)
            c_post = np.transpose(ds_post.read([1, 2, 3], window=win_p), (1, 2, 0))
            c_pre = np.transpose(ds_pre.read([1, 2, 3], window=win_m), (1, 2, 0))

            ax_cam_post = plt.subplot2grid((5, 6), (3, c_idx * 2), rowspan=2, facecolor="#1e1e2e")
            ax_cam_post.imshow(c_post)
            ax_cam_post.set_title("O que o Drone Vê\n(Câmera Pós-Enchente)", color="#f38ba8", fontsize=8.5, fontweight="bold")
            ax_cam_post.axis("off")

            ax_cam_pre = plt.subplot2grid((5, 6), (3, c_idx * 2 + 1), rowspan=2, facecolor="#1e1e2e")
            ax_cam_pre.imshow(c_pre)
            ax_cam_pre.set_title("O que Veria no Mapa\n(Original Pré-Enchente)", color="#a6e3a1", fontsize=8.5, fontweight="bold")
            ax_cam_pre.axis("off")

        plt.suptitle(
            "Vale do Taquari / RS (Ambiente de Treino - Cenário S2 Choque Morfológico)\n"
            "Comparação Sincronizada: Reta vs Primeiro Treino (PPO PoC) vs Treino MO (MOPPO)\n"
            "Com Nuvem de Pontos da Crença e Câmera do Drone vs Mapa Original (Rio Taquari)",
            color="#cdd6f4",
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout()

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
        plt.close(fig)
        frames_rgb.append(rgb)

    ds_post.close()
    ds_pre.close()

    # Pause on end
    for _ in range(int(fps * 2.5)):
        frames_rgb.append(frames_rgb[-1])

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    h, w, _ = frames_rgb[0].shape

    temp_raw_mp4 = out_mp4.with_name("temp_taquari_nuvem_raw.mp4")
    print(f"[video] Encoding {len(frames_rgb)} frames to raw MP4...")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(temp_raw_mp4), fourcc, fps, (w, h))
    for f in frames_rgb:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()

    # Convert to WhatsApp-ready H.264
    print(f"[video] Converting with FFmpeg to H.264 / faststart: {out_mp4}...")
    cmd = [
        "ffmpeg", "-y", "-i", str(temp_raw_mp4),
        "-c:v", "libx264", "-profile:v", "main", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(out_mp4)
    ]
    subprocess.run(cmd, check=True)
    if temp_raw_mp4.exists():
        temp_raw_mp4.unlink()

    # Copy alias for whatsapp
    whatsapp_mp4 = out_mp4.parent / "taquari_comparison_whatsapp.mp4"
    import shutil
    shutil.copy(out_mp4, whatsapp_mp4)

    if out_gif:
        print(f"[video] Encoding GIF: {out_gif}...")
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

    print(f"[video] Saved {out_mp4} and {out_gif} successfully!")


if __name__ == "__main__":
    generate_taquari_nuvem_comparison_video()

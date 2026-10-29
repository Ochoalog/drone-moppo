"""Renders high-resolution videos for Porto Alegre Ilhas focusing on:
1. Vanished islands (Ilhas que sumiram na enchente de Maio 2024).
2. Point cloud of hypotheses (Nuvem de pontos / partículas da crença b_t).
3. Synchronized camera view: "O que o drone vê" (Pós-Enchente) vs "O que veria no mapa original" (Pré-Enchente).
4. Synchronized comparison of:
   - Linha Reta (Straight)
   - Primeiro Treino (PPO PoC S2)
   - Treino MO (MOPPO com Memória)
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
from .env import Episode
from .mo_eval import load_mo_model
from .mo_poa_eval import MOPPOPolicyWrapper
from .evaluate import PPOPolicy
from .baselines import StraightPolicy
from .filter import entropy


def generate_poa_ilhas_nuvem_comparison_video(
    seed: int = 2037, # Episode 13
    out_mp4: Path = ROOT / "results_mo/videos/poa_ilhas_comparison_nuvem_pontos_camera.mp4",
    out_gif: Path = ROOT / "results_mo/videos/poa_ilhas_comparison_nuvem_pontos_camera.gif",
    fps: int = 4,
):
    print("=" * 60)
    print(f"Generating POA Ilhas Nuvem de Pontos & Câmera Video (Seed {seed})")
    print("=" * 60)

    cfg_poa = load_config("configs_poa/poa.yaml")
    grid = poa_grid(cfg_poa)
    slices = region_slices(cfg_poa, grid)
    rs, cs = slices["ilhas"]

    beta = beta_for(cfg_poa, "ilhas", "P1", "zeroshot")
    data = region_data(cfg_poa, "ilhas", "P1", beta, pca_from="S2")

    # Load 3 models
    net_mo = load_mo_model(ROOT / "runs_mo/moppo_S2_hand_s0/final_model.pt", memory=True)
    pol_mo = MOPPOPolicyWrapper(net_mo, np.array([0.15, 0.15, 0.15, 0.55], dtype=np.float32), memory=True)
    pol_ppo = PPOPolicy(str(ROOT / "runs/overnight/S2_hand_s0/final_model.zip"), deterministic=True)
    pol_str = StraightPolicy()

    # Episode initialization to get window coordinates
    ep_init = Episode(data, cfg_poa["filter"], "test", cfg_poa["data"]["cell_m"])
    ep_init.reset(seed=seed)
    r0, c0 = ep_init.r0, ep_init.c0
    print(f"[video] Episode window on Ilhas: r0={r0}, c0={c0}")

    # Run 3 policies
    runs = {}
    for key, pol, label in [
        ("straight", pol_str, "Linha Reta (Straight)"),
        ("ppo_poc", pol_ppo, "Primeiro Treino (PPO PoC S2)"),
        ("moppo", pol_mo, "Treino MO (MOPPO com Memória)"),
    ]:
        ep = Episode(data, cfg_poa["filter"], "test", cfg_poa["data"]["cell_m"])
        ep.reset(seed=seed)
        if hasattr(pol, "reset"):
            pol.reset(ep=ep, rng=np.random.default_rng([seed, 7]))
        traj, beliefs, map_est = [], [], []
        while not ep.done:
            traj.append(ep.pos)
            beliefs.append(ep.filter.b.copy())
            map_est.append(ep.filter.map_estimate())
            a = pol.act(ep)
            ep.step(a)
        traj.append(ep.pos)
        beliefs.append(ep.filter.b.copy())
        map_est.append(ep.filter.map_estimate())
        runs[key] = {
            "label": label,
            "traj": np.array(traj),
            "beliefs": np.array(beliefs),
            "map_est": np.array(map_est),
            "success": bool(ep.last.success),
            "false_loc": bool(ep.last.false_loc),
            "dist": float(ep.distance_m),
        }
        print(f"  {label}: steps={len(traj)-1}, succ={ep.last.success}, dist={ep.distance_m:.0f}m")

    # Load high-res rasters for camera crops and window background
    cell = grid.cell
    r_glob = rs.start + r0
    c_glob = cs.start + c0
    xmin = grid.x0 + c_glob * cell
    xmax = grid.x0 + (c_glob + 64) * cell
    ymin = grid.y1 - (r_glob + 64) * cell
    ymax = grid.y1 - r_glob * cell

    # Pre-flood reference map background
    with rasterio.open("data_poa/raw/poa_pre_feb24_thumb.tif") as ds:
        win = from_bounds(xmin, ymin, xmax, ymax, ds.transform)
        rgb_map = np.transpose(ds.read([1, 2, 3], window=win), (1, 2, 0))
    rgb_map = cv2.resize(rgb_map, (512, 512), interpolation=cv2.INTER_CUBIC)

    # High-resolution rasters for camera footprints
    ds_post = rasterio.open("data_poa/raw/poa_post.tif")
    ds_pre = rasterio.open("data_poa/raw/poa_pre_feb24.tif")

    max_t = max(len(runs["straight"]["traj"]), len(runs["ppo_poc"]["traj"]), len(runs["moppo"]["traj"]))
    print(f"[video] Rendering {max_t} synchronized comparison frames...")

    cols_config = [
        ("straight", "#cba6f7", 0),
        ("ppo_poc", "#fab387", 1),
        ("moppo", "#a6e3a1", 2),
    ]

    frames_rgb = []
    rng_pts = np.random.default_rng(12345)

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
            # Top Panel: Map + Point Cloud (Nuvem de Pontos) + Trajectory
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
            cur_rg = rs.start + r0 + cur_pos[0]
            cur_cg = cs.start + c0 + cur_pos[1]
            cur_x = grid.x0 + (cur_cg + 0.5) * cell
            cur_y = grid.y1 - (cur_rg + 0.5) * cell

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
            "Porto Alegre (Delta do Jacuí / Ilhas - Enchente Histórica de Maio 2024)\n"
            "Comparação Sincronizada: Reta vs Primeiro Treino (PPO PoC) vs Treino MO (MOPPO)\n"
            "Com Nuvem de Pontos da Crença e Câmera do Drone vs Mapa Original (Ilhas Submersas)",
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

    print(f"[video] Encoding {len(frames_rgb)} frames to MP4: {out_mp4}...")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_mp4), fourcc, fps, (w, h))
    for f in frames_rgb:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()

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
    generate_poa_ilhas_nuvem_comparison_video()

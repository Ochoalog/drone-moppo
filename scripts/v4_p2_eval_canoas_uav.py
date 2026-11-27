"""Plano 2 (ICUAS 2027) — Avaliação ativa pareada sobre o ortomosaico REAL de UAV pós-enchente
do CBPMESP em Canoas (Mathias Velho, 25–26/05/2024) contra o mapa pré-enchente Maxar (02/02/2024).

Reaproveita 100% da infraestrutura validada da v3 (`load_site_tables`, `RealisticBatchedEnv`,
`BASELINES`, `build_network`, `NetworkPolicy`, `resolve_run_dir`, `_episodes`):
1. Carrega `canoas` (`poa_pre_feb24`, `dinov2_vits14_ftA`) via `load_site_tables` (mesmo mapa pré-evento,
   mesma calibração β da dobra A de Taquari e mesma distintividade).
2. Baixa/reprojeta os 3 ortomosaicos UAV reais do CBPMESP para UTM 22S (0,610 m/px) com correção do
   deslocamento residual medido no Gate P2-G0 (`dx=+3.5 m, dy=-4.3 m`).
3. Extrai os descritores `dinov2_vits14_ftA` (60, 100 e 150 m; K=4 variantes) sobre o ortomosaico UAV
   e substitui `tables.obs_desc` e `tables.obs_ok` pelas observações reais de UAV.
4. Roda 1000 episódios pareados (seeds 2024..3023) comparando as 5 heurísticas e os 10 checkpoints
   `mx_A_ppo_s0..s9` (zero-shot, modos `sampled` e `argmax`).
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

from dal.v3.agents.baselines import BASELINES, run_policy
from dal.v3.agents.ppo_batched import NetworkPolicy, PolicyAdapter, PPOTrainConfig, build_network
from dal.v3.descriptors.extract import DinoV2Extractor
from dal.v3.envs.realistic_env import RealisticBatchedEnv, RealisticConfig
from dal.v3.jobs import _episodes, load_pca, load_site_tables, resolve_run_dir, work_dir
from dal.v3.sensor.render import RasterGeoref, batch_from_spec, render_crops
from dal.v3.sensor.specs import (
    ALTITUDES_M,
    BLUR_SIGMA_RANGE,
    BRIGHTNESS_RANGE,
    CONTRAST_RANGE,
    JITTER_M,
    N_OBS_VARIANTS,
    NOISE_STD,
    OUT_PX,
    SOURCE_RES_M,
    TILT_MAX_DEG,
    YAW_SD_DEG,
    footprint_m,
    load_site,
    site_grid,
)

TARGET_CRS = "EPSG:32722"
UAV_URLS = [
    ("leste_20240525_15cm", "/vsicurl/https://oin-hotosm-temp.s3.us-east-1.amazonaws.com/66531b8b49dd860001fae1fb/0/66531b8b49dd860001fae1fc.tif"),
    ("oeste_20240525_15cm", "/vsicurl/https://oin-hotosm-temp.s3.us-east-1.amazonaws.com/6653074d49dd860001fae1f7/0/6653074d49dd860001fae1f8.tif"),
    ("leste_20240526_4cm", "/vsicurl/https://oin-hotosm-temp.s3.us-east-1.amazonaws.com/6653dbc349dd860001fae221/0/6653dbc349dd860001fae222.tif"),
]
# Deslocamento residual medido no Gate P2-G0 (CBPMESP UAV vs Maxar poa_pre_feb24)
UAV_DX_M, UAV_DY_M = 3.5, -4.3


def build_uav_raster_on_canoas_grid(cache_tif: Path) -> tuple[np.ndarray, RasterGeoref]:
    """Monta o mosaico UAV CBPMESP em UTM 22S cobrindo Mathias Velho, Canoas (0,61035 m/px)."""
    x0, y0, x1, y1 = 476500.0, 6690900.0, 482700.0, 6693050.0
    w = int(round((x1 - x0) / SOURCE_RES_M))
    h = int(round((y1 - y0) / SOURCE_RES_M))
    transform = from_origin(x0, y1, SOURCE_RES_M, SOURCE_RES_M)

    if cache_tif.exists():
        with rasterio.open(cache_tif) as src:
            arr = np.transpose(src.read(), (1, 2, 0))
            return arr, RasterGeoref(src.transform.c, src.transform.f, src.transform.a)

    cache_tif.parent.mkdir(parents=True, exist_ok=True)
    mosaic = np.zeros((3, h, w), dtype=np.uint8)
    for name, src_path in UAV_URLS:
        open_kwargs = {}
        with rasterio.open(src_path) as probe:
            ovs = probe.overviews(1)
            nat_m = probe.res[0] * (111000.0 if probe.crs and probe.crs.is_geographic else 1.0)
            chosen = None
            for idx, ov in enumerate(ovs):
                if nat_m * ov <= SOURCE_RES_M * 1.15:
                    chosen = idx
            if chosen is not None:
                open_kwargs["overview_level"] = chosen
        dst = np.zeros((3, h, w), dtype=np.uint8)
        with rasterio.open(src_path, **open_kwargs) as src:
            reproject(
                source=rasterio.band(src, [1, 2, 3]),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=TARGET_CRS,
                resampling=Resampling.bilinear,
                src_nodata=0,
                dst_nodata=0,
            )
        # Exclui bordas pretas (<=3) e brancas (>=252) de nodata
        valid = (dst.max(axis=0) > 3) & (dst.min(axis=0) < 252)
        mosaic[:, valid] = dst[:, valid]
        print(f"  Warped {name}: {valid.mean() * 100:.1f}% da caixa", flush=True)

    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 3,
        "dtype": "uint8",
        "crs": TARGET_CRS,
        "transform": transform,
        "compress": "deflate",
        "tiled": True,
    }
    with rasterio.open(cache_tif, "w", **profile) as out:
        out.write(mosaic)
    return np.transpose(mosaic, (1, 2, 0)), RasterGeoref(x0, y1, SOURCE_RES_M)


def extract_uav_obs_descriptors(
    uav_img: np.ndarray,
    georef: RasterGeoref,
    weights_path: Path,
    cache_dir: Path,
    device: str,
    batch_size: int = 256,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Extrai descritores dinov2_vits14_ftA nas células de 25 m da CANOAS_AOI_UTM cobertas pelo UAV."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    site = load_site("canoas")
    grid = site_grid(site)
    n_cells = grid.nrows * grid.ncols

    ok_path = cache_dir / "uav_obs_ok.npy"
    if ok_path.exists() and all((cache_dir / f"uav_obs_{alt}m.npy").exists() for alt in ALTITUDES_M):
        obs_ok = np.load(ok_path)
        obs_desc = {alt: np.load(cache_dir / f"uav_obs_{alt}m.npy").astype(np.float32) for alt in ALTITUDES_M}
        return obs_desc, obs_ok

    rows, cols = np.divmod(np.arange(n_cells), grid.ncols)
    cx, cy = grid.center(rows, cols)
    h_img, w_img, _ = uav_img.shape
    x_min, x_max = georef.x_origin + 150.0, georef.x_origin + w_img * georef.res - 150.0
    y_max, y_min = georef.y_origin - 150.0, georef.y_origin - h_img * georef.res + 150.0
    in_box = (cx >= x_min) & (cx <= x_max) & (cy >= y_min) & (cy <= y_max)
    cand_idx = np.nonzero(in_box)[0]
    print(f"Células candidatas dentro da caixa UAV: {len(cand_idx)} / {n_cells}", flush=True)

    extractor = DinoV2Extractor("dinov2_vits14", OUT_PX, device, weights=str(weights_path), outputs=("ft",))
    img_t = torch.as_tensor(uav_img, device=device)
    rng = np.random.default_rng(20261127)

    obs_desc: dict[int, np.ndarray] = {}
    valid_per_alt = []
    for alt in ALTITUDES_M:
        fp = footprint_m(alt)
        full = np.zeros((N_OBS_VARIANTS, n_cells, 256), dtype=np.float16)
        vfrac_all = np.zeros((N_OBS_VARIANTS, n_cells), dtype=np.float32)
        for v in range(N_OBS_VARIANTS):
            spec_df = pd.DataFrame({
                "crop_id": np.arange(len(cand_idx), dtype=np.int64) + v * len(cand_idx),
                "x": cx[cand_idx] + UAV_DX_M + rng.uniform(-JITTER_M, JITTER_M, len(cand_idx)),
                "y": cy[cand_idx] + UAV_DY_M + rng.uniform(-JITTER_M, JITTER_M, len(cand_idx)),
                "footprint_m": fp,
                "out_px": OUT_PX,
                "yaw_deg": rng.normal(0.0, YAW_SD_DEG, len(cand_idx)),
                "roll_deg": rng.uniform(-TILT_MAX_DEG, TILT_MAX_DEG, len(cand_idx)),
                "pitch_deg": rng.uniform(-TILT_MAX_DEG, TILT_MAX_DEG, len(cand_idx)),
                "brightness": rng.uniform(*BRIGHTNESS_RANGE, len(cand_idx)),
                "contrast": rng.uniform(*CONTRAST_RANGE, len(cand_idx)),
                "blur_sigma": rng.uniform(*BLUR_SIGMA_RANGE, len(cand_idx)),
                "noise_std": NOISE_STD,
            })
            feats_list, vfrac_list = [], []
            for start in range(0, len(spec_df), batch_size):
                b_rows = spec_df.iloc[start : start + batch_size]
                crops = render_crops(img_t, georef, batch_from_spec(b_rows, device))
                feats_list.append(extractor(crops)["ft"])
                valid_px = ((crops.amax(dim=-1) > 3) & (crops.amin(dim=-1) < 252)).float().mean(dim=(1, 2))
                vfrac_list.append(valid_px.cpu().numpy())
            full[v, cand_idx] = np.concatenate(feats_list, axis=0)
            vfrac_all[v, cand_idx] = np.concatenate(vfrac_list, axis=0)
        np.save(cache_dir / f"uav_obs_{alt}m.npy", full)
        obs_desc[alt] = full.astype(np.float32)
        valid_per_alt.append((vfrac_all.min(axis=0) >= 0.85).reshape(grid.nrows, grid.ncols))
        print(f"  Alt {alt}m extraída: {int(valid_per_alt[-1].sum())} células válidas (>=85% UAV)", flush=True)

    obs_ok = np.logical_and.reduce(valid_per_alt)
    np.save(ok_path, obs_ok)
    print(f"Total células válidas em todas as altitudes: {int(obs_ok.sum())}", flush=True)
    return obs_desc, obs_ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-root", type=Path, required=True, help="ex.: /content/drive/MyDrive/drone_v3")
    parser.add_argument("--results-dir", type=Path, required=True, help="pasta de saída dos CSVs")
    parser.add_argument("--n-episodes", type=int, default=1000)
    parser.add_argument("--window-cells", type=int, default=40, help="40x40 = 1.0x1.0 km (cabe dentro do ortomosaico UAV)")
    parser.add_argument("--min-window-valid", type=float, default=0.70)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    t0 = time.time()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.out_root / "v4_canoas_uav"
    weights_path = args.out_root / "finetune" / "finetune_A" / "backbone.pt"

    print("1/4 Montando/carregando ortomosaico UAV CBPMESP...", flush=True)
    uav_img, georef = build_uav_raster_on_canoas_grid(cache_dir / "canoas_uav_0.61m.tif")

    print("2/4 Extraindo descritores dinov2_vits14_ftA sobre imagens reais de UAV...", flush=True)
    obs_desc, obs_ok = extract_uav_obs_descriptors(uav_img, georef, weights_path, cache_dir / "features", args.device)

    print("3/4 Carregando tabelas canoas (mapa Maxar pré-evento + calibração Taquari fold A)...", flush=True)
    job = {"descriptor": "dinov2_vits14_ftA", "fold": "A", "feature": "ft"}
    directory = work_dir(args.out_root, "dinov2_vits14_ftA", "A")
    tables_sat, _ = load_site_tables(job, args.out_root, "canoas", "C1", "test", args.device)
    # Aplica a mesma PCA de Taquari (256 -> 128 dim + L2 norm) usada no map_desc de tables_sat
    pca = load_pca(directory, ALTITUDES_M)
    obs_desc_proj = {alt: pca[alt].transform(obs_desc[alt]) for alt in ALTITUDES_M}
    tables_uav = copy.copy(tables_sat)
    tables_uav.obs_desc = obs_desc_proj
    tables_uav.obs_ok = obs_ok
    # Em Mathias Velho (poa_pre_feb24, céu limpo), telhados claros metálicos acionam o detector simples de nuvem;
    # liberamos map_ok nas células válidas do UAV onde o mapa tem vetor não-nulo (norma > 0.5).
    map_has_data = (np.linalg.norm(tables_uav.map_desc[100], axis=-1) > 0.5).reshape(tables_uav.grid_shape)
    tables_uav.map_ok = tables_uav.map_ok | (obs_ok & map_has_data)
    both_ok = tables_uav.map_ok & tables_uav.obs_ok
    print(f"  Máscaras: map_ok={int(tables_uav.map_ok.sum())}, obs_ok={int(tables_uav.obs_ok.sum())}, both={int(both_ok.sum())}", flush=True)

    # Sanity check rápido de retrieval (100 m, variante 0) nas células válidas do UAV
    valid_flat = np.nonzero(both_ok.ravel())[0]
    rng_chk = np.random.default_rng(42)
    sample_idx = rng_chk.choice(valid_flat, size=min(500, len(valid_flat)), replace=False)
    q = torch.as_tensor(obs_desc_proj[100][0, sample_idx], device=args.device)
    m = torch.as_tensor(tables_uav.map_desc[100][valid_flat], device=args.device)
    sim = q @ m.T
    top1_cell = valid_flat[sim.argmax(dim=1).cpu().numpy()]
    ncols = tables_uav.grid_shape[1]
    dr = (top1_cell // ncols) - (sample_idx // ncols)
    dc = (top1_cell % ncols) - (sample_idx % ncols)
    dist_m = np.hypot(dr, dc) * CELL_M
    print(f"  [Sanity check retrieval 100m, {len(sample_idx)} queries / {len(valid_flat)} cells]: "
          f"top1_exato={(top1_cell == sample_idx).mean()*100:.1f}% | top1<=100m={(dist_m <= 100).mean()*100:.1f}%", flush=True)

    W = args.window_cells
    cfg = RealisticConfig(window_cells=W, min_window_valid=args.min_window_valid)
    env = RealisticBatchedEnv(tables_uav, cfg, device=args.device)
    print(f"  Janelas válidas de {W}x{W} ({W*25}m, min_valid={args.min_window_valid}): {len(env.windows)}", flush=True)

    seeds = [2024 + i for i in range(args.n_episodes)]
    frames = []

    print("4/4 Rodando avaliação pareada (5 heurísticas + 10 seeds PPO zero-shot)...", flush=True)
    for name in ("random", "greedy_map", "lawnmower", "straight", "straight_150"):
        env_run = RealisticBatchedEnv(
            tables_uav,
            RealisticConfig(window_cells=W, min_window_valid=args.min_window_valid, altitude_actions=(name == "straight_150")),
            device=args.device,
        )
        env_run = run_policy(env_run, BASELINES[name](), seeds)
        df = _episodes(env_run, name, -1, "canoas_uav", "C1").assign(split="test", mode="—")
        frames.append(df)
        succ_t = df.loc[df["success"], "flight_time_s"].mean() if df["success"].any() else float("nan")
        print(f"  [baseline] {name:14s}: succ={df['success'].mean()*100:5.1f}% | false={df['false_loc'].mean()*100:4.1f}% | time_succ={succ_t:5.1f}s", flush=True)

    class PaddedAdapter(PolicyAdapter):
        """Envolve a observação WxW em padding central 128x128 para avaliar checkpoints PPO treinados em 128x128."""

        def observe(self, env_cur: RealisticBatchedEnv) -> dict[str, torch.Tensor]:
            obs = super().observe(env_cur)
            if env_cur.W == 128:
                return obs
            pad_total = 128 - env_cur.W
            p0 = pad_total // 2
            p1 = pad_total - p0
            obs["map"] = torch.nn.functional.pad(obs["map"], (p0, p1, p0, p1), value=0.0)
            return obs

    for seed_num in range(10):
        run_dir = resolve_run_dir(directory, f"mx_A_ppo_s{seed_num}", include_invalid_gate=True)
        if run_dir is None:
            continue
        for mode in ("sampled", "argmax"):
            env_run = RealisticBatchedEnv(tables_uav, cfg, device=args.device)
            # Constrói a rede com shape 128x128 (mesmo dos checkpoints de Taquari)
            dummy_env = RealisticBatchedEnv(
                tables_uav,
                RealisticConfig(window_cells=128, min_window_valid=0.05),
                device=args.device,
            )
            net = build_network(dummy_env, PPOTrainConfig(change_prior_input=False), args.device)
            del dummy_env
            net.load_state_dict(torch.load(run_dir / "final_model.pt", map_location=args.device))
            policy = NetworkPolicy(net.eval(), PaddedAdapter(False), deterministic=(mode == "argmax"))
            env_run = run_policy(env_run, policy, seeds)
            df = _episodes(env_run, "ppo", seed_num, "canoas_uav", "C1").assign(split="test", mode=mode, run=run_dir.name)
            frames.append(df)
        print(f"  [ppo] seed {seed_num}: sampled={frames[-2]['success'].mean()*100:5.1f}% | argmax={frames[-1]['success'].mean()*100:5.1f}%", flush=True)

    episodes = pd.concat(frames, ignore_index=True)
    episodes.to_csv(args.results_dir / "episodes_canoas_uav.csv", index=False)

    summary = episodes.groupby(["method", "mode"]).agg(
        n_episodes=("success", "size"),
        success_pct=("success", lambda s: round(s.mean() * 100, 2)),
        false_loc_pct=("false_loc", lambda s: round(s.mean() * 100, 2)),
        mean_time_s=("flight_time_s", "mean"),
    ).reset_index()
    summary.to_csv(args.results_dir / "summary_canoas_uav.csv", index=False)
    print("\n=== SUMMARY: CANOAS REAL UAV (CBPMESP) -> MAXAR PRE-FLOOD ===")
    print(summary.to_string(index=False))
    print(f"Concluído em {(time.time() - t0)/60:.1f} min.", flush=True)


if __name__ == "__main__":
    main()

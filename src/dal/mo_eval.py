"""Avaliação multiobjetivo: métricas de Pareto, hipervolume, EUM e baselines."""
from __future__ import annotations

import argparse
import itertools
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import DATA, RESULTS, RUNS, load_config
from .mo_env import DroneLocMOEnv, make_mo_env
from .mo_ppo import MOPPONetwork


# ============================================================================
# Grade sobre o simplex (§5.4)
# ============================================================================
def generate_simplex_grid(step: float = 0.25) -> list[np.ndarray]:
    """Gera a grade do simplex 4D com o passo dado (35 pontos para step=0,25) + os presets fast e safe = 37 pontos
    (o unif já está na grade)."""
    k = int(round(1.0 / step))
    grid = []
    # Todas as partições inteiras não negativas que somam k
    for p0 in range(k + 1):
        for p1 in range(k + 1 - p0):
            for p2 in range(k + 1 - p0 - p1):
                p3 = k - p0 - p1 - p2
                pt = np.array([p0, p1, p2, p3], dtype=np.float32) * step
                grid.append(pt)

    # Acrescenta os 3 presets da §5.3 (os que já estão na grade não se repetem)
    presets = [
        np.array([0.15, 0.55, 0.15, 0.15], dtype=np.float32),  # w_fast (custo)
        np.array([0.15, 0.15, 0.15, 0.55], dtype=np.float32),  # w_safe (risco)
        np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32),  # w_unif
    ]

    out = []
    for pt in grid:
        out.append(pt)

    for p in presets:
        if not any(np.allclose(p, existing, atol=1e-4) for existing in out):
            out.append(p)

    return out


# ============================================================================
# Hipervolume e EUM (§6.2)
# ============================================================================
def compute_hypervolume_2d(points: np.ndarray, ref_point: np.ndarray) -> float:
    """Hipervolume 2D exato para problemas de maximização."""
    # Mantém só os pontos que dominam o ponto de referência
    valid = [p for p in points if np.all(p >= ref_point)]
    if not valid:
        return 0.0

    # Ordena de forma decrescente pelo primeiro objetivo
    pts = sorted(valid, key=lambda x: -x[0])
    hv = 0.0
    cur_y = ref_point[1]

    # Soma a área em degraus retangulares
    for p in pts:
        if p[1] > cur_y:
            hv += (p[0] - ref_point[0]) * (p[1] - cur_y)
            cur_y = p[1]

    return float(hv)


def compute_hypervolume_3d(points: np.ndarray, ref_point: np.ndarray) -> float:
    """Hipervolume 3D exato, por fatias 2D, para problemas de maximização."""
    valid = np.array([p for p in points if np.all(p >= ref_point)])
    if len(valid) == 0:
        return 0.0

    # Ordena os valores únicos de z de forma crescente
    z_vals = np.unique(valid[:, 2])
    z_all = np.sort(np.concatenate([[ref_point[2]], z_vals]))

    total_hv = 0.0
    for i in range(len(z_all) - 1):
        z_lo = z_all[i]
        z_hi = z_all[i + 1]
        delta_z = z_hi - z_lo
        if delta_z <= 0:
            continue

        # Os pontos com z >= z_hi contribuem nesta fatia
        slice_pts = valid[valid[:, 2] >= z_hi][:, :2]
        if len(slice_pts) > 0:
            area_2d = compute_hypervolume_2d(slice_pts, ref_point[:2])
            total_hv += area_2d * delta_z

    return float(total_hv)


def compute_eum(
    utilities: np.ndarray,
    n_samples: int = 1000,
    seed: int = 42,
) -> float:
    """Métrica de utilidade esperada (EUM) sobre pesos Dirichlet(alpha=1) (AMOR / MORL-Baselines).

    utilities: matriz (N, M) com os pontos de utilidade da fronteira de Pareto.
    """
    rng = np.random.default_rng(seed)
    M = utilities.shape[1]
    weights = rng.dirichlet(np.ones(M), size=n_samples)  # (1000, M)

    # Para cada w sorteado, pega a maior utilidade escalarizada entre todas as políticas
    scalarized = weights @ utilities.T  # (1000, N)
    max_util = scalarized.max(axis=1)   # (1000,)
    return float(np.mean(max_util))


# ============================================================================
# Intervalos de confiança por bootstrap
# ============================================================================
def boot_ci(x: np.ndarray | list[float], n: int = 1000, seed: int = 0, stat=np.mean) -> tuple[float, float]:
    arr = np.asarray(x, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(arr), size=(n, len(arr)))
    boot_stats = stat(arr[idx], axis=1)
    return float(np.percentile(boot_stats, 2.5)), float(np.percentile(boot_stats, 97.5))


# ============================================================================
# Pipeline de avaliação
# ============================================================================
def load_mo_model(model_path: Path, memory: bool = True) -> MOPPONetwork:
    """Carrega o state dict de uma MOPPONetwork treinada."""
    c_in = 7
    h, w = 64, 64
    nv = 69 if memory else 5
    net = MOPPONetwork(c_in, h, w, nv, action_dim=8, num_objectives=4)
    state = torch.load(model_path, map_location="cpu")
    net.load_state_dict(state)
    net.eval()
    return net


def run_episode(
    env: DroneLocMOEnv,
    policy,
    w: np.ndarray,
    seed: int,
    is_ppo_mo: bool = True,
    save_traj: bool = False,
    deterministic: bool = True,
) -> tuple[dict, dict | None]:
    """Roda um episódio pareado.

    Com `deterministic=False` a ação é amostrada de π(·|s, w), como no treino. A semente do torch
    é fixada pelo seed do episódio: todos os w veem os mesmos números aleatórios (pareamento).
    """
    if not deterministic:
        torch.manual_seed(seed)
    obs, _ = env.reset(seed=seed, options={"w": w})
    done = False
    decision_times = []

    beliefs, map_estimates = ([env.ep.filter.b.copy()], [env.ep.filter.map_estimate()]) if save_traj else (None, None)

    while not done:
        t0 = time.perf_counter()
        if is_ppo_mo:
            t_obs = {
                "map": torch.from_numpy(obs["map"]).unsqueeze(0).float(),
                "vec": torch.from_numpy(obs["vec"]).unsqueeze(0).float(),
            }
            action = policy.act(t_obs, deterministic=deterministic)
        else:
            action = policy.act(env.ep)
        dt = time.perf_counter() - t0
        decision_times.append(dt)

        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc

        if save_traj:
            beliefs.append(env.ep.filter.b.astype(np.float32))
            map_estimates.append(env.ep.filter.map_estimate())

    r = env.ep.last
    ep_info = {
        "seed": seed,
        "success": bool(r.success),
        "false_loc": bool(r.false_loc),
        "steps": int(env.ep.t),
        "steps_to_success": int(env.ep.t) if r.success else np.nan,
        "final_error_m": float(r.error_m),
        "distance_m": float(env.ep.distance_m),
        "n_blocked": int(env.ep.n_blocked),
        "final_entropy": float(r.entropy),
        "decision_ms": float(np.mean(decision_times) * 1000.0) if decision_times else 0.0,
    }

    traj_data = None
    if save_traj:
        traj_data = {
            "traj": np.array(env.ep.traj),
            "map_est": np.array(map_estimates),
            "beliefs": np.array(beliefs, dtype=np.float32),
            "window": np.array([env.ep.r0, env.ep.c0]),
            "success": bool(r.success),
            "false_loc": bool(r.false_loc),
        }

    return ep_info, traj_data


def evaluate_weight_grid(
    net: MOPPONetwork,
    cfg: dict,
    grid: list[np.ndarray],
    episodes: int = 300,
    seed_base: int = 2024,
    memory: bool = True,
    traj_dir: Path | None = None,
    save_traj_count: int = 10,
    label: str = "mo_ppo",
    deterministic: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Avalia uma política MOPPO sobre a grade de pesos, com seeds de episódio pareados."""
    env = make_mo_env("hand", "S2", cfg, split="test", memory=memory)
    all_episodes = []
    summary_rows = []

    presets = {
        "fast": np.array([0.15, 0.55, 0.15, 0.15], dtype=np.float32),
        "safe": np.array([0.15, 0.15, 0.15, 0.55], dtype=np.float32),
        "unif": np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32),
    }

    t_start = time.time()
    for w_idx, w in enumerate(grid):
        w_str = f"[{w[0]:.2f},{w[1]:.2f},{w[2]:.2f},{w[3]:.2f}]"
        is_preset = False
        preset_name = ""
        for name, pw in presets.items():
            if np.allclose(w, pw, atol=1e-3):
                is_preset = True
                preset_name = name
                break

        ep_results = []
        for i in range(episodes):
            ep_seed = seed_base + i
            save_traj = bool(traj_dir and is_preset and (i < save_traj_count) and memory)

            ep_info, traj_data = run_episode(
                env, net, w, ep_seed, is_ppo_mo=True, save_traj=save_traj, deterministic=deterministic
            )
            ep_info["weight_idx"] = w_idx
            ep_info["w_str"] = w_str
            ep_info["w0"] = w[0]
            ep_info["w1"] = w[1]
            ep_info["w2"] = w[2]
            ep_info["w3"] = w[3]
            ep_info["method"] = label
            ep_info["preset"] = preset_name
            ep_results.append(ep_info)
            all_episodes.append(ep_info)

            if save_traj and traj_data is not None:
                traj_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    traj_dir / f"test_S2_{label}_{preset_name}_ep{i}.npz",
                    **traj_data,
                )

        df_w = pd.DataFrame(ep_results)
        s_lo, s_hi = boot_ci(df_w["success"])
        f_lo, f_hi = boot_ci(df_w["false_loc"])
        d_lo, d_hi = boot_ci(df_w["distance_m"])

        summary_rows.append({
            "method": label,
            "weight_idx": w_idx,
            "w_str": w_str,
            "w_info": float(w[0]),
            "w_cost": float(w[1]),
            "w_success": float(w[2]),
            "w_risk": float(w[3]),
            "preset": preset_name,
            "episodes": episodes,
            "success": float(df_w["success"].mean()),
            "success_lo": s_lo,
            "success_hi": s_hi,
            "false_loc": float(df_w["false_loc"].mean()),
            "false_loc_lo": f_lo,
            "false_loc_hi": f_hi,
            "steps_median": float(df_w["steps"].median()),
            "distance_m": float(df_w["distance_m"].mean()),
            "distance_lo": d_lo,
            "distance_hi": d_hi,
            "decision_ms": float(df_w["decision_ms"].mean()),
        })

        if (w_idx + 1) % 5 == 0 or (w_idx + 1) == len(grid):
            print(f"[{label}] Evaluated {w_idx + 1}/{len(grid)} weights ({time.time() - t_start:.0f}s)")

    env.close()
    return pd.DataFrame(summary_rows), pd.DataFrame(all_episodes)


def evaluate_reference_methods(
    cfg: dict,
    episodes: int = 300,
    seed_base: int = 2024,
    traj_dir: Path | None = None,
    save_traj_count: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reavalia a linha reta e os PPOs da PoC (runs/overnight/) nos mesmos 300 episódios."""
    from .baselines import StraightPolicy
    from .evaluate import PPOPolicy

    env = make_mo_env("hand", "S2", cfg, split="test", memory=False)
    summary_rows = []
    all_episodes = []

    # 1. Baseline da linha reta
    straight = StraightPolicy()
    ep_results = []
    for i in range(episodes):
        seed = seed_base + i
        save_traj = bool(traj_dir and (i < save_traj_count))
        env.reset(seed=seed, options={"w": [0.25, 0.25, 0.25, 0.25]})
        straight.reset(env.ep, np.random.default_rng([seed, 7]))
        ep_info, traj_data = run_episode(env, straight, np.array([0.25]*4), seed, is_ppo_mo=False, save_traj=save_traj)
        ep_info["method"] = "straight"
        ep_results.append(ep_info)
        all_episodes.append(ep_info)
        if save_traj and traj_data is not None:
            traj_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(traj_dir / f"test_S2_straight_ep{i}.npz", **traj_data)

    df_s = pd.DataFrame(ep_results)
    s_lo, s_hi = boot_ci(df_s["success"])
    f_lo, f_hi = boot_ci(df_s["false_loc"])
    d_lo, d_hi = boot_ci(df_s["distance_m"])
    summary_rows.append({
        "method": "straight",
        "weight_idx": -1,
        "w_str": "N/A",
        "w_info": 0.0,
        "w_cost": 0.0,
        "w_success": 0.0,
        "w_risk": 0.0,
        "preset": "baseline",
        "episodes": episodes,
        "success": float(df_s["success"].mean()),
        "success_lo": s_lo,
        "success_hi": s_hi,
        "false_loc": float(df_s["false_loc"].mean()),
        "false_loc_lo": f_lo,
        "false_loc_hi": f_hi,
        "steps_median": float(df_s["steps"].median()),
        "distance_m": float(df_s["distance_m"].mean()),
        "distance_lo": d_lo,
        "distance_hi": d_hi,
        "decision_ms": float(df_s["decision_ms"].mean()),
    })

    # 2. Modelos PPO da PoC (seeds 0, 1 e 2)
    poc_root = RUNS / "overnight"
    poc_models = [poc_root / f"S2_hand_s{s}" / "final_model.zip" for s in range(3)]
    for s_idx, model_path in enumerate(poc_models):
        if not model_path.exists():
            continue
        ppo_pol = PPOPolicy(str(model_path), deterministic=True)
        ep_results = []
        for i in range(episodes):
            seed = seed_base + i
            env.reset(seed=seed, options={"w": [0.25, 0.25, 0.25, 0.25]})
            ep_info, _ = run_episode(env, ppo_pol, np.array([0.25]*4), seed, is_ppo_mo=False, save_traj=False)
            ep_info["method"] = f"ppo_poc_s{s_idx}"
            ep_results.append(ep_info)
            all_episodes.append(ep_info)

        df_p = pd.DataFrame(ep_results)
        s_lo, s_hi = boot_ci(df_p["success"])
        f_lo, f_hi = boot_ci(df_p["false_loc"])
        d_lo, d_hi = boot_ci(df_p["distance_m"])
        summary_rows.append({
            "method": f"ppo_poc_s{s_idx}",
            "weight_idx": -1,
            "w_str": "[0.25,0.25,0.25,0.25]",
            "w_info": 0.25,
            "w_cost": 0.25,
            "w_success": 0.25,
            "w_risk": 0.25,
            "preset": "uniform",
            "episodes": episodes,
            "success": float(df_p["success"].mean()),
            "success_lo": s_lo,
            "success_hi": s_hi,
            "false_loc": float(df_p["false_loc"].mean()),
            "false_loc_lo": f_lo,
            "false_loc_hi": f_hi,
            "steps_median": float(df_p["steps"].median()),
            "distance_m": float(df_p["distance_m"].mean()),
            "distance_lo": d_lo,
            "distance_hi": d_hi,
            "decision_ms": float(df_p["decision_ms"].mean()),
        })

    env.close()
    return pd.DataFrame(summary_rows), pd.DataFrame(all_episodes)

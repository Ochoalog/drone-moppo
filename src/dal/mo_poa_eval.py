"""Avaliação zero-shot da política multiobjetivo em Porto Alegre (H7).

Avalia o MOPPO com memória (seed 0) em POA P1 (centro e ilhas) nos presets fast, safe e unif,
de forma pareada com o PPO-S2 e a linha reta. Os modelos do Taquari e os dados de POA
são só lidos, nunca alterados.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .baselines import StraightPolicy
from .config import ROOT, load_config
from .env import Episode
from .evaluate import PPOPolicy
from .mo_eval import boot_ci, load_mo_model
from .poa import beta_for, region_data


class MOPPOPolicyWrapper:
    """Adapta a MOPPONetwork à interface de política do Episode (`reset` / `act`).

    Com `deterministic=False` a ação é amostrada de π(·|s, w), como no treino.
    """

    def __init__(self, net, w: np.ndarray, memory: bool = True, deterministic: bool = True):
        self.net = net
        self.w = np.array(w, dtype=np.float32)
        self.memory = memory
        self.deterministic = deterministic
        self.action_history = np.zeros((8, 8), dtype=np.float32)

    def reset(self, ep=None, rng=None):
        self.action_history.fill(0.0)

    def act(self, ep) -> int:
        obs = ep.policy_obs()
        t_frac = np.float32(1.0 - ep.t / ep.max_steps)
        if self.memory:
            vec = np.concatenate([self.w, self.action_history.flatten(), [t_frac]], dtype=np.float32)
        else:
            vec = np.concatenate([self.w, [t_frac]], dtype=np.float32)

        t_obs = {
            "map": torch.from_numpy(obs["map"]).unsqueeze(0).float(),
            "vec": torch.from_numpy(vec).unsqueeze(0).float(),
        }
        action = self.net.act(t_obs, deterministic=self.deterministic)
        if self.memory:
            self.action_history[:-1] = self.action_history[1:]
            one_hot = np.zeros(8, dtype=np.float32)
            one_hot[int(action)] = 1.0
            self.action_history[-1] = one_hot
        return int(action)


def evaluate_poa_p1(
    mo_model_path: Path,
    episodes: int = 300,
    seed_base: int = 2024,
    out_csv: Path | None = None,
    deterministic: bool = True,
    include_references: bool = True,
    episodes_csv: Path | None = None,
) -> pd.DataFrame:
    """Roda a avaliação zero-shot pareada em POA P1, nas regiões centro e ilhas.

    `deterministic` vale para **todas** as políticas aprendidas (MO e PPO-S2), como manda a regra A2
    do DECISIONS_MO_V2.md; com `False`, a semente do torch é fixada pelo seed do episódio
    (números aleatórios comuns). `include_references=False` pula só a reta, que não é aprendida e
    não depende do modo. `episodes_csv` grava uma linha por episódio (para ICs pareados).
    """
    cfg_poa = load_config("configs_poa/poa.yaml")
    net = load_mo_model(mo_model_path, memory=True)

    ppo_s2_path = ROOT / "runs/overnight/S2_hand_s0/final_model.zip"
    pol_ppo_s2 = PPOPolicy(str(ppo_s2_path), deterministic=deterministic) if ppo_s2_path.exists() else None
    pol_straight = StraightPolicy()

    presets = {
        "mo_safe": np.array([0.15, 0.15, 0.15, 0.55], dtype=np.float32),
        "mo_fast": np.array([0.15, 0.55, 0.15, 0.15], dtype=np.float32),
        "mo_unif": np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32),
    }

    regions = ["centro", "ilhas"]
    rows = []
    episode_rows = []

    for region in regions:
        print(f"\n[poa_eval] Evaluating region {region} (P1, zeroshot, {episodes} episodes)...")
        beta = beta_for(cfg_poa, region, "P1", "zeroshot")
        data = region_data(cfg_poa, region, "P1", beta, pca_from="S2")
        ep = Episode(data, cfg_poa["filter"], "test", cfg_poa["data"]["cell_m"])

        # Políticas avaliadas nesta região
        policies = {
            name: MOPPOPolicyWrapper(net, weights, memory=True, deterministic=deterministic)
            for name, weights in presets.items()
        }
        if include_references:
            policies["straight"] = pol_straight
        if pol_ppo_s2:
            policies["ppo_s2"] = pol_ppo_s2

        for method_name, pol in policies.items():
            t0 = time.time()
            succs, flocs, dists, steps = [], [], [], []
            dts = []

            for i in range(episodes):
                ep_seed = seed_base + i
                if not deterministic:
                    torch.manual_seed(ep_seed)
                ep.reset(seed=ep_seed)
                if hasattr(pol, "reset"):
                    pol.reset(ep=ep, rng=np.random.default_rng([ep_seed, 7]))

                while not ep.done:
                    dt0 = time.perf_counter()
                    a = pol.act(ep)
                    dt = time.perf_counter() - dt0
                    dts.append(dt)
                    ep.step(a)

                r = ep.last
                succs.append(float(r.success))
                flocs.append(float(r.false_loc))
                dists.append(float(ep.distance_m))
                steps.append(int(ep.t))
                episode_rows.append({
                    "region": region, "method": method_name, "episode": i, "seed": ep_seed,
                    "mode": "argmax" if deterministic else "sampled",
                    "window_r0": int(ep.r0), "window_c0": int(ep.c0),
                    "success": bool(r.success), "false_loc": bool(r.false_loc),
                    "steps": int(ep.t), "distance_m": float(ep.distance_m),
                })

            s_lo, s_hi = boot_ci(succs)
            f_lo, f_hi = boot_ci(flocs)
            d_lo, d_hi = boot_ci(dists)

            row = {
                "region": region,
                "scenario": "P1",
                "variant": "zeroshot",
                "method": method_name,
                "episodes": episodes,
                "success": float(np.mean(succs)),
                "success_lo": s_lo,
                "success_hi": s_hi,
                "false_loc": float(np.mean(flocs)),
                "false_loc_lo": f_lo,
                "false_loc_hi": f_hi,
                "steps_median": float(np.median(steps)),
                "distance_mean_m": float(np.mean(dists)),
                "distance_lo": d_lo,
                "distance_hi": d_hi,
                "decision_ms": float(np.mean(dts) * 1000.0) if dts else 0.0,
            }
            rows.append(row)
            elapsed = time.time() - t0
            print(
                f"  [{method_name}] succ: {row['success']*100:.1f}%, "
                f"floc: {row['false_loc']*100:.1f}%, dist: {row['distance_mean_m']:.1f}m ({elapsed:.1f}s)"
            )

    df_res = pd.DataFrame(rows)
    if episodes_csv:
        episodes_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(episode_rows).to_csv(episodes_csv, index=False)
    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_res.to_csv(out_csv, index=False)
        print(f"[poa_eval] Saved summary to {out_csv}")

    return df_res


def main() -> None:
    """CLI. `--mode` escolhe argmax ou amostrado; `--out` grava resumo e episódios (padrão: saída v1)."""
    import argparse

    parser = argparse.ArgumentParser(description="Avaliação zero-shot do MOPPO em Porto Alegre (H7)")
    parser.add_argument("--mode", choices=["argmax", "sampled"], default="argmax")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--out", help="pasta de saída; sem ela, grava o resumo v1 em results_mo/poa_summary.csv")
    args = parser.parse_args()

    mo_model = ROOT / "runs_mo/moppo_S2_hand_s0/final_model.pt"
    if args.out is None:
        evaluate_poa_p1(mo_model, episodes=args.episodes, out_csv=ROOT / "results_mo/poa_summary.csv")
        return
    out_dir = ROOT / args.out
    evaluate_poa_p1(
        mo_model, episodes=args.episodes, deterministic=args.mode == "argmax",
        out_csv=out_dir / f"poa_summary_{args.mode}.csv", episodes_csv=out_dir / f"poa_episodes_{args.mode}.csv",
    )


if __name__ == "__main__":
    main()

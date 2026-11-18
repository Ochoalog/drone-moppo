"""Piloto local da v3 com dados reais (descritor hand nas specs v3, uma altitude, CPU). Não gera veredito.

1. **G3 preliminar:** controle S0 (mapa = observação), dobra A, divisão de teste: a reta com filtro padrão
   precisa de ≥ 90% de sucesso (β calibrado no treino, regra da PoC).
2. **S2:** calibra β₁, κ e α (prior do EMSR720, fontes ≤ 06/05/2024) no treino da dobra A e roda as baselines
   com filtro padrão e CA em episódios de teste.

Saídas: results_v3/pilot_local/{calibration.json, episodes.csv, summary.csv, g3.json}.
Uso: .venv/bin/python scripts/pilot_local_v3.py --altitude 100 --episodes 300
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

import numpy as np
import pandas as pd
import torch

from dal.config import ROOT
from dal.v3.agents.baselines import BASELINES, run_policy
from dal.v3.envs.realistic_env import RealisticBatchedEnv, RealisticConfig
from dal.v3.localization import calibration
from dal.v3.pipeline import DataRequest, load_tables
from dal.v3.reporting.provenance import with_provenance

OUT = ROOT / "results_v3" / "pilot_local"
FEATURES = ROOT / "data_v3" / "features" / "v3"
G3_THRESHOLD = 0.90
BASELINE_NAMES = ("straight", "lawnmower", "random", "greedy_map")


def calibrated_tables(scenario: str, altitude: int, split: str, prior, episodes: int):
    """Tabelas da divisão pedida com β calibrado no treino (β₁ por máxima verossimilhança, κ pela regra da PoC)."""
    config = RealisticConfig(altitudes_m=(altitude,), start_altitude_index=0)
    train, _ = load_tables(FEATURES, DataRequest("taquari", scenario, "A", "hand", (altitude,), "train", "hand"),
                           {altitude: 1.0}, change_prior=prior)
    beta1 = calibration.calibrate_beta1(train, altitude, config.window_cells)
    kappa, kappa_rows = calibration.calibrate_kappa(train, config, {altitude: beta1["beta1"]}, episodes)
    train.beta = {altitude: kappa * beta1["beta1"]}
    alpha, alpha_rows = (calibration.calibrate_alpha(train, config, episodes) if prior is not None else (0.0, []))
    target, _ = load_tables(FEATURES, DataRequest("taquari", scenario, "A", "hand", (altitude,), split, "hand"),
                            train.beta, change_prior=prior)
    result = {"scenario": scenario, "beta1": beta1, "kappa": kappa, "beta": train.beta[altitude], "alpha": alpha,
              "kappa_rows": kappa_rows, "alpha_rows": alpha_rows}
    return target, config, result


def episodes_frame(env, method: str, scenario: str) -> pd.DataFrame:
    return pd.DataFrame({"method": method, "scenario": scenario, "episode": np.arange(env.n),
                         "window_r0": env.origin[:, 0].numpy(), "window_c0": env.origin[:, 1].numpy(),
                         "success": env.success.numpy(), "false_loc": env.false_loc.numpy(), "steps": env.t.numpy(),
                         "flight_time_s": env.flight_time_s.numpy(), "distance_m": env.distance_m.numpy()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--altitude", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--calibration-episodes", type=int, default=100)
    args = parser.parse_args()
    torch.set_num_threads(8)
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = [2024 + i for i in range(args.episodes)]
    prior = np.load(ROOT / "results_v3/prior/taquari_prior_25m.npy").astype(np.float64)
    started, frames, calibrations = time.time(), [], {}

    # 1. G3 preliminar: controle S0 (sem prior: o mapa é a própria observação).
    tables, config, calibrations["S0"] = calibrated_tables("S0", args.altitude, "test", None,
                                                           args.calibration_episodes)
    env = run_policy(RealisticBatchedEnv(tables, config), BASELINES["straight"](), seeds)
    frames.append(episodes_frame(env, "straight", "S0"))
    g3 = {"success": float(env.success.float().mean()), "threshold": G3_THRESHOLD,
          "passed": bool(env.success.float().mean() >= G3_THRESHOLD), "episodes": args.episodes,
          "descriptor": "hand (specs v3, 25 m)", "altitude_m": args.altitude,
          "note": "G3 preliminar: o oficial usa o descritor escolhido no G1"}
    (OUT / "g3.json").write_text(json.dumps(g3, indent=1))
    print("G3", g3, f"({(time.time() - started) / 60:.1f} min)", flush=True)

    # 2. S2: baselines com filtro padrão e CA.
    tables, config, calibrations["S2"] = calibrated_tables("S2", args.altitude, "test", prior,
                                                           args.calibration_episodes)
    for name in BASELINE_NAMES:
        for alpha in (0.0, calibrations["S2"]["alpha"]):
            policy = BASELINES[name](samples=8) if name == "greedy_map" else BASELINES[name]()
            env = run_policy(RealisticBatchedEnv(tables, replace(config, alpha=alpha)), policy, seeds)
            frames.append(episodes_frame(env, f"{name}{'_ca' if alpha > 0 else ''}", "S2"))
            print(name, alpha, f"sucesso {env.success.float().mean():.3f} falsa {env.false_loc.float().mean():.3f}",
                  f"({(time.time() - started) / 60:.1f} min)", flush=True)
            if alpha == calibrations["S2"]["alpha"] == 0.0:
                break
    (OUT / "calibration.json").write_text(json.dumps(calibrations, indent=1, default=float))
    episodes = with_provenance(pd.concat(frames, ignore_index=True), source_file="results_v3/pilot_local/episodes.csv",
                               protocol=f"piloto local v3; hand; {args.altitude} m; deriva 2%; seeds 2024+i",
                               config=vars(args), n_episodes=args.episodes, seeds="—", fold="A", region="taquari")
    episodes.to_csv(OUT / "episodes.csv", index=False)
    summary = episodes.groupby(["scenario", "method"])[["success", "false_loc", "steps", "flight_time_s"]].mean()
    summary.reset_index().to_csv(OUT / "summary.csv", index=False)
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()

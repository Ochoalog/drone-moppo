"""Gate G2 oficial: paridade do ambiente em lote (modo legado) nos 1000 episódios de teste de S2.

Grava results_v3/gates/G2.json. Critério (§3.2 do plano v3): reta e zigue-zague com trajetórias
idênticas e crenças até 1e-4; PPO da PoC (seeds 0–2, argmax) com sucesso dentro do IC do REPORT.md
(e aqui também a concordância episódio a episódio com results/episodes_test.csv).

Uso: .venv/bin/python scripts/gate_g2_parity_v3.py [--device cpu|cuda] [--episodes 1000]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from dal.config import ROOT, load_config
from dal.env import ScenarioData
from dal.v3.envs.parity import batched_baseline, batched_ppo, compare_baseline, run_cpu_baseline
from dal.v3.reporting.provenance import config_hash, git_commit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--episodes", type=int, default=1000)
    args = parser.parse_args()
    config = load_config("configs/default.yaml")
    data = ScenarioData.load("hand", "S2", config)
    seeds = [int(config["eval"]["seed"]) + i for i in range(args.episodes)]
    report = {"gate": "G2", "episodes": args.episodes, "device": args.device, "git_commit": git_commit(),
              "config_hash": config_hash(config), "baselines": {}, "ppo": {}}

    for name in ("straight", "lawnmower"):
        report["baselines"][name] = compare_baseline(run_cpu_baseline(data, config, name, seeds),
                                                     batched_baseline(data, config, name, seeds, args.device))
        print(name, report["baselines"][name], flush=True)

    recorded = pd.read_csv(ROOT / "results/episodes_test.csv")
    canonical = pd.read_csv(ROOT / "paper/tables/v1_rq1_taquari.csv")
    ci_row = canonical[(canonical.scenario == "S2") & (canonical.method == "ppo") & (canonical.metric == "success")
                       & (canonical.seed_label == "all")].iloc[0]
    successes = []
    for seed in (0, 1, 2):
        env = batched_ppo(data, config, str(ROOT / f"runs/overnight/S2_hand_s{seed}/final_model.zip"), seeds,
                          args.device)
        ref = recorded[(recorded.scenario == "S2") & (recorded.method == "ppo") & (recorded.model_seed == seed)
                       & (recorded.episode < args.episodes)].sort_values("episode")
        success = env.last.success.cpu().numpy()
        successes.append(success)
        report["ppo"][f"seed{seed}"] = {
            "success_batched": float(success.mean()), "success_recorded": float(ref.success.mean()),
            "episodes_identical_outcome": int(((success == ref.success.to_numpy())
                                               & (env.t.cpu().numpy() == ref.steps.to_numpy())).sum())}
        print(f"ppo seed{seed}", report["ppo"][f"seed{seed}"], flush=True)
    pooled = float(np.mean(successes))
    report["ppo"]["pooled_success"] = pooled
    report["ppo"]["report_ci"] = [float(ci_row.ci_low), float(ci_row.ci_high)]
    ppo_ok = bool(ci_row.ci_low <= pooled <= ci_row.ci_high)
    report["passed"] = bool(all(b["passed"] for b in report["baselines"].values()) and ppo_ok)
    out = ROOT / "results_v3" / "gates"
    out.mkdir(parents=True, exist_ok=True)
    (out / "G2.json").write_text(json.dumps(report, indent=1))
    print("G2", "PASSOU" if report["passed"] else "FALHOU")


if __name__ == "__main__":
    main()

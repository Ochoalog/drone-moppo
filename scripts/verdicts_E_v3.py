"""Vereditos e saídas do experimento E (DECISIONS_V3.md V46, V50, V51), gerados por código a partir dos episódios.

V51: o E-real saiu (E1 e E2 cortados). O E é E-synth + L0 (E3–E5), mais um item descritivo, a transferência
zero-shot Taquari → Canoas das 10 `mx_A_ppo` (PPO − cada heurística, com IC, sem veredito).

Entradas (cópias da branch `colab/results`, feitas por este script com `--colab-out`):
- `results_v3/E/<job>/episodes.parquet` das avaliações `E_eval_*`;
- `results_v3/E/<job>/DONE.json` dos treinos (gate aos 25%) e das avaliações (commit, hash da config).

Saídas:
- `results_v3/E/canoas_zeroshot.csv` e `paper/tables/E_canoas_zeroshot.csv`: item descritivo da V51;
- `results_v3/E/verdicts_E.json`: E3 (amostrado = principal; argmax ao lado), melhor heurística por regime;
- `results_v3/E/curve.csv`: vantagem da PPO sobre a aleatória e sobre a melhor heurística, por regime, e E5;
- `results_v3/E/table3.csv`: E4 (sucesso, localização falsa, tempo de voo nos sucessos);
- `paper/tables/E_curve.csv`, `paper/tables/E_table3.csv` (com proveniência) e `paper/figures/fig3_E.pdf`.

Uso: .venv/bin/python scripts/verdicts_E_v3.py [--colab-out <worktree da colab/results>/colab_out]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from dal.config import ROOT
from dal.v3.evaluation import experiment_e as exp_e

OUT = ROOT / "results_v3" / "E"
PAPER_TABLES = ROOT / "paper" / "tables"
PAPER_FIGURES = ROOT / "paper" / "figures"


def regime_specs() -> list[dict]:
    """Regimes do E, na ordem em que aparecem (a figura reordena pelo sucesso da aleatória no treino)."""
    specs = [{"name": "Taquari L0", "eval": "E_eval_taquari_L0", "region": "taquari",
              "runs": [f"mx_A_ppo_s{s}" for s in range(10)], "kind": "taquari"}]
    levels = OUT / "noise_levels.json"
    if levels.exists():
        for level, info in json.loads(levels.read_text())["levels"].items():
            specs.append({"name": f"Taquari {level} (σ={info['sigma']:g})", "eval": f"E_eval_synth_{level}",
                          "region": "taquari", "runs": [f"E_synth_{level}_ppo_s{s}" for s in range(5)],
                          "kind": "taquari", "level": level})
    return specs


ZEROSHOT = {"name": "Canoas zero-shot", "eval": "E_eval_canoas_zeroshot", "region": "canoas",
            "runs": [f"mx_A_ppo_s{s}" for s in range(10)], "kind": "descriptive"}


def canoas_zeroshot() -> pd.DataFrame:
    """V51, descritivo: PPO de Taquari nas 31 janelas de Canoas contra cada heurística (sem treino em Canoas,
    então não há "melhor heurística no treino"; nenhuma é escolhida, todas são reportadas)."""
    path = OUT / ZEROSHOT["eval"] / "episodes.parquet"
    if not path.exists():
        return pd.DataFrame()
    episodes = exp_e.with_method_key(pd.read_parquet(path))
    regime = exp_e.Regime(ZEROSHOT["name"], episodes[episodes.region == "canoas"], episodes.iloc[0:0])
    rows = []
    for mode in ("sampled", "argmax"):
        for index, heuristic in enumerate(exp_e.HEURISTICS):
            diff = exp_e.advantage(regime, heuristic, mode, seed=400 + index)
            rows.append({"mode": mode, "over": heuristic, "advantage": diff.estimate, "low": diff.low,
                         "high": diff.high, **provenance(ZEROSHOT)})
    for method in (*exp_e.HEURISTICS, regime.policy_key("sampled"), regime.policy_key("argmax")):
        summary = exp_e.method_summary(regime, method, seed=499)
        rows.append({"mode": "summary", "over": method, "advantage": summary["success"],
                     "low": summary["success_low"], "high": summary["success_high"], **provenance(ZEROSHOT)})
    return pd.DataFrame(rows)


def sync(colab_out: Path, specs: list[dict]) -> None:
    """Copia da colab/results só o que a análise usa (episódios e DONE.json)."""
    names = {s["eval"] for s in specs} | {ZEROSHOT["eval"]} | {r for s in specs for r in s["runs"]} | {"scan_noise_A"}
    names |= {f"mx_A_{m}_s{s}" for m in ("ppo", "ppo_ca", "ppo_ca_alt") for s in range(10)}
    for name in names:
        source = colab_out / name
        if not source.exists():
            continue
        target = OUT / name
        target.mkdir(parents=True, exist_ok=True)
        for file in ("DONE.json", "episodes.parquet", "summary.csv", "scan.csv"):
            if (source / file).exists():
                shutil.copy2(source / file, target / file)


def load_regime(spec: dict) -> exp_e.Regime | None:
    path = OUT / spec["eval"] / "episodes.parquet"
    if not path.exists():
        return None
    episodes = exp_e.with_method_key(pd.read_parquet(path))
    episodes = episodes[episodes.region == spec["region"]]
    return exp_e.Regime(spec["name"], episodes[episodes.split == "test"], episodes[episodes.split == "train"])


def load_gates(runs: list[str]) -> list[dict]:
    gates = []
    for run in runs:
        done = OUT / run / "DONE.json"
        if done.exists():
            gates.append(json.loads(done.read_text())["metrics"]["gate"])
    return gates


def provenance(spec: dict) -> dict:
    done_path = OUT / spec["eval"] / "DONE.json"
    done = json.loads(done_path.read_text()) if done_path.exists() else {}
    return {"source_file": f"results_v3/E/{spec['eval']}/episodes.parquet",
            "protocol": "E (V46, exploratório); teste; 1000 episódios pareados; bootstrap hierárquico B=2000",
            "n_episodes": 1000, "seeds": "episódios 2024+i; " + ",".join(spec["runs"]), "fold": "A",
            "region": spec["region"], "git_commit": done.get("git_commit", ""),
            "config_hash": done.get("config_hash", "")}


def analyse(specs: list[dict]) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    regimes, curve_rows, table_rows = {}, [], []
    for index, spec in enumerate(specs):
        regime = load_regime(spec)
        if regime is None or regime.test.empty:
            continue
        regimes[spec["name"]] = (spec, regime)
        gates = exp_e.gate_rate(load_gates(spec["runs"]))
        row = {"regime": spec["name"], "random_train_success": regime.random_train_success(),
               "best_heuristic": regime.best, **gates, **provenance(spec)}
        for mode in ("sampled", "argmax"):
            for label, over in (("random", "random"), ("best", regime.best)):
                diff = exp_e.advantage(regime, over, mode, seed=index)
                row.update({f"adv_{label}_{mode}": diff.estimate, f"adv_{label}_{mode}_low": diff.low,
                            f"adv_{label}_{mode}_high": diff.high})
        curve_rows.append(row)
        for method in ("random", regime.best, regime.policy_key("sampled"), regime.policy_key("argmax")):
            table_rows.append({**exp_e.method_summary(regime, method, seed=index), **provenance(spec)})
    curve = pd.DataFrame(curve_rows)
    if not curve.empty:
        curve = curve.sort_values("random_train_success", ascending=False)
    verdicts = {"status": "exploratório (V46); E1 e E2 cortados com o E-real (V51)", "main_mode": exp_e.MAIN_MODE,
                "regimes": list(regimes)}
    for mode in ("sampled", "argmax"):
        key = "" if mode == exp_e.MAIN_MODE else "_argmax"
        l3 = next((name for name in regimes if " L3 " in f" {name} "), None)
        if l3 and "Taquari L0" in regimes:
            e3 = exp_e.difference_of_paired(regimes[l3][1], regimes["Taquari L0"][1], mode, seed=301)
            verdicts[f"E3{key}"] = {**e3.summary(), "criterion": "Δ(L3) − Δ(L0) > 0, IC exclui 0",
                                    "regime_high": l3, "verdict": exp_e.classify(e3)}
    return verdicts, curve, pd.DataFrame(table_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--colab-out", help="pasta colab_out de um worktree da branch colab/results")
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()
    specs = regime_specs()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.colab_out:
        sync(Path(args.colab_out), specs)
    verdicts, curve, table = analyse(specs)
    (OUT / "verdicts_E.json").write_text(json.dumps(verdicts, indent=1, ensure_ascii=False, default=float))
    curve.to_csv(OUT / "curve.csv", index=False)
    table.to_csv(OUT / "table3.csv", index=False)
    PAPER_TABLES.mkdir(parents=True, exist_ok=True)
    curve.to_csv(PAPER_TABLES / "E_curve.csv", index=False)
    table.to_csv(PAPER_TABLES / "E_table3.csv", index=False)
    zeroshot = canoas_zeroshot()
    if not zeroshot.empty:
        zeroshot.to_csv(OUT / "canoas_zeroshot.csv", index=False)
        zeroshot.to_csv(PAPER_TABLES / "E_canoas_zeroshot.csv", index=False)
    if not args.no_figure and not curve.empty:
        from dal.v3.reporting.figures_e import figure_3

        PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
        figure_3(curve, PAPER_FIGURES / "fig3_E.pdf")
    print(json.dumps(verdicts, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()

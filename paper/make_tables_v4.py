"""Tabelas do artigo v4 (PLANO_ARTIGO.md v4 §4), lidas só de CSVs versionados. NÃO editar os .tex à mão.

- Tab. II (`tables/tab2_v1.tex`): estudo v1, PPO × heurísticas em S1 e S2 (sucesso, localização falsa, distância
  nos sucessos), com o IC hierárquico da diferença para a reta (`tables/v1_rq1_taquari.csv`).
- Baselines v3 nos 6 alvos (`tables/v3_baselines.csv` e `tables/tab_v3_baselines.tex`), a partir do
  `results_v3/eval/eval_baselines_A/summary.csv` (filtro padrão; com α = 0 o CA é idêntico).

Uso: .venv/bin/python paper/make_tables_v4.py
"""
from __future__ import annotations

import json

import pandas as pd

from dal.config import ROOT

TABLES = ROOT / "paper" / "tables"
V1_METHODS = {"ppo": "PPO", "straight": "Straight", "lawnmower": "Lawnmower", "greedy_map": "Greedy (map)",
              "random": "Random"}
V3_METHODS = {"random": "Random", "straight": "Straight", "straight_150": "Straight 150\\,m",
              "lawnmower": "Lawnmower", "greedy_map": "Greedy (map)"}
V3_TARGETS = [("taquari", "S2", "test", "Taquari S2 (test)"), ("taquari", "S2", "train", "Taquari S2 (train)"),
              ("taquari", "S1", "test", "Taquari S1 (test)"), ("canoas", "C1", "test", "Canoas C1"),
              ("poa_ilhas", "P1", "test", "POA islands P1"), ("poa_centro", "P1", "test", "POA center P1")]


def pct(value: float) -> str:
    return f"{100 * value:.1f}"


def tab2_v1() -> None:
    source = "paper/tables/v1_rq1_taquari.csv"
    table = pd.read_csv(ROOT / source)
    table = table[table.seed_label == "all"].set_index(["scenario", "method", "metric"])
    lines = [f"% Gerado por paper/make_tables_v4.py a partir de {source}. NÃO EDITAR.",
             "\\begin{tabular}{@{}llrrrr@{}}", "\\toprule",
             "Map & Method & Succ. (\\%) & $\\Delta$ vs. straight (p.p.) & False loc. (\\%) & Dist. succ. (km) \\\\",
             "\\midrule"]
    for scenario in ("S1", "S2"):
        for method, name in V1_METHODS.items():
            success = table.loc[(scenario, method, "success")]
            false_loc = table.loc[(scenario, method, "false_localization")]
            distance = table.loc[(scenario, method, "distance_mean_success")]
            diff = ("---" if method == "straight" else
                    f"{100 * success.diff_vs_straight:+.1f} [{100 * success.diff_ci_low:+.1f}; "
                    f"{100 * success.diff_ci_high:+.1f}]")
            label = scenario if method == "ppo" else ""
            seeds = f" ({int(success.seeds)} seeds)" if int(success.seeds) > 1 else ""
            lines.append(f"{label} & {name}{seeds} & {pct(success.estimate)} & {diff} & {pct(false_loc.estimate)} & "
                         f"{distance.estimate / 1000:.2f} \\\\")
        lines.append("\\midrule" if scenario == "S1" else "\\bottomrule")
    lines.append("\\end{tabular}")
    (TABLES / "tab2_v1.tex").write_text("\n".join(lines) + "\n")


def v3_baselines() -> None:
    source = "results_v3/eval/eval_baselines_A/summary.csv"
    done = json.loads((ROOT / "results_v3/eval/eval_baselines_A/DONE.json").read_text())
    summary = pd.read_csv(ROOT / source)
    summary = summary[summary.method.isin(V3_METHODS)]
    csv = summary.assign(source_file=source, protocol="v3 realista; filtro padrão; seeds 2024+i",
                         n_episodes=1000, seeds="episódios 2024+i (heurísticas: 1 execução)", fold="A",
                         git_commit=done["git_commit"], config_hash=done["config_hash"])
    csv.to_csv(TABLES / "v3_baselines.csv", index=False)
    index = summary.set_index(["region", "scenario", "split", "method"])
    header = " & ".join(V3_METHODS.values())
    lines = [f"% Gerado por paper/make_tables_v4.py a partir de {source} (commit {done['git_commit'][:7]}). "
             "NÃO EDITAR.", "\\begin{tabular}{@{}l" + "r" * len(V3_METHODS) + "@{}}", "\\toprule",
             f"Target & {header} \\\\", "\\midrule"]
    for region, scenario, split, name in V3_TARGETS:
        cells = [pct(index.loc[(region, scenario, split, m)].success) for m in V3_METHODS]
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TABLES / "tab_v3_baselines.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    tab2_v1()
    v3_baselines()
    print("tabelas geradas em", TABLES)


if __name__ == "__main__":
    main()

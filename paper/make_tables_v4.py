"""Tabelas do artigo v4 (PLANO_ARTIGO.md v4 §4), lidas só de CSVs versionados. NÃO editar os .tex à mão.

- Tab. II (`tables/tab2_v1.tex`): estudo v1, PPO × heurísticas em S1 e S2 (sucesso, localização falsa, distância
  nos sucessos), com o IC hierárquico da diferença para a reta (`tables/v1_rq1_taquari.csv`).
- Tab. III (`tables/tab3_e.tex`): experimento E (E4 e E5), de `results_v3/E/table3.csv` e `curve.csv`.
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


def tab3_e() -> None:
    """Tab. III: por nível de ruído, sucesso (IC), localização falsa e tempo de voo nos sucessos; gate (E5)."""
    import re

    source = "results_v3/E/table3.csv"
    table = pd.read_csv(ROOT / source)
    curve = pd.read_csv(ROOT / "results_v3/E/curve.csv").set_index("regime")
    levels = json.loads((ROOT / "results_v3/E/noise_levels.json").read_text())["levels"]
    methods = {"random": "Random", "straight_150": "Straight 150\\,m", "ppo@sampled": "PPO (sampled)",
               "ppo@argmax": "PPO (argmax)"}
    lines = [f"% Gerado por paper/make_tables_v4.py a partir de {source} e results_v3/E/curve.csv. NÃO EDITAR.",
             "\\begin{tabular}{@{}llrrrr@{}}", "\\toprule",
             "Level ($\\sigma$) & Method & Succ. (\\%) [95\\% CI] & False loc. (\\%) & Time succ. (s) & Gate \\\\",
             "\\midrule"]
    regimes = list(dict.fromkeys(table.regime))
    for index, regime in enumerate(regimes):
        level = re.search(r"L\d", regime).group(0)
        sigma = "0" if level == "L0" else f"{levels[level]['sigma']:g}"
        part = table[table.regime == regime].set_index("method")
        gate = f"{int(curve.loc[regime].gate_original_passed)}/{int(curve.loc[regime].n_runs)}"
        for row_index, (method, name) in enumerate(methods.items()):
            row = part.loc[method]
            label = f"{level} ({sigma})" if row_index == 0 else ""
            gate_cell = gate if method == "ppo@sampled" else ""
            lines.append(f"{label} & {name} & {pct(row.success)} [{pct(row.success_low)}; {pct(row.success_high)}] & "
                         f"{pct(row.false_loc)} & {row.flight_time_success_s:.0f} & {gate_cell} \\\\")
        lines.append("\\midrule" if index < len(regimes) - 1 else "\\bottomrule")
    lines.append("\\end{tabular}")
    (TABLES / "tab3_e.tex").write_text("\n".join(lines) + "\n")


VERDICT_EN = {"suportada": "supported", "não suportada": "not supported", "inconclusiva": "inconclusive",
              "contrária": "contrary", "não avaliável": "not evaluable"}


def verdict_en(text: str) -> str:
    return VERDICT_EN.get(text, text)


def tab_verdicts() -> None:
    """Tab. A2: veredito de cada hipótese pré-especificada (v1, MO, v3) e dos critérios exploratórios do E.

    Os vereditos são lidos dos arquivos gerados por código; só os "não avaliável" por decisão registrada
    (H2–H5 da v3, E1/E2) citam a decisão em vez de um arquivo.
    """
    v1 = json.loads((ROOT / "results_final/stats/verdicts.json").read_text())
    reports = v1["recorded_in_reports"]
    mo = json.loads((ROOT / "results_mo_v1fix/verdicts_v2.json").read_text())
    m1 = pd.read_csv(ROOT / "results_final/mechanism/m1.csv").m1_verdict.unique()
    assert len(m1) == 1, m1
    h1 = json.loads((ROOT / "results_v3/verdicts/h1.json").read_text())
    e = json.loads((ROOT / "results_v3/E/verdicts_E.json").read_text())
    e3 = e["E3"]

    def ci(block: dict) -> str:
        return (f"${100 * block['estimate']:+.1f}$ [${100 * block['low']:+.1f}$; ${100 * block['high']:+.1f}$]~p.p.")

    groups = [
        ("v1 study (weak descriptor)", [
            ("H1", "PPO beats every simple heuristic in S1 and S2", v1["H1"]["hierarchical_ci"]),
            ("H2", "Advantage over the best heuristic grows S0 $<$ S1 $<$ S2", v1["H2"]["hierarchical_ci"]),
            ("H3", "Advantage holds in the unseen (south) region", v1["H3"]["hierarchical_ci"]),
            ("HP1", "PPO--straight relation transfers zero-shot to Porto Alegre", v1["HP1"]["hierarchical_ci"]),
            ("HP2", "Success falls as the map changes", reports["HP2"]),
            ("HP3", "Active advantage is larger on the islands", reports["HP3"]),
            ("M1", "False localizations occur at cells distinctive on the old map", verdict_en(m1[0])),
        ]),
        ("v1 multi-objective (sampled actions)", [
            ("H4", "One conditioned policy replaces fixed-weight policies", verdict_en(mo["H4"]["verdict"])),
            ("H5", "Some preference weight dominates the straight line", verdict_en(mo["H5"]["verdict_holm"])),
            ("H6", "Weights control false-localization risk and cost", verdict_en(mo["H6"]["verdict"])),
            ("H7", "The \\emph{safe} preset localizes falsely less on the islands", verdict_en(mo["H7"]["verdict"])),
        ]),
        ("v3 (strong descriptor)", [
            ("H1", "Fine-tuned descriptor beats the hand-crafted one (top-1, S1 and S2)",
             verdict_en(h1["verdict"])),
            ("H2", "Filter with the flood prior reduces false localization",
             "not evaluable ($\\alpha = 0$, V36)"),
            ("H3--H5", "PPO vs.\\ best heuristic; altitude; zero-shot transfer",
             "not evaluable (no run passed the gate, V45)"),
        ]),
        ("Experiment E (exploratory)", [
            ("E1, E2", "PPO learns in Canoas; Canoas advantage exceeds L0", "cut (V51)"),
            ("E3", "$\\Delta(\\mathrm{L3}) - \\Delta(\\mathrm{L0}) > 0$ vs.\\ best heuristic",
             f"{verdict_en(e3['verdict'])}, {ci(e3)}"),
            ("E4, E5", "Per-level metrics and gate pass rate", "descriptive (Tab.~\\ref{tab:e})"),
        ]),
    ]
    lines = ["% Gerado por paper/make_tables_v4.py (tab_verdicts) a partir de results_final/stats/verdicts.json, "
             "results_mo_v1fix/verdicts_v2.json,", "% results_final/mechanism/m1.csv, results_v3/verdicts/h1.json "
             "e results_v3/E/verdicts_E.json. NÃO EDITAR.",
             "\\begin{tabular}{@{}lp{0.47\\columnwidth}p{0.3\\columnwidth}@{}}", "\\toprule",
             "Id & Hypothesis & Verdict \\\\"]
    for title, rows in groups:
        lines += ["\\midrule", f"\\multicolumn{{3}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\"]
        lines += [f"{ident} & {text} & {verdict} \\\\" for ident, text, verdict in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TABLES / "tab_verdicts.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    tab2_v1()
    tab3_e()
    tab_verdicts()
    v3_baselines()
    print("tabelas geradas em", TABLES)


if __name__ == "__main__":
    main()

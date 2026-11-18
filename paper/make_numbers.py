"""Gera paper/numbers.tex: macros LaTeX com os números do artigo, lidos só de CSVs versionados.

Nenhum número do texto é digitado à mão: o texto usa `\\vOneDiffSuccessSTwo` etc. Cada macro tem um
comentário com o arquivo e a linha de origem. Rodar de novo depois de qualquer mudança nos CSVs.

Uso: .venv/bin/python paper/make_numbers.py
"""
from __future__ import annotations

import json

import pandas as pd

from dal.config import ROOT

OUT = ROOT / "paper" / "numbers.tex"


def pct(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}"


def km(value_m: float) -> str:
    return f"{value_m / 1000:.2f}"


def taquari_v1(macros: list) -> None:
    source = "paper/tables/v1_rq1_taquari.csv"
    table = pd.read_csv(ROOT / source)
    s2 = table[(table.scenario == "S2") & (table.seed_label == "all")].set_index(["method", "metric"])
    ppo_success = s2.loc[("ppo", "success")]
    macros += [
        ("vOneSuccessPpoSTwo", pct(ppo_success.estimate), f"{source}: S2 ppo success"),
        ("vOneSuccessStraightSTwo", pct(s2.loc[("straight", "success")].estimate), f"{source}: S2 straight success"),
        ("vOneDiffSuccessSTwoLow", pct(ppo_success.diff_ci_low), f"{source}: S2 ppo − straight, IC hierárquico"),
        ("vOneDiffSuccessSTwoHigh", pct(ppo_success.diff_ci_high), f"{source}: S2 ppo − straight, IC hierárquico"),
        ("vOneFalseLocPpoSTwo", pct(s2.loc[("ppo", "false_localization")].estimate), f"{source}: S2 ppo false loc"),
        ("vOneFalseLocStraightSTwo", pct(s2.loc[("straight", "false_localization")].estimate),
         f"{source}: S2 straight false loc"),
        ("vOneDistSuccessPpoSTwo", km(s2.loc[("ppo", "distance_mean_success")].estimate),
         f"{source}: S2 ppo distance (successes), km"),
        ("vOneDistSuccessStraightSTwo", km(s2.loc[("straight", "distance_mean_success")].estimate),
         f"{source}: S2 straight distance (successes), km"),
    ]


def change_index_v1(macros: list) -> None:
    source = "results_final/change_index/independent_covariate.csv"
    table = pd.read_csv(ROOT / source).set_index("method")
    macros += [
        ("vOneRhoColorMin", f"{table.spearman_color_success.min():.2f}", f"{source}: menor Spearman cor × sucesso"),
        ("vOneRhoColorMax", f"{table.spearman_color_success.max():.2f}", f"{source}: maior Spearman cor × sucesso"),
    ]
    quintiles = pd.read_csv(ROOT / "results_final/change_index/success_by_change_quintile.csv")
    taquari = quintiles[quintiles.location == "taquari"].set_index(["method", "quintile"])
    for method, label in (("ppo", "Ppo"), ("straight", "Straight")):
        for q in (1, 5):
            macros.append((f"vOneSuccessQ{'One' if q == 1 else 'Five'}{label}", pct(taquari.loc[(method, q)].success),
                           f"results_final/change_index/success_by_change_quintile.csv: taquari {method} Q{q}"))


def mechanism_v1(macros: list) -> None:
    source = "results_final/mechanism/m1.csv"
    table = pd.read_csv(ROOT / source)
    row = table[(table.location == "taquari_S2") & (table.measure == "visited_change")
                & (table.comparison == "a_ppo_false_minus_ppo_success")].iloc[0]
    macros += [
        ("vOneMOneChangeDiff", f"{row.estimate:.2f}", f"{source}: taquari, mudança visitada, falsa − sucesso"),
        ("vOneMOneChangeDiffLow", f"{row.ci_low:.2f}", f"{source}: IC"),
        ("vOneMOneChangeDiffHigh", f"{row.ci_high:.2f}", f"{source}: IC"),
        ("vOneMOneVerdict", str(row.m1_verdict), f"{source}: veredito pelo critério F3"),
    ]


def mo_appendix(macros: list) -> None:
    source = "results_mo_v1fix/verdicts_v2.json"
    verdicts = json.loads((ROOT / source).read_text())
    eum = pd.read_csv(ROOT / "results_mo_v1fix/eum.csv").set_index(["mode", "set"])
    macros += [
        ("moHFourVerdict", verdicts["H4"]["verdict"], f"{source}: H4"),
        ("moHSixVerdict", verdicts["H6"]["verdict"], f"{source}: H6"),
        ("moEumSampled", f"{eum.loc[('sampled', 'mo_ppo (3 presets)')].eum:.3f}", "results_mo_v1fix/eum.csv"),
        ("moEumPocSampled", f"{eum.loc[('sampled', 'ppo_poc (3 seeds)')].eum:.3f}", "results_mo_v1fix/eum.csv"),
    ]


def main() -> None:
    macros: list[tuple[str, str, str]] = []
    for builder in (taquari_v1, change_index_v1, mechanism_v1, mo_appendix):
        builder(macros)
    lines = ["% Gerado por paper/make_numbers.py a partir dos CSVs. NÃO EDITAR À MÃO."]
    for name, value, origin in macros:
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}  % {origin}")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"{len(macros)} macros em {OUT}")


if __name__ == "__main__":
    main()

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


VERDICT_EN = {"suportada": "supported", "não suportada": "not supported", "não avaliável": "not evaluable",
              "inconclusiva": "inconclusive", "contrária": "contrary", "parcial": "partial"}


def en(verdict: str) -> str:
    """Veredito em inglês para o texto do artigo (o CSV/JSON de origem fica em português)."""
    return VERDICT_EN[str(verdict)]


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
        ("vOneMOneVerdict", en(row.m1_verdict), f"{source}: veredito pelo critério F3"),
    ]


def mo_appendix(macros: list) -> None:
    source = "results_mo_v1fix/verdicts_v2.json"
    verdicts = json.loads((ROOT / source).read_text())
    eum = pd.read_csv(ROOT / "results_mo_v1fix/eum.csv").set_index(["mode", "set"])
    macros += [
        ("moHFourVerdict", en(verdicts["H4"]["verdict"]), f"{source}: H4"),
        ("moHSixVerdict", en(verdicts["H6"]["verdict"]), f"{source}: H6"),
        ("moHSevenVerdict", en(verdicts["H7"]["verdict"]), f"{source}: H7"),
        ("moEumSampled", f"{eum.loc[('sampled', 'mo_ppo (3 presets)')].eum:.3f}", "results_mo_v1fix/eum.csv"),
        ("moEumPocSampled", f"{eum.loc[('sampled', 'ppo_poc (3 seeds)')].eum:.3f}", "results_mo_v1fix/eum.csv"),
    ]


def taquari_v1_extra(macros: list) -> None:
    """S1 e as linhas do S2 que o texto v4 usa (aleatória, zigue-zague, gulosa)."""
    source = "paper/tables/v1_rq1_taquari.csv"
    table = pd.read_csv(ROOT / source)
    table = table[table.seed_label == "all"].set_index(["scenario", "method", "metric"])
    for scenario, tag in (("S1", "SOne"), ("S2", "STwo")):
        for method, label in (("ppo", "Ppo"), ("straight", "Straight"), ("random", "Random"),
                              ("lawnmower", "Lawnmower"), ("greedy_map", "Greedy")):
            row = table.loc[(scenario, method, "success")]
            name = f"vOneSucc{label}{tag}"
            if name not in {m[0] for m in macros}:
                macros.append((name, pct(row.estimate), f"{source}: {scenario} {method} success"))
        row = table.loc[(scenario, "ppo", "success")]
        macros += [(f"vOneDiffPpoStraight{tag}", f"{100 * row.diff_vs_straight:+.1f}", f"{source}: {scenario} Δ"),
                   (f"vOneDiffPpoStraight{tag}Low", pct(row.diff_ci_low), f"{source}: {scenario} IC"),
                   (f"vOneDiffPpoStraight{tag}High", pct(row.diff_ci_high), f"{source}: {scenario} IC")]


def naive_v1(macros: list) -> None:
    """IC do REPORT.md (episódios como independentes) e o Wilcoxon dos passos penalizados, S2, PPO × reta."""
    source = "results/paired_tests_test.csv"
    row = pd.read_csv(ROOT / source).query("scenario == 'S2' and method == 'ppo' and vs == 'straight'").iloc[0]
    macros += [("vOneNaiveDiffLow", pct(row.success_diff_lo), f"{source}: S2 ppo − straight, IC por episódio"),
               ("vOneNaiveDiffHigh", pct(row.success_diff_hi), f"{source}: idem"),
               ("vOneWilcoxonP", f"{row.wilcoxon_p:.1e}".replace("e-0", "\\times10^{-").replace("e-", "\\times10^{-") + "}",
                f"{source}: Wilcoxon, passos penalizados"),
               ("vOnePenStepsPpo", f"{row.pen_steps_ref:.1f}", f"{source}: mediana dos passos penalizados, PPO"),
               ("vOnePenStepsStraight", f"{row.pen_steps_other:.1f}", f"{source}: idem, reta")]


def poa_v1(macros: list) -> None:
    source = "paper/tables/v1_rq1_poa.csv"
    table = pd.read_csv(ROOT / source)
    table = table[(table.seed_label == "all") & (table.metric == "success")].set_index(["region", "scenario", "method"])
    for region, label in (("centro", "Center"), ("ilhas", "Islands")):
        ppo = table.loc[(region, "P1", "ppo_S2")]
        macros += [
            (f"vOnePoa{label}Ppo", pct(ppo.estimate), f"{source}: {region} P1 ppo_S2 success (zero-shot)"),
            (f"vOnePoa{label}Straight", pct(table.loc[(region, "P1", "straight")].estimate),
             f"{source}: {region} P1 straight"),
            (f"vOnePoa{label}DiffLow", pct(ppo.diff_ci_low), f"{source}: {region} P1 ppo_S2 − straight, IC"),
            (f"vOnePoa{label}DiffHigh", pct(ppo.diff_ci_high), f"{source}: {region} P1 ppo_S2 − straight, IC"),
        ]


def baselines_v3(macros: list) -> None:
    source = "paper/tables/v3_baselines.csv"
    table = pd.read_csv(ROOT / source)
    table = table[table.split == "test"].set_index(["region", "scenario", "method"])
    places = {("taquari", "S2"): "TaqSTwo", ("taquari", "S1"): "TaqSOne", ("canoas", "C1"): "Canoas",
              ("poa_ilhas", "P1"): "Islands", ("poa_centro", "P1"): "Center"}
    methods = {"random": "Random", "straight": "Straight", "straight_150": "StraightHigh", "lawnmower": "Lawnmower",
               "greedy_map": "Greedy"}
    for (region, scenario), place in places.items():
        for method, label in methods.items():
            row = table.loc[(region, scenario, method)]
            macros.append((f"vThree{place}{label}", pct(row.success), f"{source}: {region} {scenario} {method} success"))
            macros.append((f"vThree{place}{label}FalseLoc", pct(row.false_loc),
                           f"{source}: {region} {scenario} {method} false loc"))


def main() -> None:
    macros: list[tuple[str, str, str]] = []
    for builder in (taquari_v1, taquari_v1_extra, naive_v1, poa_v1, baselines_v3, change_index_v1, mechanism_v1, mo_appendix):
        builder(macros)
    lines = ["% Gerado por paper/make_numbers.py a partir dos CSVs. NÃO EDITAR À MÃO."]
    for name, value, origin in macros:
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}  % {origin}")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"{len(macros)} macros em {OUT}")


if __name__ == "__main__":
    main()

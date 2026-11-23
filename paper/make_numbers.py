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


E_LEVELS = {"L0": "LZero", "L1": "LOne", "L2": "LTwo", "L3": "LThree"}


def signed(value: float) -> str:
    return f"{100 * value:+.1f}"


def experiment_e(macros: list) -> None:
    """Experimento E (V46, V50–V53): curva, E3, Tab. III, gate (E5), escala do ruído e Canoas zero-shot (V51)."""
    import re

    source = "results_v3/E/curve.csv"
    curve = pd.read_csv(ROOT / source)
    curve["level"] = curve.regime.map(lambda name: re.search(r"L\d", name).group(0))
    for _, row in curve.iterrows():
        tag = E_LEVELS[row.level]
        macros += [(f"eRandTrain{tag}", pct(row.random_train_success), f"{source}: {row.regime}, aleatória no treino"),
                   (f"eGate{tag}", f"{int(row.gate_original_passed)}/{int(row.n_runs)}",
                    f"{source}: {row.regime}, aprovadas no gate (critério original)"),
                   (f"eGateAmended{tag}", f"{int(row.gate_amended_passed)}/{int(row.n_runs)}",
                    f"{source}: {row.regime}, gate emendado (V38)")]
        for label, key in (("Rand", "random"), ("Best", "best")):
            for mode, suffix in (("sampled", ""), ("argmax", "Argmax")):
                column = f"adv_{key}_{mode}"
                macros += [(f"eAdv{label}{tag}{suffix}", signed(row[column]), f"{source}: {row.regime} {column}"),
                           (f"eAdv{label}{tag}{suffix}Low", signed(row[f"{column}_low"]), f"{source}: IC"),
                           (f"eAdv{label}{tag}{suffix}High", signed(row[f"{column}_high"]), f"{source}: IC")]
    best = set(curve.best_heuristic)
    macros.append(("eBestName", "straight line at 150~m" if best == {"straight_150"} else "/".join(sorted(best)),
                   f"{source}: melhor heurística no treino, todos os níveis"))
    levels = json.loads((ROOT / "results_v3/E/noise_levels.json").read_text())["levels"]
    for level, info in levels.items():
        macros.append((f"eSigma{E_LEVELS[level]}", f"{info['sigma']:g}", "results_v3/E/noise_levels.json"))
    scans = pd.concat([pd.read_csv(ROOT / f"results_v3/E/{n}/scan.csv") for n in ("scan_noise_A", "scan_noise_A_ext")])
    scans = scans.set_index("sigma")
    lowest = scans.loc[scans.index.min()]
    macros += [("eTopOneLowSigma", pct(lowest.top1_100m), "scan_noise_A: top-1 a 100 m, menor σ"),
               ("eLowSigma", f"{scans.index.min():g}", "scan_noise_A: menor σ"),
               ("eTopOneLThree", f"{100 * scans.loc[levels['L3']['sigma']].top1_100m:.2f}",
                "scan_noise_A_ext: top-1 a 100 m em L3"),
               ("eMaxSigmaScanThree", f"{scans.loc[3.0].random_success * 100:.1f}",
                "scan_noise_A: aleatória no treino em σ = 3"),
               ("eNSigmas", str(len(scans)), "scan_noise_A + _ext: número de σ")]
    verdicts = json.loads((ROOT / "results_v3/E/verdicts_E.json").read_text())
    for key, suffix in (("E3", ""), ("E3_argmax", "Argmax")):
        e3 = verdicts[key]
        macros += [(f"eThree{suffix}", signed(e3["estimate"]), f"results_v3/E/verdicts_E.json: {key}"),
                   (f"eThree{suffix}Low", signed(e3["low"]), "idem, IC"),
                   (f"eThree{suffix}High", signed(e3["high"]), "idem, IC"),
                   (f"eThree{suffix}Verdict", en(e3["verdict"]), "idem, veredito")]
    table = pd.read_csv(ROOT / "results_v3/E/table3.csv")
    table["level"] = table.regime.map(lambda name: re.search(r"L\d", name).group(0))
    names = {"random": "Rand", "straight_150": "Best", "ppo@sampled": "Ppo", "ppo@argmax": "PpoArgmax"}
    for _, row in table[table.method.isin(names)].iterrows():
        tag, name = E_LEVELS[row.level], names[row.method]
        macros += [(f"eSucc{name}{tag}", pct(row.success), f"results_v3/E/table3.csv: {row.regime} {row.method}"),
                   (f"eFalseLoc{name}{tag}", pct(row.false_loc), "idem, localização falsa"),
                   (f"eTime{name}{tag}", f"{row.flight_time_success_s:.0f}", "idem, tempo de voo nos sucessos (s)")]
    # Heurísticas a 100 m por nível (a PPO do E voa a 100 m; a melhor heurística, a 150 m).
    evals = {"L0": "E_eval_taquari_L0", "L1": "E_eval_synth_L1", "L2": "E_eval_synth_L2", "L3": "E_eval_synth_L3"}
    for level, job in evals.items():
        summary = pd.read_csv(ROOT / f"results_v3/E/{job}/summary.csv")
        test = summary[(summary.split == "test") & (summary["mode"] == "—")].set_index("method")
        for method, name in (("straight", "Straight"), ("lawnmower", "Lawnmower"), ("greedy_map", "Greedy")):
            macros.append((f"eSucc{name}{E_LEVELS[level]}", pct(test.loc[method].success),
                           f"results_v3/E/{job}/summary.csv: teste, {method}"))
    # Matriz v3 (V44/V45): aprovadas no gate entre as 30 execuções mx_A.
    runs = sorted((ROOT / "results_v3/E").glob("mx_A_*/DONE.json"))
    gates = [json.loads(path.read_text())["metrics"]["gate"] for path in runs]
    macros += [("vThreeMatrixRuns", str(len(gates)), "results_v3/E/mx_A_*/DONE.json"),
               ("vThreeMatrixPassed", str(sum(bool(g["passed"]) for g in gates)), "idem, gate (original ou emendado)"),
               ("vThreeMatrixAdvMin", signed(min(g["advantage"] for g in gates)), "idem, menor vantagem no gate"),
               ("vThreeMatrixAdvMax", signed(max(g["advantage"] for g in gates)), "idem, maior vantagem no gate")]
    diag = pd.read_csv(ROOT / "results_v3/E/diag_canoas_split/split_diagnostic.csv")
    whole = diag[(diag.side == "all") & (diag.min_valid == 0.7)].n_windows.iloc[0]
    macros.append(("eCanoasWindows", str(int(whole)), "results_v3/E/diag_canoas_split/split_diagnostic.csv: limiar 0,7"))
    # Canoas zero-shot (V51, descritivo).
    source = "results_v3/E/canoas_zeroshot.csv"
    zs = pd.read_csv(ROOT / source)
    labels = {"random": "Random", "greedy_map": "Greedy", "lawnmower": "Lawnmower", "straight": "Straight",
              "straight_150": "AltStraight"}  # sem "High" no fim: colidiria com o limite superior do IC
    for _, row in zs[zs["mode"] == "sampled"].iterrows():
        tag = labels[row.over]
        macros += [(f"eZsDiff{tag}", signed(row.advantage), f"{source}: PPO − {row.over}"),
                   (f"eZsDiff{tag}Low", signed(row.low), "idem, IC"),
                   (f"eZsDiff{tag}High", signed(row.high), "idem, IC")]
    summary = zs[zs["mode"] == "summary"].set_index("over")
    macros += [("eZsPpo", pct(summary.loc["ppo@sampled"].advantage), f"{source}: sucesso da PPO amostrada"),
               ("eZsPpoLow", pct(summary.loc["ppo@sampled"].low), "idem, IC"),
               ("eZsPpoHigh", pct(summary.loc["ppo@sampled"].high), "idem, IC")]


def main() -> None:
    macros: list[tuple[str, str, str]] = []
    builders = (taquari_v1, taquari_v1_extra, naive_v1, poa_v1, baselines_v3, change_index_v1, mechanism_v1,
                mo_appendix,
                experiment_e)
    for builder in builders:
        builder(macros)
    lines = ["% Gerado por paper/make_numbers.py a partir dos CSVs. NÃO EDITAR À MÃO."]
    for name, value, origin in macros:
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}  % {origin}")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"{len(macros)} macros em {OUT}")


if __name__ == "__main__":
    main()

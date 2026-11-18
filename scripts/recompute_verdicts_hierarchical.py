"""P5.5: vereditos H1–H3 (PoC) e HP1 (Porto Alegre) recalculados com ICs hierárquicos.

Reusa as funções de critério originais (`dal.report.verdicts`, `dal.poa_report.verdict_hp1`) sem
alterá-las: só as colunas de IC da tabela de testes pareados são trocadas pelos ICs do bootstrap
hierárquico (seeds → janelas). Os vereditos originais são recalculados ao lado, para conferência.
HP2 e HP3 usam contrastes entre regiões e ficam com o IC original (nota no JSON).

Saída: results_final/stats/{verdicts.json, paired_tests_taquari_*.csv, paired_tests_poa.csv}
Uso: .venv/bin/python scripts/recompute_verdicts_hierarchical.py [--n-boot 2000]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from dal import poa_report, report
from dal.config import ROOT, load_config
from dal.evaluation.statistics import MethodMatrix, hierarchical_bootstrap, percentile_ci, reconstruct_episode_window

OUT = ROOT / "results_final" / "stats"


def _matrix(frame: pd.DataFrame, column: str, episodes: list[int]) -> MethodMatrix:
    table = frame.pivot_table(index="model_seed", columns="episode", values=column, aggfunc="mean")
    return MethodMatrix(table.reindex(columns=episodes).to_numpy(dtype=float))


def hierarchical_diff_ci(group: pd.DataFrame, ref: str, other: str, column: str, clusters: np.ndarray,
                         episodes: list[int], n_boot: int) -> tuple[float, float]:
    """IC hierárquico pareado de média(ref) − média(other)."""
    matrices = {name: _matrix(group[group.method == name].assign(**{column: group[column].astype(float)}),
                              column, episodes) for name in (ref, other)}
    replicates = hierarchical_bootstrap(matrices, clusters, "mean", n_boot, seed=1)
    return percentile_ci(replicates[ref] - replicates[other])


def taquari_windows(split: str) -> dict[str, np.ndarray]:
    """Janela de cada episódio, por cenário, reconstruída a partir da semente."""
    from dal.env import ScenarioData

    config = load_config("configs/default.yaml")
    out = {}
    for scenario in ("S0", "S1", "S2"):
        data = ScenarioData.load("hand", scenario, config)
        windows = data.windows(split, int(config["filter"]["window"]), config["filter"]["min_window_valid"])
        out[scenario] = windows
    return out


def taquari_tests(episodes: pd.DataFrame, split: str, max_steps: int, n_boot: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Testes pareados originais e a versão com ICs hierárquicos de Δsucesso."""
    original = report.paired_tests(episodes, max_steps, n_boot)
    hierarchical = original.copy()
    seed0 = int(load_config("configs/default.yaml")["eval"]["seed"])
    windows = taquari_windows(split)
    for index, row in hierarchical.iterrows():
        group = episodes[episodes.scenario == row.scenario]
        episode_ids = sorted(group.episode.unique())
        clusters = np.array(["%d_%d" % reconstruct_episode_window(windows[row.scenario], seed0 + e)
                             for e in episode_ids])
        low, high = hierarchical_diff_ci(group, "ppo", row.vs, "success", clusters, episode_ids, n_boot)
        hierarchical.loc[index, ["success_diff_lo", "success_diff_hi"]] = [low, high]
    return original, hierarchical


def poa_tests(max_steps: int, n_boot: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Testes pareados de Porto Alegre (P1, zero-shot) com ICs hierárquicos de Δsucesso e Δdistância."""
    episodes = pd.read_csv(ROOT / "results_poa/episodes.csv")
    episodes = episodes[(episodes.scenario == "P1") & (episodes.variant == "zeroshot")]
    original = poa_report.paired_tests(episodes, max_steps, n_boot)
    hierarchical = original.copy()
    for index, row in hierarchical.iterrows():
        group = episodes[episodes.region == row.region]
        first = group.drop_duplicates("episode").sort_values("episode")
        episode_ids = first.episode.tolist()
        clusters = (first.window_r0.astype(str) + "_" + first.window_c0.astype(str)).to_numpy()
        for column, prefix in (("success", "success_diff"), ("distance_m", "distance_diff")):
            low, high = hierarchical_diff_ci(group, row.ref, row.vs, column, clusters, episode_ids, n_boot)
            hierarchical.loc[index, [f"{prefix}_lo", f"{prefix}_hi"]] = [low, high]
    return original, hierarchical


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    config = load_config("configs/default.yaml")
    max_steps = int(config["filter"]["max_steps"])

    test = pd.read_csv(ROOT / "results/episodes_test.csv")
    train = pd.read_csv(ROOT / "results/episodes_train.csv")
    summary_test, summary_train = report.summary_table(test, args.n_boot), report.summary_table(train, args.n_boot)
    tests_test, tests_test_h = taquari_tests(test, "test", max_steps, args.n_boot)
    tests_train, tests_train_h = taquari_tests(train, "train", max_steps, args.n_boot)
    for name, frame in (("test", tests_test_h), ("train", tests_train_h)):
        frame.to_csv(OUT / f"paired_tests_taquari_{name}_hierarchical.csv", index=False)
    original = report.verdicts(summary_test, tests_test, summary_train, tests_train)
    hierarchical = report.verdicts(summary_test, tests_test_h, summary_train, tests_train_h)

    poa_original, poa_hier = poa_tests(int(load_config("configs_poa/poa.yaml")["filter"]["max_steps"]), args.n_boot)
    poa_hier.to_csv(OUT / "paired_tests_poa_P1_hierarchical.csv", index=False)
    reference = pd.read_csv(ROOT / "results_poa/taquari_reference.csv")
    hp1_original, _ = poa_report.verdict_hp1(poa_original, reference)
    hp1_hier, hp1_lines = poa_report.verdict_hp1(poa_hier, reference)

    result = {
        "H1": {"original_ci": original["H1"][0], "hierarchical_ci": hierarchical["H1"][0]},
        "H2": {"original_ci": original["H2"][0], "hierarchical_ci": hierarchical["H2"][0]},
        "H3": {"original_ci": original["H3"][0], "hierarchical_ci": hierarchical["H3"][0]},
        "HP1": {"original_ci": hp1_original, "hierarchical_ci": hp1_hier},
        "HP2_HP3": "IC original mantido: contrastes entre regiões com bootstrap independente (ver docstring)",
        "details_hierarchical": {k: v[1] for k, v in hierarchical.items()} | {"HP1": "\n".join(hp1_lines)},
        "recorded_in_reports": json.loads((ROOT / "results/hypotheses.json").read_text())
        | json.loads((ROOT / "results_poa/hypotheses.json").read_text()),
    }
    (OUT / "verdicts.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    for key in ("H1", "H2", "H3", "HP1"):
        print(f"{key}: original={result[key]['original_ci']} | hierárquico={result[key]['hierarchical_ci']}")
    print("gravados nos relatórios:", result["recorded_in_reports"])


if __name__ == "__main__":
    main()

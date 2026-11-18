"""Tabelas canônicas do artigo (P6 do plano v2) → paper/tables/*.csv.

Lê só CSVs de episódios versionados e calcula, para cada método, as métricas do glossário
(`docs/PLANO_DESENVOLVIMENTO.md` §2.6) com IC 95% por bootstrap hierárquico (seeds → janelas),
resultado por seed e diferença pareada contra a reta. Toda linha tem source_file, protocol,
n_episodes, seeds, region e scenario.

Tabelas:
- rq1_taquari.csv: PoC, teste (sul), S0–S2, 1000 episódios pareados (results/episodes_test.csv);
- rq1_poa.csv: Porto Alegre, P0–P2, variante zero-shot, centro e ilhas, 500 episódios (results_poa/episodes.csv);
- rq3_mo_v1.csv: MO v1 e referências nos dois modos, Taquari S2, 300 episódios (results_mo_v1fix/), se existir.

Uso: .venv/bin/python scripts/make_canonical_tables.py [--n-boot 2000]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dal.config import ROOT, load_config
from dal.evaluation.statistics import (
    MethodMatrix,
    hierarchical_bootstrap,
    percentile_ci,
    point_estimate,
    reconstruct_episode_window,
)

MAX_STEPS = 80
REFERENCE_METHOD = "straight"
# métrica -> (coluna derivada, estatística, filtro de episódios)
METRICS = {
    "success": ("success", "mean", None),
    "false_localization": ("false_loc", "mean", None),
    "timeout": ("timeout", "mean", None),
    "steps_success_median": ("steps", "median", "success"),
    "steps_penalized_median": ("steps_penalized", "median", None),
    "distance_mean_all": ("distance_m", "mean", None),
    "distance_mean_success": ("distance_m", "mean", "success"),
}


def add_derived_columns(episodes: pd.DataFrame) -> pd.DataFrame:
    """Colunas do glossário: timeout e passos penalizados (falha = max_steps + 1)."""
    episodes = episodes.copy()
    episodes["success"] = episodes["success"].astype(bool)
    episodes["false_loc"] = episodes["false_loc"].astype(bool)
    episodes["timeout"] = ~episodes["success"] & ~episodes["false_loc"]
    episodes["steps_penalized"] = np.where(episodes["success"], episodes["steps"], MAX_STEPS + 1)
    return episodes


def metric_matrix(frame: pd.DataFrame, column: str, only: str | None, episodes: list[int],
                  seed_column: str) -> MethodMatrix:
    """Matriz (seed × episódio) de uma métrica; episódios fora do filtro viram NaN."""
    values = frame[column].astype(float).where(frame[only].astype(bool)) if only else frame[column].astype(float)
    table = frame.assign(_v=values).pivot_table(index=seed_column, columns="episode", values="_v", aggfunc="mean",
                                                dropna=False)
    return MethodMatrix(table.reindex(columns=episodes).to_numpy(dtype=float))


def summarize_group(group: pd.DataFrame, clusters: np.ndarray, episodes: list[int], context: dict,
                    n_boot: int, seed_column: str) -> list[dict]:
    """Linhas (agregado e por seed) de um grupo (cenário/região/modo), com diferença pareada vs a reta."""
    methods = sorted(group["method"].unique())
    rows = []
    for metric, (column, statistic, only) in METRICS.items():
        matrices = {m: metric_matrix(group[group.method == m], column, only, episodes, seed_column) for m in methods}
        replicates = hierarchical_bootstrap(matrices, clusters, statistic, n_boot, seed=0)
        for method in methods:
            matrix = matrices[method]
            low, high = percentile_ci(replicates[method])
            row = {**context, "method": method, "metric": metric, "seed_label": "all",
                   "estimate": point_estimate(matrix, statistic), "ci_low": low, "ci_high": high,
                   "n_episodes": int(np.isfinite(matrix.values).sum()), "seeds": matrix.n_seeds}
            if REFERENCE_METHOD in matrices and method != REFERENCE_METHOD:
                diff = replicates[method] - replicates[REFERENCE_METHOD]
                row["diff_vs_straight"] = row["estimate"] - point_estimate(matrices[REFERENCE_METHOD], statistic)
                row["diff_ci_low"], row["diff_ci_high"] = percentile_ci(diff)
            rows.append(row)
            if matrix.n_seeds > 1:
                seed_labels = sorted(group.loc[group.method == method, seed_column].unique())
                for k, label in enumerate(seed_labels):
                    single = MethodMatrix(matrix.values[k : k + 1])
                    rows.append({**context, "method": method, "metric": metric, "seed_label": str(label),
                                 "estimate": point_estimate(single, statistic),
                                 "n_episodes": int(np.isfinite(single.values).sum()), "seeds": 1})
    return rows


def taquari_table(n_boot: int) -> pd.DataFrame:
    """PoC no teste (sul): janelas reconstruídas a partir da semente do episódio."""
    from dal.env import ScenarioData

    source = "results/episodes_test.csv"
    config = load_config("configs/default.yaml")
    episodes_all = add_derived_columns(pd.read_csv(ROOT / source))
    rows = []
    for scenario, group in episodes_all.groupby("scenario"):
        data = ScenarioData.load("hand", scenario, config)
        windows = data.windows("test", int(config["filter"]["window"]), config["filter"]["min_window_valid"])
        episodes = sorted(group["episode"].unique())
        clusters = np.array(["%d_%d" % reconstruct_episode_window(windows, int(config["eval"]["seed"]) + e)
                             for e in episodes])
        context = {"source_file": source, "protocol": "argmax; teste sul; seeds pareados 2024+i",
                   "region": "taquari_sul", "scenario": scenario, "mode": "argmax", "n_windows": len(set(clusters))}
        rows += summarize_group(group, clusters, episodes, context, n_boot, "model_seed")
    return pd.DataFrame(rows)


def poa_table(n_boot: int) -> pd.DataFrame:
    """Porto Alegre, variante zero-shot (modelos do Taquari sem retreino)."""
    source = "results_poa/episodes.csv"
    episodes_all = add_derived_columns(pd.read_csv(ROOT / source))
    episodes_all = episodes_all[episodes_all["variant"] == "zeroshot"]
    rows = []
    for (scenario, region), group in episodes_all.groupby(["scenario", "region"]):
        first = group.drop_duplicates("episode").sort_values("episode")
        episodes = first["episode"].tolist()
        clusters = (first["window_r0"].astype(str) + "_" + first["window_c0"].astype(str)).to_numpy()
        context = {"source_file": source, "protocol": "argmax; zero-shot; seeds pareados 2024+i",
                   "region": region, "scenario": scenario, "mode": "argmax", "n_windows": len(set(clusters))}
        rows += summarize_group(group, clusters, episodes, context, n_boot, "model_seed")
    return pd.DataFrame(rows)


def mo_v1_table(n_boot: int) -> pd.DataFrame | None:
    """MO v1 e referências nos dois modos (Taquari S2), a partir da avaliação v1fix."""
    source = "results_mo_v1fix/episodes_presets.csv"
    if not (ROOT / source).exists():
        return None
    episodes_all = add_derived_columns(pd.read_csv(ROOT / source))
    # Uma "seed" por método aqui (seeds da PoC viram métodos separados e também um agregado).
    episodes_all["model_seed"] = 0
    poc = episodes_all[episodes_all.method.str.startswith("ppo_poc_s")].copy()
    poc["model_seed"] = poc["method"].str[-1].astype(int)
    poc["method"] = "ppo_poc"
    episodes_all["method"] = episodes_all["method"] + np.where(episodes_all["preset"].fillna("") != "",
                                                               "@" + episodes_all["preset"].fillna(""), "")
    straight = episodes_all[episodes_all.method.str.startswith("straight")]
    rows = []
    for mode in ("argmax", "sampled"):
        group = pd.concat([episodes_all[episodes_all["mode"] == mode], poc[poc["mode"] == mode],
                           straight.assign(mode=mode)], ignore_index=True).drop_duplicates(
            ["method", "model_seed", "episode"])
        group.loc[group.method.str.startswith("straight"), "method"] = "straight"
        first = group.drop_duplicates("episode").sort_values("episode")
        episodes = first["episode"].tolist()
        clusters = (first["window_r0"].astype(str) + "_" + first["window_c0"].astype(str)).to_numpy()
        context = {"source_file": source, "protocol": f"{mode}; teste sul S2; 300 episódios; seeds 2024+i e torch",
                   "region": "taquari_sul", "scenario": "S2", "mode": mode, "n_windows": len(set(clusters))}
        rows += summarize_group(group, clusters, episodes, context, n_boot, "model_seed")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--only", choices=["taquari", "poa", "mo"], help="gera só uma tabela")
    args = parser.parse_args()

    output_dir = ROOT / "paper" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    builders = {"taquari": ("rq1_taquari.csv", taquari_table), "poa": ("rq1_poa.csv", poa_table),
                "mo": ("rq3_mo_v1.csv", mo_v1_table)}
    for key, (filename, builder) in builders.items():
        if args.only and key != args.only:
            continue
        table = builder(args.n_boot)
        if table is None:
            print(f"[tables] {filename}: fonte ausente, pulado")
            continue
        table.to_csv(output_dir / filename, index=False)
        print(f"[tables] {filename}: {len(table)} linhas")


if __name__ == "__main__":
    main()

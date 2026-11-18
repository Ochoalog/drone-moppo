"""Estatística e vereditos da v3, gerados por código a partir dos episódios (nunca escritos à mão).

- rliable: IQM do sucesso entre seeds e probabilidade de melhora P(X > Y), com bootstrap estratificado.
- Diferenças pareadas: bootstrap hierárquico (seeds → janelas) de `dal.evaluation.statistics`.
- Critérios: `DECISIONS_V3.md` §2 (enquanto não commitados lá, os do rascunho
  `docs/DECISIONS_V3_hipoteses_RASCUNHO.md`).

Formato dos episódios: o de `dal.v3.jobs.evaluate_job` (method, model_seed, region, scenario, fold, episode,
window_r0, window_c0, success, false_loc, flight_time_s, ...).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dal.evaluation.statistics import MethodMatrix, hierarchical_bootstrap, percentile_ci

N_BOOT = 2000
NON_INFERIORITY_MARGIN = -0.03
H2_SUCCESS_MARGIN = -0.02


def _rliable_metrics():
    """Métricas do rliable (IQM, probabilidade de melhora). O bootstrap é o nosso (ver `stratified_bootstrap`)."""
    from rliable import metrics

    return metrics


def stratified_bootstrap(func, score_arrays: tuple[np.ndarray, ...], reps: int = N_BOOT,
                         seed: int = 0) -> tuple[float, float, float]:
    """Estimativa e IC 95% com bootstrap estratificado (execuções reamostradas dentro de cada tarefa).

    É o mesmo esquema do `rliable.library.get_interval_estimates`, reimplementado porque essa função depende
    do pacote `arch`, incompatível com o pandas 3 neste ambiente (DECISIONS_V3.md, V19).
    """
    rng = np.random.default_rng(seed)
    point = float(np.ravel(func(*score_arrays))[0])
    replicates = np.empty(reps)
    for b in range(reps):
        resampled = []
        for scores in score_arrays:
            runs, tasks = scores.shape
            index = rng.integers(runs, size=(runs, tasks))
            resampled.append(np.take_along_axis(scores, index, axis=0))
        replicates[b] = float(np.ravel(func(*resampled))[0])
    low, high = np.percentile(replicates, [2.5, 97.5])
    return point, float(low), float(high)


@dataclass
class PairedDiff:
    estimate: float
    low: float
    high: float


def method_matrix(episodes: pd.DataFrame, method: str, column: str, episode_ids: list[int],
                  only_success: bool = False) -> MethodMatrix:
    """(seeds × episódios) de uma métrica; com `only_success`, episódios sem sucesso viram NaN."""
    part = episodes[episodes.method == method]
    values = part[column].astype(float)
    if only_success:
        values = values.where(part.success.astype(bool))
    table = part.assign(_v=values).pivot_table(index="model_seed", columns="episode", values="_v", aggfunc="mean",
                                               dropna=False)
    return MethodMatrix(table.reindex(columns=episode_ids).to_numpy(float))


def paired_difference(episodes: pd.DataFrame, method_a: str, method_b: str, column: str,
                      only_success: bool = False, seed: int = 0) -> PairedDiff:
    """média(a) − média(b) nos mesmos episódios, com IC hierárquico (seeds → janelas)."""
    first = episodes.drop_duplicates("episode").sort_values("episode")
    episode_ids = first.episode.tolist()
    clusters = (first.window_r0.astype(str) + "_" + first.window_c0.astype(str)).to_numpy()
    matrices = {m: method_matrix(episodes, m, column, episode_ids, only_success) for m in (method_a, method_b)}
    replicates = hierarchical_bootstrap(matrices, clusters, "mean", N_BOOT, seed)
    estimate = float(np.nanmean(matrices[method_a].values) - np.nanmean(matrices[method_b].values))
    low, high = percentile_ci(replicates[method_a] - replicates[method_b])
    return PairedDiff(estimate, low, high)


def run_scores(episodes: pd.DataFrame, method: str, metric: str = "success") -> np.ndarray:
    """Matriz rliable (execuções × tarefas) com a taxa por seed; tarefa = (região, cenário)."""
    part = episodes[episodes.method == method]
    table = part.groupby(["model_seed", "region", "scenario"])[metric].mean().unstack(["region", "scenario"])
    return table.to_numpy(float)


def iqm_with_ci(scores: dict[str, np.ndarray], reps: int = N_BOOT) -> pd.DataFrame:
    """IQM entre execuções (rliable) e IC do bootstrap estratificado."""
    metrics = _rliable_metrics()
    rows = []
    for method, array in scores.items():
        point, low, high = stratified_bootstrap(metrics.aggregate_iqm, (array,), reps)
        rows.append({"method": method, "iqm": point, "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


def probability_of_improvement(scores_x: np.ndarray, scores_y: np.ndarray, reps: int = N_BOOT) -> PairedDiff:
    """P(X > Y) do rliable, com IC do bootstrap estratificado."""
    metrics = _rliable_metrics()
    return PairedDiff(*stratified_bootstrap(metrics.probability_of_improvement, (scores_x, scores_y), reps))


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """p-valores ajustados por Holm."""
    names = sorted(p_values, key=p_values.get)
    adjusted, running = {}, 0.0
    for rank, name in enumerate(names):
        running = max(running, (len(names) - rank) * p_values[name])
        adjusted[name] = min(1.0, running)
    return adjusted


# ============================================================================ hipóteses
def verdict_h2(episodes_by_place: dict[str, pd.DataFrame], methods=("straight", "ppo_ca")) -> dict:
    """Filtro CA reduz a localização falsa (CA − padrão < 0) sem perder mais de 2 p.p. de sucesso.

    Para a reta, compara `straight_ca` × `straight`; para a PPO-CA, a mesma política com filtro CA × padrão
    (`ppo_ca` × `ppo_ca_stdfilter` na avaliação).
    """
    pairs = {"straight": ("straight_ca", "straight"), "ppo_ca": ("ppo_ca", "ppo_ca_stdfilter")}
    rows = []
    for place, episodes in episodes_by_place.items():
        for method in methods:
            ca, standard = pairs[method]
            false_loc = paired_difference(episodes, ca, standard, "false_loc")
            success = paired_difference(episodes, ca, standard, "success")
            rows.append({"place": place, "method": method, "false_loc_diff": false_loc.estimate,
                         "false_loc_low": false_loc.low, "false_loc_high": false_loc.high,
                         "success_diff": success.estimate, "success_low": success.low,
                         "holds": false_loc.high < 0 and success.low >= H2_SUCCESS_MARGIN,
                         "contradicts": false_loc.low >= 0})
    table = pd.DataFrame(rows)
    verdict = ("suportada" if table.holds.all() else "não suportada" if table.contradicts.any() else "inconclusiva")
    return {"verdict": verdict, "table": table}


def verdict_h3_for_place(episodes: pd.DataFrame, policy: str, heuristic: str) -> dict:
    """Superioridade (P(melhora) > 0,5 com IC) ou não inferioridade com eficiência (tempo de voo)."""
    improvement = probability_of_improvement(run_scores(episodes, policy), run_scores(episodes, heuristic))
    success = paired_difference(episodes, policy, heuristic, "success")
    time = paired_difference(episodes, policy, heuristic, "flight_time_s", only_success=True)
    superiority = improvement.low > 0.5
    non_inferior_efficient = success.low >= NON_INFERIORITY_MARGIN and time.high < 0
    return {"prob_improvement": improvement.estimate, "prob_low": improvement.low, "prob_high": improvement.high,
            "success_diff": success.estimate, "success_low": success.low, "success_high": success.high,
            "time_diff_s": time.estimate, "time_low": time.low, "time_high": time.high,
            "superiority": superiority, "non_inferior_efficient": non_inferior_efficient,
            "supported": superiority or non_inferior_efficient,
            "mode": "superioridade" if superiority else "não inferioridade + eficiência" if non_inferior_efficient
            else "—"}


def verdict_h5(h3_by_place: dict[str, dict], zero_shot_places=("poa_ilhas", "poa_centro", "third_area")) -> dict:
    available = [p for p in zero_shot_places if p in h3_by_place]
    supported = [p for p in available if h3_by_place[p]["supported"]]
    return {"verdict": "suportada" if len(supported) >= 2 else "não suportada" if len(available) >= 2 else
            "não avaliável", "supported_places": supported, "evaluated_places": available}

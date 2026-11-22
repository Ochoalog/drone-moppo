"""Análise do experimento E (DECISIONS_V3.md V46): E1–E5, gerados por código a partir dos episódios.

Regime = (local ou nível de ruído). Para cada regime há episódios de teste e de treino com as heurísticas
(modo "—") e a PPO (modos `sampled`, o principal, e `argmax`). A melhor heurística sai do **treino**.
Diferenças pareadas: bootstrap hierárquico seeds → janelas (`dal.evaluation.statistics`), B = 2000.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dal.evaluation.statistics import hierarchical_bootstrap, percentile_ci
from dal.v3.evaluation.verdicts import N_BOOT, PairedDiff, method_matrix, paired_difference

HEURISTICS = ("straight", "lawnmower", "random", "greedy_map", "straight_150")
E1_MIN_ADVANTAGE = 0.05
MAIN_MODE = "sampled"


def with_method_key(episodes: pd.DataFrame) -> pd.DataFrame:
    """Coluna `key`: o nome da heurística, ou `<política>@<modo>` para as políticas."""
    mode = episodes["mode"].fillna("—").astype(str)
    key = np.where(mode.isin(["—", "", "nan"]), episodes.method, episodes.method + "@" + mode)
    return episodes.assign(key=key, method=key)


def best_heuristic(train: pd.DataFrame, candidates=HEURISTICS) -> str:
    """Maior sucesso no treino; empate → menor localização falsa; depois, ordem alfabética (V46)."""
    part = train[train.method.isin(candidates)]
    table = part.groupby("method").agg(success=("success", "mean"), false_loc=("false_loc", "mean")).reset_index()
    table["success"] = table.success.round(9)
    return str(table.sort_values(["success", "false_loc", "method"], ascending=[False, True, True]).method.iloc[0])


def classify(diff: PairedDiff, threshold: float = 0.0) -> str:
    """"suportada" se o critério vale; "contrária" se o IC está inteiro abaixo de 0; senão "inconclusiva"."""
    if diff.low > 0 and diff.estimate >= threshold:
        return "suportada"
    if diff.high < 0:
        return "contrária"
    return "inconclusiva"


@dataclass
class Regime:
    name: str
    test: pd.DataFrame  # episódios de teste com a coluna `method` já no formato de `with_method_key`
    train: pd.DataFrame
    policy: str = "ppo"

    @property
    def best(self) -> str:
        return best_heuristic(self.train)

    def policy_key(self, mode: str = MAIN_MODE) -> str:
        return f"{self.policy}@{mode}"

    def random_train_success(self) -> float:
        return float(self.train.loc[self.train.method == "random", "success"].mean())


def advantage(regime: Regime, over: str, mode: str = MAIN_MODE, column: str = "success", seed: int = 0,
              only_success: bool = False) -> PairedDiff:
    return paired_difference(regime.test, regime.policy_key(mode), over, column, only_success, seed)


def difference_of_independent(a: PairedDiff, b: PairedDiff) -> PairedDiff:
    """a − b com réplicas independentes (regimes em locais diferentes, janelas diferentes)."""
    replicates = a.replicates - b.replicates
    low, high = percentile_ci(replicates)
    return PairedDiff(a.estimate - b.estimate, low, high, replicates)


def difference_of_paired(high_regime: Regime, low_regime: Regime, mode: str = MAIN_MODE, seed: int = 0) -> PairedDiff:
    """Δ(alto) − Δ(baixo) com bootstrap conjunto: as mesmas janelas reamostradas nos dois níveis (E3).

    Exige os mesmos episódios (semente → janela) nos dois regimes, o que vale para os níveis de ruído do Taquari.
    """
    frames = []
    for tag, regime in (("hi", high_regime), ("lo", low_regime)):
        keep = regime.test[regime.test.method.isin([regime.policy_key(mode), regime.best])]
        renamed = keep.assign(method=np.where(keep.method == regime.best, f"{tag}|best", f"{tag}|policy"))
        frames.append(renamed)
    joint = pd.concat(frames, ignore_index=True)
    windows = joint.groupby("episode")[["window_r0", "window_c0"]].nunique()
    if (windows > 1).any().any():
        raise ValueError("os regimes não têm os mesmos episódios: use difference_of_independent")
    first = joint.drop_duplicates("episode").sort_values("episode")
    episode_ids = first.episode.tolist()
    clusters = (first.window_r0.astype(str) + "_" + first.window_c0.astype(str)).to_numpy()
    names = ["hi|policy", "hi|best", "lo|policy", "lo|best"]
    matrices = {m: method_matrix(joint, m, "success", episode_ids) for m in names}
    reps = hierarchical_bootstrap(matrices, clusters, "mean", N_BOOT, seed)
    replicates = (reps["hi|policy"] - reps["hi|best"]) - (reps["lo|policy"] - reps["lo|best"])
    means = {m: float(np.nanmean(matrices[m].values)) for m in names}
    estimate = (means["hi|policy"] - means["hi|best"]) - (means["lo|policy"] - means["lo|best"])
    low, high = percentile_ci(replicates)
    return PairedDiff(estimate, low, high, replicates)


def method_summary(regime: Regime, method: str, seed: int = 0) -> dict:
    """Sucesso, localização falsa e tempo de voo nos sucessos de um método, com IC hierárquico (E4)."""
    part = regime.test
    first = part.drop_duplicates("episode").sort_values("episode")
    episode_ids = first.episode.tolist()
    clusters = (first.window_r0.astype(str) + "_" + first.window_c0.astype(str)).to_numpy()
    out = {"regime": regime.name, "method": method,
           "n_episodes": int(part.loc[part.method == method, "episode"].nunique()),
           "n_seeds": int(part.loc[part.method == method, "model_seed"].nunique())}
    for column, only_success in (("success", False), ("false_loc", False), ("flight_time_s", True)):
        matrix = method_matrix(part, method, column, episode_ids, only_success)
        reps = hierarchical_bootstrap({method: matrix}, clusters, "mean", N_BOOT, seed)[method]
        low, high = percentile_ci(reps)
        name = "flight_time_success_s" if only_success else column
        out.update({name: float(np.nanmean(matrix.values)), f"{name}_low": low, f"{name}_high": high})
    return out


def gate_rate(gates: list[dict]) -> dict:
    """E5: fração de execuções aprovadas no gate (critério original e emendado)."""
    original = [g.get("advantage", -1.0) >= g.get("threshold", 0.05) for g in gates]
    amended = [bool(g.get("passed", False)) or o for g, o in zip(gates, original, strict=True)]
    n = len(gates)
    return {"n_runs": n, "gate_original_passed": int(sum(original)), "gate_amended_passed": int(sum(amended)),
            "gate_original_rate": sum(original) / n if n else np.nan,
            "gate_amended_rate": sum(amended) / n if n else np.nan}

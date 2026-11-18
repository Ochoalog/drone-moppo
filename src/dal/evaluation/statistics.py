"""Bootstrap hierárquico (seeds → janelas) e reconstrução da janela de cada episódio (P5 do plano v2).

Por que hierárquico: os ICs antigos reamostravam episódios como se fossem independentes. Isso ignora
(i) a variação entre seeds de treino da política (7,3 p.p. em S2) e (ii) a dependência espacial:
episódios na mesma janela compartilham o mapa. Aqui, cada réplica:
1. reamostra **janelas** com reposição (todos os episódios da janela saem juntos). As janelas são as
   mesmas para todos os métodos, o que mantém o pareamento;
2. para cada método com ≥ 2 seeds, reamostra **seeds** com reposição.

A função antiga (`dal.mo_eval.boot_ci`) continua existindo; os vereditos novos são reportados ao lado.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_BOOTSTRAP_SAMPLES = 2000
CONFIDENCE = 0.95


def reconstruct_episode_window(windows: np.ndarray, seed: int) -> tuple[int, int]:
    """Janela (r0, c0) sorteada por `Episode.reset(seed=seed)`, sem rodar o episódio.

    Reproduz a ordem dos sorteios de `dal.env.Episode.reset`: o primeiro inteiro alimenta o gerador
    da inicialização (janela e partida), e a janela é o primeiro sorteio desse gerador.
    """
    rng = np.random.default_rng(seed)
    init_rng = np.random.default_rng(rng.integers(2**63))
    r0, c0 = windows[init_rng.integers(len(windows))]
    return int(r0), int(c0)


def window_ids(frame: pd.DataFrame) -> pd.Series:
    """Identificador textual da janela a partir das colunas window_r0 e window_c0."""
    return frame["window_r0"].astype(int).astype(str) + "_" + frame["window_c0"].astype(int).astype(str)


@dataclass(frozen=True)
class MethodMatrix:
    """Valores de uma métrica por (seed, episódio) de um método; NaN onde o episódio não conta."""

    values: np.ndarray  # (n_seeds, n_episodes)

    @property
    def n_seeds(self) -> int:
        return self.values.shape[0]


def pivot_method(frame: pd.DataFrame, metric: str, episodes: Sequence[int], seed_column: str = "model_seed") -> MethodMatrix:
    """Monta a matriz (seed × episódio) de um método, na ordem de episódios dada."""
    table = frame.pivot_table(index=seed_column, columns="episode", values=metric, aggfunc="mean")
    table = table.reindex(columns=list(episodes))
    return MethodMatrix(table.to_numpy(dtype=float))


def _replicate_weights(cluster_of_episode: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Pesos por episódio numa réplica: quantas vezes a janela do episódio foi sorteada."""
    clusters, inverse = np.unique(cluster_of_episode, return_inverse=True)
    counts = np.bincount(rng.integers(len(clusters), size=len(clusters)), minlength=len(clusters))
    return counts[inverse].astype(float)


def _weighted_statistic(matrix: MethodMatrix, episode_weights: np.ndarray, rng: np.random.Generator,
                        statistic: str) -> float:
    """Estatística de uma réplica: reamostra seeds e aplica os pesos de episódio."""
    seed_rows = rng.integers(matrix.n_seeds, size=matrix.n_seeds) if matrix.n_seeds > 1 else np.zeros(1, int)
    values = matrix.values[seed_rows]  # (n_seeds, n_episodes)
    weights = np.broadcast_to(episode_weights, values.shape)
    valid = ~np.isnan(values)
    if statistic == "mean":
        total = (weights * valid).sum()
        return float((np.where(valid, values, 0.0) * weights).sum() / total) if total > 0 else np.nan
    if statistic == "median":
        repeated = np.repeat(values[valid], weights[valid].astype(int))
        return float(np.median(repeated)) if repeated.size else np.nan
    raise ValueError(f"estatística desconhecida: {statistic}")


def point_estimate(matrix: MethodMatrix, statistic: str = "mean") -> float:
    """Estimativa pontual: média (ou mediana) sobre todos os seeds e episódios válidos."""
    values = matrix.values[~np.isnan(matrix.values)]
    if values.size == 0:
        return np.nan
    return float(values.mean() if statistic == "mean" else np.median(values))


def hierarchical_bootstrap(
    matrices: dict[str, MethodMatrix],
    cluster_of_episode: np.ndarray,
    statistic: str = "mean",
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Réplicas bootstrap da estatística de cada método, com as mesmas janelas em todos (pareado)."""
    rng = np.random.default_rng(seed)
    replicates = {name: np.empty(n_boot) for name in matrices}
    for b in range(n_boot):
        episode_weights = _replicate_weights(cluster_of_episode, rng)
        for name, matrix in matrices.items():
            replicates[name][b] = _weighted_statistic(matrix, episode_weights, rng, statistic)
    return replicates


def percentile_ci(replicates: np.ndarray, confidence: float = CONFIDENCE) -> tuple[float, float]:
    """IC por percentis das réplicas (ignora réplicas NaN)."""
    alpha = (1.0 - confidence) / 2.0
    finite = replicates[np.isfinite(replicates)]
    if finite.size == 0:
        return np.nan, np.nan
    return float(np.quantile(finite, alpha)), float(np.quantile(finite, 1.0 - alpha))


def summarize_with_ci(
    matrices: dict[str, MethodMatrix],
    cluster_of_episode: np.ndarray,
    statistic: str = "mean",
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = 0,
    reference: str | None = None,
) -> pd.DataFrame:
    """Uma linha por método: estimativa, IC hierárquico e, se `reference`, a diferença pareada com IC."""
    replicates = hierarchical_bootstrap(matrices, cluster_of_episode, statistic, n_boot, seed)
    rows = []
    for name, matrix in matrices.items():
        low, high = percentile_ci(replicates[name])
        row = {"method": name, "estimate": point_estimate(matrix, statistic), "ci_low": low, "ci_high": high,
               "n_seeds": matrix.n_seeds}
        if reference is not None and name != reference:
            diff = replicates[name] - replicates[reference]
            row["diff_vs_reference"] = point_estimate(matrix, statistic) - point_estimate(matrices[reference], statistic)
            row["diff_ci_low"], row["diff_ci_high"] = percentile_ci(diff)
        rows.append(row)
    return pd.DataFrame(rows)

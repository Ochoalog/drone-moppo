"""Validação cruzada espacial no Taquari (§3.6 do plano v3): 3 faixas de latitude de largura igual,
com margem de 200 m entre elas; cada dobra treina em 2 faixas e testa na terceira.

A dobra A testa na faixa sul (a linha da v1, lat −29,26°, fica perto do limite do terço sul).
As linhas da margem não entram em nenhum lado (nem treino nem teste).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MARGIN_M = 200.0
FOLD_TEST_BAND = {"A": 2, "B": 1, "C": 0}  # linha 0 = norte; banda 2 = sul


@dataclass(frozen=True)
class Fold:
    name: str
    train_rows: np.ndarray
    test_rows: np.ndarray


def band_edges(n_rows: int, n_bands: int = 3) -> list[tuple[int, int]]:
    """Faixas [início, fim) de largura igual (a última absorve o resto)."""
    size = n_rows // n_bands
    return [(i * size, (i + 1) * size if i < n_bands - 1 else n_rows) for i in range(n_bands)]


def make_fold(name: str, n_rows: int, cell_m: float, window_cells: int) -> Fold:
    """Linhas de treino e de teste de uma dobra, sem as margens entre faixas."""
    margin = int(np.ceil(MARGIN_M / cell_m))
    bands = band_edges(n_rows)
    test_band = FOLD_TEST_BAND[name]
    if min(end - start for start, end in bands) < window_cells:
        raise ValueError("faixa menor que a janela do episódio")

    def trimmed(index: int) -> np.ndarray:
        start, end = bands[index]
        # Tira a margem nos lados que encostam em outra faixa.
        start += margin // 2 if index > 0 else 0
        end -= margin - margin // 2 if index < len(bands) - 1 else 0
        return np.arange(start, end)

    test = trimmed(test_band)
    train_bands = [trimmed(i) for i in range(len(bands)) if i != test_band]
    # Duas faixas de treino vizinhas se juntam (a margem entre elas não é necessária para o vazamento),
    # mas as janelas nunca podem cruzar a faixa de teste: só linhas de treino contíguas viram candidatas.
    return Fold(name, np.concatenate(train_bands), test)

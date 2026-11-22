"""Divisão espacial dentro de um local zero-shot (experimento E-real, Canoas; DECISIONS_V3.md V46).

A caixa de Canoas tem 320 × 480 células de 25 m, e a divisão é ao longo das colunas (o eixo mais longo):
- teste = as últimas `w` colunas (leste);
- margem = as 8 colunas (200 m) antes do teste, fora dos dois lados;
- treino = as colunas restantes, a oeste.

`w` é a menor largura ≥ 180 células (1,4 janela) com ≥ 20 janelas válidas distintas no teste. Se não houver,
a largura cresce célula a célula até 256 (2 janelas). A escolha usa só as máscaras de validade.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np

from dal.v3.envs.realistic_env import PerceptionTables, valid_windows

MARGIN_CELLS = 8  # 200 m
MIN_TEST_WIDTH = 180  # ≥ 1,4 janela de 128 células
MAX_TEST_WIDTH = 256  # 2 janelas
MIN_TEST_WINDOWS = 20


@dataclass(frozen=True)
class ColumnSplit:
    n_cols: int
    test_width: int
    margin: int
    n_test_windows: int
    n_train_windows: int

    @property
    def test_start(self) -> int:
        return self.n_cols - self.test_width

    @property
    def train_end(self) -> int:
        """Primeira coluna fora do treino (o treino é [0, train_end))."""
        return self.test_start - self.margin

    def cols(self, split: str) -> np.ndarray:
        if split == "train":
            return np.arange(0, self.train_end)
        if split == "test":
            return np.arange(self.test_start, self.n_cols)
        raise ValueError(f"divisão desconhecida: {split}")

    def to_dict(self) -> dict:
        return {**asdict(self), "train_cols": [0, self.train_end], "test_cols": [self.test_start, self.n_cols]}

    @classmethod
    def from_dict(cls, data: dict) -> ColumnSplit:
        return cls(int(data["n_cols"]), int(data["test_width"]), int(data["margin"]), int(data["n_test_windows"]),
                   int(data["n_train_windows"]))


def with_columns(tables: PerceptionTables, cols: np.ndarray) -> PerceptionTables:
    """Cópia rasa das tabelas com as janelas restritas às colunas dadas."""
    restricted = copy.copy(tables)
    restricted.cols_allowed = np.asarray(cols)
    return restricted


def count_windows(tables: PerceptionTables, cols: np.ndarray, window: int, min_valid: float) -> int:
    return len(valid_windows(with_columns(tables, cols), window, min_valid))


def choose_column_split(tables: PerceptionTables, window: int = 128, min_valid: float = 0.7,
                        min_test_windows: int = MIN_TEST_WINDOWS, min_width: int = MIN_TEST_WIDTH,
                        max_width: int = MAX_TEST_WIDTH, margin: int = MARGIN_CELLS) -> ColumnSplit | None:
    """A menor largura de teste com janelas suficientes; None se nem `max_width` basta (parar e perguntar)."""
    n_cols = tables.grid_shape[1]
    for width in range(min_width, max_width + 1):
        test_cols = np.arange(n_cols - width, n_cols)
        n_test = count_windows(tables, test_cols, window, min_valid)
        if n_test >= min_test_windows:
            train_cols = np.arange(0, n_cols - width - margin)
            n_train = count_windows(tables, train_cols, window, min_valid)
            return ColumnSplit(n_cols, width, margin, n_test, n_train)
    return None


def with_rows(tables: PerceptionTables, rows: np.ndarray) -> PerceptionTables:
    restricted = copy.copy(tables)
    restricted.rows_allowed = np.asarray(rows)
    return restricted


def split_diagnostic(tables: PerceptionTables, window: int = 128,
                     min_valids: tuple[float, ...] = (0.7, 0.6, 0.5)) -> list[dict]:
    """Janelas válidas por lado (leste/oeste nas colunas, sul/norte nas linhas), largura da faixa e limiar.

    Só máscaras de validade, sem desempenho: subsídio para o Vitor quando a regra da V46 falha. Não escolhe nada.
    """
    n_rows, n_cols = tables.grid_shape
    rows = []
    for min_valid in min_valids:
        whole = len(valid_windows(tables, window, min_valid))
        rows.append({"side": "all", "width": None, "min_valid": min_valid, "n_windows": whole})
        for side, n in (("east", n_cols), ("west", n_cols), ("south", n_rows), ("north", n_rows)):
            for width in range(window, n + 1, 16):
                band = np.arange(n - width, n) if side in ("east", "south") else np.arange(0, width)
                restricted = with_columns(tables, band) if side in ("east", "west") else with_rows(tables, band)
                rows.append({"side": side, "width": width, "min_valid": min_valid,
                             "n_windows": len(valid_windows(restricted, window, min_valid))})
    return rows

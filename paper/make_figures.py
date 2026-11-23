"""Regenera, a partir dos CSVs versionados, todas as figuras e tabelas geradas por código e as macros do artigo.

Ordem: macros → tabelas → Fig. 1 (visão geral, `make_fig1.py`) → Fig. 2 → Fig. 3.
A Fig. 3 é desenhada pelo `scripts/verdicts_E_v3.py` a partir de `results_v3/E/curve.csv`; aqui só é redesenhada.

Uso: PYTHONPATH=src .venv/bin/python paper/make_figures.py
"""
from __future__ import annotations

import runpy

import pandas as pd

from dal.config import ROOT
from dal.v3.reporting.figures_e import figure_3


def main() -> None:
    for script in ("make_numbers.py", "make_tables_v4.py", "make_fig1.py", "make_fig2_v4.py"):
        runpy.run_path(str(ROOT / "paper" / script), run_name="__main__")
    figure_3(pd.read_csv(ROOT / "results_v3/E/curve.csv"), ROOT / "paper/figures/fig3_E.pdf")
    print("Fig. 3 regenerada")


if __name__ == "__main__":
    main()

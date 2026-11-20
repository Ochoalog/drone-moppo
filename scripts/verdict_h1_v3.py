"""Veredito do H1 (§2.2 do DECISIONS_V3.md): top-1 do descritor aprendido × hand nas janelas de teste, S1 e S2.

Entrada: os `_queries.npz` do job `h1_retrieval_ftA` (publicados em `colab_out/h1_retrieval_ftA/`).
Saída: `results_v3/verdicts/h1.json` (estimativa e IC 95% por bootstrap de células consulta, B = 2000).
Uso: .venv/bin/python scripts/verdict_h1_v3.py --dir results_v3/h1/h1_retrieval_ftA
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dal.config import ROOT
from dal.v3.evaluation.verdicts import N_BOOT, query_cell_difference, verdict_h1
from dal.v3.reporting.provenance import git_commit

LEARNED = "h1_retrieval_ftA_ft_test_queries.npz"
HAND = "h1_retrieval_ftA_dinov2_vits14_hand_test_queries.npz"


def load(path: Path, scenario: str) -> dict:
    data = np.load(path)
    return {key: data[f"{scenario}_{key}"] for key in ("window", "variant", "cell", "hit")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", default="results_v3/h1/h1_retrieval_ftA")
    args = parser.parse_args()
    directory = ROOT / args.dir
    differences, counts = {}, {}
    for scenario in ("S1", "S2"):
        learned, hand = load(directory / LEARNED, scenario), load(directory / HAND, scenario)
        differences[scenario] = query_cell_difference(learned, hand)
        counts[scenario] = {"queries": int(len(learned["hit"])), "cells": int(len(np.unique(learned["cell"]))),
                            "top1_learned": float(learned["hit"].mean()), "top1_hand": float(hand["hit"].mean())}
    result = verdict_h1(differences)
    result.update({"counts": counts, "source_file": [str(directory / LEARNED), str(directory / HAND)],
                   "protocol": "window_stats v1, 20 janelas de teste 64x64, semente 0; bootstrap de células consulta",
                   "n_boot": N_BOOT, "fold": "A (teste sul)", "region": "taquari", "git_commit": git_commit()})
    out = ROOT / "results_v3" / "verdicts"
    out.mkdir(parents=True, exist_ok=True)
    (out / "h1.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps(result, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

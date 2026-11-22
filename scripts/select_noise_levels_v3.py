"""Escolhe os níveis L1–L3 do E-synth a partir da varredura `scan_noise_A` (regra da V46, fixada antes).

Regra: L1, L2 e L3 = os σ com sucesso da aleatória (treino da dobra A, κ escolhido) mais perto de 75%, 55%
e 35%, nessa ordem, sem repetir σ; empate → menor σ. σ inválido (aleatória < 5% no κ escolhido) fica de fora,
e vale o σ válido mais próximo. Se nenhum σ válido der aleatória < 75%: parar e perguntar ao Vitor.

Uso: .venv/bin/python scripts/select_noise_levels_v3.py [--scan <scan.csv>]
Saída: results_v3/E/noise_levels.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dal.config import ROOT

TARGETS = {"L1": 0.75, "L2": 0.55, "L3": 0.35}
MIN_RANDOM_SUCCESS = 0.05
DEFAULT_SCAN = ROOT / "results_v3" / "E" / "scan_noise_A" / "scan.csv"
OUT = ROOT / "results_v3" / "E" / "noise_levels.json"


class NoValidLevelError(RuntimeError):
    """Nenhum σ válido abaixo de 75% de sucesso da aleatória (V46: parar e perguntar)."""


def select_levels(scan: pd.DataFrame, targets: dict[str, float] = TARGETS,
                  min_random_success: float = MIN_RANDOM_SUCCESS) -> dict:
    """Função pura: {nível: {sigma, random_success, target, ...}} e a lista de σ inválidos."""
    valid = scan[scan.random_success >= min_random_success].sort_values("sigma")
    invalid = sorted(scan.loc[scan.random_success < min_random_success, "sigma"].astype(float).tolist())
    if not (valid.random_success < 0.75).any():
        raise NoValidLevelError("nenhum σ válido com sucesso da aleatória < 75%")
    chosen, levels = set(), {}
    for level, target in targets.items():
        candidates = valid[~valid.sigma.isin(chosen)].copy()
        if candidates.empty:
            raise NoValidLevelError(f"sem σ válido sobrando para {level}")
        candidates["distance"] = (candidates.random_success - target).abs().round(9)  # empate numérico → menor σ
        best = candidates.sort_values(["distance", "sigma"]).iloc[0]
        chosen.add(best.sigma)
        levels[level] = {"sigma": float(best.sigma), "target_random_success": target,
                         "random_success": float(best.random_success), "kappa": float(best.kappa),
                         "calibration_file": str(best.get("calibration_file", ""))}
    return {"levels": levels, "invalid_sigmas": invalid, "rule": "V46", "min_random_success": min_random_success}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scan", default=str(DEFAULT_SCAN))
    args = parser.parse_args()
    scan = pd.read_csv(args.scan)
    result = select_levels(scan)
    scan_path = Path(args.scan)
    result["source_file"] = str(scan_path.relative_to(ROOT)) if scan_path.is_relative_to(ROOT) else args.scan
    for column in ("git_commit", "config_hash"):
        if column in scan.columns:
            result[f"scan_{column}"] = str(scan[column].iloc[0])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

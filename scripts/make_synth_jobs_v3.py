"""Gera as configs e as linhas da fila do E-synth (V46) a partir de `results_v3/E/noise_levels.json`.

Para cada nível L1–L3: `E_synth_<L>_ppo_s0–4` (treino igual ao `mx_A_ppo`, mais o ruído do nível e a calibração
recalibrada dele) e `E_eval_synth_<L>` (teste e treino da dobra A, heurísticas com filtro padrão, dois modos).
Prioridade: L3 primeiro (a ordem de corte da V46 mantém L0 e L3), depois L1 e L2.

Uso: .venv/bin/python scripts/make_synth_jobs_v3.py
"""
from __future__ import annotations

import json

from dal.config import ROOT

LEVELS = ROOT / "results_v3" / "E" / "noise_levels.json"
CONFIGS = ROOT / "configs_v3" / "jobs" / "E"
QUEUE = ROOT / "colab" / "jobs_v3.csv"
PRIORITY = {"L3": 8, "L1": 10, "L2": 10}
PRIOR = "/content/drive/MyDrive/drone_v3/prior/taquari_prior_emsr720_le0605_10m.tif"
SEEDS = range(5)


def train_yaml(level: str, info: dict, seed: int) -> str:
    rate = info["random_success"]
    return f"""# Experimento E-synth (V46), nível {level} (σ = {info['sigma']:g}; aleatória no treino {rate:.3f}).
# Gerado por scripts/make_synth_jobs_v3.py a partir de results_v3/E/noise_levels.json. NÃO EDITAR.
# Igual ao mx_A_ppo (matriz), mais o ruído na observação e a calibração recalibrada do nível.
kind: train
descriptor: dinov2_vits14_ftA
feature: ft
fold: A
scenario: S2
method: ppo
seed: {seed}
total_timesteps: 1400000
n_envs: 512
calibration_file: {info['calibration_file']}
obs_noise_sigma: {info['sigma']}
obs_noise_seed: 0
gate_flight_time_ratio: 0.75
prior:
  taquari: {PRIOR}
"""


def eval_yaml(level: str, info: dict) -> str:
    models = "\n".join(f"  - {{method: ppo, seed: {s}, run: E_synth_{level}_ppo_s{s}, label: ppo}}" for s in SEEDS)
    return f"""# Experimento E-synth (V46), avaliação do nível {level} (σ = {info['sigma']:g}). Gerado por
# scripts/make_synth_jobs_v3.py. NÃO EDITAR. Mesmo protocolo do E_eval_taquari_L0, com o ruído do nível.
kind: evaluate
descriptor: dinov2_vits14_ftA
feature: ft
fold: A
episodes: 1000
calibration_file: {info['calibration_file']}
obs_noise_sigma: {info['sigma']}
obs_noise_seed: 0
include_invalid_gate: true
publish_parquet: true
baseline_filters: [standard]
modes: [sampled, argmax]
baselines: [straight, lawnmower, random, greedy_map, straight_150]
targets:
  - {{site: taquari, scenario: S2, split: test}}
  - {{site: taquari, scenario: S2, split: train}}
prior:
  taquari: {PRIOR}
models:
{models}
"""


def main() -> None:
    levels = json.loads(LEVELS.read_text())["levels"]
    rows = []
    for level, info in levels.items():
        if not info.get("calibration_file"):
            raise ValueError(f"{level}: noise_levels.json sem calibration_file")
        runs = []
        for seed in SEEDS:
            job = f"E_synth_{level}_ppo_s{seed}"
            (CONFIGS / f"{job}.yaml").write_text(train_yaml(level, info, seed))
            rows.append(f"{job},colab/04_train_matrix.ipynb,configs_v3/jobs/E/{job}.yaml,scan_noise_A,8,"
                        f"{PRIORITY[level]},queued")
            runs.append(job)
        job = f"E_eval_synth_{level}"
        (CONFIGS / f"{job}.yaml").write_text(eval_yaml(level, info))
        rows.append(f"{job},colab/05_evaluate.ipynb,configs_v3/jobs/E/{job}.yaml,{';'.join(runs)},25,"
                    f"{PRIORITY[level]},queued")
    lines = QUEUE.read_text().splitlines()
    existing = {line.split(",")[0] for line in lines}
    new = [r for r in rows if r.split(",")[0] not in existing]
    QUEUE.write_text("\n".join(lines + new) + "\n")
    print(f"{len(rows)} configs; {len(new)} linhas novas na fila")


if __name__ == "__main__":
    main()

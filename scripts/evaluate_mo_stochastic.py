"""Reavalia as políticas MOPPO amostrando ações de π(·|s, w), em paralelo.

Motivação (pendência 5.1): com argmax, a ação escolhida é a mesma para quase todo w
(ver scripts/diagnose_mo_conditioning.py), embora a distribuição π(·|s, w) mude com w.
A política foi treinada estocástica, então a avaliação amostrada é a mais fiel ao treino.

Os episódios usam os mesmos seeds (2024 + i) da avaliação original. Assim, os resultados
continuam pareados com a linha reta e com os PPOs da PoC, que são reaproveitados do
summary_test.csv original. A saída fica em results_mo/stochastic/ e não sobrescreve nada.

Uso:
    nohup .venv/bin/python scripts/evaluate_mo_stochastic.py --workers 8 > runs_mo/stochastic_eval.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Um thread por processo: o paralelismo vem dos workers, não do torch/BLAS.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from dal.config import ROOT, load_config  # noqa: E402
from dal.mo_eval import evaluate_weight_grid, generate_simplex_grid, load_mo_model  # noqa: E402
from dal.mo_report import (  # noqa: E402
    compute_normalized_hv_eum,
    generate_behavior_adaptation_figure,
    generate_pareto_figures,
)

FAST_WEIGHTS = np.array([0.15, 0.55, 0.15, 0.15], dtype=np.float32)
SAFE_WEIGHTS = np.array([0.15, 0.15, 0.15, 0.55], dtype=np.float32)
REFERENCE_METHODS = ("straight", "ppo_poc_s0", "ppo_poc_s1", "ppo_poc_s2")


@dataclass(frozen=True)
class EvaluationTask:
    """Uma unidade de trabalho: um modelo avaliado em um único vetor de pesos."""

    label: str
    model_path: str
    use_memory: bool
    weight_index: int
    weights: tuple[float, ...]
    preset: str = ""


_worker_cache: dict = {}


def run_task(task: EvaluationTask, config_path: str, episodes: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Executa uma tarefa num worker, reaproveitando modelo e config já carregados no processo."""
    import torch

    torch.set_num_threads(1)
    if "config" not in _worker_cache:
        _worker_cache["config"] = load_config(config_path)
    if task.model_path not in _worker_cache:
        _worker_cache[task.model_path] = load_mo_model(Path(task.model_path), memory=task.use_memory)

    summary, episodes_df = evaluate_weight_grid(
        _worker_cache[task.model_path],
        _worker_cache["config"],
        [np.array(task.weights, dtype=np.float32)],
        episodes=episodes,
        memory=task.use_memory,
        label=task.label,
        deterministic=False,
    )
    # Cada tarefa avalia um único w; o índice global é restaurado aqui.
    for frame in (summary, episodes_df):
        frame["weight_idx"] = task.weight_index
        if task.preset:
            frame["preset"] = task.preset
    return summary, episodes_df


def build_tasks(runs_dir: Path) -> list[EvaluationTask]:
    """Monta as tarefas: grade completa para as políticas MO e o preset de cada PPO de peso fixo."""
    grid = generate_simplex_grid(step=0.25)
    tasks = []
    for label, run_name, use_memory in (
        ("mo_ppo", "moppo_S2_hand_s0", True),
        ("mo_ppo_nomem", "moppo_nomem_S2_hand_s0", False),
    ):
        model_path = runs_dir / run_name / "final_model.pt"
        for weight_index, weights in enumerate(grid):
            tasks.append(EvaluationTask(label, str(model_path), use_memory, weight_index, tuple(map(float, weights))))

    for label, run_name, weights, preset in (
        ("ppo_fixed_fast", "ppo_fixed_fast_s0", FAST_WEIGHTS, "fast"),
        ("ppo_fixed_safe", "ppo_fixed_safe_s0", SAFE_WEIGHTS, "safe"),
    ):
        model_path = runs_dir / run_name / "final_model.pt"
        tasks.append(EvaluationTask(label, str(model_path), True, 0, tuple(map(float, weights)), preset))

    # As tarefas mais caras (sem memória: episódios mais longos) vão primeiro para equilibrar a carga.
    return sorted(tasks, key=lambda task: task.use_memory)


def run_all(tasks: list[EvaluationTask], config_path: str, episodes: int, workers: int):
    """Distribui as tarefas entre os workers e junta os resultados."""
    summaries, episode_frames = [], []
    started_at = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_task, task, config_path, episodes): task for task in tasks}
        for done_count, future in enumerate(as_completed(futures), start=1):
            summary, episodes_df = future.result()
            summaries.append(summary)
            episode_frames.append(episodes_df)
            task = futures[future]
            elapsed_min = (time.time() - started_at) / 60
            print(f"[stochastic-eval] {done_count}/{len(tasks)} {task.label} w={task.weights} ({elapsed_min:.1f} min)",
                  flush=True)
    summary = pd.concat(summaries, ignore_index=True).sort_values(["method", "weight_idx"], ignore_index=True)
    return summary, pd.concat(episode_frames, ignore_index=True)


def write_metrics_and_figures(summary: pd.DataFrame, reference: pd.DataFrame, output_dir: Path) -> dict:
    """Recalcula fronteiras de Pareto, adaptação (H6), hipervolume e EUM com a avaliação estocástica."""
    figures_dir = output_dir / "figs"
    figures_dir.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([summary, reference], ignore_index=True)
    combined.to_csv(output_dir / "summary_test.csv", index=False)

    generate_pareto_figures(combined, figures_dir)
    mo_memory = summary[summary["method"] == "mo_ppo"]
    _, adaptation_stats = generate_behavior_adaptation_figure(mo_memory, figures_dir)
    hv_eum = compute_normalized_hv_eum(
        mo_memory,
        summary[summary["method"] == "mo_ppo_nomem"],
        summary[summary["method"].str.startswith("ppo_fixed")],
        reference,
    )
    (output_dir / "hypervolume_eum.json").write_text(json.dumps(hv_eum, indent=2))
    (output_dir / "behavior_adaptation.json").write_text(json.dumps(adaptation_stats, indent=2, default=float))
    return hv_eum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs_mo/default.yaml")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--out", default="results_mo/stochastic")
    parser.add_argument("--limit-tasks", type=int, help="Só para teste rápido do script")
    args = parser.parse_args()

    output_dir = ROOT / args.out
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(ROOT / "runs_mo")
    if args.limit_tasks:
        tasks = tasks[: args.limit_tasks]

    summary, episodes_df = run_all(tasks, args.config, args.episodes, args.workers)
    episodes_df.to_csv(output_dir / "episodes_test.csv", index=False)

    original_summary = pd.read_csv(ROOT / "results_mo" / "summary_test.csv")
    reference = original_summary[original_summary["method"].isin(REFERENCE_METHODS)]
    hv_eum = write_metrics_and_figures(summary, reference, output_dir)
    print(json.dumps(hv_eum, indent=2))


if __name__ == "__main__":
    main()

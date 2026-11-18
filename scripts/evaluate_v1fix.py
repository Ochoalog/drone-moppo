"""Avaliação v1fix: modelos v1 nos dois modos (argmax e amostrado) e varredura 1-D do H6.

Protocolo pré-especificado em DECISIONS_MO_V2.md (A1–A3, A5 e §2):
- mesmos episódios para todos (seed do ambiente = 2024 + i);
- números aleatórios comuns: `torch.manual_seed(2024 + i)` antes de cada episódio;
- todas as políticas aprendidas nos dois modos; a reta (não aprendida) roda uma vez.

Saídas em results_mo_v1fix/:
- episodes_presets.csv: MO (com e sem memória) nos presets fast/safe/unif, PPOs fixos no próprio
  preset, PPO da PoC (seeds 0–2) e reta;
- episodes_h6_sweep.csv: MO com memória nas varreduras de w_risco e w_custo (11 pontos cada).
Cada linha é um episódio, com a janela (window_r0, window_c0) para o bootstrap por janela.

Uso:
    nohup .venv/bin/python scripts/evaluate_v1fix.py --workers 8 > results_mo_v1fix/eval.log 2>&1 &
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dal.config import ROOT, load_config  # noqa: E402

PRESETS = {
    "fast": (0.15, 0.55, 0.15, 0.15),
    "safe": (0.15, 0.15, 0.15, 0.55),
    "unif": (0.25, 0.25, 0.25, 0.25),
}
MODES = {"argmax": True, "sampled": False}
SWEEP_VALUES = [round(0.1 * k, 1) for k in range(11)]
OBJECTIVE_INDEX = {"risk": 3, "cost": 1}


@dataclass(frozen=True)
class Task:
    """Um método avaliado em um w, num modo, sobre todos os episódios."""

    group: str  # "presets" ou "h6_sweep"
    method: str
    kind: str  # "mo", "poc" ou "straight"
    model_path: str
    use_memory: bool
    weights: tuple[float, ...]
    mode: str
    preset: str = ""
    sweep: str = ""
    sweep_value: float = float("nan")


def sweep_weights(objective: str, value: float) -> tuple[float, ...]:
    """w com `value` no objetivo varrido e (1 - value)/3 nos outros três."""
    rest = (1.0 - value) / 3.0
    weights = [rest] * 4
    weights[OBJECTIVE_INDEX[objective]] = value
    return tuple(float(x) for x in weights)


def build_tasks(runs_mo: Path, runs_poc: Path) -> list[Task]:
    """Monta todas as tarefas das duas partes da avaliação."""
    mo_runs = {"mo_ppo": ("moppo_S2_hand_s0", True), "mo_ppo_nomem": ("moppo_nomem_S2_hand_s0", False)}
    fixed_runs = {"ppo_fixed_fast": ("ppo_fixed_fast_s0", "fast"), "ppo_fixed_safe": ("ppo_fixed_safe_s0", "safe")}
    tasks = [Task("presets", "straight", "straight", "", False, PRESETS["unif"], "argmax", "baseline")]
    for mode in MODES:
        for method, (run, memory) in mo_runs.items():
            for preset, weights in PRESETS.items():
                tasks.append(Task("presets", method, "mo", str(runs_mo / run / "final_model.pt"), memory,
                                  weights, mode, preset))
        for method, (run, preset) in fixed_runs.items():
            tasks.append(Task("presets", method, "mo", str(runs_mo / run / "final_model.pt"), True,
                              PRESETS[preset], mode, preset))
        for seed in (0, 1, 2):
            tasks.append(Task("presets", f"ppo_poc_s{seed}", "poc",
                              str(runs_poc / f"S2_hand_s{seed}" / "final_model.zip"), False,
                              PRESETS["unif"], mode, "uniform"))
        for objective in OBJECTIVE_INDEX:
            for value in SWEEP_VALUES:
                tasks.append(Task("h6_sweep", "mo_ppo", "mo", str(runs_mo / "moppo_S2_hand_s0" / "final_model.pt"),
                                  True, sweep_weights(objective, value), mode, sweep=objective, sweep_value=value))
    # Tarefas lentas primeiro (sem memória e PoC têm episódios mais longos), para equilibrar a carga.
    return sorted(tasks, key=lambda t: (t.kind != "poc", t.use_memory))


_cache: dict = {}


def _policy_for(task: Task):
    """Carrega (uma vez por processo) o modelo da tarefa."""
    if task.kind == "straight":
        from dal.baselines import StraightPolicy

        return StraightPolicy()
    key = (task.model_path, task.mode)
    if key not in _cache:
        if task.kind == "poc":
            from dal.evaluate import PPOPolicy

            _cache[key] = PPOPolicy(task.model_path, deterministic=MODES[task.mode])
        else:
            from dal.mo_eval import load_mo_model

            _cache[key] = load_mo_model(Path(task.model_path), memory=task.use_memory)
    return _cache[key]


def run_task(task: Task, config_path: str, episodes: int, seed_base: int) -> pd.DataFrame:
    """Roda os episódios de uma tarefa e devolve uma linha por episódio."""
    import torch

    from dal.mo_env import make_mo_env
    from dal.mo_eval import run_episode

    torch.set_num_threads(1)
    config = _cache.setdefault("config", load_config(config_path))
    env_key = ("env", task.use_memory)
    if env_key not in _cache:
        _cache[env_key] = make_mo_env("hand", "S2", config, split="test", memory=task.use_memory)
    env = _cache[env_key]
    policy = _policy_for(task)
    weights = np.array(task.weights, dtype=np.float32)
    deterministic = MODES[task.mode]

    rows = []
    for i in range(episodes):
        seed = seed_base + i
        if task.kind == "mo":
            info, _ = run_episode(env, policy, weights, seed, is_ppo_mo=True, deterministic=deterministic)
        else:
            # Políticas que agem sobre o Episode: o reset do ambiente precisa vir antes do reset da política.
            if not deterministic:
                torch.manual_seed(seed)
            env.reset(seed=seed, options={"w": weights})
            policy.reset(env.ep, np.random.default_rng([seed, 7]))
            info, _ = run_episode(env, policy, weights, seed, is_ppo_mo=False, deterministic=deterministic)
        info.update(group=task.group, method=task.method, mode=task.mode, preset=task.preset,
                    sweep=task.sweep, sweep_value=task.sweep_value, episode=i,
                    window_r0=int(env.ep.r0), window_c0=int(env.ep.c0),
                    w_info=weights[0], w_cost=weights[1], w_success=weights[2], w_risk=weights[3])
        rows.append(info)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs_mo/default.yaml")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed-base", type=int, default=2024)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--out", default="results_mo_v1fix")
    parser.add_argument("--limit-tasks", type=int, help="só para teste rápido do script")
    args = parser.parse_args()

    output_dir = ROOT / args.out
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(ROOT / "runs_mo", ROOT / "runs" / "overnight")
    if args.limit_tasks:
        tasks = tasks[: args.limit_tasks]

    frames, started_at = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_task, t, args.config, args.episodes, args.seed_base): t for t in tasks}
        for done, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            frames.append(future.result())
            print(f"[v1fix] {done}/{len(tasks)} {task.group} {task.method} {task.mode} "
                  f"{task.preset or task.sweep} {task.sweep_value} ({(time.time() - started_at) / 60:.1f} min)",
                  flush=True)

    episodes = pd.concat(frames, ignore_index=True)
    for group, frame in episodes.groupby("group"):
        frame.to_csv(output_dir / f"episodes_{group}.csv", index=False)
    print(f"[v1fix] salvo em {output_dir}")


if __name__ == "__main__":
    main()

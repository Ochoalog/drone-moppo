"""Teste de regressão "golden": trava os números durante a refatoração.

Roda poucos episódios com seeds fixos e compara as métricas com CSVs gravados em tests/golden/.
Se um PR de refatoração (só de forma) mudar algum número, este teste falha.

Para regravar os arquivos de referência (só quando a mudança de comportamento for intencional):
    DAL_UPDATE_GOLDEN=1 .venv/bin/pytest tests/test_regression_golden.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from dal.baselines import make_baselines
from dal.config import ROOT, load_config
from dal.env import Episode
from helpers import small_cfg, synthetic_data

GOLDEN_DIR = Path(__file__).parent / "golden"
UPDATE_GOLDEN = os.environ.get("DAL_UPDATE_GOLDEN") == "1"
# Tempo de decisão depende da máquina; todo o resto precisa bater exatamente.
VOLATILE_COLUMNS = ("decision_ms",)
EPISODE_SEEDS = range(2024, 2030)
MO_MODEL_PATH = ROOT / "runs_mo" / "moppo_S2_hand_s0" / "final_model.pt"


def summarize_episode(method: str, seed: int, episode: Episode) -> dict:
    """Extrai as métricas de um episódio terminado, no mesmo formato dos CSVs de avaliação."""
    result = episode.last
    return {
        "method": method,
        "seed": seed,
        "success": bool(result.success),
        "false_loc": bool(result.false_loc),
        "steps": int(episode.t),
        "final_error_m": round(float(result.error_m), 6),
        "distance_m": round(float(episode.distance_m), 6),
        "final_entropy": round(float(result.entropy), 6),
        "trajectory": " ".join(f"{row},{col}" for row, col in episode.traj),
    }


def assert_matches_golden(frame: pd.DataFrame, name: str) -> None:
    """Compara com o arquivo golden, ou o regrava quando DAL_UPDATE_GOLDEN=1."""
    golden_path = GOLDEN_DIR / f"{name}.csv"
    frame = frame.drop(columns=[c for c in VOLATILE_COLUMNS if c in frame.columns])
    if UPDATE_GOLDEN or not golden_path.exists():
        GOLDEN_DIR.mkdir(exist_ok=True)
        frame.to_csv(golden_path, index=False)
        if not UPDATE_GOLDEN:
            pytest.skip(f"golden criado agora em {golden_path}; rode de novo para comparar")
        return
    expected = pd.read_csv(golden_path)
    actual = pd.read_csv(pd.io.common.StringIO(frame.to_csv(index=False)))
    pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-6)


def run_baselines(episode: Episode, config: dict, methods: tuple[str, ...]) -> pd.DataFrame:
    """Roda as baselines nos mesmos seeds; o ruído de cada política também é fixado pelo seed."""
    policies = make_baselines(config)
    rows = []
    for method in methods:
        for seed in EPISODE_SEEDS:
            episode.reset(seed=seed)
            policies[method].reset(episode, np.random.default_rng([seed, 7]))
            while not episode.done:
                episode.step(policies[method].act(episode))
            rows.append(summarize_episode(method, seed, episode))
    return pd.DataFrame(rows)


def test_golden_synthetic_baselines():
    """Núcleo (filtro, ambiente, baselines) em dados sintéticos: roda sem baixar nada."""
    config = small_cfg()
    episode = Episode(synthetic_data(), config["filter"], "train", config["data"]["cell_m"])
    frame = run_baselines(episode, config, ("straight", "lawnmower", "random", "greedy", "greedy_map"))
    assert_matches_golden(frame, "synthetic_baselines")


@pytest.mark.data
def test_golden_taquari_s2_baselines():
    """Mesmo teste nos dados reais do Taquari (S2, região de teste)."""
    from dal.env import ScenarioData

    config = load_config("configs/default.yaml")
    data = ScenarioData.load("hand", "S2", config)
    episode = Episode(data, config["filter"], "test", config["data"]["cell_m"])
    frame = run_baselines(episode, config, ("straight", "lawnmower", "greedy_map"))
    assert_matches_golden(frame, "taquari_s2_baselines")


@pytest.mark.data
@pytest.mark.skipif(not MO_MODEL_PATH.exists(), reason="modelo MOPPO treinado ausente em runs_mo/")
@pytest.mark.parametrize("deterministic", [True, False], ids=["argmax", "sampled"])
def test_golden_moppo_policy(deterministic: bool):
    """Política MOPPO treinada no ambiente multiobjetivo, com argmax e com amostragem pareada."""
    from dal.mo_env import make_mo_env
    from dal.mo_eval import load_mo_model, run_episode

    torch.set_num_threads(1)
    config = load_config("configs_mo/default.yaml")
    env = make_mo_env("hand", "S2", config, split="test", memory=True)
    net = load_mo_model(MO_MODEL_PATH, memory=True)
    rows = []
    for preset, weights in {"fast": [0.15, 0.55, 0.15, 0.15], "safe": [0.15, 0.15, 0.15, 0.55]}.items():
        for seed in EPISODE_SEEDS:
            info, _ = run_episode(env, net, np.array(weights, dtype=np.float32), seed, deterministic=deterministic)
            info["preset"] = preset
            info["trajectory"] = " ".join(f"{row},{col}" for row, col in env.ep.traj)
            rows.append(info)
    suffix = "argmax" if deterministic else "sampled"
    assert_matches_golden(pd.DataFrame(rows), f"moppo_{suffix}")

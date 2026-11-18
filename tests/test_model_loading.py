"""Teste de carga dos modelos treinados (§4.4, item 3, do PLANO_DESENVOLVIMENTO.md v2).

Carrega os 9 modelos PPO da PoC (`runs/overnight/`) e os 4 modelos da MO v1 (`runs_mo/`) e roda
20 episódios de cada um, com ação mais provável. O resultado precisa ser idêntico à referência
gerada com o código da tag `v0-pre-refactor` (antes de qualquer refatoração).

Por que existe: o SB3 guarda a classe do extrator CNN pelo caminho do módulo. Mover `dal.train`
quebra o `PPO.load`, e mudar a rede da MO quebra o `load_state_dict`. Nunca retreinar por isso:
use `src/dal/_compat.py`.

Para regerar a referência a partir da tag (ver docs/REVISAO_TRABALHO_v1.md):
    git worktree add /tmp/dal_v0 v0-pre-refactor
    PYTHONPATH=/tmp/dal_v0/src DAL_DATA_DIR=$PWD/data DAL_UPDATE_GOLDEN=1 \
        .venv/bin/pytest tests/test_model_loading.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = Path(__file__).parent / "golden" / "model_loading.csv"
UPDATE_GOLDEN = os.environ.get("DAL_UPDATE_GOLDEN") == "1"
EPISODES_PER_MODEL = 20
SEED_BASE = 2024
POC_MODELS = [(scenario, seed) for scenario in ("S0", "S1", "S2") for seed in (0, 1, 2)]
# (nome da execução, usa memória)
MO_MODELS = [("moppo_S2_hand_s0", True), ("moppo_nomem_S2_hand_s0", False),
             ("ppo_fixed_fast_s0", True), ("ppo_fixed_safe_s0", True)]
UNIFORM_WEIGHTS = np.full(4, 0.25, dtype=np.float32)

pytestmark = pytest.mark.data


def _episode_row(model: str, seed: int, episode) -> dict:
    """Métricas e trajetória de um episódio terminado."""
    result = episode.last
    return {
        "model": model,
        "seed": seed,
        "success": bool(result.success),
        "false_loc": bool(result.false_loc),
        "steps": int(episode.t),
        "distance_m": round(float(episode.distance_m), 6),
        "final_entropy": round(float(result.entropy), 6),
        "trajectory": " ".join(f"{row},{col}" for row, col in episode.traj),
    }


def _run_poc_models() -> list[dict]:
    """PPO da PoC: carrega pelo SB3 e roda no simulador de episódios do cenário do modelo."""
    from dal.config import load_config
    from dal.env import Episode, ScenarioData
    from dal.evaluate import PPOPolicy

    config = load_config(REPO_ROOT / "configs" / "default.yaml")
    rows = []
    for scenario in ("S0", "S1", "S2"):
        data = ScenarioData.load("hand", scenario, config)
        episode = Episode(data, config["filter"], "test", config["data"]["cell_m"])
        for model_scenario, model_seed in POC_MODELS:
            if model_scenario != scenario:
                continue
            policy = PPOPolicy(str(REPO_ROOT / "runs" / "overnight" / f"{scenario}_hand_s{model_seed}" /
                                   "final_model.zip"), deterministic=True)
            for i in range(EPISODES_PER_MODEL):
                episode.reset(seed=SEED_BASE + i)
                while not episode.done:
                    episode.step(policy.act(episode))
                rows.append(_episode_row(f"poc_{scenario}_s{model_seed}", SEED_BASE + i, episode))
    return rows


def _run_mo_models() -> list[dict]:
    """Modelos da MO v1: carrega o state dict e roda no ambiente multiobjetivo com w uniforme."""
    from dal.config import load_config
    from dal.mo_env import make_mo_env
    from dal.mo_eval import load_mo_model, run_episode

    config = load_config(REPO_ROOT / "configs_mo" / "default.yaml")
    rows = []
    for run_name, use_memory in MO_MODELS:
        env = make_mo_env("hand", "S2", config, split="test", memory=use_memory)
        net = load_mo_model(REPO_ROOT / "runs_mo" / run_name / "final_model.pt", memory=use_memory)
        for i in range(EPISODES_PER_MODEL):
            # Sem o argumento `deterministic`: a tag v0 não o tinha, e o padrão é argmax nas duas versões.
            run_episode(env, net, UNIFORM_WEIGHTS, SEED_BASE + i)
            rows.append(_episode_row(f"mo_{run_name}", SEED_BASE + i, env.ep))
        env.close()
    return rows


@pytest.mark.skipif(not (REPO_ROOT / "runs" / "overnight").exists(), reason="modelos da PoC ausentes")
def test_trained_models_load_and_reproduce_reference():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    frame = pd.DataFrame(_run_poc_models() + _run_mo_models())
    if UPDATE_GOLDEN or not GOLDEN_PATH.exists():
        GOLDEN_PATH.parent.mkdir(exist_ok=True)
        frame.to_csv(GOLDEN_PATH, index=False)
        pytest.skip(f"referência gravada em {GOLDEN_PATH}")
    expected = pd.read_csv(GOLDEN_PATH)
    actual = pd.read_csv(pd.io.common.StringIO(frame.to_csv(index=False)))
    pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-6)

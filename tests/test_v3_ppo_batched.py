"""Treinador PPO em lote: roda, retoma do checkpoint e marca INVALID_GATE quando o gate falha."""
import json

import torch

from dal.v3.agents.ppo_batched import INVALID_GATE_MARKER, PPOTrainConfig, train
from dal.v3.envs.realistic_env import RealisticBatchedEnv
from test_v3_realistic_env import _config, _tables


def _cfg(**overrides) -> PPOTrainConfig:
    base = dict(total_timesteps=256, n_envs=8, n_steps=8, n_epochs=1, batch_size=32, checkpoint_minutes=0.0,
                gate_episodes=8, features_dim=32)
    base.update(overrides)
    return PPOTrainConfig(**base)


def _envs(change_prior_input=False):
    config = _config(alpha=0.5 if change_prior_input else 0.0)
    return RealisticBatchedEnv(_tables(False), config), RealisticBatchedEnv(_tables(False), config)


def test_training_runs_writes_outputs_and_resumes(tmp_path):
    env, gate_env = _envs()
    out = train(env, gate_env, _cfg(gate_min_advantage_over_random=-1.0), tmp_path / "run")
    assert (out / "final_model.pt").exists() and (out / "gate_25pct.json").exists()
    rows = (out / "progress.csv").read_text().strip().splitlines()
    assert len(rows) == 1 + 256 // 64
    # Retomada: com o orçamento já cumprido, nada é treinado de novo.
    summary_before = json.loads((out / "train_summary.json").read_text())
    train(*_envs(), _cfg(gate_min_advantage_over_random=-1.0), out)
    assert json.loads((out / "train_summary.json").read_text())["global_step"] == summary_before["global_step"]


def test_failed_gate_marks_the_run_invalid(tmp_path):
    env, gate_env = _envs()
    out = train(env, gate_env, _cfg(gate_min_advantage_over_random=2.0), tmp_path / "run")
    assert out.name.endswith(INVALID_GATE_MARKER) and (out / INVALID_GATE_MARKER).exists()


def test_change_aware_policy_gets_the_extra_channel(tmp_path):
    env, gate_env = _envs(change_prior_input=True)
    out = train(env, gate_env, _cfg(change_prior_input=True, gate_min_advantage_over_random=-1.0), tmp_path / "ca")
    weights = torch.load(out / "final_model.pt")
    assert weights["extractor.cnn.0.weight"].shape[1] == 3  # crença, máscara, prior

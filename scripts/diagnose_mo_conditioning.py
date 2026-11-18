"""Diagnóstico do condicionamento da política MOPPO no vetor de preferências w.

Responde a duas perguntas, usadas na pendência 5.1 do plano de desenvolvimento:
1. Os quatro objetivos (info, custo, sucesso, risco) têm escalas comparáveis?
   Se um deles domina o retorno, w praticamente não muda a recompensa escalarizada.
2. A distribuição de ações da política muda quando w muda, nos mesmos estados?

Uso:
    .venv/bin/python scripts/diagnose_mo_conditioning.py \
        --model runs_mo/moppo_S2_hand_s0/final_model.pt --out results_mo/diagnostics
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dal.config import load_config
from dal.mo_env import make_mo_env
from dal.mo_eval import load_mo_model

OBJECTIVE_NAMES = ("info", "cost", "success", "risk")
UNIFORM_WEIGHTS = np.full(4, 0.25, dtype=np.float32)
PROBE_WEIGHTS = {
    "unif": UNIFORM_WEIGHTS,
    "fast": np.array([0.15, 0.55, 0.15, 0.15], dtype=np.float32),
    "safe": np.array([0.15, 0.15, 0.15, 0.55], dtype=np.float32),
    "only_info": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    "only_cost": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    "only_success": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    "only_risk": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
}


def to_tensor_obs(obs: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """Converte a observação do ambiente para tensores com dimensão de lote."""
    return {key: torch.from_numpy(value).unsqueeze(0).float() for key, value in obs.items()}


def collect_rollouts(net, env, episodes: int, seed_base: int) -> tuple[pd.DataFrame, list[dict]]:
    """Roda a política com w uniforme e guarda os retornos por objetivo e os estados visitados."""
    returns_rows = []
    visited_states = []
    for episode_idx in range(episodes):
        obs, _ = env.reset(seed=seed_base + episode_idx, options={"w": UNIFORM_WEIGHTS})
        episode_return = np.zeros(len(OBJECTIVE_NAMES), dtype=np.float64)
        done = False
        while not done:
            visited_states.append({key: value.copy() for key, value in obs.items()})
            action = net.act(to_tensor_obs(obs), deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            # Usa a recompensa bruta: queremos a escala natural de cada objetivo.
            episode_return += info["r_vec_raw"]
            done = terminated or truncated
        returns_rows.append(dict(zip(OBJECTIVE_NAMES, episode_return)))
    return pd.DataFrame(returns_rows), visited_states


def action_probabilities(net, state: dict[str, np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Calcula π(·|s, w) trocando só o bloco de pesos do vetor de observação."""
    probe = {key: value.copy() for key, value in state.items()}
    probe["vec"][: len(weights)] = weights
    tensor_obs = to_tensor_obs(probe)
    with torch.no_grad():
        features = net.get_features(tensor_obs)
        logits = net.actor(net._get_head_input(features, tensor_obs["vec"]))
    return torch.softmax(logits, dim=-1).squeeze(0).numpy()


def measure_weight_sensitivity(net, states: list[dict]) -> pd.DataFrame:
    """Mede, por par de w, a variação total média entre políticas e a concordância do argmax."""
    probabilities = {
        name: np.stack([action_probabilities(net, state, weights) for state in states])
        for name, weights in PROBE_WEIGHTS.items()
    }
    rows = []
    for name_a, name_b in combinations(PROBE_WEIGHTS, 2):
        prob_a, prob_b = probabilities[name_a], probabilities[name_b]
        total_variation = 0.5 * np.abs(prob_a - prob_b).sum(axis=1)
        argmax_agreement = (prob_a.argmax(axis=1) == prob_b.argmax(axis=1)).mean()
        rows.append({
            "weights_a": name_a,
            "weights_b": name_b,
            "mean_total_variation": float(total_variation.mean()),
            "max_total_variation": float(total_variation.max()),
            "argmax_agreement": float(argmax_agreement),
        })
    return pd.DataFrame(rows)


def summarize_objective_scales(returns: pd.DataFrame) -> dict:
    """Resume a escala (média e desvio) e a correlação entre os retornos dos objetivos."""
    return {
        "mean": returns.mean().to_dict(),
        "std": returns.std().to_dict(),
        # A razão entre desvios mostra quanto um objetivo domina a vantagem escalarizada.
        "std_ratio_to_cost": (returns.std() / max(returns["cost"].std(), 1e-9)).to_dict(),
        "spearman": returns.corr(method="spearman").round(3).to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs_mo/default.yaml")
    parser.add_argument("--model", default="runs_mo/moppo_S2_hand_s0/final_model.pt")
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=7000)
    parser.add_argument("--out", default="results_mo/diagnostics")
    args = parser.parse_args()

    config = load_config(args.config)
    use_memory = not args.no_memory
    net = load_mo_model(Path(args.model), memory=use_memory)
    # Episódios na região de treino (norte), para não tocar no conjunto de teste.
    env = make_mo_env("hand", "S2", config, split="train", memory=use_memory)

    returns, states = collect_rollouts(net, env, args.episodes, args.seed_base)
    sensitivity = measure_weight_sensitivity(net, states)
    scales = summarize_objective_scales(returns)

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    returns.to_csv(output_dir / "objective_returns.csv", index=False)
    sensitivity.to_csv(output_dir / "weight_sensitivity.csv", index=False)
    (output_dir / "objective_scales.json").write_text(json.dumps(scales, indent=2))

    print(f"estados analisados: {len(states)}")
    print(json.dumps({k: scales[k] for k in ("mean", "std", "std_ratio_to_cost")}, indent=2))
    print(sensitivity.to_string(index=False))


if __name__ == "__main__":
    main()

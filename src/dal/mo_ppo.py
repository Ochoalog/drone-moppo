"""Implementação do MOPPO (PPO multiobjetivo) em PyTorch (AMOR, Alegre et al., SIGGRAPH 2025)."""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.distributions.categorical import Categorical

from .config import RUNS, load_config
from .mo_env import make_mo_env


# ============================================================================
# Rede neural
# ============================================================================
class LocExtractor(nn.Module):
    """3 convoluções com stride sobre o mapa (C, W, W) + tronco com projeção balanceada do vetor."""

    def __init__(self, c_in: int, h: int, w: int, nv: int, features_dim: int = 256):
        super().__init__()
        self.cnn = nn.Sequential(
            # ~1,3 M MAC por amostra: 64 -> 16 -> 8 -> 4
            nn.Conv2d(c_in, 16, 4, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, c_in, h, w)
            n_flat = self.cnn(dummy).shape[1]

        # Projeções balanceadas (128 da CNN + 128 do vetor) para a CNN não abafar o w
        self.cnn_proj = nn.Sequential(
            nn.Linear(n_flat, features_dim // 2),
            nn.ReLU(),
        )
        self.vec_proj = nn.Sequential(
            nn.Linear(nv, features_dim // 2),
            nn.ReLU(),
        )
        self.features_dim = features_dim

    def forward(self, obs_map: torch.Tensor, obs_vec: torch.Tensor) -> torch.Tensor:
        cnn_feat = self.cnn_proj(self.cnn(obs_map))
        # Normaliza o w para média zero e escala unitária (condicionamento, §8)
        if obs_vec.shape[1] >= 4:
            w_norm = (obs_vec[:, :4] - 0.25) * 4.0
            vec_norm = torch.cat([w_norm, obs_vec[:, 4:]], dim=1)
        else:
            vec_norm = obs_vec
        vec_feat = self.vec_proj(vec_norm)
        return torch.cat([cnn_feat, vec_feat], dim=1)


class MOPPONetwork(nn.Module):
    """Ator-crítico multiobjetivo com crítico vetorial V(s, w) em R^M (AMOR, Eq. 4)."""

    def __init__(
        self,
        c_in: int,
        h: int,
        w: int,
        nv: int,
        action_dim: int = 8,
        num_objectives: int = 4,
        features_dim: int = 256,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.num_objectives = num_objectives
        self.action_dim = action_dim
        self.extractor = LocExtractor(c_in, h, w, nv, features_dim=features_dim)

        # As cabeças do ator e do crítico recebem as features e também o w normalizado diretamente (AMOR §4.1 / §5)
        in_dim = features_dim + num_objectives
        self.actor = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_objectives),
        )

    def _get_head_input(self, feat: torch.Tensor, obs_vec: torch.Tensor) -> torch.Tensor:
        w_raw = obs_vec[:, :self.num_objectives]
        if self.num_objectives == 4:
            w_norm = (w_raw - 0.25) * 4.0
        else:
            w_norm = w_raw
        return torch.cat([feat, w_norm], dim=-1)

    def get_features(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.extractor(obs["map"], obs["vec"])

    def get_value(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Retorna V(s, w) com forma (B, num_objectives)."""
        feat = self.get_features(obs)
        head_in = self._get_head_input(feat, obs["vec"])
        return self.critic(head_in)

    def get_action_and_value(
        self, obs: dict[str, torch.Tensor], action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.get_features(obs)
        head_in = self._get_head_input(feat, obs["vec"])
        logits = self.actor(head_in)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(head_in)
        return action, log_prob, entropy, value

    def act(self, obs: dict[str, torch.Tensor], deterministic: bool = True) -> int:
        with torch.no_grad():
            feat = self.get_features(obs)
            head_in = self._get_head_input(feat, obs["vec"])
            logits = self.actor(head_in)
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = Categorical(logits=logits).sample()
            return int(action.item())


# ============================================================================
# GAE e funções de perda
# ============================================================================
def compute_vector_gae(
    rewards: np.ndarray | torch.Tensor,  # (T, num_envs, M)
    values: np.ndarray | torch.Tensor,   # (T + 1, num_envs, M)
    dones: np.ndarray | torch.Tensor,    # (T + 1, num_envs)
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calcula a estimativa de vantagem generalizada (GAE) vetorial sobre os M objetivos.

    Retorna:
        advantages: (T, num_envs, M)
        returns:    (T, num_envs, M)
    """
    if isinstance(rewards, np.ndarray):
        rewards = torch.from_numpy(rewards).float()
    if isinstance(values, np.ndarray):
        values = torch.from_numpy(values).float()
    if isinstance(dones, np.ndarray):
        dones = torch.from_numpy(dones).float()

    T, num_envs, M = rewards.shape
    advantages = torch.zeros((T, num_envs, M), dtype=torch.float32, device=rewards.device)
    last_gae = torch.zeros((num_envs, M), dtype=torch.float32, device=rewards.device)

    for t in reversed(range(T)):
        next_non_terminal = 1.0 - dones[t + 1].unsqueeze(-1)
        next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

    returns = advantages + values[:-1]
    return advantages, returns


def compute_scalar_gae(
    rewards: np.ndarray | torch.Tensor,  # (T, num_envs)
    values: np.ndarray | torch.Tensor,   # (T + 1, num_envs)
    dones: np.ndarray | torch.Tensor,    # (T + 1, num_envs)
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE escalar padrão, usado para verificação (especificação §3a)."""
    if isinstance(rewards, np.ndarray):
        rewards = torch.from_numpy(rewards).float()
    if isinstance(values, np.ndarray):
        values = torch.from_numpy(values).float()
    if isinstance(dones, np.ndarray):
        dones = torch.from_numpy(dones).float()

    T, num_envs = rewards.shape
    advantages = torch.zeros((T, num_envs), dtype=torch.float32, device=rewards.device)
    last_gae = torch.zeros((num_envs,), dtype=torch.float32, device=rewards.device)

    for t in reversed(range(T)):
        next_non_terminal = 1.0 - dones[t + 1]
        next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

    returns = advantages + values[:-1]
    return advantages, returns


def compute_losses(
    net: MOPPONetwork,
    obs: dict[str, torch.Tensor],
    actions: torch.Tensor,
    old_logprobs: torch.Tensor,
    adv_scalar: torch.Tensor,
    target_returns: torch.Tensor,
    clip_range: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    return_logratio: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Calcula as perdas do MOPPO: política, valor, bônus de entropia e perda total.

    Com `return_logratio=True`, devolve também log(π_novo/π_antigo) do mesmo forward, para medir
    KL e clipfrac sem um segundo forward (convenção do CleanRL/SB3: antes do passo do otimizador).
    """
    _, new_logprob, entropy, new_value = net.get_action_and_value(obs, actions)

    # Normaliza a vantagem escalarizada por mini-lote (AMOR §4.2)
    adv_norm = (adv_scalar - adv_scalar.mean()) / (adv_scalar.std() + 1e-8)

    # Perda da política (objetivo substituto com clipping)
    logratio = new_logprob - old_logprobs
    ratio = torch.exp(logratio)
    pg_loss1 = -adv_norm * ratio
    pg_loss2 = -adv_norm * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    # Perda do valor: soma sobre os objetivos de MSE(V_i, R_i) (especificação §3)
    if net.num_objectives == 1 and target_returns.dim() == 1:
        v_loss = 0.5 * ((new_value.squeeze(-1) - target_returns) ** 2).mean()
    else:
        v_loss = 0.5 * ((new_value - target_returns) ** 2).sum(dim=-1).mean()

    ent_loss = entropy.mean()
    total_loss = pg_loss - ent_coef * ent_loss + vf_coef * v_loss

    if return_logratio:
        return total_loss, pg_loss, v_loss, ent_loss, logratio.detach()
    return total_loss, pg_loss, v_loss, ent_loss


# ============================================================================
# Fábrica de ambientes para o AsyncVectorEnv
# ============================================================================
def _make_env_fn(
    descriptor: str,
    scenario: str,
    cfg: dict,
    split: str,
    rank: int,
    seed: int,
    memory: bool = True,
    default_w: np.ndarray | None = None,
    scale_factors: np.ndarray | None = None,
):
    def thunk():
        from threadpoolctl import threadpool_limits

        # O processo do ambiente nasce por fork com o numpy já importado: OMP_NUM_THREADS não vale mais.
        # Sem este limite, cada ambiente abre 8 threads do OpenBLAS e os 8 ambientes disputam a CPU.
        threadpool_limits(limits=1)
        torch.set_num_threads(1)
        base = make_mo_env(
            descriptor,
            scenario,
            cfg,
            split=split,
            memory=memory,
            default_w=default_w,
            scale_factors=scale_factors,
        )
        base.reset(seed=seed + 1000 * rank)
        base.action_space.seed(seed + rank)
        return base

    return thunk


# ============================================================================
# Avaliação e gates
# ============================================================================
def evaluate_moppo(
    net: MOPPONetwork,
    cfg: dict,
    scenario: str,
    descriptor: str,
    split: str,
    w: np.ndarray | list[float],
    n_episodes: int = 50,
    seed_base: int = 2024,
    memory: bool = True,
    scale_factors: np.ndarray | None = None,
    deterministic: bool = True,
) -> dict[str, float | list[float]]:
    """Avalia uma rede MOPPO com um w fixo em episódios pareados.

    Com `deterministic=False` as ações são amostradas, com a semente do torch fixada pelo seed do episódio.
    """
    net.eval()
    env = make_mo_env(
        descriptor,
        scenario,
        cfg,
        split=split,
        memory=memory,
        default_w=w,
        scale_factors=scale_factors,
    )

    successes = []
    false_locs = []
    distances = []
    steps_list = []

    with torch.no_grad():
        for i in range(n_episodes):
            ep_seed = seed_base + i
            if not deterministic:
                torch.manual_seed(ep_seed)
            obs, _ = env.reset(seed=ep_seed, options={"w": w})
            done = False
            while not done:
                t_obs = {
                    "map": torch.from_numpy(obs["map"]).unsqueeze(0).float(),
                    "vec": torch.from_numpy(obs["vec"]).unsqueeze(0).float(),
                }
                action = net.act(t_obs, deterministic=deterministic)
                obs, _, term, trunc, info = env.step(action)
                done = term or trunc

            r = env.ep.last
            successes.append(float(r.success))
            false_locs.append(float(r.false_loc))
            distances.append(float(env.ep.distance_m))
            steps_list.append(int(env.ep.t))

    env.close()
    net.train()

    return {
        "success": float(np.mean(successes)),
        "false_loc": float(np.mean(false_locs)),
        "distance_mean": float(np.mean(distances)),
        "steps_median": float(np.median(steps_list)),
        "raw_successes": successes,
        "raw_false_locs": false_locs,
        "raw_distances": distances,
    }


GATE_WEIGHTS = {
    "unif": np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32),
    "fast": np.array([0.15, 0.55, 0.15, 0.15], dtype=np.float32),
    "safe": np.array([0.15, 0.15, 0.15, 0.55], dtype=np.float32),
}
INVALID_GATE_MARKER = "INVALID_GATE"
LEARNING_GATE_THRESHOLD = 0.50
GATE_EPISODES = 50


def _bootstrap_mean_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    """IC de 95% da média por bootstrap simples."""
    rng = np.random.default_rng(seed)
    boot_means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def _conditioning_gate(net, cfg, scenario, descriptor, split, memory, scale_factors, deterministic: bool) -> dict:
    """Gate de condicionamento num modo: a distância de `fast` e `safe` difere (IC da diferença exclui 0)."""
    results = {
        name: evaluate_moppo(
            net, cfg, scenario, descriptor, split, GATE_WEIGHTS[name], n_episodes=GATE_EPISODES, seed_base=6000,
            memory=memory, scale_factors=scale_factors, deterministic=deterministic,
        )
        for name in ("fast", "safe")
    }
    diff_dist = np.array(results["safe"]["raw_distances"]) - np.array(results["fast"]["raw_distances"])
    ci_lo, ci_hi = _bootstrap_mean_ci(diff_dist)
    return {
        "dist_fast_mean": results["fast"]["distance_mean"],
        "dist_safe_mean": results["safe"]["distance_mean"],
        "diff_mean": float(diff_dist.mean()),
        "ci_95": [ci_lo, ci_hi],
        "passed": bool(ci_lo > 0.0 or ci_hi < 0.0),
    }


def run_gate_checks(
    net: MOPPONetwork,
    cfg: dict,
    scenario: str,
    descriptor: str,
    split: str = "train",
    memory: bool = True,
    scale_factors: np.ndarray | None = None,
    log_file: Path | None = None,
    fixed_weights: np.ndarray | None = None,
) -> dict:
    """Verifica os gates aos 25% do treino (§8 da v1; regra A4 do DECISIONS_MO_V2.md).

    - Gate de aprendizado: sucesso no treino >= 0,5 em S2 (com w uniforme na MO, ou com o w fixo do
      PPO de peso fixo); a política aleatória faz ~0,42. Decide no modo amostrado.
    - Gate de condicionamento (só na MO): a distância difere entre w_fast e w_safe (IC exclui 0).
      É medido nos dois modos; decide no modo amostrado, e o argmax fica registrado.
    """
    learning_weights = GATE_WEIGHTS["unif"] if fixed_weights is None else np.asarray(fixed_weights, np.float32)
    learning = {}
    for mode, deterministic in (("sampled", False), ("argmax", True)):
        result = evaluate_moppo(
            net, cfg, scenario, descriptor, split, learning_weights, n_episodes=GATE_EPISODES, seed_base=5000,
            memory=memory, scale_factors=scale_factors, deterministic=deterministic,
        )
        learning[mode] = {"success": result["success"], "passed": bool(result["success"] >= LEARNING_GATE_THRESHOLD)}
    report = {
        "learning_gate": {"threshold": LEARNING_GATE_THRESHOLD, "decision_mode": "sampled", **learning,
                          "passed": learning["sampled"]["passed"]},
    }

    if fixed_weights is None:
        conditioning = {
            mode: _conditioning_gate(net, cfg, scenario, descriptor, split, memory, scale_factors, deterministic)
            for mode, deterministic in (("sampled", False), ("argmax", True))
        }
        report["conditioning_gate"] = {"decision_mode": "sampled", **conditioning,
                                       "passed": conditioning["sampled"]["passed"]}

    report["valid"] = all(gate["passed"] for gate in report.values() if isinstance(gate, dict))
    lines = [f"[GATE] learning (success >= {LEARNING_GATE_THRESHOLD}): "
             + ", ".join(f"{m}={learning[m]['success']:.3f}" for m in learning)
             + f" -> {'PASSED' if report['learning_gate']['passed'] else 'FAILED'}"]
    if "conditioning_gate" in report:
        lines.append("[GATE] conditioning (distance safe - fast, CI excl. 0): " + ", ".join(
            f"{m}=[{c['ci_95'][0]:.1f}, {c['ci_95'][1]:.1f}] m" for m, c in conditioning.items())
            + f" -> {'PASSED' if report['conditioning_gate']['passed'] else 'FAILED'}")
    if not report["valid"]:
        lines.append(f"[GATE] {INVALID_GATE_MARKER}: esta execução não pode gerar vereditos")
    msg = "\n".join(lines)
    print(msg)
    if log_file:
        with open(log_file, "a") as f:
            f.write(msg + "\n")

    return report


# ============================================================================
# Rotina de treino
# ============================================================================
def train_moppo(
    cfg: dict,
    out_dir: Path,
    memory: bool = True,
    default_w: np.ndarray | list[float] | None = None,
    scale_factors: np.ndarray | list[float] | None = None,
    total_timesteps: int | None = None,
    seed: int | None = None,
) -> Path:
    """Treina uma política MOPPO no Taquari S2 (norte) com AsyncVectorEnv."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    ppo_cfg = cfg["ppo"]
    env_cfg = cfg["env"]
    seed = seed if seed is not None else int(ppo_cfg.get("seed", 0))
    total_timesteps = (
        total_timesteps if total_timesteps is not None else int(ppo_cfg["total_timesteps"])
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(int(os.environ.get("DAL_TORCH_THREADS", max(1, (os.cpu_count() or 2) // 2))))

    n_envs = int(ppo_cfg.get("n_envs", os.cpu_count() or 4))
    n_steps = int(ppo_cfg.get("n_steps", 256))
    batch_size = int(ppo_cfg.get("batch_size", 2048))
    n_epochs = int(ppo_cfg.get("n_epochs", 5))
    lr = float(ppo_cfg.get("learning_rate", 3.0e-4))
    gamma = float(ppo_cfg.get("gamma", 0.99))
    gae_lambda = float(ppo_cfg.get("gae_lambda", 0.95))
    clip_range = float(ppo_cfg.get("clip_range", 0.2))
    ent_coef = float(ppo_cfg.get("ent_coef", 0.01))
    vf_coef = float(ppo_cfg.get("vf_coef", 0.5))
    features_dim = int(ppo_cfg.get("features_dim", 256))
    max_grad_norm = float(ppo_cfg.get("max_grad_norm", 0.5))

    desc = env_cfg["descriptor"]
    scen = env_cfg["scenario"]
    split = env_cfg.get("split", "train")

    env_fns = [
        _make_env_fn(
            desc, scen, cfg, split, i, seed,
            memory=memory, default_w=default_w, scale_factors=scale_factors
        )
        for i in range(n_envs)
    ]
    envs = gym.vector.AsyncVectorEnv(env_fns)

    # Dimensões do espaço de observação
    c_in, h, w = envs.single_observation_space["map"].shape
    nv = envs.single_observation_space["vec"].shape[0]
    action_dim = int(envs.single_action_space.n)
    num_objectives = 4 if default_w is None or len(default_w) == 4 else len(default_w)

    net = MOPPONetwork(
        c_in, h, w, nv,
        action_dim=action_dim,
        num_objectives=num_objectives,
        features_dim=features_dim,
    )
    optimizer = optim.Adam(net.parameters(), lr=lr, eps=1e-5)

    # Buffers do rollout
    obs_map = torch.zeros((n_steps, n_envs, c_in, h, w), dtype=torch.float32)
    obs_vec = torch.zeros((n_steps, n_envs, nv), dtype=torch.float32)
    actions = torch.zeros((n_steps, n_envs), dtype=torch.long)
    logprobs = torch.zeros((n_steps, n_envs), dtype=torch.float32)
    rewards_vec = torch.zeros((n_steps, n_envs, num_objectives), dtype=torch.float32)
    dones = torch.zeros((n_steps, n_envs), dtype=torch.float32)
    values = torch.zeros((n_steps + 1, n_envs, num_objectives), dtype=torch.float32)
    step_weights = torch.zeros((n_steps, n_envs, num_objectives), dtype=torch.float32)

    # Logs em CSV
    progress_file = out_dir / "progress.csv"
    progress_cols = [
        "step", "timesteps", "fps", "time_elapsed", "policy_loss", "value_loss",
        "entropy", "total_loss", "approx_kl", "clipfrac", "mean_return_unif"
    ]
    progress_f = open(progress_file, "w")
    progress_f.write(",".join(progress_cols) + "\n")

    # Checkpoints: no máximo 3 por execução (§0)
    ck_dir = out_dir / "checkpoints"
    ck_dir.mkdir(parents=True, exist_ok=True)
    num_rollouts = total_timesteps // (n_steps * n_envs)
    ck_steps = [
        int(num_rollouts * 0.25),
        int(num_rollouts * 0.50),
        int(num_rollouts * 0.75),
    ]

    gate_results = {}
    gate_checked_25 = False

    t0 = time.time()
    next_obs, _ = envs.reset(seed=seed)
    next_obs_map = torch.from_numpy(next_obs["map"]).float()
    next_obs_vec = torch.from_numpy(next_obs["vec"]).float()
    next_done = torch.zeros(n_envs, dtype=torch.float32)

    global_step = 0

    for rollout in range(num_rollouts):
        # 1. Coleta do rollout (1 thread evita disputa com os processos dos ambientes)
        torch.set_num_threads(1)
        net.eval()
        for step in range(n_steps):
            global_step += n_envs
            obs_map[step] = next_obs_map
            obs_vec[step] = next_obs_vec
            dones[step] = next_done

            # O w ativo de cada ambiente são as 4 primeiras dimensões de obs_vec
            step_weights[step] = next_obs_vec[:, :num_objectives]

            with torch.no_grad():
                cur_obs = {"map": next_obs_map, "vec": next_obs_vec}
                action, logprob, _, val = net.get_action_and_value(cur_obs)
                values[step] = val

            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = torch.from_numpy((terminations | truncations).astype(np.float32))
            rewards_vec[step] = torch.from_numpy(infos["r_vec"]).float()

            next_obs_map = torch.from_numpy(next_obs["map"]).float()
            next_obs_vec = torch.from_numpy(next_obs["vec"]).float()

        # Valor de bootstrap no fim do rollout
        with torch.no_grad():
            cur_obs = {"map": next_obs_map, "vec": next_obs_vec}
            values[n_steps] = net.get_value(cur_obs)

        # 2. GAE vetorial
        all_dones = torch.cat([dones, next_done.unsqueeze(0)], dim=0)
        advantages, returns = compute_vector_gae(
            rewards_vec, values, all_dones, gamma=gamma, gae_lambda=gae_lambda
        )

        # Escalariza as vantagens: A_escalar = A · w (AMOR §4.2)
        adv_scalar = (advantages * step_weights).sum(dim=-1)

        # 3. Atualizações por mini-lote
        b_obs_map = obs_map.reshape(-1, c_in, h, w)
        b_obs_vec = obs_vec.reshape(-1, nv)
        b_actions = actions.reshape(-1)
        b_logprobs = logprobs.reshape(-1)
        b_adv_scalar = adv_scalar.reshape(-1)
        b_returns = returns.reshape(-1, num_objectives)

        batch_total = n_steps * n_envs
        inds = np.arange(batch_total)

        net.train()
        # Durante o update os processos dos ambientes estão parados: usa todos os núcleos.
        torch.set_num_threads(os.cpu_count() or 4)
        epoch_pg_losses, epoch_v_losses, epoch_ent_losses, epoch_tot_losses = [], [], [], []
        epoch_kls, epoch_clips = [], []

        for _ in range(n_epochs):
            np.random.shuffle(inds)
            for start in range(0, batch_total, batch_size):
                end = start + batch_size
                mb_inds = inds[start:end]

                mb_obs = {"map": b_obs_map[mb_inds], "vec": b_obs_vec[mb_inds]}
                mb_actions = b_actions[mb_inds]
                mb_logprobs = b_logprobs[mb_inds]
                mb_adv = b_adv_scalar[mb_inds]
                mb_returns = b_returns[mb_inds]

                tot_loss, pg_loss, v_loss, ent_loss, logratio = compute_losses(
                    net, mb_obs, mb_actions, mb_logprobs, mb_adv, mb_returns,
                    clip_range=clip_range, vf_coef=vf_coef, ent_coef=ent_coef, return_logratio=True,
                )

                optimizer.zero_grad()
                tot_loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
                optimizer.step()

                # Métricas
                # Medido no forward da perda (antes do passo); antes havia um forward extra só para isto.
                with torch.no_grad():
                    approx_kl = ((torch.exp(logratio) - 1) - logratio).mean().item()
                    clipfrac = ((torch.exp(logratio) - 1.0).abs() > clip_range).float().mean().item()

                epoch_pg_losses.append(pg_loss.item())
                epoch_v_losses.append(v_loss.item())
                epoch_ent_losses.append(ent_loss.item())
                epoch_tot_losses.append(tot_loss.item())
                epoch_kls.append(approx_kl)
                epoch_clips.append(clipfrac)

        elapsed = time.time() - t0
        fps = global_step / max(1e-3, elapsed)

        # Registro
        progress_f.write(
            f"{rollout},{global_step},{fps:.1f},{elapsed:.1f},"
            f"{np.mean(epoch_pg_losses):.5f},{np.mean(epoch_v_losses):.5f},"
            f"{np.mean(epoch_ent_losses):.5f},{np.mean(epoch_tot_losses):.5f},"
            f"{np.mean(epoch_kls):.5f},{np.mean(epoch_clips):.5f},0.0\n"
        )
        progress_f.flush()

        if (rollout + 1) % max(1, num_rollouts // 20) == 0:
            print(
                f"[train_moppo] Rollout {rollout + 1}/{num_rollouts} | Step {global_step}/{total_timesteps} "
                f"| FPS: {fps:.1f} | TotLoss: {np.mean(epoch_tot_losses):.4f} "
                f"| PGLoss: {np.mean(epoch_pg_losses):.4f} | VLoss: {np.mean(epoch_v_losses):.4f}"
            )

        # Checkpoints (no máximo 3: ~25%, ~50%, ~75%)
        if rollout in ck_steps:
            ck_idx = ck_steps.index(rollout) + 1
            ck_file = ck_dir / f"moppo_step_{global_step}_ck{ck_idx}.pt"
            torch.save(net.state_dict(), ck_file)
            print(f"[checkpoint] Saved checkpoint {ck_idx}/3 to {ck_file.name}")

        # Gates aos 25% do treino (§8); agora também nos PPOs de peso fixo e na MO sem memória
        if not gate_checked_25 and rollout >= ck_steps[0]:
            gate_checked_25 = True
            print("[GATE] Checking gates at 25% training...")
            gate_results = run_gate_checks(
                net, cfg, scen, desc, split=split, memory=memory,
                scale_factors=scale_factors, log_file=out_dir / "gates.log", fixed_weights=default_w,
            )
            (out_dir / "gate_check_25pct.json").write_text(json.dumps(gate_results, indent=2))
            if not gate_results["valid"]:
                # Na v1, a falha do gate passou em silêncio e o problema só apareceu no relatório final.
                (out_dir / INVALID_GATE_MARKER).write_text(json.dumps(gate_results, indent=2))
                warnings.warn(
                    f"{INVALID_GATE_MARKER} em {out_dir.name}: um gate falhou aos 25% do treino. "
                    "Não gere vereditos a partir desta execução.",
                    stacklevel=1,
                )

    envs.close()
    progress_f.close()

    # Salva o modelo final
    torch.save(net.state_dict(), out_dir / "final_model.pt")

    train_meta = {
        "scenario": scen,
        "descriptor": desc,
        "seed": seed,
        "memory": memory,
        "default_w": [float(x) for x in default_w] if default_w is not None else None,
        "total_timesteps": global_step,
        "seconds": elapsed,
        "fps": fps,
        "gate_results": gate_results,
    }
    (out_dir / "train_meta.json").write_text(json.dumps(train_meta, indent=2))
    print(f"[train_moppo] Finished {global_step} timesteps in {elapsed:.1f}s ({fps:.1f} FPS) -> {out_dir}")

    return out_dir


# ============================================================================
# CLI
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--scenario", default="S2")
    ap.add_argument("--descriptor", default="hand")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timesteps", type=int)
    ap.add_argument("--no-memory", action="store_true", help="Memory ablation")
    ap.add_argument("--w", nargs="*", type=float, help="Fixed weights for single-weight PPO control")
    ap.add_argument("--out", help="Output run directory")
    args = ap.parse_args()

    cfg = load_config(args.config, {"env": {"scenario": args.scenario, "descriptor": args.descriptor}})
    out = Path(args.out) if args.out else RUNS / f"moppo_{args.scenario}_{args.descriptor}_s{args.seed}"
    train_moppo(
        cfg,
        out,
        memory=not args.no_memory,
        default_w=np.array(args.w, dtype=np.float32) if args.w else None,
        total_timesteps=args.timesteps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

"""PPO em lote no ambiente realista (o MOPPO da v1/v2 com m = 1, §3.2 do plano v3).

Reaproveita `MOPPONetwork`, `compute_vector_gae` e `compute_losses` (o teste de redução da v2 garante
que, com m = 1, o MOPPO é o PPO escalar). Os episódios são contínuos: quem termina é reiniciado logo
depois de a transição ser guardada (sem a transição espúria do autoreset NEXT_STEP, ver A7 da v2).

Retomável: `checkpoint.pt` a cada ≤ `checkpoint_minutes` (rede, otimizador, passo global, contador de
episódios); ao retomar, os ambientes recomeçam episódios novos. Gate de aprendizado aos 25% do orçamento:
se falhar, grava o marcador `INVALID_GATE` e, no fim, renomeia a pasta com o sufixo `_INVALID_GATE`.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dal.mo_ppo import MOPPONetwork, compute_losses, compute_vector_gae
from dal.v3.agents.baselines import RandomPolicy, run_policy
from dal.v3.envs.realistic_env import RealisticBatchedEnv

INVALID_GATE_MARKER = "INVALID_GATE"
TRAIN_SEED_BASE = 10_000_000  # episódios de treino; nunca colidem com os de avaliação (2024 + i)
GATE_SEED_BASE = 9_000_000


@dataclass
class PPOTrainConfig:
    total_timesteps: int = 1_400_000
    n_envs: int = 512
    n_steps: int = 32
    n_epochs: int = 5
    batch_size: int = 4096
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    features_dim: int = 256
    change_prior_input: bool = False  # política CA: o prior de mudança entra como canal
    # Recompensa (mesmas constantes da v1; o custo por passo vira custo por tempo de voo: 0,01 a cada 6 s)
    time_cost_per_6s: float = 0.01
    border_penalty: float = 0.1
    success_reward: float = 1.0
    false_loc_penalty: float = 1.0
    seed: int = 0
    checkpoint_minutes: float = 15.0
    gate_fraction: float = 0.25
    gate_episodes: int = 200
    gate_min_advantage_over_random: float = 0.05


class PolicyAdapter:
    """Monta a observação da rede (vetor com w = 1 na frente, como espera a MOPPONetwork com m = 1)."""

    def __init__(self, change_prior_input: bool):
        self.change_prior_input = change_prior_input

    def observe(self, env: RealisticBatchedEnv) -> dict[str, torch.Tensor]:
        obs = env.policy_obs(include_change_prior=self.change_prior_input)
        ones = torch.ones(env.n, 1, device=obs["vec"].device)
        return {"map": obs["map"], "vec": torch.cat([ones, obs["vec"]], dim=1)}


class NetworkPolicy:
    """Usa a rede treinada como política (para gates e avaliação): argmax ou amostrado."""

    name = "ppo"

    def __init__(self, net: MOPPONetwork, adapter: PolicyAdapter, deterministic: bool = True):
        self.net, self.adapter, self.deterministic = net, adapter, deterministic

    def reset(self, env, seeds):
        self.generator = torch.Generator(device="cpu").manual_seed(int(seeds[0]) if seeds else 0)

    @torch.no_grad()
    def act(self, env):
        obs = self.adapter.observe(env)
        logits = self.net.actor(self.net._get_head_input(self.net.get_features(obs), obs["vec"]))
        if self.deterministic:
            return logits.argmax(1).cpu()
        return torch.multinomial(torch.softmax(logits.cpu().float(), 1), 1, generator=self.generator).squeeze(1)


def build_network(env: RealisticBatchedEnv, cfg: PPOTrainConfig, device: str) -> MOPPONetwork:
    adapter = PolicyAdapter(cfg.change_prior_input)
    env.reset([0])
    sample = adapter.observe(env)
    channels, height, width = sample["map"].shape[1:]
    return MOPPONetwork(channels, height, width, sample["vec"].shape[1], action_dim=env.n_actions, num_objectives=1,
                        features_dim=cfg.features_dim).to(device)


def learning_gate(env: RealisticBatchedEnv, net: MOPPONetwork, cfg: PPOTrainConfig) -> dict:
    """Sucesso da política (amostrada) − sucesso da aleatória nos mesmos episódios de treino."""
    seeds = [GATE_SEED_BASE + i for i in range(cfg.gate_episodes)]
    policy = NetworkPolicy(net, PolicyAdapter(cfg.change_prior_input), deterministic=False)
    policy_success = float(run_policy(env, policy, seeds).success.float().mean())
    random_success = float(run_policy(env, RandomPolicy(), seeds).success.float().mean())
    advantage = policy_success - random_success
    return {"policy_success": policy_success, "random_success": random_success, "advantage": advantage,
            "threshold": cfg.gate_min_advantage_over_random, "passed": advantage >= cfg.gate_min_advantage_over_random}


def _reward(cfg: PPOTrainConfig, entropy_before, entropy_after, outcome) -> torch.Tensor:
    reward = (entropy_before - entropy_after).float()
    reward -= cfg.time_cost_per_6s * outcome["step_time_s"].float() / 6.0
    reward -= cfg.border_penalty * outcome["blocked"].float()
    reward += cfg.success_reward * outcome["success"].float() - cfg.false_loc_penalty * outcome["false_loc"].float()
    return reward


def train(env: RealisticBatchedEnv, gate_env: RealisticBatchedEnv, cfg: PPOTrainConfig, out_dir: Path,
          device: str = "cpu", log=print) -> Path:
    """Treina (ou retoma) e devolve a pasta final (com sufixo `_INVALID_GATE` se o gate falhar)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
    torch.manual_seed(cfg.seed)
    net = build_network(env, cfg, device)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate, eps=1e-5)
    adapter = PolicyAdapter(cfg.change_prior_input)
    state = {"global_step": 0, "episodes_started": 0, "gate": None}
    checkpoint = out_dir / "checkpoint.pt"
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        net.load_state_dict(saved["net"])
        optimizer.load_state_dict(saved["optimizer"])
        state = saved["state"]
        log(f"[ppo] retomando de {state['global_step']} passos")

    def next_seeds(count: int) -> list[int]:
        seeds = [TRAIN_SEED_BASE + cfg.seed * 1_000_000 + state["episodes_started"] + i for i in range(count)]
        state["episodes_started"] += count
        return seeds

    n, T = cfg.n_envs, cfg.n_steps
    env.reset(next_seeds(n))
    progress = (out_dir / "progress.csv").open("a")
    if progress.tell() == 0:
        progress.write("global_step,steps_per_s,mean_return_recent,success_recent,false_loc_recent,policy_loss,value_loss,entropy\n")
    started, last_checkpoint = time.time(), time.time()
    recent = {"success": [], "false_loc": []}
    while state["global_step"] < cfg.total_timesteps:
        segment_start = time.time()
        buffers = {"map": [], "vec": [], "actions": [], "logprobs": [], "rewards": [], "dones": [], "values": []}
        net.eval()
        for _ in range(T):
            obs = adapter.observe(env)
            with torch.no_grad():
                action, logprob, _, value = net.get_action_and_value(obs)
            entropy_before = env.entropy()
            outcome = env.step(action)
            reward = _reward(cfg, entropy_before, env.entropy(), outcome)
            finished = env.done.clone()
            for key, item in (("map", obs["map"]), ("vec", obs["vec"]), ("actions", action), ("logprobs", logprob),
                              ("rewards", reward.view(-1, 1)), ("dones", finished.float()), ("values", value)):
                buffers[key].append(item)
            if bool(finished.any()):
                rows = torch.nonzero(finished).flatten()
                recent["success"] += env.success[rows].float().tolist()
                recent["false_loc"] += env.false_loc[rows].float().tolist()
                env.reset_rows(rows, next_seeds(int(rows.numel())))
            state["global_step"] += n
        with torch.no_grad():
            last_value = net.get_value(adapter.observe(env))
        rewards = torch.stack(buffers["rewards"])  # (T, N, 1)
        values = torch.cat([torch.stack(buffers["values"]), last_value.unsqueeze(0)])
        # dones[t] = o episódio terminou no passo t; o GAE espera "o estado t+1 é início de episódio".
        dones = torch.cat([torch.zeros(1, n, device=rewards.device), torch.stack(buffers["dones"])])
        advantages, returns = compute_vector_gae(rewards, values, dones, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda)
        flat = {"map": torch.stack(buffers["map"]).flatten(0, 1), "vec": torch.stack(buffers["vec"]).flatten(0, 1)}
        actions = torch.stack(buffers["actions"]).flatten()
        logprobs = torch.stack(buffers["logprobs"]).flatten()
        adv = advantages.squeeze(-1).flatten()
        ret = returns.flatten(0, 1)
        net.train()
        losses = []
        for _ in range(cfg.n_epochs):
            order = torch.randperm(adv.numel(), device=adv.device)
            for start in range(0, adv.numel(), cfg.batch_size):
                idx = order[start:start + cfg.batch_size]
                minibatch = {"map": flat["map"][idx], "vec": flat["vec"][idx]}
                total, pg, v, ent = compute_losses(net, minibatch, actions[idx], logprobs[idx], adv[idx], ret[idx],
                                                   clip_range=cfg.clip_range, vf_coef=cfg.vf_coef,
                                                   ent_coef=cfg.ent_coef)
                optimizer.zero_grad()
                total.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                optimizer.step()
                losses.append((pg.item(), v.item(), ent.item()))
        steps_per_s = n * T / (time.time() - segment_start)
        pg, v, ent = np.mean(losses, axis=0)
        window = slice(-500, None)
        progress.write(f"{state['global_step']},{steps_per_s:.1f},{float(rewards.sum(0).mean()):.4f},"
                       f"{np.mean(recent['success'][window] or [np.nan]):.4f},"
                       f"{np.mean(recent['false_loc'][window] or [np.nan]):.4f},{pg:.5f},{v:.5f},{ent:.4f}\n")
        progress.flush()

        if state["gate"] is None and state["global_step"] >= cfg.gate_fraction * cfg.total_timesteps:
            state["gate"] = learning_gate(gate_env, net, cfg)
            (out_dir / "gate_25pct.json").write_text(json.dumps(state["gate"], indent=1))
            log(f"[ppo] gate de aprendizado: {state['gate']}")
            if not state["gate"]["passed"]:
                (out_dir / INVALID_GATE_MARKER).write_text(json.dumps(state["gate"]))
                log(f"[ppo] {INVALID_GATE_MARKER}: a execução não gera vereditos")
        if time.time() - last_checkpoint >= cfg.checkpoint_minutes * 60:
            torch.save({"net": net.state_dict(), "optimizer": optimizer.state_dict(), "state": state}, checkpoint)
            last_checkpoint = time.time()
    progress.close()
    torch.save(net.state_dict(), out_dir / "final_model.pt")
    torch.save({"net": net.state_dict(), "optimizer": optimizer.state_dict(), "state": state}, checkpoint)
    summary = {"global_step": state["global_step"], "minutes": (time.time() - started) / 60, "gate": state["gate"]}
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=1))
    if (out_dir / INVALID_GATE_MARKER).exists() and not out_dir.name.endswith(INVALID_GATE_MARKER):
        final_dir = out_dir.with_name(out_dir.name + "_" + INVALID_GATE_MARKER)
        out_dir.rename(final_dir)
        return final_dir
    return out_dir

"""PPO training (stable-baselines3) with a small custom CNN extractor."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from torch import nn

from .config import RUNS, load_config


class LocExtractor(BaseFeaturesExtractor):
    """3 strided convs over the (C, W, W) map tensor + MLP, concatenated with the vector input."""

    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        C, H, W = observation_space["map"].shape
        nv = observation_space["vec"].shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(C, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n = self.cnn(torch.zeros(1, C, H, W)).shape[1]
        self.mlp = nn.Sequential(nn.Linear(n + nv, features_dim), nn.ReLU())

    def forward(self, obs):
        return self.mlp(torch.cat([self.cnn(obs["map"]), obs["vec"]], dim=1))


class InfoLogger(BaseCallback):
    """Logs success / false-localization rates of recent training episodes."""

    def _on_rollout_end(self) -> None:
        buf = list(self.model.ep_info_buffer)
        if buf and "success" in buf[0]:
            self.logger.record("rollout/success_rate", float(np.mean([e["success"] for e in buf])))
            self.logger.record("rollout/false_loc_rate", float(np.mean([e["false_loc"] for e in buf])))

    def _on_step(self) -> bool:
        return True


def _env_fn(descriptor: str, scenario: str, cfg: dict, rank: int, seed: int):
    def f():
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        torch.set_num_threads(1)
        from .env import make_env

        env = make_env(descriptor, scenario, cfg, split="train")
        env.reset(seed=seed + 1000 * rank)
        env.action_space.seed(seed + rank)
        return env

    return f


def run_name(cfg: dict) -> str:
    e = cfg["env"]
    return f"{e['scenario']}_{e['descriptor']}_s{cfg['ppo']['seed']}"


def train(cfg: dict, out_root: Path) -> Path:
    p, e = cfg["ppo"], cfg["env"]
    out = out_root / run_name(cfg)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    seed = int(p["seed"])
    set_random_seed(seed)
    torch.set_num_threads(int(os.environ.get("DAL_TORCH_THREADS", max(1, (os.cpu_count() or 2) // 2))))
    n_envs = int(p["n_envs"])
    fns = [_env_fn(e["descriptor"], e["scenario"], cfg, i, seed) for i in range(n_envs)]
    venv = SubprocVecEnv(fns) if n_envs > 1 else DummyVecEnv(fns)
    venv = VecMonitor(venv, str(out / "monitor.csv"), info_keywords=("success", "false_loc", "error_m"))
    venv.seed(seed)
    model = PPO(
        "MultiInputPolicy", venv, n_steps=p["n_steps"], batch_size=p["batch_size"], n_epochs=p["n_epochs"],
        learning_rate=p["learning_rate"], gamma=p["gamma"], gae_lambda=p["gae_lambda"], clip_range=p["clip_range"],
        ent_coef=p["ent_coef"], vf_coef=p["vf_coef"], seed=seed, verbose=0, device="cpu",
        policy_kwargs=dict(features_extractor_class=LocExtractor,
                           features_extractor_kwargs=dict(features_dim=p["features_dim"]),
                           net_arch=dict(pi=[128], vf=[128])),
    )
    model.set_logger(configure(str(out), ["csv", "tensorboard", "stdout"]))
    total = int(p["total_timesteps"])
    ck = CheckpointCallback(save_freq=max(1, total // p["checkpoints"] // n_envs), save_path=str(out / "checkpoints"),
                            name_prefix="ppo")
    t0 = time.time()
    model.learn(total_timesteps=total, callback=CallbackList([ck, InfoLogger()]), progress_bar=False)
    dt = time.time() - t0
    model.save(out / "final_model")
    meta = dict(name=run_name(cfg), seconds=dt, timesteps=int(model.num_timesteps), fps=model.num_timesteps / dt)
    (out / "train_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[train] {meta}")
    venv.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--scenario")
    ap.add_argument("--descriptor")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--timesteps", type=int)
    ap.add_argument("--out", help="run root (default runs/<config name>)")
    a = ap.parse_args()
    ov: dict = {"env": {}, "ppo": {}}
    if a.scenario:
        ov["env"]["scenario"] = a.scenario
    if a.descriptor:
        ov["env"]["descriptor"] = a.descriptor
    if a.seed is not None:
        ov["ppo"]["seed"] = a.seed
    if a.timesteps:
        ov["ppo"]["total_timesteps"] = a.timesteps
    cfg = load_config(a.config, ov)
    root = Path(a.out) if a.out else RUNS / Path(a.config).stem
    train(cfg, root)


if __name__ == "__main__":
    main()

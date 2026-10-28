"""Paired evaluation: every method sees the same episodes (window, start and dynamics noise are
fixed by the episode seed). Writes one row per episode."""
from __future__ import annotations

import argparse
import os
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import Policy, make_baselines
from .config import RESULTS, RUNS, load_config

COLUMNS = ["method", "model_seed", "scenario", "descriptor", "split", "episode", "seed", "success", "false_loc",
           "steps", "steps_to_success", "final_error_m", "distance_m", "n_blocked", "final_entropy",
           "decision_ms"]


class PPOPolicy(Policy):
    def __init__(self, path: str, deterministic: bool = True):
        from stable_baselines3 import PPO

        self.model = PPO.load(path, device="cpu")
        self.deterministic = deterministic
        self.name = "ppo"

    def act(self, ep) -> int:
        a, _ = self.model.predict(ep.policy_obs(), deterministic=self.deterministic)
        return int(a)


_CACHE: dict = {}


def _data(desc: str, scen: str, cfg: dict):
    from .env import ScenarioData

    k = (desc, scen)
    if k not in _CACHE:
        _CACHE[k] = ScenarioData.load(desc, scen, cfg)
    return _CACHE[k]


def _policy(spec: tuple, cfg: dict) -> Policy:
    k = ("policy",) + spec
    if k not in _CACHE:
        if spec[0] == "ppo":
            _CACHE[k] = PPOPolicy(spec[1], cfg["eval"].get("ppo_deterministic", True))
        else:
            _CACHE[k] = make_baselines(cfg)[spec[0]]
    return _CACHE[k]


def run_episodes(task: tuple) -> list[dict]:
    """Worker: (cfg, scenario, split, method spec, model_seed, episode indices, n_traj_to_save, traj dir)."""
    import torch

    torch.set_num_threads(1)
    from .env import Episode

    cfg, scen, split, spec, model_seed, idx, save_traj, traj_dir = task
    desc = cfg["env"]["descriptor"]
    data = _data(desc, scen, cfg)
    ep = Episode(data, cfg["filter"], split, cfg["data"]["cell_m"])
    pol = _policy(spec, cfg)
    label = spec[0]
    rows = []
    for i in idx:
        seed = int(cfg["eval"]["seed"]) + int(i)
        ep.reset(seed=seed)
        pol.reset(ep, np.random.default_rng([seed, 7]))
        keep = i < save_traj
        beliefs, maps = ([ep.filter.b.copy()], [ep.filter.map_estimate()]) if keep else (None, None)
        dts = []
        while not ep.done:
            t0 = time.perf_counter()
            a = pol.act(ep)
            dts.append(time.perf_counter() - t0)
            ep.step(a)
            if keep:
                beliefs.append(ep.filter.b.astype(np.float32))
                maps.append(ep.filter.map_estimate())
        r = ep.last
        rows.append(dict(method=label, model_seed=model_seed, scenario=scen, descriptor=desc, split=split,
                         episode=int(i), seed=seed, success=bool(r.success), false_loc=bool(r.false_loc),
                         steps=ep.t, steps_to_success=ep.t if r.success else np.nan, final_error_m=r.error_m,
                         distance_m=ep.distance_m, n_blocked=ep.n_blocked, final_entropy=r.entropy,
                         decision_ms=1000 * float(np.mean(dts)) if dts else np.nan))
        if keep:
            traj_dir.mkdir(parents=True, exist_ok=True)
            tag = f"{label}{'' if model_seed < 0 else model_seed}"
            np.savez_compressed(traj_dir / f"{split}_{scen}_{tag}_ep{i}.npz", traj=np.array(ep.traj),
                                map_est=np.array(maps), beliefs=np.array(beliefs, np.float32),
                                window=np.array([ep.r0, ep.c0]), success=r.success, false_loc=r.false_loc)
    return rows


def find_models(run_root: Path, scenario: str, descriptor: str) -> list[tuple[int, str]]:
    out = []
    for d in sorted(run_root.glob(f"{scenario}_{descriptor}_s*")):
        f = d / "final_model.zip"
        if f.exists():
            out.append((int(d.name.rsplit("_s", 1)[1]), str(f)))
    return out


def evaluate(cfg: dict, scenarios: list[str], methods: list[str], run_root: Path | None, episodes: int,
             split: str, out_csv: Path, workers: int, save_traj: int = 3, chunk: int = 50) -> pd.DataFrame:
    tasks = []
    for scen in scenarios:
        specs: list[tuple[tuple, int]] = [((m,), -1) for m in methods if m != "ppo"]
        if "ppo" in methods and run_root is not None:
            specs += [(("ppo", path), s) for s, path in find_models(run_root, scen, cfg["env"]["descriptor"])]
        for spec, ms in specs:
            for s in range(0, episodes, chunk):
                tasks.append((cfg, scen, split, spec, ms, list(range(s, min(s + chunk, episodes))), save_traj,
                              out_csv.parent / "traj"))
    # slow methods first for better load balancing
    tasks.sort(key=lambda t: 0 if t[3][0].startswith("greedy") else 1)
    t0 = time.time()
    rows: list[dict] = []
    if workers > 1:
        with get_context("forkserver").Pool(workers) as pool:
            for i, r in enumerate(pool.imap_unordered(run_episodes, tasks)):
                rows += r
                if (i + 1) % max(1, len(tasks) // 10) == 0:
                    print(f"[eval] {i + 1}/{len(tasks)} tasks, {time.time() - t0:.0f}s", flush=True)
    else:
        for t in tasks:
            rows += run_episodes(t)
    df = pd.DataFrame(rows, columns=COLUMNS).sort_values(["scenario", "method", "model_seed", "episode"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[eval] {len(df)} episodes -> {out_csv} in {time.time() - t0:.0f}s")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--scenarios", nargs="*", default=["S0", "S1", "S2"])
    ap.add_argument("--methods", nargs="*", default=["straight", "lawnmower", "random", "greedy", "greedy_map", "ppo"])
    ap.add_argument("--runs", help="root folder with trained runs (default runs/<config name>)")
    ap.add_argument("--episodes", type=int)
    ap.add_argument("--split")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--out", help="output CSV (default results/episodes_<split>.csv)")
    ap.add_argument("--save-traj", type=int, default=3)
    ap.add_argument("--descriptor")
    a = ap.parse_args()
    cfg = load_config(a.config, {"env": {"descriptor": a.descriptor}} if a.descriptor else None)
    ev = cfg["eval"]
    split = a.split or ev["split"]
    out = Path(a.out) if a.out else RESULTS / f"episodes_{split}.csv"
    root = Path(a.runs) if a.runs else RUNS / Path(a.config).stem
    evaluate(cfg, a.scenarios, a.methods, root, a.episodes or ev["episodes"], split, out,
             a.workers or ev.get("workers") or os.cpu_count() or 1, a.save_traj)


if __name__ == "__main__":
    main()

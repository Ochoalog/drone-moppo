"""Jobs de calibração, treino e avaliação da v3 (notebooks 04 e 05). Rodam na GPU do Colab ou na CPU.

Todos partem de um YAML de job. A pasta de trabalho no Drive é `<out_root>/v3/<descritor>/fold_<dobra>/`:
- `calibration.json` (β₁ por altitude, κ, α) e `pca_<h>m.npz` (PCA da faixa de treino, reaplicada no zero-shot);
- `runs/<job_id>/` (modelo, progresso, gate);
- `eval/<job_id>/episodes.csv` (uma linha por episódio, com proveniência).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dal.config import ROOT
from dal.v3.agents.baselines import BASELINES, run_policy
from dal.v3.agents.ppo_batched import NetworkPolicy, PolicyAdapter, PPOTrainConfig, build_network, train
from dal.v3.descriptors.tables import PCAFit
from dal.v3.envs.realistic_env import RealisticBatchedEnv, RealisticConfig
from dal.v3.localization import calibration
from dal.v3.localization.change_prior import rasterize_prior
from dal.v3.pipeline import DataRequest, load_tables
from dal.v3.reporting.colab_io import load_job_config, now_iso, write_done
from dal.v3.reporting.provenance import config_hash, git_commit
from dal.v3.sensor import specs

EVAL_SEED_BASE = 2024
METHODS = {  # nome → (filtro CA, canal m na política, altitude como ação)
    "ppo": (False, False, False), "ppo_ca": (True, True, False), "ppo_ca_alt": (True, True, True)}


def work_dir(out_root: Path, descriptor: str, fold: str) -> Path:
    return Path(out_root) / "v3" / descriptor / f"fold_{fold}"


def site_prior(job: dict, site: str) -> np.ndarray | None:
    """Prior de mudança do local na grade de 25 m, a partir do raster indicado no job (fonte independente)."""
    path = job.get("prior", {}).get(site)
    if not path:
        return None
    grid = specs.site_grid(specs.load_site(site))
    return rasterize_prior(path, grid.x0, grid.y1, grid.nrows, grid.ncols, specs.CELL_M)


def calibration_file(job: dict) -> str:
    """Arquivo da calibração da dobra (V33: `calibration_v2.json` não sobrescreve a primeira)."""
    return job.get("calibration_file", "calibration.json")


def save_pca(fits: dict[int, PCAFit], directory: Path) -> None:
    for altitude, fit in fits.items():
        np.savez(directory / f"pca_{altitude}m.npz", mean=fit.mean, components=fit.components)


def load_pca(directory: Path, altitudes) -> dict[int, PCAFit]:
    return {h: PCAFit(**dict(np.load(directory / f"pca_{h}m.npz"))) for h in altitudes}


# ============================================================================ calibração
def calibrate_job(job: dict, out_root: Path, device: str) -> dict:
    """β₁ por altitude, κ e α, só em janelas de treino da dobra; grava também a PCA da dobra."""
    directory = work_dir(out_root, job["descriptor"], job["fold"])
    directory.mkdir(parents=True, exist_ok=True)
    prior = site_prior(job, "taquari")
    request = DataRequest("taquari", job.get("scenario", "S2"), job["fold"], job["descriptor"], split="train",
                          feature=job.get("feature", "gem"))
    tables, fits = load_tables(Path(out_root) / "features" / "v3", request, beta={h: 1.0 for h in specs.ALTITUDES_M},
                               change_prior=prior)
    save_pca(fits, directory)
    beta1 = {h: calibration.calibrate_beta1(tables, h, 128)["beta1"] for h in specs.ALTITUDES_M}
    config = RealisticConfig()
    kappas = tuple(job.get("kappas", calibration.KAPPAS))  # V33: grade estendível (κ no limite da grade original)
    kappa, kappa_rows = calibration.calibrate_kappa(tables, config, beta1, job.get("episodes", 200), device=device,
                                                    kappas=kappas)
    tables.beta = {h: kappa * b for h, b in beta1.items()}
    alpha, alpha_rows = (calibration.calibrate_alpha(tables, config, job.get("episodes", 200), device=device)
                         if prior is not None else (0.0, []))
    result = {"beta1": beta1, "kappa": kappa, "beta": tables.beta, "alpha": alpha, "kappa_rows": kappa_rows,
              "alpha_rows": alpha_rows, "has_prior": prior is not None}
    (directory / calibration_file(job)).write_text(json.dumps(result, indent=1, default=float))
    return result


# ============================================================================ contexto comum
_TABLES_CACHE: dict = {}


def load_site_tables(job: dict, out_root: Path, site: str, scenario: str, split: str):
    """Tabelas de um local/divisão com a calibração da dobra (zero-shot: PCA e β da dobra do Taquari).

    Ficam em cache por (local, cenário, divisão): o filtro CA e a altitude mudam só a config do ambiente.
    """
    key = (job["descriptor"], job["fold"], site, scenario, split)
    if key not in _TABLES_CACHE:
        directory = work_dir(out_root, job["descriptor"], job["fold"])
        calib = json.loads((directory / calibration_file(job)).read_text())
        beta = {int(h): float(b) for h, b in calib["beta"].items()}
        zero_shot = site != "taquari"
        request = DataRequest(site, scenario, None if zero_shot else job["fold"], job["descriptor"], split=split,
                              feature=job.get("feature", "gem"))
        pca_saved = (directory / f"pca_{specs.ALTITUDES_M[0]}m.npz").exists()
        pca = load_pca(directory, specs.ALTITUDES_M) if (zero_shot or pca_saved) else None
        tables, _ = load_tables(Path(out_root) / "features" / "v3", request, beta, change_prior=site_prior(job, site),
                                pca_fits=pca)
        _TABLES_CACHE[key] = (tables, float(calib["alpha"]))
    return _TABLES_CACHE[key]


def make_env(job: dict, out_root: Path, site: str, scenario: str, split: str, method_ca: bool, altitude: bool,
             device: str, drift: float = 0.02) -> RealisticBatchedEnv:
    """Ambiente com filtro padrão ou CA (α da calibração da dobra) e, opcionalmente, altitude como ação."""
    tables, alpha = load_site_tables(job, out_root, site, scenario, split)
    config = RealisticConfig(altitude_actions=altitude, alpha=alpha if method_ca else 0.0, drift_fraction=drift)
    return RealisticBatchedEnv(tables, config, device)


# ============================================================================ treino
def train_job(job: dict, out_root: Path, device: str, job_id: str) -> dict:
    method_ca, prior_channel, altitude = METHODS[job["method"]]
    drift = float(job.get("drift", 0.02))
    env = make_env(job, out_root, "taquari", job.get("scenario", "S2"), "train", method_ca, altitude, device, drift)
    gate_env = make_env(job, out_root, "taquari", job.get("scenario", "S2"), "train", method_ca, altitude, device,
                        drift)
    # Compartilha os descritores de percepção na GPU entre env e gate_env (economiza ~4.4 GB de VRAM)
    gate_env.map_desc = env.map_desc
    gate_env.obs_desc = env.obs_desc
    gate_env.map_ok = env.map_ok
    gate_env.obs_ok = env.obs_ok
    gate_env.prior = env.prior
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    cfg = PPOTrainConfig(total_timesteps=int(job["total_timesteps"]), n_envs=int(job.get("n_envs", 512)),
                         seed=int(job["seed"]), change_prior_input=prior_channel,
                         gate_min_advantage_over_random=float(job.get("gate_min_advantage", 0.05)),
                         gate_flight_time_ratio=job.get("gate_flight_time_ratio"))
    run_dir = work_dir(out_root, job["descriptor"], job["fold"]) / "runs" / job_id
    final = train(env, gate_env, cfg, run_dir, device)
    summary = json.loads((final / "train_summary.json").read_text())
    progress = pd.read_csv(final / "progress.csv")
    return {"run_dir": str(final), "global_step": summary["global_step"], "gate": summary["gate"],
            "steps_per_s_median": float(progress.steps_per_s.median()),
            "invalid_gate": final.name.endswith("INVALID_GATE")}


# ============================================================================ avaliação
def evaluate_job(job: dict, out_root: Path, device: str, job_id: str) -> pd.DataFrame:
    """Avalia baselines (filtro padrão e CA) e modelos treinados nos mesmos episódios de cada local."""
    seeds = [EVAL_SEED_BASE + i for i in range(int(job.get("episodes", 1000)))]
    directory = work_dir(out_root, job["descriptor"], job["fold"])
    frames = []
    for target in job["targets"]:  # ex.: {"site": "taquari", "scenario": "S2"}
        site, scenario = target["site"], target["scenario"]
        split = "test"
        for name in job.get("baselines", []):
            for use_ca in (False, True):
                altitude = name == "straight_150"
                env = make_env(job, out_root, site, scenario, split, use_ca, altitude, device)
                env = run_policy(env, BASELINES[name](), seeds)
                frames.append(_episodes(env, f"{name}{'_ca' if use_ca else ''}", -1, site, scenario))
        for model in job.get("models", []):  # {"method": "ppo_ca", "seed": 0, "run": "<job_id do treino>"}
            run_dir = directory / "runs" / model["run"]
            if not run_dir.exists():
                continue  # execução INVALID_GATE tem outro nome de pasta: nunca entra na avaliação
            method_ca, prior_channel, altitude = METHODS[model["method"]]
            env = make_env(job, out_root, site, scenario, split, method_ca, altitude, device)
            net = build_network(env, PPOTrainConfig(change_prior_input=prior_channel), device)
            net.load_state_dict(torch.load(run_dir / "final_model.pt", map_location=device))
            deterministic = bool(job.get("argmax", True))
            policy = NetworkPolicy(net.eval(), PolicyAdapter(prior_channel), deterministic=deterministic)
            env = run_policy(env, policy, seeds)
            frames.append(_episodes(env, model["method"], int(model["seed"]), site, scenario))
    episodes = pd.concat(frames, ignore_index=True)
    for column, value in {"source_file": f"eval/{job_id}/episodes.csv", "protocol": "v3 realista; seeds 2024+i",
                          "n_episodes": len(seeds), "fold": job["fold"], "git_commit": git_commit(),
                          "config_hash": config_hash(job)}.items():
        episodes[column] = value
    return episodes


def _episodes(env: RealisticBatchedEnv, method: str, seed: int, site: str, scenario: str) -> pd.DataFrame:
    return pd.DataFrame({
        "method": method, "model_seed": seed, "region": site, "scenario": scenario, "episode": np.arange(env.n),
        "window_r0": env.origin[:, 0].cpu().numpy(), "window_c0": env.origin[:, 1].cpu().numpy(),
        "success": env.success.cpu().numpy(), "false_loc": env.false_loc.cpu().numpy(), "steps": env.t.cpu().numpy(),
        "flight_time_s": env.flight_time_s.cpu().numpy(), "distance_m": env.distance_m.cpu().numpy(),
        "final_altitude_index": env.altitude_index.cpu().numpy()})


# ============================================================================ CLI
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("kind", nargs="?", choices=["calibrate", "train", "evaluate"],
                        help="tipo do job; se omitido, vem da chave `kind` do YAML")
    parser.add_argument("--job-config", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--colab-out", default="colab_out")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    job = load_job_config(ROOT / args.job_config)
    args.kind = args.kind or job["kind"]
    started_at, started = now_iso(), time.time()
    out = Path(args.colab_out) / args.job_id
    out.mkdir(parents=True, exist_ok=True)
    status, reason = "OK", ""
    if args.kind == "calibrate":
        metrics = calibrate_job(job, Path(args.out_root), args.device)
        # A calibração completa (com as linhas de κ e α) vai também para colab_out: publicada na colab/results.
        (out / "calibration.json").write_text(json.dumps(metrics, indent=1, default=float))
        metrics = {k: v for k, v in metrics.items() if not k.endswith("_rows")}
    elif args.kind == "train":
        metrics = train_job(job, Path(args.out_root), args.device, args.job_id)
        if metrics["invalid_gate"]:
            status, reason = "INVALID_GATE", "gate de aprendizado aos 25% reprovado"
    else:
        episodes = evaluate_job(job, Path(args.out_root), args.device, args.job_id)
        directory = work_dir(Path(args.out_root), job["descriptor"], job["fold"]) / "eval" / args.job_id
        directory.mkdir(parents=True, exist_ok=True)
        episodes.to_csv(directory / "episodes.csv", index=False)
        summary = episodes.groupby(["region", "scenario", "method"])[["success", "false_loc", "flight_time_s"]].mean()
        summary.reset_index().to_csv(out / "summary.csv", index=False)
        if (out_size := episodes.memory_usage().sum()) < 4e6:
            episodes.to_csv(out / "episodes.csv", index=False)
        metrics = {"rows": len(episodes), "episodes_bytes": int(out_size)}
    write_done(out, job_id=args.job_id, config=job, started_at=started_at,
               gpu_minutes=(time.time() - started) / 60 if args.device.startswith("cuda") else 0.0, metrics=metrics,
               status=status, reason=reason)
    print(json.dumps(metrics, indent=1, default=str))


if __name__ == "__main__":
    main()

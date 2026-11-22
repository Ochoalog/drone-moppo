"""Jobs de calibração, treino e avaliação da v3 (notebooks 04 e 05). Rodam na GPU do Colab ou na CPU.

Todos partem de um YAML de job. A pasta de trabalho no Drive é `<out_root>/v3/<descritor>/fold_<dobra>/`:
- `calibration.json` (β₁ por altitude, κ, α) e `pca_<h>m.npz` (PCA da faixa de treino, reaplicada no zero-shot);
- `runs/<job_id>/` (modelo, progresso, gate);
- `eval/<job_id>/episodes.csv` (uma linha por episódio, com proveniência).
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dal.config import ROOT, load_config
from dal.grid import lonlat_to_utm
from dal.v3.agents.baselines import BASELINES, run_policy
from dal.v3.agents.ppo_batched import NetworkPolicy, PolicyAdapter, PPOTrainConfig, build_network, train
from dal.v3.descriptors.obs_noise import add_observation_noise
from dal.v3.descriptors.tables import PCAFit
from dal.v3.envs.realistic_env import RealisticBatchedEnv, RealisticConfig
from dal.v3.evaluation.site_split import ColumnSplit, choose_column_split, split_diagnostic, with_columns
from dal.v3.localization import calibration
from dal.v3.localization.change_prior import rasterize_prior
from dal.v3.pipeline import DataRequest, load_tables
from dal.v3.reporting.colab_io import load_job_config, now_iso, write_done
from dal.v3.reporting.provenance import config_hash, git_commit
from dal.v3.sensor import specs

EVAL_SEED_BASE = 2024
METHODS = {  # nome → (filtro CA, canal m na política, altitude como ação)
    "ppo": (False, False, False), "ppo_ca": (True, True, False), "ppo_ca_alt": (True, True, True),
    "ppo_alt": (False, False, True)}  # E-alt (V46): altitude sem filtro CA nem canal de prior
NOISE_SIGMAS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)  # E-synth (V46)
DEFAULT_SCENARIO = {"taquari": "S2", "poa": "P1", "canoas": "C1"}


def work_dir(out_root: Path, descriptor: str, fold: str) -> Path:
    return Path(out_root) / "v3" / descriptor / f"fold_{fold}"


def site_prior(job: dict, site: str) -> np.ndarray | None:
    """Prior de mudança do local na grade de 25 m, a partir do raster indicado no job (fonte independente)."""
    path = job.get("prior", {}).get(site)
    if not path:
        return None
    grid = specs.site_grid(specs.load_site(site))
    return rasterize_prior(path, grid.x0, grid.y1, grid.nrows, grid.ncols, specs.CELL_M)


def calibration_file(job: dict, site: str | None = None) -> str:
    """Arquivo da calibração (V33: `calibration_v2.json` não sobrescreve a primeira).

    `site_calibration` (V46) dá um arquivo próprio por local, ex.: `{canoas: calibration_canoas.json}`.
    """
    per_site = job.get("site_calibration") or {}
    if site is not None and site in per_site:
        return per_site[site]
    return job.get("calibration_file", "calibration.json")


def noise_settings(job: dict) -> tuple[float, int]:
    """(σ, semente) do ruído na observação (E-synth); σ = 0 = sem ruído."""
    return float(job.get("obs_noise_sigma", 0.0)), int(job.get("obs_noise_seed", 0))


def save_pca(fits: dict[int, PCAFit], directory: Path) -> None:
    for altitude, fit in fits.items():
        np.savez(directory / f"pca_{altitude}m.npz", mean=fit.mean, components=fit.components)


def load_pca(directory: Path, altitudes) -> dict[int, PCAFit]:
    return {h: PCAFit(**dict(np.load(directory / f"pca_{h}m.npz"))) for h in altitudes}


# ============================================================================ calibração
class SplitUnavailableError(RuntimeError):
    """A divisão do local não rende janelas de teste suficientes (V46: parar e perguntar ao Vitor)."""


def calibrate_job(job: dict, out_root: Path, device: str) -> dict:
    """β₁ por altitude, κ e α, só em janelas de treino; grava também a PCA da dobra (só no Taquari).

    Num local zero-shot com `column_split: true` (Canoas, V46), escolhe a divisão por colunas pelas máscaras,
    calibra só na faixa de treino e grava a divisão no arquivo da calibração; a PCA da dobra A fica intacta.
    """
    directory = work_dir(out_root, job["descriptor"], job["fold"])
    directory.mkdir(parents=True, exist_ok=True)
    site = job.get("site", "taquari")
    split_info = None
    if site == "taquari":
        prior = site_prior(job, "taquari")
        request = DataRequest("taquari", job.get("scenario", "S2"), job["fold"], job["descriptor"], split="train",
                              feature=job.get("feature", "gem"))
        tables, fits = load_tables(Path(out_root) / "features" / "v3", request,
                                   beta={h: 1.0 for h in specs.ALTITUDES_M}, change_prior=prior)
        save_pca(fits, directory)
    else:
        if not job.get("column_split"):
            raise ValueError("calibração fora do Taquari exige `column_split: true` (V46)")
        prior = None  # E-real usa o filtro padrão: α não é calibrado
        request = DataRequest(site, job.get("scenario", DEFAULT_SCENARIO[site]), None, job["descriptor"],
                              feature=job.get("feature", "gem"))
        tables, _ = load_tables(Path(out_root) / "features" / "v3", request, beta={h: 1.0 for h in specs.ALTITUDES_M},
                                pca_fits=load_pca(directory, specs.ALTITUDES_M))
        column_split = choose_column_split(tables, RealisticConfig().window_cells, RealisticConfig().min_window_valid)
        if column_split is None:
            error = SplitUnavailableError(f"{site}: < 20 janelas de teste mesmo com 2 janelas de largura")
            error.diagnostic = split_diagnostic(tables, RealisticConfig().window_cells)
            ok = tables.map_ok & tables.obs_ok
            error.column_valid = pd.DataFrame({"col": np.arange(ok.shape[1]), "valid_frac": ok.mean(0),
                                               "map_ok_frac": tables.map_ok.mean(0),
                                               "obs_ok_frac": tables.obs_ok.mean(0)})
            raise error
        split_info = column_split.to_dict()
        tables = with_columns(tables, column_split.cols("train"))
    sigma, noise_seed = noise_settings(job)
    tables = add_observation_noise(tables, sigma, noise_seed, device=device)
    result = calibrate_tables(tables, job, device, prior is not None)
    result.update({"site": site, "column_split": split_info, "obs_noise_sigma": sigma, "obs_noise_seed": noise_seed})
    (directory / calibration_file(job, site)).write_text(json.dumps(result, indent=1, default=float))
    return result


def calibrate_tables(tables, job: dict, device: str, with_alpha: bool) -> dict:
    """Regra da PoC: β₁ (log-verossimilhança), κ (utilidade da aleatória) e, com prior, α. Só janelas permitidas."""
    beta_info = {h: calibration.calibrate_beta1(tables, h, 128) for h in specs.ALTITUDES_M}
    beta1 = {h: info["beta1"] for h, info in beta_info.items()}
    config = RealisticConfig()
    kappas = tuple(job.get("kappas", calibration.KAPPAS))  # V33: grade estendível (κ no limite da grade original)
    kappa, kappa_rows = calibration.calibrate_kappa(tables, config, beta1, job.get("episodes", 200), device=device,
                                                    kappas=kappas)
    tables.beta = {h: kappa * b for h, b in beta1.items()}
    alpha, alpha_rows = (calibration.calibrate_alpha(tables, config, job.get("episodes", 200), device=device)
                         if with_alpha else (0.0, []))
    return {"beta1": beta1, "kappa": kappa, "beta": tables.beta, "alpha": alpha, "kappa_rows": kappa_rows,
            "alpha_rows": alpha_rows, "has_prior": with_alpha,
            "top1_by_altitude": {h: info["top1"] for h, info in beta_info.items()}}


def scan_noise_job(job: dict, out_root: Path, device: str) -> pd.DataFrame:
    """E-synth (V46): para cada σ, recalibra β₁ e κ no treino da dobra e mede a aleatória e o top-1.

    Grava `calibration_noise_<σ>.json` na pasta da dobra (usado pelos treinos e avaliações do nível escolhido).
    """
    directory = work_dir(out_root, job["descriptor"], job["fold"])
    request = DataRequest("taquari", job.get("scenario", "S2"), job["fold"], job["descriptor"], split="train",
                          feature=job.get("feature", "gem"))
    pca = load_pca(directory, specs.ALTITUDES_M)  # a PCA da dobra (gravada pela calibração), sem reajuste
    base, _ = load_tables(Path(out_root) / "features" / "v3", request, beta={h: 1.0 for h in specs.ALTITUDES_M},
                          pca_fits=pca)
    seed = int(job.get("obs_noise_seed", 0))
    rows = []
    for sigma in job.get("sigmas", NOISE_SIGMAS):
        t0 = time.time()
        tables = add_observation_noise(base, float(sigma), seed, device=device)
        result = calibrate_tables(tables, job, device, with_alpha=False)
        result.update({"site": "taquari", "column_split": None, "obs_noise_sigma": float(sigma),
                       "obs_noise_seed": seed})
        (directory / noise_calibration_file(float(sigma))).write_text(json.dumps(result, indent=1, default=float))
        chosen = next(r for r in result["kappa_rows"] if r["kappa"] == result["kappa"])
        rows.append({"sigma": float(sigma), "kappa": result["kappa"], "random_success": chosen["success"],
                     "random_false_loc": chosen["false_loc"], "top1_100m": result["top1_by_altitude"][100],
                     **{f"beta1_{h}m": b for h, b in result["beta1"].items()},
                     "calibration_file": noise_calibration_file(float(sigma)), "seconds": time.time() - t0})
        print(f"[scan_noise] σ={sigma}: {rows[-1]}", flush=True)
        del tables
    return pd.DataFrame(rows)


def noise_calibration_file(sigma: float) -> str:
    return f"calibration_noise_{sigma:g}.json"


# ============================================================================ contexto comum
_TABLES_CACHE: dict = {}


def load_site_tables(job: dict, out_root: Path, site: str, scenario: str, split: str, device: str = "cpu"):
    """Tabelas de um local/divisão com a calibração da dobra (zero-shot: PCA e β da dobra do Taquari).

    Ficam em cache por (local, cenário, divisão): o filtro CA e a altitude mudam só a config do ambiente.
    """
    sigma, noise_seed = noise_settings(job)
    key = (job["descriptor"], job["fold"], site, scenario, split, calibration_file(job, site), sigma, noise_seed)
    if key not in _TABLES_CACHE:
        directory = work_dir(out_root, job["descriptor"], job["fold"])
        calib = json.loads((directory / calibration_file(job, site)).read_text())
        if float(calib.get("obs_noise_sigma", 0.0)) != sigma:
            raise ValueError(f"calibração {calibration_file(job, site)} é de σ = {calib.get('obs_noise_sigma')}, "
                             f"o job pede σ = {sigma}")
        beta = {int(h): float(b) for h, b in calib["beta"].items()}
        zero_shot = site != "taquari"
        request = DataRequest(site, scenario, None if zero_shot else job["fold"], job["descriptor"], split=split,
                              feature=job.get("feature", "gem"))
        pca_saved = (directory / f"pca_{specs.ALTITUDES_M[0]}m.npz").exists()
        pca = load_pca(directory, specs.ALTITUDES_M) if (zero_shot or pca_saved) else None
        tables, _ = load_tables(Path(out_root) / "features" / "v3", request, beta, change_prior=site_prior(job, site),
                                pca_fits=pca)
        if calib.get("column_split"):  # E-real (V46): divisão por colunas gravada pela calibração do local
            tables = with_columns(tables, ColumnSplit.from_dict(calib["column_split"]).cols(split))
        tables = add_observation_noise(tables, sigma, noise_seed, device=device)
        _TABLES_CACHE[key] = (tables, float(calib["alpha"]))
    return _TABLES_CACHE[key]


def restrict_to_region(tables, site: str, region: str | None):
    """Cópia das tabelas com as janelas restritas a uma sub-região (ex.: ilhas ou Centro de POA, v1)."""
    if region is None:
        return tables
    box = load_config("configs_poa/poa.yaml")["regions"][region] if site == "poa" else None
    if box is None:
        raise ValueError(f"sub-região desconhecida: {site}/{region}")
    grid = specs.site_grid(specs.load_site(site))
    xs, ys = lonlat_to_utm(np.array([box["lon0"], box["lon1"]]), np.array([box["lat1"], box["lat0"]]))
    cols = np.arange(grid.ncols)
    rows = np.arange(grid.nrows)
    col_center = grid.x0 + (cols + 0.5) * grid.cell
    row_center = grid.y1 - (rows + 0.5) * grid.cell
    restricted = copy.copy(tables)
    restricted.cols_allowed = cols[(col_center >= xs.min()) & (col_center <= xs.max())]
    in_rows = rows[(row_center >= ys.min()) & (row_center <= ys.max())]
    restricted.rows_allowed = np.intersect1d(tables.rows_allowed, in_rows)
    return restricted


def make_env(job: dict, out_root: Path, site: str, scenario: str, split: str, method_ca: bool, altitude: bool,
             device: str, drift: float = 0.02, region: str | None = None) -> RealisticBatchedEnv:
    """Ambiente com filtro padrão ou CA (α da calibração da dobra) e, opcionalmente, altitude como ação."""
    tables, alpha = load_site_tables(job, out_root, site, scenario, split, device)
    tables = restrict_to_region(tables, site, region)
    config = RealisticConfig(altitude_actions=altitude, alpha=alpha if method_ca else 0.0, drift_fraction=drift)
    return RealisticBatchedEnv(tables, config, device)


# ============================================================================ treino
def train_job(job: dict, out_root: Path, device: str, job_id: str) -> dict:
    method_ca, prior_channel, altitude = METHODS[job["method"]]
    drift = float(job.get("drift", 0.02))
    site = job.get("site", "taquari")  # E-real (V46): treino na faixa de treino de Canoas
    scenario = job.get("scenario", DEFAULT_SCENARIO[site])
    env = make_env(job, out_root, site, scenario, "train", method_ca, altitude, device, drift)
    gate_env = make_env(job, out_root, site, scenario, "train", method_ca, altitude, device, drift)
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
    import gc
    seeds = [EVAL_SEED_BASE + i for i in range(int(job.get("episodes", 1000)))]
    directory = work_dir(out_root, job["descriptor"], job["fold"])
    part_dir = directory / "eval" / job_id / "parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for target_idx, target in enumerate(job["targets"]):  # ex.: {"site": "taquari", "scenario": "S2"}
        site, scenario, region = target["site"], target["scenario"], target.get("region")
        split = target.get("split", "test")  # "train": escolha da melhor heurística CA no conjunto de calibração
        label = f"{site}_{region}" if region else site
        part_path = part_dir / f"part_{target_idx}_{label}_{scenario}_{split}.parquet"
        if part_path.exists():
            print(f"[evaluate] {label} ({scenario}/{split}): carregando de {part_path.name} (já computado)", flush=True)
            frames.append(pd.read_parquet(part_path))
            continue
        print(f"[evaluate] {label} ({scenario}/{split}): avaliando ({target_idx + 1}/{len(job['targets'])})...", flush=True)
        target_frames = []
        filters = job.get("baseline_filters", ["standard", "ca"])  # E (V46): só o padrão (α = 0)
        for name in job.get("baselines", []):
            for use_ca in [f == "ca" for f in filters]:
                altitude = name == "straight_150"
                t0 = time.time()
                env = make_env(job, out_root, site, scenario, split, use_ca, altitude, device, region=region)
                env = run_policy(env, BASELINES[name](), seeds)
                method = f"{name}{'_ca' if use_ca else ''}"
                df = _episodes(env, method, -1, label, scenario).assign(split=split, mode="—")
                target_frames.append(df)
                elapsed = time.time() - t0
                succ = float(env.success.float().mean())
                print(f"  [baseline] {method:18s} | succ: {succ:.3f} | time: {float(env.flight_time_s.mean()):.1f}s | {elapsed:.1f}s", flush=True)
                del env
        for model in job.get("models", []):  # {"method": "ppo_ca", "seed": 0, "run": "<job_id do treino>"}
            run_dir = resolve_run_dir(directory, model["run"], bool(job.get("include_invalid_gate", False)))
            if run_dir is None:
                continue  # execução INVALID_GATE tem outro nome de pasta: fora da avaliação (exceto no E, V46)
            method_ca, prior_channel, altitude = METHODS[model["method"]]
            method_ca = method_ca and model.get("filter", "ca") != "standard"  # H2: PPO-CA com filtro padrão
            drift = float(model.get("drift", 0.02))  # ablação de deriva: avaliada na deriva do treino
            gate = json.loads((run_dir / "train_summary.json").read_text()).get("gate") or {}
            # Os dois critérios do gate (V38): o original é só a vantagem de sucesso sobre a aleatória.
            gate_original = bool(gate.get("advantage", -1.0) >= gate.get("threshold", 0.05))
            gate_amended = bool(gate.get("passed", False)) or gate_original
            invalid_gate = run_dir.name.endswith("INVALID_GATE")
            for mode in job.get("modes", ["argmax"]):  # argmax e/ou sampled (lição do estudo v1)
                t0 = time.time()
                env = make_env(job, out_root, site, scenario, split, method_ca, altitude, device, drift, region)
                net = build_network(env, PPOTrainConfig(change_prior_input=prior_channel), device)
                net.load_state_dict(torch.load(run_dir / "final_model.pt", map_location=device))
                policy = NetworkPolicy(net.eval(), PolicyAdapter(prior_channel), deterministic=mode == "argmax")
                env = run_policy(env, policy, seeds)
                name = model.get("label", model["method"])
                df = _episodes(env, name, int(model["seed"]), label, scenario).assign(
                    split=split, mode=mode, gate_original_passed=gate_original, gate_amended_passed=gate_amended,
                    invalid_gate=invalid_gate, run=run_dir.name)
                target_frames.append(df)
                elapsed = time.time() - t0
                succ = float(env.success.float().mean())
                print(f"  [model] {name:15s} ({mode:7s}) | succ: {succ:.3f} | time: {float(env.flight_time_s.mean()):.1f}s | {elapsed:.1f}s", flush=True)
                del env, net, policy
        target_df = pd.concat(target_frames, ignore_index=True)
        target_df.to_parquet(part_path, index=False)
        frames.append(target_df)
        _TABLES_CACHE.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    episodes = pd.concat(frames, ignore_index=True)
    sigma, noise_seed = noise_settings(job)
    for column, value in {"source_file": f"eval/{job_id}/episodes.csv", "protocol": "v3 realista; seeds 2024+i",
                          "obs_noise_sigma": sigma, "obs_noise_seed": noise_seed,
                          "n_episodes": len(seeds), "fold": job["fold"], "git_commit": git_commit(),
                          "config_hash": config_hash(job)}.items():
        episodes[column] = value
    return episodes


def resolve_run_dir(directory: Path, run: str, include_invalid_gate: bool) -> Path | None:
    """Pasta do treino; a versão `_INVALID_GATE` só entra com `include_invalid_gate` (exceção do E, V46)."""
    run_dir = directory / "runs" / run
    if run_dir.exists():
        return run_dir
    invalid = directory / "runs" / f"{run}_INVALID_GATE"
    return invalid if include_invalid_gate and invalid.exists() else None


def _episodes(env: RealisticBatchedEnv, method: str, seed: int, site: str, scenario: str) -> pd.DataFrame:
    return pd.DataFrame({
        "method": method, "model_seed": seed, "region": site, "scenario": scenario, "episode": np.arange(env.n),
        "window_r0": env.origin[:, 0].cpu().numpy(), "window_c0": env.origin[:, 1].cpu().numpy(),
        "success": env.success.cpu().numpy(), "false_loc": env.false_loc.cpu().numpy(), "steps": env.t.cpu().numpy(),
        "flight_time_s": env.flight_time_s.cpu().numpy(), "distance_m": env.distance_m.cpu().numpy(),
        "final_altitude_index": env.altitude_index.cpu().numpy(),
        "mean_altitude_m": (env.altitude_sum_m / env.t.clamp(min=1)).cpu().numpy(),
        # Índice de mudança da janela (H4): fração inundada média do prior independente na janela (NaN sem prior).
        "window_change_prior": env.window_change_prior().cpu().numpy()})


# ============================================================================ CLI
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("kind", nargs="?", choices=["calibrate", "train", "evaluate", "scan_noise"],
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
        try:
            metrics = calibrate_job(job, Path(args.out_root), args.device)
        except SplitUnavailableError as error:  # V46: parar e perguntar (sem calibração)
            metrics, status, reason = {}, "FAILED", str(error)
            if getattr(error, "diagnostic", None):  # só máscaras: subsídio para a decisão do Vitor
                pd.DataFrame(error.diagnostic).to_csv(out / "split_diagnostic.csv", index=False)
                error.column_valid.to_csv(out / "column_valid.csv", index=False)
                metrics = {"split_diagnostic": "split_diagnostic.csv"}
        else:
            # A calibração completa (com as linhas de κ e α) vai também para colab_out: publicada na colab/results.
            name = calibration_file(job, job.get("site"))
            (out / name).write_text(json.dumps(metrics, indent=1, default=float))
            metrics = {k: v for k, v in metrics.items() if not k.endswith("_rows")}
    elif args.kind == "scan_noise":
        scan = scan_noise_job(job, Path(args.out_root), args.device)
        scan.assign(source_file=f"colab_out/{args.job_id}/scan.csv", protocol="E-synth V46; treino dobra A",
                    n_episodes=int(job.get("episodes", 200)), seeds="calibração 0..episodes-1", fold=job["fold"],
                    region="taquari", git_commit=git_commit(), config_hash=config_hash(job)
                    ).to_csv(out / "scan.csv", index=False)
        directory = work_dir(Path(args.out_root), job["descriptor"], job["fold"])
        for name in scan.calibration_file:  # as calibrações por σ também vão para a colab/results (pequenas)
            (out / name).write_text((directory / name).read_text())
        metrics = {"rows": len(scan)}
    elif args.kind == "train":
        metrics = train_job(job, Path(args.out_root), args.device, args.job_id)
        if metrics["invalid_gate"]:
            status, reason = "INVALID_GATE", "gate de aprendizado aos 25% reprovado"
    else:
        episodes = evaluate_job(job, Path(args.out_root), args.device, args.job_id)
        directory = work_dir(Path(args.out_root), job["descriptor"], job["fold"]) / "eval" / args.job_id
        directory.mkdir(parents=True, exist_ok=True)
        episodes.to_csv(directory / "episodes.csv", index=False)
        keys = ["region", "scenario", "split", "method", "mode"]
        summary = episodes.groupby(keys)[["success", "false_loc", "flight_time_s"]].mean()
        summary.reset_index().to_csv(out / "summary.csv", index=False)
        if (out_size := episodes.memory_usage().sum()) < 4e6:
            episodes.to_csv(out / "episodes.csv", index=False)
        if job.get("publish_parquet"):  # E (V46): os episódios vão para a colab/results, compactados
            episodes.to_parquet(out / "episodes.parquet", index=False, compression="zstd")
        metrics = {"rows": len(episodes), "episodes_bytes": int(out_size)}
    write_done(out, job_id=args.job_id, config=job, started_at=started_at,
               gpu_minutes=(time.time() - started) / 60 if args.device.startswith("cuda") else 0.0, metrics=metrics,
               status=status, reason=reason)
    print(json.dumps(metrics, indent=1, default=str))


if __name__ == "__main__":
    main()

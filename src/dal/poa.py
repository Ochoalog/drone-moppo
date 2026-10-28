"""Porto Alegre battery: descriptors, Taquari-PCA reuse, per-region scenario data, beta
recalibration (cross-region) and the paired zero-shot evaluation of the Taquari models.

Taquari artefacts (`paths.taquari_data`, `paths.taquari_runs`) are only ever read.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT, load_config, paths
from .env import Episode, ScenarioData, distinctiveness
from .poa_data import load_date, load_masks, load_offsets, poa_grid, region_slices

DESC = "hand"


def _p(cfg: dict, key: str) -> Path:
    v = cfg["paths"][key]
    return Path(v) if Path(v).is_absolute() else ROOT / v


def feat_dir(cfg: dict) -> Path:
    return paths(cfg)["data"] / "features"


def cache_dir(cfg: dict) -> Path:
    return paths(cfg)["data"] / "cache" / DESC


# ============================================================================ descriptors
def precompute(cfg: dict) -> dict:
    """Map descriptor per cell for every POA date and K augmented observation variants of the
    observation date, with exactly the PoC crop/augmentation parameters."""
    from .features import compute_set, hand_descriptor

    d, fcfg = cfg["data"], cfg["features"]
    grid = poa_grid(cfg)
    offsets = load_offsets(cfg)
    rng = np.random.default_rng(fcfg["seed"])
    out = feat_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    timing, n = {}, grid.nrows * grid.ncols
    for date in d["sources"]:
        r = load_date(cfg, date)
        jobs = [("map", False)] + ([("obs", True)] if date == d["obs_date"] else [])
        for kind, aug in jobs:
            reps = fcfg["obs_variants"] if kind == "obs" else 1
            t = time.time()
            res = [compute_set(r, grid, fcfg, offsets[date], {DESC: hand_descriptor}, aug, rng)[DESC] for _ in range(reps)]
            dt = time.time() - t
            timing[f"{date}_{kind}"] = dict(seconds=dt, crops=n * reps)
            print(f"[poa-features] {date} {kind}: {n * reps} crops in {dt:.0f}s", flush=True)
            np.save(out / f"{DESC}_{date}_{kind}.npy", np.stack(res) if kind == "obs" else res[0])
        del r
    meta = dict(grid=grid.to_dict(), timing=timing, cfg=fcfg, total_seconds=sum(v["seconds"] for v in timing.values()))
    (out / f"meta_{DESC}.json").write_text(json.dumps(meta, indent=1))
    return meta


def standardize(x: np.ndarray, ref_idx: np.ndarray) -> np.ndarray:
    """PoC per-date z-score (statistics over `ref_idx` cells) + L2 normalization."""
    from .features import _standardize

    return _standardize(x, ref_idx)


# ============================================================================ Taquari PCA (read-only reuse)
def taquari_pca_params(cfg: dict, map_date: str) -> dict[str, np.ndarray]:
    """Recompute, from the Taquari files (read-only), the PCA that `env.build_cache` fitted on the
    Taquari TRAIN cells of `map_date` (mu, Vt, sd); it is not saved there, but it is a
    deterministic function of those files. Checked against the stored Taquari PCA channels."""
    from .config import default_config
    from .features import load_descriptors
    from .grid import Grid

    tq = _p(cfg, "taquari_data")
    tcfg = default_config()
    td = tcfg["data"]
    grid = Grid.from_aoi(**td["aoi"], cell=td["cell_m"])
    train, _ = grid.split_rows(td["split_lat"], td["split_margin_m"])
    m, _ = load_descriptors(DESC, map_date, grid, train, feat_dir=tq / "features")
    ok = np.load(tq / "masks.npz")[f"{map_date}_ok"]
    P = tcfg["env"]["pca_components"]
    tr = np.zeros(ok.shape, bool)
    tr[train] = True
    X = m[tr & ok]
    mu = X.mean(0)
    _, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    sd = S[:P] / np.sqrt(len(X))
    params = dict(mu=mu, Vt=Vt[:P], sd=sd)
    ref = np.load(tq / "cache" / DESC / f"aux_{map_date}.npz")["pca"]
    err = float(np.abs(apply_pca(params, m, ok) - ref).max())
    assert err < 1e-4, f"Taquari PCA reproduction failed ({err})"
    params["reproduction_max_abs_err"] = np.array(err)
    return params


def apply_pca(params: dict, m: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Same transform as env.build_cache, with fixed (not refitted) parameters -> (P, R, C)."""
    proj = ((m - params["mu"]) @ params["Vt"].T) / params["sd"]
    pca = np.clip(proj / 3.0, -1, 1).transpose(2, 0, 1).astype(np.float32)
    pca[:, ~ok] = 0.0
    return pca


def load_taquari_pca(cfg: dict, map_date: str) -> dict[str, np.ndarray]:
    p = paths(cfg)["data"] / f"taquari_pca_{DESC}_{map_date}.npz"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, **taquari_pca_params(cfg, map_date))
    z = np.load(p)
    return {k: z[k] for k in z.files}


def taquari_beta(cfg: dict, scenario: str) -> dict:
    b = json.loads((_p(cfg, "taquari_data") / "beta.json").read_text())
    return b[f"{DESC}_{cfg['taquari_beta'][scenario]}"]


# ============================================================================ cache
def change_index(m: np.ndarray, o: np.ndarray) -> np.ndarray:
    """1 - cos(map descriptor, mean of the K observation descriptors) per cell, in [0, 2]."""
    om = o.mean(0)
    num = (m * om).sum(-1)
    den = np.linalg.norm(m, axis=-1) * np.linalg.norm(om, axis=-1) + 1e-12
    return np.clip(1.0 - num / den, 0.0, 2.0).astype(np.float32)


def build_cache(cfg: dict) -> None:
    """Standardized descriptors (per date, statistics over that date's valid cells of the whole POA
    box), distinctiveness on the POA map, Taquari-PCA channels and the change index."""
    d = cfg["data"]
    grid = poa_grid(cfg)
    masks = load_masks(cfg)
    fd, cd = feat_dir(cfg), cache_dir(cfg)
    cd.mkdir(parents=True, exist_ok=True)
    R, C = grid.nrows, grid.ncols
    obs = d["obs_date"]
    ref = lambda date: np.flatnonzero(masks[f"{date}_valid_frac"].ravel() >= d["valid_min"])  # noqa: E731
    o = standardize(np.load(fd / f"{DESC}_{obs}_obs.npy"), ref(obs))
    o = o.reshape(o.shape[0], R, C, -1)
    np.save(cd / "obs.npy", o.astype(np.float16))
    r_in, r_out = cfg["env"]["distinct_radius"]
    stats = {}
    for date in d["sources"]:
        m = standardize(np.load(fd / f"{DESC}_{date}_map.npy"), ref(date)).reshape(R, C, -1)
        ok = masks[f"{date}_ok"]
        np.save(cd / f"map_{date}.npy", m.astype(np.float16))
        np.save(cd / f"distinct_{date}.npy", distinctiveness(m, ok, r_in, r_out).astype(np.float32))
        ch = change_index(m, o)
        np.save(cd / f"change_{date}.npy", ch)
        for tscen, tdate in cfg["taquari_models"].items():
            np.save(cd / f"pca_{tscen}_{date}.npy", apply_pca(load_taquari_pca(cfg, tdate), m, ok))
        both = ok & masks[f"{obs}_ok"]
        stats[date] = dict(change_mean=float(ch[both].mean()) if both.any() else None)
        print(f"[poa-cache] {date}: change index mean {stats[date]['change_mean']}", flush=True)
    (cd / "meta.json").write_text(json.dumps(dict(grid=grid.to_dict(), stats=stats), indent=1))


# ============================================================================ region data
_MEM: dict = {}


def _arr(cfg: dict, name: str) -> np.ndarray:
    k = (str(cache_dir(cfg)), name)
    if k not in _MEM:
        _MEM[k] = np.load(cache_dir(cfg) / name)
        if _MEM[k].dtype == np.float16:
            _MEM[k] = _MEM[k].astype(np.float32)
    return _MEM[k]


def region_data(cfg: dict, region: str, scenario: str, beta: float, pca_from: str | None = None) -> ScenarioData:
    """ScenarioData cropped to a region; every row of the crop is usable (no train/test split here)."""
    date = cfg["scenarios"][scenario]
    rs, cs = region_slices(cfg)[region]
    masks = load_masks(cfg)
    obs = cfg["data"]["obs_date"]
    pca_from = pca_from or next(iter(cfg["taquari_models"]))
    pca = _arr(cfg, f"pca_{pca_from}_{date}.npy")[:, rs, cs]
    n = rs.stop - rs.start
    return ScenarioData(
        name=f"{DESC}_{region}_{scenario}", map_desc=_arr(cfg, f"map_{date}.npy")[rs, cs],
        obs_desc=_arr(cfg, "obs.npy")[:, rs, cs], map_ok=masks[f"{date}_ok"][rs, cs], obs_ok=masks[f"{obs}_ok"][rs, cs],
        distinct=_arr(cfg, f"distinct_{date}.npy")[rs, cs], pca=np.ascontiguousarray(pca),
        train_rows=np.arange(n), test_rows=np.arange(n), beta=float(beta),
        meta=dict(descriptor=DESC, scenario=scenario, map_date=date, region=region, pca_from=pca_from))


def region_covariates(cfg: dict, region: str, scenario: str) -> dict[str, np.ndarray]:
    date = cfg["scenarios"][scenario]
    rs, cs = region_slices(cfg)[region]
    masks = load_masks(cfg)
    obs = cfg["data"]["obs_date"]
    return dict(change=_arr(cfg, f"change_{date}.npy")[rs, cs], water_obs=masks[f"{obs}_water_frac"][rs, cs],
                water_map=masks[f"{date}_water_frac"][rs, cs],
                both=(masks[f"{date}_ok"] & masks[f"{obs}_ok"])[rs, cs])


def n_windows(cfg: dict, region: str, scenario: str) -> int:
    f = cfg["filter"]
    return len(region_data(cfg, region, scenario, 1.0).windows("test", f["window"], f["min_window_valid"]))


def eval_units(cfg: dict) -> list[tuple[str, str]]:
    """(region, scenario) pairs with enough valid windows; the others are reported as skipped."""
    out = []
    for region in cfg["regions"]:
        for scen in cfg["scenarios"]:
            if n_windows(cfg, region, scen) >= cfg["eval"]["min_windows"]:
                out.append((region, scen))
    return out


# ============================================================================ beta recalibration
def other_region(cfg: dict, region: str) -> str:
    return next(r for r in cfg["regions"] if r != region)


def recalibrate(cfg: dict, units: list[tuple[str, str]], n_windows_: int = 30) -> dict:
    """Variant "beta recalibrated": beta_1 (max. likelihood) and kappa (random-policy utility)
    estimated on the OTHER region, for every scenario; written to data_poa/beta_recal.json."""
    from .calibrate import calibrate_kappa, window_stats

    P = paths(cfg)
    f = cfg["filter"]
    kc = cfg.get("calibration", {})
    out, rows = {}, []
    for region, scen in units:
        src = other_region(cfg, region)
        if (src, scen) not in units and n_windows(cfg, src, scen) == 0:
            print(f"[poa-calib] {region} {scen}: no windows in {src}, recalibrated variant skipped")
            continue
        data = region_data(cfg, src, scen, 1.0)
        st = window_stats(data, f["window"], n_windows_, 0, f["min_window_valid"])
        data.beta = st["beta"]
        kappa, krows = calibrate_kappa(data, f, kc.get("episodes", 200), kc.get("workers", 8))
        rec = dict(region=region, scenario=scen, calibrated_on=src, beta1=st["beta"], kappa=kappa,
                   beta=st["beta"] * kappa, top1=st["top1"], top5=st["top5"], top1_within1=st["top1_within1"],
                   loglik_true=st["loglik_true"], loglik_uniform=st["loglik_uniform"], n_windows=st["n_windows"])
        out[f"{region}_{scen}"] = rec
        rows.append(rec)
        pd.DataFrame(krows).assign(region=region, scenario=scen, calibrated_on=src).to_csv(
            P["results"] / f"kappa_{region}_{scen}.csv", index=False)
        print(f"[poa-calib] {region} {scen} (on {src}): top1 {st['top1']:.3f} beta1 {st['beta']} kappa {kappa} "
              f"-> beta {rec['beta']:.2f}", flush=True)
    (P["data"] / "beta_recal.json").write_text(json.dumps(out, indent=1))
    return out


def retrieval_table(cfg: dict, units: list[tuple[str, str]], n_windows_: int = 30) -> pd.DataFrame:
    """Descriptor retrieval in each evaluated (region, scenario) (diagnostic only; nothing is tuned on it)."""
    from .calibrate import window_stats

    f = cfg["filter"]
    rows = []
    for region, scen in units:
        st = window_stats(region_data(cfg, region, scen, 1.0), f["window"], n_windows_, 0, f["min_window_valid"])
        st.pop("beta")
        rows.append(dict(region=region, scenario=scen, **st))
    df = pd.DataFrame(rows)
    df.to_csv(paths(cfg)["results"] / "retrieval.csv", index=False)
    return df


def beta_table(cfg: dict, units: list[tuple[str, str]]) -> pd.DataFrame:
    """beta per (region, scenario, variant) as used by the filter -> results_poa/beta.csv."""
    p = paths(cfg)["data"] / "beta_recal.json"
    rec = json.loads(p.read_text()) if p.exists() else {}
    rows = []
    for region, scen in units:
        t = taquari_beta(cfg, scen)
        rows.append(dict(region=region, scenario=scen, variant="zeroshot", source=f"Taquari {cfg['taquari_beta'][scen]}",
                         beta1=t["beta1"], kappa=t["kappa"], beta=t["beta"]))
        r = rec.get(f"{region}_{scen}")
        if r:
            rows.append(dict(region=region, scenario=scen, variant="recal", source=f"POA {r['calibrated_on']}",
                             beta1=r["beta1"], kappa=r["kappa"], beta=r["beta"]))
    df = pd.DataFrame(rows)
    df.to_csv(paths(cfg)["results"] / "beta.csv", index=False)
    return df


def beta_for(cfg: dict, region: str, scen: str, variant: str) -> float | None:
    if variant == "zeroshot":
        return float(taquari_beta(cfg, scen)["beta"])
    p = paths(cfg)["data"] / "beta_recal.json"
    rec = json.loads(p.read_text()).get(f"{region}_{scen}") if p.exists() else None
    return None if rec is None else float(rec["beta"])


# ============================================================================ evaluation
COLUMNS = ["region", "scenario", "variant", "method", "model_seed", "episode", "seed", "success", "false_loc",
           "steps", "steps_to_success", "final_error_m", "distance_m", "n_blocked", "final_entropy", "decision_ms",
           "beta", "window_r0", "window_c0", "change_window", "change_start", "water_obs", "water_map",
           "water_diff"]
_W: dict = {}


def _policy(cfg: dict, method: str, model_path: str | None):
    k = (method, model_path)
    if k not in _W:
        if model_path:
            from .evaluate import PPOPolicy

            _W[k] = PPOPolicy(model_path, cfg["eval"].get("ppo_deterministic", True))
        else:
            from .baselines import make_baselines

            _W[k] = make_baselines(cfg)[method]
    return _W[k]


def run_episodes(task: tuple) -> list[dict]:
    """Worker: one (region, scenario, variant, method, model) and a chunk of paired episode indices."""
    import torch

    torch.set_num_threads(1)
    from .filter import disk_offsets

    cfg, region, scen, variant, method, model_seed, model_path, pca_from, idx, save_traj, traj_dir = task
    beta = beta_for(cfg, region, scen, variant)
    key = ("data", region, scen, variant, pca_from)
    if key not in _W:
        _W[key] = region_data(cfg, region, scen, beta, pca_from)
        _W[("cov", region, scen)] = region_covariates(cfg, region, scen)
    data, cov = _W[key], _W[("cov", region, scen)]
    ep = Episode(data, cfg["filter"], "test", cfg["data"]["cell_m"])
    pol = _policy(cfg, method, model_path)
    W = ep.W
    disk = disk_offsets(5)
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
        sl = (slice(ep.r0, ep.r0 + W), slice(ep.c0, ep.c0 + W))
        both = cov["both"][sl]
        ch = cov["change"][sl]
        sr = np.clip(ep.start[0] + disk[:, 0], 0, W - 1)
        sc = np.clip(ep.start[1] + disk[:, 1], 0, W - 1)
        sb = both[sr, sc]
        wo, wm = float(cov["water_obs"][sl].mean()), float(cov["water_map"][sl].mean())
        rows.append(dict(region=region, scenario=scen, variant=variant, method=method, model_seed=model_seed,
                         episode=int(i), seed=seed, success=bool(r.success), false_loc=bool(r.false_loc), steps=ep.t,
                         steps_to_success=ep.t if r.success else np.nan, final_error_m=r.error_m,
                         distance_m=ep.distance_m, n_blocked=ep.n_blocked, final_entropy=r.entropy,
                         decision_ms=1000 * float(np.mean(dts)) if dts else np.nan, beta=beta,
                         window_r0=int(ep.r0), window_c0=int(ep.c0),
                         change_window=float(ch[both].mean()) if both.any() else np.nan,
                         change_start=float(ch[sr, sc][sb].mean()) if sb.any() else np.nan,
                         water_obs=wo, water_map=wm, water_diff=wo - wm))
        if keep:
            traj_dir.mkdir(parents=True, exist_ok=True)
            tag = f"{method}{'' if model_seed < 0 else model_seed}"
            np.savez_compressed(traj_dir / f"{region}_{scen}_{variant}_{tag}_ep{i}.npz", traj=np.array(ep.traj),
                                map_est=np.array(maps), beliefs=np.array(beliefs, np.float32),
                                window=np.array([ep.r0, ep.c0]), success=r.success, false_loc=r.false_loc)
    return rows


def find_taquari_models(cfg: dict) -> list[tuple[str, int, str]]:
    from .evaluate import find_models

    root = _p(cfg, "taquari_runs")
    out = []
    for tscen in cfg["taquari_models"]:
        out += [(f"ppo_{tscen}", s, p) for s, p in find_models(root, tscen, DESC)]
    return out


def evaluate(cfg: dict, units: list[tuple[str, str]], methods: list[str], variants: list[str], episodes: int,
             out_csv: Path, workers: int, save_traj: int, chunk: int = 25, extra_models: list | None = None) -> pd.DataFrame:
    """`methods`: baseline names and/or "ppo" (= every Taquari model). `extra_models`: further
    (method, seed, path, pca_from) PPO entries (e.g. models trained in Porto Alegre)."""
    models = find_taquari_models(cfg) if "ppo" in methods else []
    specs = [(m, -1, None, None) for m in methods if m != "ppo"]
    specs += [(m, s, p, m.split("_", 1)[1]) for m, s, p in models]
    specs += list(extra_models or [])
    traj_dir = out_csv.parent / "traj"
    tasks = []
    for region, scen in units:
        for variant in variants:
            if beta_for(cfg, region, scen, variant) is None:
                print(f"[poa-eval] skip {region} {scen} {variant}: no beta")
                continue
            for method, ms, path, pca_from in specs:
                # trajectories: first episodes of the zero-shot variant, PPO seed 0 (or first seed)
                keep = save_traj if variant == variants[0] and ms <= 0 else 0
                for s in range(0, episodes, chunk):
                    tasks.append((cfg, region, scen, variant, method, ms, path, pca_from,
                                  list(range(s, min(s + chunk, episodes))), keep, traj_dir))
    tasks.sort(key=lambda t: 0 if t[4].startswith("greedy") else 1)
    t0 = time.time()
    rows: list[dict] = []
    print(f"[poa-eval] {len(tasks)} tasks, {workers} workers", flush=True)
    if workers > 1:
        with get_context("forkserver").Pool(workers) as pool:
            for i, r in enumerate(pool.imap_unordered(run_episodes, tasks)):
                rows += r
                if (i + 1) % max(1, len(tasks) // 20) == 0:
                    print(f"[poa-eval] {i + 1}/{len(tasks)} tasks, {time.time() - t0:.0f}s", flush=True)
    else:
        for t in tasks:
            rows += run_episodes(t)
    df = pd.DataFrame(rows, columns=COLUMNS).sort_values(["region", "scenario", "variant", "method", "model_seed", "episode"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[poa-eval] {len(df)} episodes -> {out_csv} in {time.time() - t0:.0f}s", flush=True)
    return df


# ============================================================================ sanity gate
def sanity_gate(cfg: dict, episodes: int = 200, workers: int = 8) -> dict:
    """P0 (map = observation date): the straight baseline must localize in >= 90 % in both regions."""
    P = paths(cfg)
    df = evaluate(cfg, [(r, "P0") for r in cfg["regions"]], ["straight", "random", "lawnmower"], ["zeroshot"],
                  episodes, P["results"] / "gate_episodes.csv", workers, 0)
    rates = df.groupby(["region", "method"]).success.mean().unstack()
    ok = bool((rates["straight"] >= 0.9).all())
    res = dict(passed=ok, threshold=0.9, episodes=episodes,
               success={r: {m: float(v) for m, v in row.items()} for r, row in rates.iterrows()},
               false_loc={r: float(v) for r, v in df[df.method == "straight"].groupby("region").false_loc.mean().items()})
    (P["results"] / "sanity_gate.json").write_text(json.dumps(res, indent=1))
    print(f"[poa-gate] sanity gate {'PASSED' if ok else 'FAILED'}: {res['success']}", flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs_poa/poa.yaml")
    ap.add_argument("--step", choices=["features", "cache", "units", "gate", "calib", "beta", "eval"], required=True)
    ap.add_argument("--episodes", type=int)
    ap.add_argument("--methods", nargs="*", default=["straight", "lawnmower", "random", "greedy_map", "ppo"])
    ap.add_argument("--variants", nargs="*", default=["zeroshot", "recal"])
    ap.add_argument("--scenarios", nargs="*")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--out")
    a = ap.parse_args()
    cfg = load_config(a.config)
    P = paths(cfg)
    P["results"].mkdir(parents=True, exist_ok=True)
    workers = a.workers or cfg["eval"].get("workers") or os.cpu_count() or 1
    if a.step == "features":
        precompute(cfg)
    elif a.step == "cache":
        build_cache(cfg)
    elif a.step == "units":
        units = {f"{r}_{s}": n_windows(cfg, r, s) for r in cfg["regions"] for s in cfg["scenarios"]}
        (P["results"] / "windows.json").write_text(json.dumps(dict(n_windows=units, min_windows=cfg["eval"]["min_windows"],
                                                                     evaluated=[f"{r}_{s}" for r, s in eval_units(cfg)]), indent=1))
        print(units)
    elif a.step == "gate":
        sanity_gate(cfg, a.episodes or 200, workers)
    elif a.step == "calib":
        units = eval_units(cfg)
        retrieval_table(cfg, units)
        recalibrate(cfg, units)
        beta_table(cfg, units)
    elif a.step == "beta":
        print(beta_table(cfg, eval_units(cfg)))
    elif a.step == "eval":
        units = [u for u in eval_units(cfg) if not a.scenarios or u[1] in a.scenarios]
        evaluate(cfg, units, a.methods, a.variants, a.episodes or cfg["eval"]["episodes"],
                 Path(a.out) if a.out else P["results"] / "episodes.csv", workers, cfg["eval"]["save_traj"])


if __name__ == "__main__":
    main()

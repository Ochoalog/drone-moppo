"""P7: índice de mudança no Taquari (F1) e mecanismo M1 (F3), com os critérios do DECISIONS_FINAL.md.

Etapas:
1. `change`: índice de mudança por episódio no Taquari (S1, S2; teste sul) a partir dos episódios
   já avaliados, e quintis de sucesso nos dois locais → results_final/change_index/.
2. `replay`: reexecuta PPO-S2 (seeds 0–2) e reta em argmax nos mesmos episódios (Taquari S2 e
   POA P1 zero-shot, ilhas e centro), registrando distintividade e mudança das células visitadas.
   Confere que cada episódio reproduz o CSV gravado → results_final/mechanism/visits.csv.
3. `verdict`: bootstrap por janela (e seeds) e veredito do M1 → results_final/mechanism/m1.csv.

Uso: .venv/bin/python scripts/analyze_mechanism.py {change,replay,verdict} [--workers 8]
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from dal.config import ROOT, load_config  # noqa: E402
from dal.evaluation.statistics import percentile_ci, reconstruct_episode_window  # noqa: E402

OUT = ROOT / "results_final"
N_BOOT = 2000
QUINTILES = 5
POA_REGIONS = ("ilhas", "centro")
PPO_SEEDS = (0, 1, 2)


# ============================================================================ F1: índice de mudança
def taquari_change_per_episode() -> pd.DataFrame:
    """Índice de mudança médio da janela de cada episódio do teste (S1, S2)."""
    from dal.env import ScenarioData
    from dal.poa import change_index

    config = load_config("configs/default.yaml")
    episodes = pd.read_csv(ROOT / "results/episodes_test.csv")
    rows = []
    for scenario in ("S1", "S2"):
        data = ScenarioData.load("hand", scenario, config)
        change = change_index(data.map_desc, data.obs_desc)
        valid = data.map_ok & data.obs_ok
        W = int(config["filter"]["window"])
        windows = data.windows("test", W, config["filter"]["min_window_valid"])
        for episode in sorted(episodes.loc[episodes.scenario == scenario, "episode"].unique()):
            r0, c0 = reconstruct_episode_window(windows, int(config["eval"]["seed"]) + int(episode))
            block, ok = change[r0:r0 + W, c0:c0 + W], valid[r0:r0 + W, c0:c0 + W]
            rows.append({"scenario": scenario, "episode": int(episode), "window_r0": r0, "window_c0": c0,
                         "change_window": float(block[ok].mean())})
    return episodes.merge(pd.DataFrame(rows), on=["scenario", "episode"])


def quintile_table(episodes: pd.DataFrame, location: str, unit_columns: list[str]) -> pd.DataFrame:
    """Sucesso por quintil do índice (quintis sobre as unidades de episódio do local) e Spearman por método."""
    units = episodes.drop_duplicates(unit_columns)
    edges = np.quantile(units["change_window"].dropna(), np.linspace(0, 1, QUINTILES + 1))
    episodes = episodes.assign(quintile=np.clip(np.searchsorted(edges, episodes["change_window"], side="right"),
                                                1, QUINTILES))
    rows = []
    for method, group in episodes.groupby("method"):
        rho = spearmanr(group["change_window"], group["success"].astype(float), nan_policy="omit").statistic
        for quintile, part in group.groupby("quintile"):
            rows.append({"location": location, "method": method, "quintile": int(quintile),
                         "lo_edge": edges[quintile - 1], "hi_edge": edges[quintile], "n": len(part),
                         "success": part["success"].mean(), "false_localization": part["false_loc"].mean(),
                         "spearman_change_success": rho})
    return pd.DataFrame(rows)


def run_change_stage() -> None:
    out = OUT / "change_index"
    out.mkdir(parents=True, exist_ok=True)
    taquari = taquari_change_per_episode()
    taquari.to_csv(out / "taquari_episodes.csv", index=False)
    poa = pd.read_csv(ROOT / "results_poa/episodes.csv")
    poa = poa[poa["variant"] == "zeroshot"]
    table = pd.concat([quintile_table(taquari, "taquari", ["scenario", "episode"]),
                       quintile_table(poa, "porto_alegre", ["scenario", "region", "episode"])], ignore_index=True)
    table.to_csv(out / "success_by_change_quintile.csv", index=False)
    print(table[table.quintile.isin([1, QUINTILES])].round(3).to_string(index=False))


# ============================================================================ F2: covariável independente
def _match_histogram(source: np.ndarray, reference: np.ndarray, source_ok: np.ndarray, reference_ok: np.ndarray):
    """Iguala, canal a canal, o histograma de `source` ao de `reference` (só pixels válidos, AOI inteira)."""
    matched = source.copy()
    for channel in range(3):
        src_hist = np.bincount(source[..., channel][source_ok], minlength=256).astype(float)
        ref_hist = np.bincount(reference[..., channel][reference_ok], minlength=256).astype(float)
        src_cdf, ref_cdf = np.cumsum(src_hist) / src_hist.sum(), np.cumsum(ref_hist) / ref_hist.sum()
        lookup = np.clip(np.round(np.interp(src_cdf, ref_cdf, np.arange(256))), 0, 255).astype(np.uint8)
        matched[..., channel] = lookup[source[..., channel]]
    return matched


def _cell_mean_lab(raster, grid, offset: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Média Lab por célula da grade (pixels válidos) e a fração de pixels válidos por célula."""
    import cv2

    from dal.data import valid_mask

    height, width = raster.img.shape[:2]
    cols = raster.x_origin + (np.arange(width) + 0.5) * raster.res - offset[0]
    rows = raster.y_origin - (np.arange(height) + 0.5) * raster.res - offset[1]
    cell_col = np.floor((cols - grid.x0) / grid.cell).astype(int)
    cell_row = np.floor((grid.y1 - rows) / grid.cell).astype(int)
    inside = ((cell_row[:, None] >= 0) & (cell_row[:, None] < grid.nrows)
              & (cell_col[None, :] >= 0) & (cell_col[None, :] < grid.ncols))
    ok = valid_mask(raster.img) & inside
    index = (cell_row[:, None] * grid.ncols + cell_col[None, :])[ok]
    lab = cv2.cvtColor(raster.img, cv2.COLOR_RGB2LAB).astype(np.float64)[ok]
    n_cells = grid.nrows * grid.ncols
    counts = np.bincount(index, minlength=n_cells)
    means = np.stack([np.bincount(index, weights=lab[:, c], minlength=n_cells) for c in range(3)], axis=1)
    return means / np.maximum(counts, 1)[:, None], counts


def run_covariate_stage() -> None:
    """F2: ΔE Lab por célula (mapa com histograma igualado ao da observação) e Spearman com o sucesso."""
    from dal.align import load_offsets
    from dal.data import load_date, valid_mask
    from dal.grid import Grid

    config = load_config("configs/default.yaml")
    grid = Grid.from_aoi(**config["data"]["aoi"], cell=config["data"]["cell_m"])
    offsets = load_offsets()
    post = load_date("post")
    post_lab, post_counts = _cell_mean_lab(post, grid, offsets["post"])
    episodes = pd.read_csv(OUT / "change_index" / "taquari_episodes.csv")
    W = int(config["filter"]["window"])
    rows = []
    for scenario in ("S1", "S2"):
        map_date = config["scenarios"][scenario]
        raster = load_date(map_date)
        raster.img = _match_histogram(raster.img, post.img, valid_mask(raster.img), valid_mask(post.img))
        map_lab, map_counts = _cell_mean_lab(raster, grid, offsets[map_date])
        min_pixels = 0.5 * (grid.cell / post.res) ** 2
        delta_e = np.where((map_counts >= min_pixels) & (post_counts >= min_pixels),
                           np.linalg.norm(map_lab - post_lab, axis=1), np.nan).reshape(grid.nrows, grid.ncols)
        np.save(OUT / "change_index" / f"taquari_delta_e_{scenario}.npy", delta_e.astype(np.float32))
        for _, ep in episodes[episodes.scenario == scenario].drop_duplicates("episode").iterrows():
            block = delta_e[ep.window_r0:ep.window_r0 + W, ep.window_c0:ep.window_c0 + W]
            rows.append({"scenario": scenario, "episode": int(ep.episode), "color_delta_window": float(np.nanmean(block))})
        del raster
    episodes = episodes.merge(pd.DataFrame(rows), on=["scenario", "episode"])
    episodes.to_csv(OUT / "change_index" / "taquari_episodes_with_color.csv", index=False)
    table = []
    for method, group in episodes.groupby("method"):
        success = group["success"].astype(float)
        table.append({"method": method,
                      "spearman_change_success": spearmanr(group.change_window, success).statistic,
                      "spearman_color_success": spearmanr(group.color_delta_window, success, nan_policy="omit").statistic,
                      "spearman_change_color": spearmanr(group.change_window, group.color_delta_window,
                                                         nan_policy="omit").statistic, "n": len(group)})
    table = pd.DataFrame(table)
    table.to_csv(OUT / "change_index" / "independent_covariate.csv", index=False)
    print(table.round(3).to_string(index=False))


# ============================================================================ F3: reexecução
def _location_setup(location: str):
    """Dados do episódio, mapas de distintividade e mudança, e config, para um local."""
    if location == "taquari_S2":
        from dal.env import ScenarioData
        from dal.poa import change_index

        config = load_config("configs/default.yaml")
        data = ScenarioData.load("hand", "S2", config)
        return config, data, data.distinct, change_index(data.map_desc, data.obs_desc)
    from dal.poa import beta_for, region_covariates, region_data

    config = load_config("configs_poa/poa.yaml")
    region = location.split("_", 1)[1]
    data = region_data(config, region, "P1", beta_for(config, region, "P1", "zeroshot"), "S2")
    return config, data, data.distinct, region_covariates(config, region, "P1")["change"]


def replay_task(task: tuple[str, str, int]) -> pd.DataFrame:
    """Reexecuta um (local, método, seed) em todos os episódios e mede as células visitadas."""
    import torch

    from dal.baselines import StraightPolicy
    from dal.env import Episode
    from dal.evaluate import PPOPolicy

    torch.set_num_threads(1)
    location, method, model_seed = task
    config, data, distinct, change = _location_setup(location)
    episode = Episode(data, config["filter"], "test", config["data"]["cell_m"])
    policy = (PPOPolicy(str(ROOT / "runs/overnight" / f"S2_hand_s{model_seed}" / "final_model.zip"), True)
              if method == "ppo" else StraightPolicy())
    n_episodes = 1000 if location == "taquari_S2" else 500
    rows = []
    for i in range(n_episodes):
        seed = int(config["eval"]["seed"]) + i
        episode.reset(seed=seed)
        policy.reset(episode, np.random.default_rng([seed, 7]))
        while not episode.done:
            episode.step(policy.act(episode))
        cells = np.array(episode.traj) + np.array([episode.r0, episode.c0])
        rows.append({"location": location, "method": method, "model_seed": model_seed, "episode": i,
                     "window_r0": episode.r0, "window_c0": episode.c0, "success": bool(episode.last.success),
                     "false_loc": bool(episode.last.false_loc), "steps": int(episode.t),
                     "visited_distinct": float(np.nanmean(distinct[cells[:, 0], cells[:, 1]])),
                     "visited_change": float(np.nanmean(change[cells[:, 0], cells[:, 1]]))})
    return pd.DataFrame(rows)


def check_reproduction(visits: pd.DataFrame) -> pd.DataFrame:
    """Compara sucesso, localização falsa e passos com os CSVs gravados; devolve as divergências."""
    taquari = pd.read_csv(ROOT / "results/episodes_test.csv")
    taquari = taquari[(taquari.scenario == "S2") & taquari.method.isin(["ppo", "straight"])].assign(location="taquari_S2")
    poa = pd.read_csv(ROOT / "results_poa/episodes.csv")
    poa = poa[(poa.scenario == "P1") & (poa.variant == "zeroshot") & poa.method.isin(["ppo_S2", "straight"])]
    poa = poa.assign(location="poa_" + poa.region, method=poa.method.replace({"ppo_S2": "ppo"}))
    recorded = pd.concat([taquari, poa])[["location", "method", "model_seed", "episode", "success", "false_loc", "steps"]]
    recorded["model_seed"] = recorded["model_seed"].where(recorded.method == "ppo", -1)
    merged = visits.assign(model_seed=visits.model_seed.where(visits.method == "ppo", -1)).merge(
        recorded, on=["location", "method", "model_seed", "episode"], suffixes=("", "_recorded"))
    mismatch = (merged.success != merged.success_recorded) | (merged.false_loc != merged.false_loc_recorded) | (
        merged.steps != merged.steps_recorded)
    print(f"[replay] {len(merged)} episódios conferidos, {int(mismatch.sum())} divergentes")
    return merged[mismatch]


def run_replay_stage(workers: int) -> None:
    out = OUT / "mechanism"
    out.mkdir(parents=True, exist_ok=True)
    locations = ["taquari_S2"] + [f"poa_{region}" for region in POA_REGIONS]
    tasks = [(loc, "ppo", s) for loc in locations for s in PPO_SEEDS] + [(loc, "straight", -1) for loc in locations]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        visits = pd.concat(pool.map(replay_task, tasks), ignore_index=True)
    visits.to_csv(out / "visits.csv", index=False)
    check_reproduction(visits).to_csv(out / "reproduction_mismatches.csv", index=False)


# ============================================================================ F3: veredito
def _bootstrap_difference(first: pd.DataFrame, second: pd.DataFrame, measure: str, rng) -> np.ndarray:
    """Réplicas de média(first) − média(second): janelas com reposição (compartilhadas) e seeds da PPO."""
    windows = np.unique(np.concatenate([first.window_id.unique(), second.window_id.unique()]))
    replicates = np.empty(N_BOOT)
    for b in range(N_BOOT):
        weight = pd.Series(np.bincount(rng.integers(len(windows), size=len(windows)), minlength=len(windows)),
                           index=windows)
        means = []
        for part in (first, second):
            seeds = part.model_seed.unique()
            seed_weight = pd.Series(np.bincount(rng.integers(len(seeds), size=len(seeds)), minlength=len(seeds)),
                                    index=seeds)
            w = weight.reindex(part.window_id).to_numpy() * seed_weight.reindex(part.model_seed).to_numpy()
            means.append(np.average(part[measure], weights=w) if w.sum() > 0 else np.nan)
        replicates[b] = means[0] - means[1]
    return replicates


def run_verdict_stage() -> None:
    visits = pd.read_csv(OUT / "mechanism" / "visits.csv")
    visits["window_id"] = visits.window_r0.astype(str) + "_" + visits.window_c0.astype(str)
    rng = np.random.default_rng(0)
    rows = []
    for location, group in visits.groupby("location"):
        ppo, straight = group[group.method == "ppo"], group[group.method == "straight"]
        comparisons = {
            "a_ppo_false_minus_ppo_success": (ppo[ppo.false_loc], ppo[ppo.success]),
            "b_ppo_false_minus_straight_false": (ppo[ppo.false_loc], straight[straight.false_loc]),
        }
        for measure in ("visited_distinct", "visited_change"):
            for name, (first, second) in comparisons.items():
                low, high = percentile_ci(_bootstrap_difference(first, second, measure, rng))
                rows.append({"location": location, "measure": measure, "comparison": name,
                             "estimate": first[measure].mean() - second[measure].mean(), "ci_low": low,
                             "ci_high": high, "n_first": len(first), "n_second": len(second),
                             "holds": bool(low > 0), "contradicts": bool(high < 0)})
    table = pd.DataFrame(rows)
    by_location = table.groupby("location").agg(holds=("holds", "all"), contradicts=("contradicts", "any"))
    poa_ok = by_location.loc[[f"poa_{r}" for r in POA_REGIONS if f"poa_{r}" in by_location.index], "holds"].any()
    if by_location.contradicts.any():
        verdict = "não suportada"
    elif by_location.loc["taquari_S2", "holds"] and poa_ok:
        verdict = "suportada"
    else:
        verdict = "inconclusiva"
    table["m1_verdict"] = verdict
    table.to_csv(OUT / "mechanism" / "m1.csv", index=False)
    print(table.round(4).to_string(index=False))
    print(f"M1: {verdict}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["change", "covariate", "replay", "verdict"])
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = parser.parse_args()
    {"change": run_change_stage, "covariate": run_covariate_stage, "replay": lambda: run_replay_stage(args.workers),
     "verdict": run_verdict_stage}[args.stage]()


if __name__ == "__main__":
    main()

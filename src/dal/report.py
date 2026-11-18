"""Tabelas, estatística, figuras e REPORT.md. Todo número sai de arquivos na pasta de resultados."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from .config import ROOT, load_config  # noqa: E402

SIMPLE = ["straight", "lawnmower", "random"]
ORDER = ["ppo", "greedy", "greedy_map", "lawnmower", "straight", "random"]
COLORS = {"ppo": "#1f77b4", "greedy": "#d62728", "greedy_map": "#ff9896", "lawnmower": "#2ca02c",
          "straight": "#9467bd", "random": "#7f7f7f"}
LABEL = {"ppo": "PPO", "greedy": "greedy IG (oracle obs.)", "greedy_map": "greedy IG (map-predicted obs.)",
         "lawnmower": "lawnmower", "straight": "straight", "random": "random"}


# ============================================================================ estatística
def boot_ci(x: np.ndarray, n: int, rng: np.random.Generator, stat=np.mean) -> tuple[float, float]:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    idx = rng.integers(len(x), size=(n, len(x)))
    v = stat(x[idx], axis=1)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def penalized_steps(df: pd.DataFrame, max_steps: int) -> pd.Series:
    """Passos até o sucesso; falhas (inclusive localização falsa) contam como max_steps + 1."""
    return df.steps_to_success.fillna(max_steps + 1)


def per_episode(df: pd.DataFrame, max_steps: int) -> pd.DataFrame:
    """Uma linha por (cenário, método, episódio); os seeds do PPO são promediados por episódio."""
    d = df.assign(pen=penalized_steps(df, max_steps), success=df.success.astype(float),
                  false_loc=df.false_loc.astype(float))
    return d.groupby(["scenario", "method", "episode"])[["success", "false_loc", "pen"]].mean().reset_index()


def summary_table(df: pd.DataFrame, n_boot: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for (scen, meth), g in df.groupby(["scenario", "method"]):
        s = g.steps_to_success.dropna()
        lo, hi = boot_ci(g.success.astype(float), n_boot, rng)
        flo, fhi = boot_ci(g.false_loc.astype(float), n_boot, rng)
        per_seed = g.groupby("model_seed").success.mean()
        rows.append(dict(
            scenario=scen, method=meth, episodes=len(g), model_seeds=int((g.model_seed >= 0).any() and g.model_seed.nunique()),
            success=g.success.mean(), success_lo=lo, success_hi=hi,
            success_seed_std=float(per_seed.std(ddof=0)) if len(per_seed) > 1 else np.nan,
            false_loc=g.false_loc.mean(), false_loc_lo=flo, false_loc_hi=fhi,
            steps_median=s.median() if len(s) else np.nan,
            steps_q1=s.quantile(0.25) if len(s) else np.nan, steps_q3=s.quantile(0.75) if len(s) else np.nan,
            final_error_median_m=g.final_error_m.median(), distance_mean_m=g.distance_m.mean(),
            decision_ms=g.decision_ms.mean()))
    t = pd.DataFrame(rows)
    t["order"] = t.method.map({m: i for i, m in enumerate(ORDER)}).fillna(99)
    return t.sort_values(["scenario", "order"]).drop(columns="order")


def paired_tests(df: pd.DataFrame, max_steps: int, n_boot: int, ref: str = "ppo", seed: int = 1) -> pd.DataFrame:
    """Wilcoxon pareado nos passos penalizados e bootstrap pareado da diferença de sucesso."""
    rng = np.random.default_rng(seed)
    pe = per_episode(df, max_steps)
    rows = []
    for scen, g in pe.groupby("scenario"):
        if ref not in set(g.method):
            continue
        a = g[g.method == ref].set_index("episode")
        for m in sorted(set(g.method) - {ref}):
            b = g[g.method == m].set_index("episode")
            common = a.index.intersection(b.index)
            x, y = a.loc[common], b.loc[common]
            diff = x.pen.values - y.pen.values
            p = float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
            ds = x.success.values - y.success.values
            lo, hi = boot_ci(ds, n_boot, rng)
            rows.append(dict(scenario=scen, method=ref, vs=m, n=len(common), success_diff=float(ds.mean()),
                             success_diff_lo=lo, success_diff_hi=hi, pen_steps_ref=float(x.pen.median()),
                             pen_steps_other=float(y.pen.median()), wilcoxon_p=p,
                             ref_better_steps=bool(np.median(diff) < 0 or (np.median(diff) == 0 and diff.mean() < 0))))
    return pd.DataFrame(rows)


# ============================================================================ figuras
def fig_success_curves(df: pd.DataFrame, max_steps: int, path: Path) -> None:
    scens = sorted(df.scenario.unique())
    fig, axes = plt.subplots(1, len(scens), figsize=(5 * len(scens), 4), sharey=True, squeeze=False)
    t = np.arange(1, max_steps + 1)
    for ax, scen in zip(axes[0], scens):
        g = df[df.scenario == scen]
        for m in [m for m in ORDER if m in set(g.method)]:
            s = g[g.method == m].steps_to_success.values
            curve = [(s <= k).mean() for k in t]  # NaN (falhas) dá False na comparação
            ax.plot(t, curve, label=LABEL[m], color=COLORS[m], lw=2 if m in ("ppo", "greedy") else 1.3)
        ax.set_title(scen)
        ax.set_xlabel("steps (1 step = 50 m)")
        ax.grid(alpha=0.3)
        ax.set_xlim(0, max_steps)
    axes[0][0].set_ylabel("cumulative success rate")
    axes[0][0].set_ylim(0, 1)
    axes[0][-1].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_bars(summ: pd.DataFrame, path: Path) -> None:
    scens = sorted(summ.scenario.unique())
    meths = [m for m in ORDER if m in set(summ.method)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    w = 0.8 / len(meths)
    x = np.arange(len(scens))
    for k, (col, ttl) in enumerate([("success", "success rate"), ("false_loc", "false-localization rate")]):
        ax = axes[k]
        for i, m in enumerate(meths):
            g = summ[summ.method == m].set_index("scenario").reindex(scens)
            err = np.vstack([g[col] - g[f"{col}_lo"], g[f"{col}_hi"] - g[col]])
            ax.bar(x + (i - len(meths) / 2 + 0.5) * w, g[col], w, yerr=err, capsize=2, color=COLORS[m], label=LABEL[m])
        ax.set_xticks(x, scens)
        ax.set_ylabel(ttl + " (95% bootstrap CI)")
        ax.set_ylim(0, 1 if col == "success" else max(0.05, float(summ[f"{col}_hi"].max()) * 1.2))
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_training(run_roots: list[Path], path: Path) -> bool:
    runs = [d for r in run_roots if r.exists() for d in sorted(r.iterdir()) if (d / "progress.csv").exists()]
    if not runs:
        return False
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    cmap = {"S0": "#1f77b4", "S1": "#ff7f0e", "S2": "#2ca02c"}
    for d in runs:
        try:
            p = pd.read_csv(d / "progress.csv")
        except pd.errors.EmptyDataError:
            continue
        if "time/total_timesteps" not in p:
            continue
        scen = d.name.split("_")[0]
        x = p["time/total_timesteps"]
        for ax, col in zip(axes, ["rollout/ep_rew_mean", "rollout/success_rate", "rollout/ep_len_mean"]):
            if col in p:
                ax.plot(x, p[col], color=cmap.get(scen, "k"), alpha=0.8, label=d.name)
    for ax, t in zip(axes, ["episode return", "success rate (train episodes)", "episode length"]):
        ax.set_title(t)
        ax.set_xlabel("timesteps")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def fig_trajectories(traj_dir: Path, cfg: dict, split: str, path: Path, methods=("ppo", "greedy", "lawnmower"),
                     episode: int | None = None) -> list[str]:
    """Uma figura por cenário; linhas = métodos, colunas = crença em t = 0, T/2, T. Por padrão mostra o
    episódio salvo com a maior trajetória média entre os métodos (mais informativo)."""
    from .data import load_date
    from .grid import Grid
    from .viz import thumb_extent_km

    d = cfg["data"]
    grid = Grid.from_aoi(**d["aoi"], cell=d["cell_m"])
    made = []
    W = cfg["filter"]["window"]
    for scen in ("S0", "S1", "S2"):
        eps: dict[int, list[int]] = {}
        for f in traj_dir.glob(f"{split}_{scen}_*_ep*.npz"):
            if f.name.split("_")[2].rstrip("0123456789") in methods:
                eps.setdefault(int(f.stem.rsplit("_ep", 1)[1]), []).append(len(np.load(f)["traj"]))
        if not eps:
            continue
        ep_i = episode if episode is not None else max(eps, key=lambda e: (len(eps[e]), np.mean(eps[e])))
        files = []
        for m in methods:
            c = sorted(traj_dir.glob(f"{split}_{scen}_{m}*_ep{ep_i}.npz"))
            c = [f for f in c if f.name.split("_")[2].rstrip("0123456789") == m]
            if c:
                files.append((m, c[0]))
        if not files:
            continue
        thumb = load_date(cfg["scenarios"][scen], thumb=True)
        ext = thumb_extent_km(thumb, grid)
        fig, axes = plt.subplots(len(files), 3, figsize=(11, 3.7 * len(files)), squeeze=False)
        for i, (m, f) in enumerate(files):
            z = np.load(f)
            tr, est, bel, (r0, c0) = z["traj"], z["map_est"], z["beliefs"], z["window"]
            T = len(tr) - 1
            for j, t in enumerate([0, T // 2, T]):
                ax = axes[i][j]
                ax.imshow(thumb.img, extent=ext)
                k = grid.cell / 1000
                win = [c0 * k, (c0 + W) * k, (r0 + W) * k, r0 * k]
                b = bel[t]
                ax.imshow(np.ma.masked_less(b / b.max(), 0.02), extent=win, cmap="magma", alpha=0.6, vmin=0, vmax=1)
                y = (r0 + tr[: t + 1, 0] + 0.5) * k
                x = (c0 + tr[: t + 1, 1] + 0.5) * k
                ax.plot(x, y, "-", color="cyan", lw=1.5)
                ax.plot(x[0], y[0], "o", color="lime", ms=6, label="start")
                ax.plot(x[-1], y[-1], "x", color="cyan", ms=8, mew=2, label="true")
                ax.plot((c0 + est[t, 1] + 0.5) * k, (r0 + est[t, 0] + 0.5) * k, "+", color="yellow", ms=10, mew=2,
                        label="MAP")
                ax.set_xlim(win[0], win[1])
                ax.set_ylim(win[2], win[3])
                ax.set_title(f"{LABEL[m]} t={t}" + (
                    f" ({'success' if z['success'] else 'FALSE LOC' if z['false_loc'] else 'timeout'})" if t == T else ""),
                    fontsize=9)
                ax.tick_params(labelsize=7)
                if j == 0:
                    ax.set_ylabel("km south")
                if i == len(files) - 1:
                    ax.set_xlabel("km east")
        axes[0][0].legend(fontsize=7, loc="lower left")
        fig.suptitle(f"{scen} ({split} episode {ep_i}): belief (magma, normalized to max) over the map thumbnail, "
                     "trajectory (cyan)")
        fig.tight_layout()
        out = path.with_name(f"{path.stem}_{scen}.png")
        fig.savefig(out, dpi=100)
        plt.close(fig)
        made.append(out.name)
    return made


# ============================================================================ hipóteses
CEILING = 0.98


def ceiling_scenarios(summ: pd.DataFrame) -> list[str]:
    """Cenários degenerados: toda baseline simples tem sucesso >= CEILING com mediana <= 2 passos, então
    nenhuma política pode se mostrar melhor em sucesso ou passos (ficam fora de H1/H3)."""
    out = []
    for scen, g in summ.groupby("scenario"):
        g = g[g.method.isin(SIMPLE)]
        if len(g) and g.success.min() >= CEILING and g.steps_median.max() <= 2:
            out.append(scen)
    return out


def verdicts(summ: pd.DataFrame, tests: pd.DataFrame, summ_train: pd.DataFrame | None, tests_train: pd.DataFrame | None,
             alpha: float = 0.05, competitive_pp: float = 0.05) -> dict[str, tuple[str, str]]:
    out = {}
    ceil = ceiling_scenarios(summ)
    ceil_note = (f"\n\nExcluded as degenerate (all simple baselines ≥ {CEILING:.2f} success in ≤ 2 steps): "
                 f"{', '.join(ceil)}." if ceil else "")
    if tests.empty or "ppo" not in set(summ.method):
        msg = "No trained PPO policy was evaluated."
        return {"H1": ("inconclusive", msg), "H2": ("inconclusive", msg), "H3": ("inconclusive", msg)}
    S = summ.set_index(["scenario", "method"])
    T = tests.set_index(["scenario", "vs"])
    # ---- H1
    lines, per = [], []
    for scen in sorted(summ.scenario.unique()):
        if scen in ceil:
            continue
        beats = []
        for m in SIMPLE:
            if (scen, m) not in T.index:
                continue
            r = T.loc[(scen, m)]
            steps_ok = r.wilcoxon_p < alpha and r.ref_better_steps
            if S.loc[(scen, m)].success >= CEILING:  # baseline no teto de sucesso: compara a rapidez
                ok = r.success_diff >= -0.01 and steps_ok
            else:
                ok = r.success_diff_lo > 0 and steps_ok
            beats.append(ok)
            lines.append(f"{scen}: PPO vs {m}: Δsuccess = {r.success_diff:+.3f} [{r.success_diff_lo:+.3f}, "
                         f"{r.success_diff_hi:+.3f}], Wilcoxon p = {r.wilcoxon_p:.2g} → {'better' if ok else 'not better'}")
        comp = None
        if (scen, "greedy") in T.index:
            r = T.loc[(scen, "greedy")]
            comp = r.success_diff >= -competitive_pp
            speed = S.loc[(scen, "greedy")].decision_ms / max(S.loc[(scen, "ppo")].decision_ms, 1e-9)
            lines.append(f"{scen}: PPO vs greedy IG: Δsuccess = {r.success_diff:+.3f} [{r.success_diff_lo:+.3f}, "
                         f"{r.success_diff_hi:+.3f}] (competitive if ≥ −{competitive_pp:.2f}: {'yes' if comp else 'no'}); "
                         f"decision time greedy/PPO = {speed:.1f}×")
        per.append(all(beats) and len(beats) == len(SIMPLE) and bool(comp))
    v = ("inconclusive" if not per else "supported" if all(per) else "not supported" if not any(per)
         else "inconclusive")
    out["H1"] = (v, "Criterion (per scenario): PPO beats every simple baseline (paired-bootstrap CI of Δsuccess > 0, "
                    "Wilcoxon p < 0.05 on penalized steps, PPO median not worse; against a baseline already at ≥ 0.98 "
                    "success: Δsuccess ≥ −1 pp and significantly fewer steps) AND Δsuccess vs greedy ≥ −5 pp. "
                    f"Holds in {sum(per)}/{len(per)} discriminative scenarios." + ceil_note + "\n\n"
                 + "\n".join(f"- {l}" for l in lines))
    # ---- H2: a vantagem sobre a melhor baseline simples cresce S0 < S1 < S2
    adv = {}
    for scen in ("S0", "S1", "S2"):
        if (scen, "ppo") not in S.index:
            continue
        best = max(SIMPLE, key=lambda m: S.loc[(scen, m)].success if (scen, m) in S.index else -1)
        r = T.loc[(scen, best)]
        adv[scen] = (best, r.success_diff, r.success_diff_lo, r.success_diff_hi)
    lines = [f"- {s}: PPO − best simple baseline ({b}) = {d:+.3f} [{lo:+.3f}, {hi:+.3f}]" for s, (b, d, lo, hi) in adv.items()]
    if len(adv) == 3:
        d = [adv[s][1] for s in ("S0", "S1", "S2")]
        mono = d[0] < d[1] < d[2]
        sep = adv["S2"][2] > adv["S0"][3]  # os ICs de S2 e S0 não se sobrepõem
        v = "supported" if mono and sep else "inconclusive" if (mono or d[2] > d[0]) else "not supported"
    else:
        v = "inconclusive"
    out["H2"] = (v, "Criterion: the advantage increases monotonically S0 < S1 < S2 and the 95% CIs of S0 and S2 do "
                    "not overlap (supported); monotone or S2 > S0 without separation (inconclusive); otherwise not "
                    "supported.\n\n" + "\n".join(lines))
    # ---- H3: generalização norte → sul
    if summ_train is None or tests_train is None or tests_train.empty:
        out["H3"] = ("inconclusive", "No evaluation on the training region was available for comparison.")
        return out
    St = summ_train.set_index(["scenario", "method"])
    Tt = tests_train.set_index(["scenario", "vs"])
    lines, per = [], []
    for scen in sorted(set(summ.scenario) & set(summ_train.scenario)):
        if scen in ceil:
            continue
        if (scen, "ppo") not in St.index or (scen, "greedy") not in St.index:
            continue
        gap_tr = St.loc[(scen, "ppo")].success - St.loc[(scen, "greedy")].success
        gap_te = S.loc[(scen, "ppo")].success - S.loc[(scen, "greedy")].success
        best = max(SIMPLE, key=lambda m: S.loc[(scen, m)].success if (scen, m) in S.index else -1)
        r = T.loc[(scen, best)]
        beats = r.success_diff_lo > 0
        drop = gap_tr - gap_te
        ok = beats and drop <= 0.05
        per.append(ok)
        lines.append(f"{scen}: PPO success north {St.loc[(scen, 'ppo')].success:.3f} → south "
                     f"{S.loc[(scen, 'ppo')].success:.3f}; PPO − greedy: north {gap_tr:+.3f}, south {gap_te:+.3f} "
                     f"(relative drop {drop:+.3f}); south PPO − {best} = {r.success_diff:+.3f} "
                     f"[{r.success_diff_lo:+.3f}, {r.success_diff_hi:+.3f}] → {'generalizes' if ok else 'does not'}")
    v = ("supported" if per and all(per) else "not supported" if per and not any(per) else "inconclusive")
    out["H3"] = (v, "Criterion (per scenario): in the unseen south region PPO still beats the best simple baseline "
                    "(CI > 0) and its gap to the training-free greedy reference shrinks by at most 5 pp relative "
                    "to the north (training) region." + ceil_note + "\n\n" + "\n".join(f"- {l}" for l in lines))
    return out


# ============================================================================ markdown
def md_table(df: pd.DataFrame, cols: list[str], fmt: dict | None = None) -> str:
    fmt = fmt or {}
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = ""
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in fmt:
                v = fmt[c](r)
            elif isinstance(v, float):
                v = "" if np.isnan(v) else f"{v:.3f}"
            cells.append(str(v))
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


def build(cfg: dict, results: Path, run_roots: list[Path], out_md: Path, title_note: str = "") -> str:
    figs = results / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    fc = cfg["filter"]
    ms = int(fc["max_steps"])
    nb = int(cfg["eval"]["bootstrap"])
    rel = lambda p: Path(os.path.relpath(Path(p).resolve(), out_md.parent.resolve())).as_posix()  # noqa: E731
    md = [f"# REPORT: active global localization of a drone with RL on outdated maps\n",
          "*Generated automatically by `python -m dal.report`. Every number below is read from files in "
          f"`{rel(results)}/`.*{title_note}\n"]
    # ---------------- dados
    ds = results / "data_stats.json"
    if ds.exists():
        s = json.loads(ds.read_text())
        md.append("## 1. Data\n")
        rows = [dict(date=k, valid=v["masks"]["aoi_valid_frac"], cloud_raw=v["masks"]["aoi_cloud_raw_frac"],
                     cloud_refined=v["masks"]["aoi_cloud_frac_of_valid"], cells_ok=v["masks"]["cells_ok_frac"],
                     offset_m=v["align"]["norm_m"], dx=v["align"]["dx_m"], dy=v["align"]["dy_m"],
                     applied=v["align"]["applied"]) for k, v in s["dates"].items()]
        md.append(f"AOI grid: {s['grid']['ncols']} × {s['grid']['nrows']} cells of {s['grid']['cell']:.0f} m; "
                  f"train rows {s['train_rows'][0]}–{s['train_rows'][1]} (north), test rows "
                  f"{s['test_rows'][0]}–{s['test_rows'][1]} (south).\n")
        md.append(md_table(pd.DataFrame(rows), ["date", "valid", "cloud_raw", "cloud_refined", "cells_ok", "dx", "dy",
                                                "offset_m", "applied"]))
        if (figs / "data_overview.png").exists():
            md.append(f"\n![data overview]({rel(figs / 'data_overview.png')})\n")
    # ---------------- recuperação
    rp = results / "retrieval.csv"
    if rp.exists():
        r = pd.read_csv(rp)
        md.append("## 2. Descriptor retrieval (train windows 64×64)\n")
        md.append("Fraction of observations whose true cell ranks top-k among the valid map cells of the window. "
                  "`top1_within1`: the best match is within 1 cell. `beta1`: single-observation maximum-likelihood "
                  "temperature; `kappa`: sequence-level tempering chosen on train episodes; `beta` = kappa·beta1 is "
                  "what the filter uses. `loglik_*`: mean log-probability of the true cell (β₁) vs uniform.\n")
        cols = [c for c in ["descriptor", "scenario", "top1", "top5", "top1_within1", "chance_top1", "beta1", "kappa",
                            "beta", "loglik_true", "loglik_uniform"] if c in r.columns]
        md.append(md_table(r, cols))
    # ---------------- gate
    gp = results / "gate.json"
    if gp.exists():
        g = json.loads(gp.read_text())
        md.append("## 3. Sanity gate (train split)\n")
        md.append(f"Gate {'**PASSED**' if g['passed'] else '**FAILED**'}; descriptor chosen: "
                  f"`{g['best_descriptor']}` ({g['episodes']} episodes per cell of the table). Thresholds in S0: "
                  f"greedy ≥ {g['thresholds']['greedy']}, lawnmower ≥ {g['thresholds']['lawnmower']}.\n")
        gc = pd.read_csv(results / "gate.csv")
        piv = gc.pivot_table(index=["descriptor", "scenario"], columns="method", values="success").reset_index()
        md.append(md_table(piv, list(piv.columns)))
        fl = gc.pivot_table(index=["descriptor", "scenario"], columns="method", values="false_loc").reset_index()
        md.append("\nFalse-localization rate:\n\n" + md_table(fl, list(fl.columns)))
    # ---------------- resultados principais
    ep_test = results / "episodes_test.csv"
    verdict = None
    if ep_test.exists():
        df = pd.read_csv(ep_test)
        summ = summary_table(df, nb)
        summ.to_csv(results / "summary_test.csv", index=False)
        tests = paired_tests(df, ms, nb)
        tests.to_csv(results / "paired_tests_test.csv", index=False)
        md.append(f"## 4. Test region (south, never seen in training): {df.episode.nunique()} paired episodes "
                  "per method × scenario\n")
        md.append("PPO rows pool all training seeds (`model_seeds`); `success_seed_std` is the std of the success "
                  "rate across seeds. Steps are over successful episodes only.\n")
        f = {"success": lambda r: f"{r.success:.3f} [{r.success_lo:.3f}, {r.success_hi:.3f}]",
             "false_loc": lambda r: f"{r.false_loc:.3f} [{r.false_loc_lo:.3f}, {r.false_loc_hi:.3f}]",
             "steps (median [IQR])": lambda r: "" if np.isnan(r.steps_median) else
             f"{r.steps_median:.0f} [{r.steps_q1:.0f}–{r.steps_q3:.0f}]",
             "decision_ms": lambda r: f"{r.decision_ms:.2f}"}
        summ2 = summ.assign(**{"steps (median [IQR])": 0})
        md.append(md_table(summ2, ["scenario", "method", "episodes", "model_seeds", "success", "success_seed_std",
                                   "false_loc", "steps (median [IQR])", "final_error_median_m", "distance_mean_m",
                                   "decision_ms"], f))
        if not tests.empty:
            md.append("\n### Paired tests: PPO vs each method\n")
            md.append("`success_diff` = PPO − other (paired bootstrap 95% CI). Wilcoxon signed-rank test on "
                      f"per-episode penalized steps (failure = {ms + 1}).\n")
            md.append(md_table(tests, ["scenario", "vs", "n", "success_diff", "success_diff_lo", "success_diff_hi",
                                       "pen_steps_ref", "pen_steps_other", "wilcoxon_p"],
                               {"wilcoxon_p": lambda r: f"{r.wilcoxon_p:.2g}"}))
        fig_success_curves(df, ms, figs / "success_vs_steps.png")
        fig_bars(summ, figs / "success_bars.png")
        md.append(f"\n![success vs steps]({rel(figs / 'success_vs_steps.png')})\n")
        md.append(f"![bars]({rel(figs / 'success_bars.png')})\n")
        made = fig_trajectories(results / "traj", cfg, "test", figs / "trajectories")
        for m in made:
            md.append(f"![trajectories]({rel(figs / m)})\n")
        summ_tr = tests_tr = None
        ep_train = results / "episodes_train.csv"
        if ep_train.exists():
            dtr = pd.read_csv(ep_train)
            summ_tr = summary_table(dtr, nb)
            tests_tr = paired_tests(dtr, ms, nb)
            summ_tr.to_csv(results / "summary_train.csv", index=False)
            md.append(f"\n## 5. Training region (north), {dtr.episode.nunique()} episodes, used for H3\n")
            md.append(md_table(summ_tr, ["scenario", "method", "episodes", "success", "false_loc",
                                         "steps_median", "decision_ms"]))
        verdict = verdicts(summ, tests, summ_tr, tests_tr)
    # ---------------- curvas de treino
    if fig_training(run_roots, figs / "training_curves.png"):
        md.append("\n## 6. Training curves\n")
        metas = [json.loads((d / "train_meta.json").read_text()) for r in run_roots if r.exists()
                 for d in sorted(r.iterdir()) if (d / "train_meta.json").exists()]
        if metas:
            md.append(md_table(pd.DataFrame(metas), ["name", "timesteps", "seconds", "fps"]))
        md.append(f"\n![training]({rel(figs / 'training_curves.png')})\n")
    # ---------------- hipóteses
    md.append("\n## 7. Hypotheses\n")
    if verdict is None:
        md.append("No test-region evaluation available: H1/H2/H3 **inconclusive**.\n")
    else:
        for h, (v, why) in verdict.items():
            md.append(f"### {h}: **{v}**\n\n{why}\n")
        (results / "hypotheses.json").write_text(json.dumps({k: v[0] for k, v in verdict.items()}, indent=1))
    text = "\n".join(md)
    out_md.write_text(text)
    print(f"[report] wrote {out_md}")
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--results", default="results")
    ap.add_argument("--runs", nargs="*", default=["runs/overnight"])
    ap.add_argument("--out", default="REPORT.md")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    cfg = load_config(a.config)
    build(cfg, ROOT / a.results, [ROOT / r for r in a.runs], ROOT / a.out, a.note)


if __name__ == "__main__":
    main()

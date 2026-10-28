"""Tables, statistics, figures and REPORT_POA.md for the Porto Alegre battery. Every number in
the report is read from files in the POA results folder (Taquari reference rows are first
copied there from the read-only Taquari results)."""
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
from scipy.stats import spearmanr, wilcoxon  # noqa: E402

from .config import ROOT, load_config, paths  # noqa: E402
from .report import boot_ci, md_table  # noqa: E402

SIMPLE = ["straight", "lawnmower", "random"]
ORDER = ["ppo_S1", "ppo_S2", "ppo_poa", "greedy_map", "greedy", "lawnmower", "straight", "random"]
COLORS = {"ppo_S1": "#1f77b4", "ppo_S2": "#17becf", "ppo_poa": "#8c564b", "greedy_map": "#ff7f0e",
          "greedy": "#d62728", "lawnmower": "#2ca02c", "straight": "#9467bd", "random": "#7f7f7f"}
LABEL = {"ppo_S1": "PPO (Taquari S1)", "ppo_S2": "PPO (Taquari S2)", "ppo_poa": "PPO (trained POA ilhas P1)",
         "greedy_map": "greedy IG (map-predicted)", "greedy": "greedy IG (oracle)", "lawnmower": "lawnmower",
         "straight": "straight", "random": "random"}
ACTIVE = ["ppo_S2", "greedy_map"]
SECTIONS = ["## 1. Data", "## 2. Sanity gate", "## 3. Descriptor retrieval and beta", "## 4. Main results",
            "## 5. Comparison with the Taquari", "## 6. Change index", "## 7. Water fraction", "## 8. Hypotheses",
            "## 9. Limitations"]
MAIN_COLS = ["region", "scenario", "variant", "method", "episodes", "model_seeds", "success", "success_seed_std",
             "false_loc", "steps (median [IQR])", "distance_mean_m", "decision_ms"]


def _order(s: pd.Series) -> pd.Series:
    return s.map({m: i for i, m in enumerate(ORDER)}).fillna(99)


# ============================================================================ statistics
def per_episode(df: pd.DataFrame, max_steps: int) -> pd.DataFrame:
    """One row per (region, scenario, variant, method, episode): PPO seeds averaged per episode."""
    d = df.assign(pen=df.steps_to_success.fillna(max_steps + 1), success=df.success.astype(float),
                  false_loc=df.false_loc.astype(float))
    keys = ["region", "scenario", "variant", "method", "episode"]
    cols = ["success", "false_loc", "pen", "distance_m", "change_window", "change_start", "water_diff", "water_obs"]
    return d.groupby(keys)[cols].mean().reset_index()


def summary(df: pd.DataFrame, n_boot: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for (reg, scen, var, meth), g in df.groupby(["region", "scenario", "variant", "method"]):
        s = g.steps_to_success.dropna()
        lo, hi = boot_ci(g.success.astype(float), n_boot, rng)
        flo, fhi = boot_ci(g.false_loc.astype(float), n_boot, rng)
        dlo, dhi = boot_ci(g.distance_m, n_boot, rng)
        per_seed = g.groupby("model_seed").success.mean()
        rows.append(dict(region=reg, scenario=scen, variant=var, method=meth, episodes=len(g),
                         model_seeds=int(g.model_seed.nunique()) if (g.model_seed >= 0).any() else 0,
                         success=g.success.mean(), success_lo=lo, success_hi=hi,
                         success_seed_std=float(per_seed.std(ddof=0)) if len(per_seed) > 1 else np.nan,
                         false_loc=g.false_loc.mean(), false_loc_lo=flo, false_loc_hi=fhi,
                         steps_median=s.median() if len(s) else np.nan, steps_q1=s.quantile(0.25) if len(s) else np.nan,
                         steps_q3=s.quantile(0.75) if len(s) else np.nan, distance_mean_m=g.distance_m.mean(),
                         distance_lo=dlo, distance_hi=dhi, decision_ms=g.decision_ms.mean(),
                         change_window_mean=g.change_window.mean()))
    t = pd.DataFrame(rows)
    return t.assign(o=_order(t.method)).sort_values(["region", "scenario", "variant", "o"]).drop(columns="o")


def paired_tests(df: pd.DataFrame, max_steps: int, n_boot: int, seed: int = 1) -> pd.DataFrame:
    """Every PPO variant vs every other method on the same episodes: paired bootstrap of Δsuccess
    and Δdistance, Wilcoxon signed-rank on penalized steps (failure = max_steps + 1)."""
    rng = np.random.default_rng(seed)
    pe = per_episode(df, max_steps)
    rows = []
    for (reg, scen, var), g in pe.groupby(["region", "scenario", "variant"]):
        meths = sorted(set(g.method), key=lambda m: ORDER.index(m) if m in ORDER else 99)
        for ref in [m for m in meths if m.startswith("ppo")]:
            a = g[g.method == ref].set_index("episode")
            for other in [m for m in meths if m != ref]:
                b = g[g.method == other].set_index("episode")
                idx = a.index.intersection(b.index)
                ds = (a.success - b.success).loc[idx].to_numpy()
                dd = (a.distance_m - b.distance_m).loc[idx].to_numpy()
                lo, hi = boot_ci(ds, n_boot, rng)
                dlo, dhi = boot_ci(dd, n_boot, rng)
                pa, pb = a.pen.loc[idx].to_numpy(), b.pen.loc[idx].to_numpy()
                try:
                    p = float(wilcoxon(pa, pb, zero_method="zsplit").pvalue) if np.any(pa != pb) else 1.0
                except ValueError:
                    p = 1.0
                rows.append(dict(region=reg, scenario=scen, variant=var, ref=ref, vs=other, n=len(idx),
                                 success_diff=float(ds.mean()), success_diff_lo=lo, success_diff_hi=hi,
                                 distance_diff_m=float(dd.mean()), distance_diff_lo=dlo, distance_diff_hi=dhi,
                                 pen_steps_ref=float(np.median(pa)), pen_steps_other=float(np.median(pb)),
                                 wilcoxon_p=p))
    return pd.DataFrame(rows)


def quintile_table(pe: pd.DataFrame, col: str, n_boot: int, seed: int = 2, q: int = 5) -> pd.DataFrame:
    """Success per quantile bin of a window covariate (bins over the pooled episodes, per method)."""
    rng = np.random.default_rng(seed)
    base = pe.drop_duplicates(["region", "scenario", "episode"])[["region", "scenario", "episode", col]].dropna()
    edges = np.unique(np.quantile(base[col], np.linspace(0, 1, q + 1)))
    pe = pe.dropna(subset=[col]).copy()
    pe["bin"] = np.clip(np.searchsorted(edges, pe[col], side="right") - 1, 0, len(edges) - 2)
    rows = []
    for (meth, b), g in pe.groupby(["method", "bin"]):
        lo, hi = boot_ci(g.success, n_boot, rng)
        rows.append(dict(method=meth, bin=int(b) + 1, lo_edge=edges[b], hi_edge=edges[b + 1], n=len(g),
                         covariate_mean=g[col].mean(), success=g.success.mean(), success_lo=lo, success_hi=hi,
                         false_loc=g.false_loc.mean(), share_ilhas=(g.region == "ilhas").mean()))
    t = pd.DataFrame(rows)
    return t.assign(o=_order(t.method)).sort_values(["o", "bin"]).drop(columns="o")


def trend_stats(pe: pd.DataFrame, qt: pd.DataFrame, col: str, n_boot: int, seed: int = 3) -> pd.DataFrame:
    """Per method: Spearman(covariate, success) over episodes and bootstrap CI of success(Q1) - success(Q5)."""
    rng = np.random.default_rng(seed)
    nb = qt.bin.max()
    edges = qt[qt.method == qt.method.iloc[0]][["lo_edge", "hi_edge"]].to_numpy()
    rows = []
    for meth, g in pe.dropna(subset=[col]).groupby("method"):
        b = np.clip(np.searchsorted(edges[:, 0], g[col], side="right"), 1, nb)
        s1, s5 = g.success[b == 1].to_numpy(), g.success[b == nb].to_numpy()
        diffs = [rng.choice(s1, len(s1)).mean() - rng.choice(s5, len(s5)).mean() for _ in range(n_boot)]
        rho, p = spearmanr(g[col], g.success)
        rows.append(dict(method=meth, n=len(g), spearman_rho=float(rho), spearman_p=float(p),
                         q1_minus_q5=float(s1.mean() - s5.mean()), q1_minus_q5_lo=float(np.percentile(diffs, 2.5)),
                         q1_minus_q5_hi=float(np.percentile(diffs, 97.5))))
    t = pd.DataFrame(rows)
    return t.assign(o=_order(t.method)).sort_values("o").drop(columns="o")


def region_contrast(pe: pd.DataFrame, scen: str, variant: str, n_boot: int, seed: int = 4) -> pd.DataFrame:
    """centro - ilhas success per method (independent bootstrap of the two regions), and the
    active-strategy advantage over the best simple baseline in each region and its difference."""
    rng = np.random.default_rng(seed)
    g = pe[(pe.scenario == scen) & (pe.variant == variant)]
    if set(g.region) != {"ilhas", "centro"}:
        return pd.DataFrame()
    piv = {r: g[g.region == r].pivot(index="episode", columns="method", values="success") for r in ("ilhas", "centro")}
    best = {r: max([m for m in SIMPLE if m in piv[r]], key=lambda m: piv[r][m].mean()) for r in piv}
    rows = []
    for meth in [m for m in ORDER if m in piv["ilhas"] and m in piv["centro"]]:
        ci, ce = piv["ilhas"][meth].to_numpy(), piv["centro"][meth].to_numpy()
        bs_d, bs_adv = [], []
        ai = piv["ilhas"][meth] - piv["ilhas"][best["ilhas"]]
        ac = piv["centro"][meth] - piv["centro"][best["centro"]]
        for _ in range(n_boot):
            ii, jj = rng.integers(len(ci), size=len(ci)), rng.integers(len(ce), size=len(ce))
            bs_d.append(ce[jj].mean() - ci[ii].mean())
            bs_adv.append(ai.to_numpy()[ii].mean() - ac.to_numpy()[jj].mean())
        rows.append(dict(method=meth, success_ilhas=ci.mean(), success_centro=ce.mean(),
                         centro_minus_ilhas=ce.mean() - ci.mean(), cmi_lo=np.percentile(bs_d, 2.5),
                         cmi_hi=np.percentile(bs_d, 97.5), best_simple_ilhas=best["ilhas"],
                         best_simple_centro=best["centro"], adv_ilhas=ai.mean(), adv_centro=ac.mean(),
                         adv_diff=ai.mean() - ac.mean(), adv_diff_lo=np.percentile(bs_adv, 2.5),
                         adv_diff_hi=np.percentile(bs_adv, 97.5)))
    return pd.DataFrame(rows)


# ============================================================================ figures
def fig_bars(summ: pd.DataFrame, path: Path) -> None:
    units = summ[["region", "scenario"]].drop_duplicates().values.tolist()
    variants = sorted(summ.variant.unique(), key=lambda v: v != "zeroshot")
    fig, axes = plt.subplots(2, len(units), figsize=(3.6 * len(units), 7), squeeze=False, sharey="row")
    for j, (reg, scen) in enumerate(units):
        for i, (col, lab) in enumerate((("success", "success rate"), ("false_loc", "false-localization rate"))):
            ax = axes[i, j]
            g = summ[(summ.region == reg) & (summ.scenario == scen)]
            meths = [m for m in ORDER if m in set(g.method)]
            w = 0.8 / len(variants)
            for k, var in enumerate(variants):
                gv = g[g.variant == var].set_index("method")
                x = np.arange(len(meths)) + (k - (len(variants) - 1) / 2) * w
                y = [gv[col].get(m, np.nan) for m in meths]
                lo = [gv[f"{col}_lo"].get(m, np.nan) for m in meths]
                hi = [gv[f"{col}_hi"].get(m, np.nan) for m in meths]
                err = np.array([np.subtract(y, lo), np.subtract(hi, y)])
                ax.bar(x, y, w, yerr=err, capsize=2, color=[COLORS.get(m, "k") for m in meths],
                       hatch=None if var == "zeroshot" else "//", edgecolor="k", lw=0.4, alpha=0.95 if var == "zeroshot" else 0.55,
                       label=var)
            ax.set_xticks(range(len(meths)))
            ax.set_xticklabels([LABEL.get(m, m) for m in meths], rotation=60, ha="right", fontsize=7)
            ax.set_ylim(0, 1.05 if col == "success" else max(0.3, float(g[f"{col}_hi"].max()) * 1.15))
            ax.set_title(f"{reg} {scen}", fontsize=9)
            if j == 0:
                ax.set_ylabel(lab)
    axes[0, 0].legend(fontsize=7, title="beta", title_fontsize=7)
    fig.suptitle("Porto Alegre: success and false localization (95% bootstrap CI; solid = zero-shot beta, "
                 "hatched = beta recalibrated on the other region)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_success_steps(df: pd.DataFrame, max_steps: int, path: Path, variant: str = "zeroshot") -> None:
    d = df[df.variant == variant]
    units = d[["region", "scenario"]].drop_duplicates().values.tolist()
    fig, axes = plt.subplots(1, len(units), figsize=(3.6 * len(units), 3.4), squeeze=False, sharey=True)
    t = np.arange(0, max_steps + 1)
    for ax, (reg, scen) in zip(axes[0], units):
        g = d[(d.region == reg) & (d.scenario == scen)]
        for m in [m for m in ORDER if m in set(g.method)]:
            gm = g[g.method == m]
            s = gm.steps_to_success.to_numpy()
            ax.plot(t, [(s <= k).sum() / len(gm) for k in t], color=COLORS.get(m, "k"), label=LABEL.get(m, m), lw=1.3)
        ax.set_title(f"{reg} {scen}", fontsize=9)
        ax.set_xlabel("steps")
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("cumulative success")
    axes[0, -1].legend(fontsize=6, loc="lower right")
    fig.suptitle(f"success vs steps ({variant} beta)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_change(qt: pd.DataFrame, path: Path, xlabel: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for m in [m for m in ORDER if m in set(qt.method)]:
        g = qt[qt.method == m]
        ax.errorbar(g.covariate_mean, g.success, yerr=[g.success - g.success_lo, g.success_hi - g.success],
                    color=COLORS.get(m, "k"), marker="o", ms=4, capsize=2, lw=1.3, label=LABEL.get(m, m))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("success rate")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_trajectories(cfg: dict, traj_dir: Path, scen: str, path: Path, methods=("ppo_S2", "straight"),
                     variant: str = "zeroshot") -> bool:
    """Same episode for each method over the poa_post thumbnail, for both regions."""
    from .poa_data import _extent, load_date, poa_grid, region_slices

    grid = poa_grid(cfg)
    sl = region_slices(cfg, grid)
    try:
        thumb = load_date(cfg, cfg["data"]["obs_date"], thumb=True)
    except Exception:  # noqa: BLE001  (synthetic test datasets have no imagery)
        thumb = None
    W = cfg["filter"]["window"]
    c = grid.cell / 1000
    regions = [r for r in cfg["regions"] if list(traj_dir.glob(f"{r}_{scen}_{variant}_*"))]
    if not regions:
        return False
    fig, axes = plt.subplots(1, len(regions), figsize=(6.2 * len(regions), 6.4), squeeze=False)
    for ax, reg in zip(axes[0], regions):
        tags = {m: (f"{m}0" if m.startswith("ppo") else m) for m in methods}
        eps = sorted({int(p.stem.rsplit("_ep", 1)[1]) for p in traj_dir.glob(f"{reg}_{scen}_{variant}_{tags[methods[0]]}_ep*.npz")})
        eps = [e for e in eps if all((traj_dir / f"{reg}_{scen}_{variant}_{tags[m]}_ep{e}.npz").exists() for m in methods)]
        if not eps:
            ax.axis("off")
            continue
        z0 = {m: np.load(traj_dir / f"{reg}_{scen}_{variant}_{tags[m]}_ep{eps[0]}.npz") for m in methods}

        def score(e):  # prefer an episode where the first method succeeds and the episode is long
            z = {m: np.load(traj_dir / f"{reg}_{scen}_{variant}_{tags[m]}_ep{e}.npz") for m in methods}
            return (bool(z[methods[0]]["success"]), len(z[methods[0]]["traj"]))

        e = max(eps, key=score)
        z0 = {m: np.load(traj_dir / f"{reg}_{scen}_{variant}_{tags[m]}_ep{e}.npz") for m in methods}
        rs, cs = sl[reg]
        r0, c0 = z0[methods[0]]["window"]
        R0, C0 = rs.start + r0, cs.start + c0
        if thumb is not None:
            ax.imshow(thumb.img, extent=_extent(thumb, grid))
        ax.add_patch(plt.Rectangle((C0 * c, R0 * c), W * c, W * c, fill=False, ec="yellow", lw=1.5))
        for m in methods:
            z = z0[m]
            tr = z["traj"]
            ax.plot((C0 + tr[:, 1] + 0.5) * c, (R0 + tr[:, 0] + 0.5) * c, color=COLORS.get(m, "w"), lw=1.6,
                    label=f"{LABEL.get(m, m)}: {'success' if z['success'] else 'false loc.' if z['false_loc'] else 'fail'} "
                          f"({len(tr) - 1} steps)")
            ax.plot((C0 + tr[-1, 1] + 0.5) * c, (R0 + tr[-1, 0] + 0.5) * c, "o", color=COLORS.get(m, "w"), ms=5)
            me = z["map_est"][-1]
            ax.plot((C0 + me[1] + 0.5) * c, (R0 + me[0] + 0.5) * c, "x", color=COLORS.get(m, "w"), ms=8, mew=2)
        ax.plot((C0 + tr[0, 1] + 0.5) * c, (R0 + tr[0, 0] + 0.5) * c, "*", color="white", ms=12, mec="k")
        pad = 0.6
        ax.set_xlim(C0 * c - pad, (C0 + W) * c + pad)
        ax.set_ylim((R0 + W) * c + pad, R0 * c - pad)
        ax.set_title(f"{reg} {scen} episode {e} (star: start, o: end, x: belief MAP)", fontsize=9)
        ax.set_xlabel("km east")
        ax.set_ylabel("km south of box top")
        ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


# ============================================================================ verdicts
def verdict_hp1(tests: pd.DataFrame, tq: pd.DataFrame) -> tuple[str, list[str]]:
    lines = []
    t = tests[(tests.scenario == "P1") & (tests.variant == "zeroshot") & (tests.vs == "straight")]
    ok_by = {}
    for ref in ("ppo_S2", "ppo_S1"):
        for _, r in t[t.ref == ref].iterrows():
            ok = r.success_diff_lo >= -0.05 and r.distance_diff_hi < 0
            if ref == "ppo_S2":
                ok_by[r.region] = ok
            lines.append(f"- {r.region} P1, {LABEL[ref]} − straight: Δsuccess = {r.success_diff:+.3f} "
                         f"[{r.success_diff_lo:+.3f}, {r.success_diff_hi:+.3f}], Δdistance = {r.distance_diff_m:+.0f} m "
                         f"[{r.distance_diff_lo:+.0f}, {r.distance_diff_hi:+.0f}] → "
                         f"{'relation kept' if ok else 'relation not kept'}{'' if ref == 'ppo_S2' else ' (secondary)'}")
    for _, r in tq.iterrows():
        lines.append(f"- Taquari {r.scenario} (reference, test split): PPO − straight Δsuccess = {r.success_diff:+.3f} "
                     f"[{r.success_diff_lo:+.3f}, {r.success_diff_hi:+.3f}], distance PPO {r.distance_ppo_m:.0f} m vs "
                     f"straight {r.distance_straight_m:.0f} m")
    n = sum(ok_by.values())
    v = "inconclusive" if not ok_by else "supported" if n == len(ok_by) else "not supported" if n == 0 else "inconclusive"
    return v, lines


def verdict_hp2(trend: pd.DataFrame, contrast: pd.DataFrame) -> tuple[str, list[str]]:
    lines = []
    dec = []
    for _, r in trend.iterrows():
        ok = r.spearman_rho < 0 and r.q1_minus_q5_lo > 0
        dec.append(ok)
        lines.append(f"- {LABEL.get(r.method, r.method)}: Spearman ρ = {r.spearman_rho:+.3f} (p = {r.spearman_p:.2g}), "
                     f"success Q1 − Q5 = {r.q1_minus_q5:+.3f} [{r.q1_minus_q5_lo:+.3f}, {r.q1_minus_q5_hi:+.3f}] → "
                     f"{'decreasing' if ok else 'not clearly decreasing'}")
    a = len(dec) > 0 and np.mean(dec) >= 0.75
    b = None
    if len(contrast):
        harder = contrast.cmi_lo > 0
        b = bool(harder.mean() >= 0.75)
        for _, r in contrast.iterrows():
            lines.append(f"- {LABEL.get(r.method, r.method)} P1: ilhas {r.success_ilhas:.3f} vs centro {r.success_centro:.3f} "
                         f"(centro − ilhas {r.centro_minus_ilhas:+.3f} [{r.cmi_lo:+.3f}, {r.cmi_hi:+.3f}])")
    parts = [a] + ([b] if b is not None else [])
    v = "supported" if all(parts) and b is not None else "not supported" if not any(parts) else "inconclusive"
    lines.insert(0, f"Part (a) success decreases with the change index for ≥ 75% of methods: {'yes' if a else 'no'}; "
                    f"part (b) ilhas harder than centro (CI) for ≥ 75% of methods: "
                    f"{'n/a' if b is None else 'yes' if b else 'no'}.")
    return v, lines


def verdict_hp3(contrast: pd.DataFrame) -> tuple[str, list[str]]:
    if not len(contrast):
        return "inconclusive", ["- P1 not available in both regions."]
    a = bool((contrast.centro_minus_ilhas > 0).all())
    lines = [f"Part (a) every method better in centro than in ilhas (point estimates): {'yes' if a else 'no'}."]
    act = contrast[contrast.method.isin(ACTIVE)]
    b_ok = []
    for _, r in act.iterrows():
        ok = r.adv_diff_lo > 0
        b_ok.append(ok)
        lines.append(f"- {LABEL.get(r.method, r.method)}: advantage over best simple baseline ilhas {r.adv_ilhas:+.3f} "
                     f"({r.best_simple_ilhas}) vs centro {r.adv_centro:+.3f} ({r.best_simple_centro}); "
                     f"difference {r.adv_diff:+.3f} [{r.adv_diff_lo:+.3f}, {r.adv_diff_hi:+.3f}] → "
                     f"{'larger in ilhas' if ok else 'not larger in ilhas'}")
    b = len(b_ok) > 0 and all(b_ok)
    none = len(act) > 0 and (act.adv_diff <= 0).all()
    v = "supported" if a and b else "not supported" if (none or not a and not any(b_ok)) else "inconclusive"
    return v, lines


# ============================================================================ report
def _fmt_main(summ: pd.DataFrame) -> pd.DataFrame:
    t = summ.copy()
    t["success"] = [f"{s:.3f} [{lo:.3f}, {hi:.3f}]" for s, lo, hi in zip(t.success, t.success_lo, t.success_hi)]
    t["false_loc"] = [f"{s:.3f} [{lo:.3f}, {hi:.3f}]" for s, lo, hi in zip(t.false_loc, t.false_loc_lo, t.false_loc_hi)]
    t["steps (median [IQR])"] = ["–" if np.isnan(m) else f"{m:.0f} [{a:.0f}–{b:.0f}]"
                                 for m, a, b in zip(t.steps_median, t.steps_q1, t.steps_q3)]
    return t


def taquari_reference(cfg: dict, out: Path) -> pd.DataFrame:
    """Copy (read-only source) the Taquari PPO − straight rows used for the comparison into results_poa."""
    root = ROOT / cfg["paths"]["taquari_results"]
    tests, summ = root / "paired_tests_test.csv", root / "summary_test.csv"
    if not (tests.exists() and summ.exists()):
        return pd.DataFrame()
    t = pd.read_csv(tests)
    s = pd.read_csv(summ).set_index(["scenario", "method"])
    t = t[(t.vs == "straight") & t.scenario.isin(["S1", "S2"])]
    rows = [dict(scenario=r.scenario, success_diff=r.success_diff, success_diff_lo=r.success_diff_lo,
                 success_diff_hi=r.success_diff_hi, success_ppo=s.loc[(r.scenario, "ppo"), "success"],
                 success_straight=s.loc[(r.scenario, "straight"), "success"],
                 distance_ppo_m=s.loc[(r.scenario, "ppo"), "distance_mean_m"],
                 distance_straight_m=s.loc[(r.scenario, "straight"), "distance_mean_m"]) for r in t.itertuples()]
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    return df


def _json(p: Path) -> dict | None:
    return json.loads(p.read_text()) if p.exists() else None


def build(cfg: dict, results: Path, out_md: Path, note: str = "") -> str:
    figs = results / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    n_boot = int(cfg["eval"]["bootstrap"])
    max_steps = int(cfg["filter"]["max_steps"])
    rel = lambda p: os.path.relpath(p, out_md.parent)  # noqa: E731
    md = [f"# REPORT_POA: zero-shot transfer of the Taquari localization policies to Porto Alegre (May 2024 flood)\n",
          "*Generated automatically by `python -m dal.poa_report`. Every number below is read from files in "
          f"`{rel(results)}/`.*{note}\n"]

    # ---------------------------------------------------------------- 1. data
    md.append(SECTIONS[0] + "\n")
    gate = _json(results / "data_gate.json")
    if gate:
        g = gate["grid"]
        md.append(f"POA box grid: {g['ncols']} × {g['nrows']} cells of {g['cell']:.0f} m. Data gate "
                  f"**{'PASSED' if gate['passed'] else 'FAILED'}** (hand-measured facts ±30 %, alignment ≤ "
                  f"{cfg['data']['align_max_m']} m).\n")
        md.append(md_table(pd.DataFrame(gate["rows"]), ["check", "expected", "measured", "rel_diff", "ok"]))
        ms = pd.DataFrame(gate["masks"]).T.reset_index(names="date")
        al = pd.DataFrame(gate["align"]).T.reset_index(names="date")
        ms = ms.merge(al[["date", "dx_m", "dy_m", "norm_m", "applied"]], on="date")
        md.append("\n" + md_table(ms, ["date", "valid_km2", "cloud_frac_of_valid", "cloud_poc_detector_frac",
                                       "water_frac_of_valid", "cells_ok_frac", "dx_m", "dy_m", "norm_m", "applied"]))
        rg = gate["regions"]
        md.append(f"\nRegions: " + ", ".join(f"`{k}` {v:.1f} km² (rows {rg['slices'][k]['rows']}, cols "
                                             f"{rg['slices'][k]['cols']})" for k, v in rg["area_km2"].items())
                  + f"; overlap {rg['overlap_km2']:.1f} km² ({rg['overlap_cells']} cells).\n")
    win = _json(results / "windows.json")
    if win:
        md.append("Valid 64×64 windows (≥ 70 % usable cells) per region × scenario: "
                  + ", ".join(f"{k}: {v}" for k, v in win["n_windows"].items())
                  + f". Evaluated (≥ {win['min_windows']} windows): {', '.join(win['evaluated'])}.\n")
    for f in ("data_overview.png", "maskcheck_post_nw_islands.png", "maskcheck_post_guaiba_centro.png",
              "maskcheck_feb24_islands.png"):
        if (figs / f).exists():
            md.append(f"![{f}]({rel(figs / f)})\n")

    # ---------------------------------------------------------------- 2. sanity gate
    md.append(SECTIONS[1] + "\n")
    sg = _json(results / "sanity_gate.json")
    if sg:
        md.append(f"P0 (map = observation date), {sg['episodes']} episodes per cell; straight must reach ≥ "
                  f"{sg['threshold']:.2f}: **{'PASSED' if sg['passed'] else 'FAILED'}**.\n")
        t = pd.DataFrame(sg["success"]).T.reset_index(names="region")
        md.append(md_table(t, list(t.columns)))
    else:
        md.append("Not run.\n")

    # ---------------------------------------------------------------- 3. retrieval / beta
    md.append("\n" + SECTIONS[2] + "\n")
    if (results / "retrieval.csv").exists():
        rt = pd.read_csv(results / "retrieval.csv")
        md.append("Top-k retrieval of the true cell inside 64×64 windows (diagnostic, nothing tuned on it).\n")
        md.append(md_table(rt, ["region", "scenario", "top1", "top5", "top1_within1", "loglik_uniform", "n_windows"]))
    if (results / "beta.csv").exists():
        bt = pd.read_csv(results / "beta.csv")
        md.append("\nβ used by the filter: `zeroshot` = Taquari β (with κ) of the mapped Taquari scenario; `recal` = "
                  "β₁ and κ re-estimated on the *other* POA region (light adaptation, no PPO retraining).\n")
        md.append(md_table(bt, ["region", "scenario", "variant", "source", "beta1", "kappa", "beta"]))

    # ---------------------------------------------------------------- 4. main results
    md.append("\n" + SECTIONS[3] + "\n")
    df = pd.read_csv(results / "episodes.csv")
    summ = summary(df, n_boot)
    summ.to_csv(results / "summary.csv", index=False)
    tests = paired_tests(df, max_steps, n_boot)
    tests.to_csv(results / "paired_tests.csv", index=False)
    md.append("Paired episodes: every method sees the same windows, starts and dynamics noise. PPO rows pool the "
              "3 training seeds (`model_seeds`); `success_seed_std` = std of the success rate across seeds. Steps "
              "over successful episodes only.\n")
    md.append(md_table(_fmt_main(summ), MAIN_COLS))
    md.append("\n### Paired tests (PPO vs each method)\n\n`success_diff`, `distance_diff_m` = PPO − other (paired "
              "bootstrap 95 % CI); Wilcoxon signed-rank on per-episode penalized steps (failure = "
              f"{max_steps + 1}).\n")
    md.append(md_table(tests, ["region", "scenario", "variant", "ref", "vs", "n", "success_diff", "success_diff_lo",
                               "success_diff_hi", "distance_diff_m", "distance_diff_lo", "distance_diff_hi",
                               "wilcoxon_p"]))
    fig_bars(summ, figs / "success_bars.png")
    fig_success_steps(df, max_steps, figs / "success_vs_steps.png")
    md.append(f"\n![bars]({rel(figs / 'success_bars.png')})\n\n![steps]({rel(figs / 'success_vs_steps.png')})\n")
    for scen in ("P1", "P2", "P0"):
        if fig_trajectories(cfg, results / "traj", scen, figs / f"trajectories_{scen}.png"):
            md.append(f"![trajectories {scen}]({rel(figs / f'trajectories_{scen}.png')})\n")

    # ---------------------------------------------------------------- 5. Taquari comparison
    md.append(SECTIONS[4] + "\n")
    tq = taquari_reference(cfg, results / "taquari_reference.csv")
    cmp_rows = []
    for _, r in tq.iterrows():
        cmp_rows.append(dict(site="Taquari (test)", scenario=r.scenario, ref="ppo", variant="-", success_diff=r.success_diff,
                             lo=r.success_diff_lo, hi=r.success_diff_hi,
                             distance_diff_m=r.distance_ppo_m - r.distance_straight_m))
    for _, r in tests[(tests.vs == "straight") & (tests.scenario == "P1")].iterrows():
        cmp_rows.append(dict(site=f"POA {r.region}", scenario="P1", ref=r.ref, variant=r.variant, success_diff=r.success_diff,
                             lo=r.success_diff_lo, hi=r.success_diff_hi, distance_diff_m=r.distance_diff_m))
    cmp = pd.DataFrame(cmp_rows)
    if len(cmp):
        cmp.to_csv(results / "comparison_taquari.csv", index=False)
        md.append("PPO − straight (Δsuccess with 95 % CI; Δdistance = mean distance PPO − straight). Taquari rows copied "
                  "from the PoC results into `taquari_reference.csv`.\n")
        md.append(md_table(cmp, ["site", "scenario", "ref", "variant", "success_diff", "lo", "hi", "distance_diff_m"]))

    # ---------------------------------------------------------------- 6. change index
    md.append("\n" + SECTIONS[5] + "\n")
    pe = per_episode(df, max_steps)
    pe.to_csv(results / "per_episode.csv", index=False)
    ch = pe[(pe.scenario != "P0") & (pe.variant == "zeroshot")]
    trend = pd.DataFrame()
    if len(ch) and ch.change_window.notna().any():
        qt = quintile_table(ch, "change_window", n_boot)
        qt.to_csv(results / "change_quintiles.csv", index=False)
        trend = trend_stats(ch, qt, "change_window", n_boot)
        trend.to_csv(results / "change_trend.csv", index=False)
        md.append("Change index per cell = 1 − cos(map descriptor, mean observation descriptor); per episode, its mean "
                  "over the window's usable cells. Quintiles over all non-control episodes (zero-shot β, "
                  f"scenarios {', '.join(sorted(ch.scenario.unique()))}, both regions pooled; `share_ilhas` = "
                  "fraction of the bin's episodes in ilhas).\n")
        md.append(md_table(qt, ["method", "bin", "lo_edge", "hi_edge", "n", "share_ilhas", "success", "success_lo",
                                "success_hi", "false_loc"]))
        md.append("\n" + md_table(trend, ["method", "n", "spearman_rho", "spearman_p", "q1_minus_q5", "q1_minus_q5_lo",
                                          "q1_minus_q5_hi"]))
        fig_change(qt, figs / "success_vs_change.png", "mean change index of the window (quintile mean)",
                   "success vs local map change (P1/P2, zero-shot)")
        md.append(f"\n![change]({rel(figs / 'success_vs_change.png')})\n")
        for reg in sorted(ch.region.unique()):
            g = ch[ch.region == reg]
            if g.change_window.nunique() > 5:
                q = quintile_table(g, "change_window", n_boot)
                q.to_csv(results / f"change_quintiles_{reg}.csv", index=False)
                md.append(f"\nWithin `{reg}` only (quintiles of that region):\n\n" +
                          md_table(q, ["method", "bin", "lo_edge", "hi_edge", "n", "success", "success_lo", "success_hi"]))

    # ---------------------------------------------------------------- 7. water
    md.append("\n" + SECTIONS[6] + "\n")
    w = pe[(pe.scenario == "P1") & (pe.variant == "zeroshot")]
    if len(w) and w.water_diff.nunique() > 5:
        wq = quintile_table(w, "water_diff", n_boot)
        wq.to_csv(results / "water_quintiles.csv", index=False)
        wt = trend_stats(w, wq, "water_diff", n_boot)
        wt.to_csv(results / "water_trend.csv", index=False)
        cc = w.drop_duplicates(["region", "episode"])
        rho = spearmanr(cc.water_diff, cc.change_window, nan_policy="omit")
        md.append("Water fraction per cell from a colour + texture classifier (smooth and dark or brown/reddish "
                  "pixels); covariate = window mean water fraction in `poa_post` − in the map date (newly flooded "
                  f"share). P1, zero-shot. Spearman(water_diff, change index) over windows = {rho.statistic:+.3f}.\n")
        md.append(md_table(wq, ["method", "bin", "lo_edge", "hi_edge", "n", "share_ilhas", "success", "success_lo", "success_hi"]))
        md.append("\n" + md_table(wt, ["method", "n", "spearman_rho", "spearman_p", "q1_minus_q5", "q1_minus_q5_lo",
                                       "q1_minus_q5_hi"]))
        fig_change(wq, figs / "success_vs_water.png", "newly flooded fraction of the window (quintile mean)",
                   "success vs new water (P1, zero-shot)")
        md.append(f"\n![water]({rel(figs / 'success_vs_water.png')})\n")
    else:
        md.append("Not available.\n")

    # ---------------------------------------------------------------- 8. hypotheses
    md.append("\n" + SECTIONS[7] + "\n")
    contrast = region_contrast(pe, "P1", "zeroshot", n_boot)
    if len(contrast):
        contrast.to_csv(results / "region_contrast_P1.csv", index=False)
    v1, l1 = verdict_hp1(tests, tq)
    v2, l2 = verdict_hp2(trend, contrast) if len(trend) else ("inconclusive", ["- no change-index data"])
    v3, l3 = verdict_hp3(contrast)
    md.append(f"### HP1 (transfer): **{v1}**\n\nCriterion: in P1 with the zero-shot β, PPO (Taquari S2) keeps the "
              "Taquari S2 relation to the straight baseline: Δsuccess CI lower bound ≥ −5 pp (≈ tie or better) AND "
              "significantly less distance (paired CI of Δdistance < 0). Supported if it holds in both regions, not "
              "supported if in neither.\n\n" + "\n".join(l1) + "\n")
    md.append(f"### HP2 (difficulty grows with change): **{v2}**\n\nCriterion: (a) for ≥ 75 % of methods, success "
              "falls with the window change index (Spearman ρ < 0 and success(Q1) − success(Q5) CI > 0); (b) in P1 "
              "ilhas are harder than centro (centro − ilhas CI > 0) for ≥ 75 % of methods. Supported if both, not "
              "supported if neither.\n\n" + "\n".join(l2) + "\n")
    md.append(f"### HP3 (structures / active advantage): **{v3}**\n\nCriterion: (a) in P1 every method succeeds more "
              "in centro than in ilhas, and (b) the advantage of the active strategies (PPO S2 and greedy_map) over "
              "the best simple baseline of each region is larger in ilhas (bootstrap CI of the difference > 0) for "
              "both. Supported if (a) and (b); not supported if the difference is ≤ 0 for both active methods or "
              "(a) and (b) both fail; otherwise inconclusive.\n\n" + "\n".join(l3) + "\n")
    (results / "hypotheses.json").write_text(json.dumps(dict(HP1=v1, HP2=v2, HP3=v3), indent=1))

    # ---------------------------------------------------------------- 9. limitations
    md.append("\n" + SECTIONS[8] + "\n")
    md.append(LIMITATIONS)
    val = results / "validation.md"
    if val.exists():
        md.append("\n## 10. Validation\n\n" + val.read_text())
    text = "\n".join(md)
    out_md.write_text(text)
    return text


LIMITATIONS = """- **Viewing angle in the Centro**: high-rise buildings lean differently in the two dates (off-nadir
  acquisitions), so façades and shadows move by several cells' worth of pixels; this is a real appearance change
  unrelated to the flood and it confounds the change index in the Centro.
- **Clouds**: `poa_post` has cumulus over the north-west islands and scattered clouds over the Centro. The
  POA cloud detector (bright/unsaturated + smooth haze, opened and dilated) misses the thinnest wisps and does
  not mask cloud *shadows*, which look like dark water.
- **Water as a uniform texture**: flooded areas are featureless; observations over open water carry almost no
  position information and the descriptor of a flooded cell resembles every other water cell. The water
  classifier is a colour/texture heuristic (validated only visually) and confuses dark vegetation and cloud
  shadows with water; grey water in `poa_pre_dec22` is mostly missed.
- **Small area**: the two regions overlap (see §1) and contain few independent 64×64 windows, so episodes
  share terrain and the CIs (over episodes) understate the uncertainty about the *site*.
- **Zero-shot inputs**: the policy's PCA channels use the Taquari PCA applied to POA descriptors standardized
  with POA statistics (the descriptor standardization is an unsupervised per-date normalization of the
  observation model); β comes from the Taquari or from the other POA region.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs_poa/poa.yaml")
    ap.add_argument("--results")
    ap.add_argument("--out")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    cfg = load_config(a.config)
    P = paths(cfg)
    res = Path(a.results) if a.results else P["results"]
    out = Path(a.out) if a.out else ROOT / cfg["paths"]["report"]
    build(cfg, res, out, a.note)
    print(f"[poa-report] wrote {out}")


if __name__ == "__main__":
    main()

"""Estatística hierárquica (seeds → janelas, B = 2000) do Plano 2 (Canoas UAV Real — CBPMESP)
e geração automática da tabela LaTeX `paper/tables/tab_v4_uav.tex`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dal.evaluation.statistics import hierarchical_bootstrap, percentile_ci
from dal.v3.evaluation.verdicts import N_BOOT, paired_difference

IN_CSV = ROOT / "results_v4" / "episodes_canoas_uav.csv"
OUT_JSON = ROOT / "results_v4" / "canoas_uav_stats.json"
OUT_TEX = ROOT / "paper" / "tables" / "tab_v4_uav.tex"


def method_key(df: pd.DataFrame) -> pd.DataFrame:
    mode = df["mode"].fillna("—").astype(str)
    key = np.where(mode.isin(["—", "", "nan"]), df["method"], df["method"] + "@" + mode)
    return df.assign(key=key, method=key, window_id=df["window_r0"].astype(str) + "_" + df["window_c0"].astype(str))


def ci_for_metric(sub: pd.DataFrame, col: str) -> tuple[float, float, float]:
    """Estimativa pontual e IC 95% por bootstrap hierárquico (seeds → janelas)."""
    seeds = sorted(sub["model_seed"].unique())
    windows = sorted(sub["window_id"].unique())
    win_to_i = {w: i for i, w in enumerate(windows)}
    # Matriz [n_seeds, n_windows] com média da métrica em cada (seed, janela)
    mat = np.full((len(seeds), len(windows)), np.nan, dtype=np.float64)
    counts = np.zeros((len(seeds), len(windows)), dtype=np.float64)
    for si, s in enumerate(seeds):
        part = sub[sub["model_seed"] == s]
        grp = part.groupby("window_id")[col].agg(["mean", "count"])
        for w, row in grp.iterrows():
            mat[si, win_to_i[w]] = float(row["mean"])
            counts[si, win_to_i[w]] = float(row["count"])
    est = float(sub[col].mean())
    rng = np.random.default_rng(20261127)
    n_s, n_w = mat.shape
    boots = np.empty(N_BOOT, dtype=np.float64)
    for b in range(N_BOOT):
        w_idx = rng.integers(0, n_w, size=n_w)
        s_idx = rng.integers(0, n_s, size=n_s)
        sub_m = mat[np.ix_(s_idx, w_idx)]
        sub_c = counts[np.ix_(s_idx, w_idx)]
        tot_c = sub_c.sum()
        boots[b] = float(np.nansum(sub_m * sub_c) / max(tot_c, 1.0))
    lo, hi = percentile_ci(boots)
    return est, lo, hi


def main() -> None:
    df = method_key(pd.read_csv(IN_CSV))
    n_win = int(df["window_id"].nunique())
    n_ep = int(df.groupby("method")["episode"].nunique().min())
    print(f"Loaded {len(df)} rows | {n_win} unique 40x40 windows | {n_ep} paired episodes/method")

    order = [
        ("straight", "Straight (100\\,m)"),
        ("lawnmower", "Lawnmower"),
        ("straight_150", "Straight 150\\,m"),
        ("random", "Random"),
        ("ppo@sampled", "PPO (sampled, 10 seeds)"),
        ("ppo@argmax", "PPO (argmax, 10 seeds)"),
        ("greedy_map", "Greedy (map)"),
    ]

    rows_stats = []
    tex_lines = [
        "% Gerado por scripts/v4_p2_stats.py a partir de results_v4/episodes_canoas_uav.csv. NÃO EDITAR.",
        "\\begin{tabular}{@{}lrrrr@{}}",
        "\\toprule",
        "Method & Succ. (\\%) [95\\% CI] & $\\Delta$ vs.\\ straight (p.p.) & False loc. (\\%) & Time succ. (s) \\\\",
        "\\midrule",
    ]

    for key, label in order:
        sub = df[df["method"] == key]
        est, lo, hi = ci_for_metric(sub, "success")
        fl = float(sub["false_loc"].mean() * 100.0)
        ts = float(sub.loc[sub["success"], "flight_time_s"].mean()) if sub["success"].any() else float("nan")
        if key == "straight":
            diff_str = "---"
            diff_dict = None
        else:
            d = paired_difference(df, key, "straight", "success", seed=20261127)
            diff_dict = {
                "estimate_pp": round(d.estimate * 100, 1),
                "low_pp": round(d.low * 100, 1),
                "high_pp": round(d.high * 100, 1),
            }
            diff_str = f"{d.estimate*100:+.1f} [{d.low*100:+.1f}; {d.high*100:+.1f}]"

        d_rand = None
        if key != "random":
            dr = paired_difference(df, key, "random", "success", seed=20261128)
            d_rand = {
                "estimate_pp": round(dr.estimate * 100, 1),
                "low_pp": round(dr.low * 100, 1),
                "high_pp": round(dr.high * 100, 1),
            }

        row_info = {
            "key": key,
            "success_pct": round(est * 100, 1),
            "ci_low_pct": round(lo * 100, 1),
            "ci_high_pct": round(hi * 100, 1),
            "false_loc_pct": round(fl, 2),
            "time_succ_s": round(ts, 1),
            "delta_vs_straight_pp": diff_dict,
            "delta_vs_random_pp": d_rand,
        }
        rows_stats.append(row_info)
        tex_lines.append(
            f"{label} & {est*100:.1f} [{lo*100:.1f}; {hi*100:.1f}] & {diff_str} & {fl:.2f} & {ts:.0f} \\\\"
        )

    tex_lines += ["\\bottomrule", "\\end{tabular}"]
    OUT_TEX.write_text("\n".join(tex_lines) + "\n")
    out_obj = {"n_windows": n_win, "n_episodes": n_ep, "methods": rows_stats}
    OUT_JSON.write_text(json.dumps(out_obj, indent=2))
    print("\n=== CANOAS REAL UAV HIERARCHICAL STATS ===")
    for r in rows_stats:
        print(f"  {r['key']:14s}: {r['success_pct']:5.1f}% [{r['ci_low_pct']:4.1f}; {r['ci_high_pct']:4.1f}] | vs straight: {r['delta_vs_straight_pp']} | false={r['false_loc_pct']:.2f}% | t={r['time_succ_s']}s")
    print(f"\nWrote {OUT_TEX} and {OUT_JSON}")


if __name__ == "__main__":
    main()

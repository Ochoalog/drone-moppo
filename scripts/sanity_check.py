"""Sanity gate (spec §6, phase 3): on TRAIN windows of S0 the greedy info-gain policy must reach
>= 60 % success within 80 steps and the lawnmower >= 40 %, with the best descriptor.
S1/S2 are run for information. Writes results/gate.csv and results/gate.json.

The best descriptor is the one that passes the gate with the highest mean greedy success over
S0-S2 (train split only)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from dal.config import RESULTS, load_config
from dal.evaluate import evaluate

GATE = {"greedy": 0.60, "lawnmower": 0.40}


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["descriptor", "scenario", "method"])
              .agg(episodes=("success", "size"), success=("success", "mean"), false_loc=("false_loc", "mean"),
                   median_steps_success=("steps_to_success", "median"), decision_ms=("decision_ms", "mean"))
              .reset_index())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--descriptors", nargs="*", default=["hand", "cnn"])
    ap.add_argument("--scenarios", nargs="*", default=["S0", "S1", "S2"])
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=999, help="episode seeds (distinct from the test seeds)")
    ap.add_argument("--report-only", action="store_true", help="only re-read existing episode CSVs")
    ap.add_argument("--tag", default="", help="suffix for the attempt (DECISIONS.md log)")
    a = ap.parse_args()
    gdir = RESULTS / "gate"
    frames = []
    for desc in a.descriptors:
        out = gdir / f"episodes_{desc}{a.tag}.csv"
        if not a.report_only:
            cfg = load_config(a.config, {"env": {"descriptor": desc}, "eval": {"seed": a.seed}})
            evaluate(cfg, a.scenarios, ["greedy", "lawnmower", "straight", "random"], None, a.episodes, "train",
                     out, a.workers, save_traj=2)
        if out.exists():
            frames.append(pd.read_csv(out))
    if not frames:
        print("no gate results found")
        return 1
    s = summarize(pd.concat(frames))
    s.to_csv(RESULTS / f"gate{a.tag}.csv", index=False)
    piv = s.pivot_table(index=["descriptor", "scenario"], columns="method", values="success")
    print(piv.round(3).to_string())
    verdict = {}
    for desc in piv.index.get_level_values(0).unique():
        s0 = piv.loc[(desc, "S0")]
        ok = all(s0[m] >= thr for m, thr in GATE.items())
        verdict[desc] = dict(passed=bool(ok), s0={m: float(s0[m]) for m in GATE},
                             mean_greedy=float(piv.loc[desc]["greedy"].mean()))
    passing = [d for d, v in verdict.items() if v["passed"]]
    best = max(passing, key=lambda d: verdict[d]["mean_greedy"]) if passing else None
    res = dict(passed=best is not None, best_descriptor=best, thresholds=GATE, per_descriptor=verdict,
               episodes=a.episodes, split="train")
    (RESULTS / f"gate{a.tag}.json").write_text(json.dumps(res, indent=1))
    print(f"GATE {'PASSED' if best else 'FAILED'}; best descriptor: {best}")
    for d, v in verdict.items():
        print(f"  {d}: S0 greedy {v['s0']['greedy']:.3f} (>= {GATE['greedy']}), lawnmower "
              f"{v['s0']['lawnmower']:.3f} (>= {GATE['lawnmower']}) -> {'pass' if v['passed'] else 'FAIL'}; "
              f"mean greedy S0-S2 {v['mean_greedy']:.3f}")
    return 0 if best else 2


if __name__ == "__main__":
    sys.exit(main())

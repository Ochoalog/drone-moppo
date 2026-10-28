"""Retrieval quality of the descriptors and calibration of the likelihood temperature beta,
both on TRAIN windows only."""
from __future__ import annotations

import argparse
import copy
import json

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from .config import DATA, RESULTS, load_config
from .env import ScenarioData

BETAS = np.round(np.geomspace(1, 300, 40), 3)


def window_stats(data: ScenarioData, W: int, n_windows: int, seed: int, min_valid: float) -> dict:
    """Top-k retrieval (exact cell and within 1 cell) and log-lik of the true cell per beta."""
    rng = np.random.default_rng(seed)
    wins = data.windows("train", W, min_valid)
    sel = wins[rng.choice(len(wins), size=min(n_windows, len(wins)), replace=False)]
    hits = {k: [] for k in (1, 5)}
    near1, ll, chance = [], [], []
    for r0, c0 in sel:
        sl = (slice(r0, r0 + W), slice(c0, c0 + W))
        mok = np.asarray(data.map_ok[sl]).ravel()
        both = mok & np.asarray(data.obs_ok[sl]).ravel()
        M = np.asarray(data.map_desc[sl], np.float32).reshape(W * W, -1)[mok]
        midx = np.flatnonzero(mok)  # map-cell index -> window cell index
        pos = {c: i for i, c in enumerate(midx)}
        q = np.flatnonzero(both)
        rr, cc = q // W, q % W
        for k in range(data.obs_desc.shape[0]):
            Z = np.asarray(data.obs_desc[k, sl[0], sl[1]], np.float32).reshape(W * W, -1)[q]
            S = Z @ M.T  # (nq, nmap)
            true = np.array([pos[c] for c in q])
            s_true = S[np.arange(len(q)), true]
            rank = (S > s_true[:, None]).sum(1)
            for kk in hits:
                hits[kk].append(rank < kk)
            best = midx[S.argmax(1)]
            near1.append(np.maximum(np.abs(best // W - rr), np.abs(best % W - cc)) <= 1)
            ll.append(np.stack([b * s_true - logsumexp(b * S, axis=1) for b in BETAS], 1))
            chance.append(np.full(len(q), 1.0 / len(midx)))
    ll = np.concatenate(ll).mean(0)
    i = int(np.argmax(ll))
    return dict(top1=float(np.concatenate(hits[1]).mean()), top5=float(np.concatenate(hits[5]).mean()),
                top1_within1=float(np.concatenate(near1).mean()), chance_top1=float(np.concatenate(chance).mean()),
                beta=float(BETAS[i]), loglik_true=float(ll[i]), loglik_uniform=float(np.log(np.concatenate(chance).mean())),
                n_windows=len(sel))


def run(cfg: dict, descriptors: list[str], n_windows: int = 30) -> pd.DataFrame:
    c = copy.deepcopy(cfg)
    c["filter"]["beta"] = 1.0  # placeholder while calibrating
    rows, betas = [], {}
    path = DATA / "beta.json"
    if path.exists():
        betas = json.loads(path.read_text())
    for desc in descriptors:
        for scen in cfg["scenarios"]:
            data = ScenarioData.load(desc, scen, c)
            st = window_stats(data, cfg["filter"]["window"], n_windows, 0, cfg["filter"]["min_window_valid"])
            rows.append(dict(descriptor=desc, scenario=scen, **st))
            betas[f"{desc}_{scen}"] = dict(beta=st["beta"], loglik_true=st["loglik_true"])
            print(f"[calib] {desc} {scen}: top1 {st['top1']:.3f} top5 {st['top5']:.3f} "
                  f"top1(<=1 cell) {st['top1_within1']:.3f} chance {st['chance_top1']:.4f} beta {st['beta']} "
                  f"ll {st['loglik_true']:.2f} (uniform {st['loglik_uniform']:.2f})", flush=True)
    path.write_text(json.dumps(betas, indent=1))
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "retrieval.csv"
    if out.exists():
        old = pd.read_csv(out)
        old = old[~old.set_index(["descriptor", "scenario"]).index.isin(df.set_index(["descriptor", "scenario"]).index)]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(out, index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--descriptors", nargs="*", default=["hand", "cnn"])
    ap.add_argument("--windows", type=int, default=30)
    a = ap.parse_args()
    run(load_config(a.config), a.descriptors, a.windows)


if __name__ == "__main__":
    main()

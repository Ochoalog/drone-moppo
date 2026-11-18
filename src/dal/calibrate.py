"""Qualidade de recuperação dos descritores e calibração da temperatura beta da verossimilhança,
ambas só em janelas de TREINO."""
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


def window_stats(data: ScenarioData, W: int, n_windows: int, seed: int, min_valid: float,
                 n_beta_q: int = 200) -> dict:
    """Recuperação top-k (célula exata e a até 1 célula) e log-verossimilhança da célula real para cada beta."""
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
        midx = np.flatnonzero(mok)  # índice da célula no mapa -> índice da célula na janela
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
            sub = rng.choice(len(q), size=min(n_beta_q, len(q)), replace=False)  # beta: subamostra as consultas
            Ss, st = S[sub], s_true[sub]
            ll.append(np.stack([b * st - logsumexp(b * Ss, axis=1) for b in BETAS], 1))
            chance.append(np.full(len(q), 1.0 / len(midx)))
    ll = np.concatenate(ll).mean(0)
    i = int(np.argmax(ll))
    return dict(top1=float(np.concatenate(hits[1]).mean()), top5=float(np.concatenate(hits[5]).mean()),
                top1_within1=float(np.concatenate(near1).mean()), chance_top1=float(np.concatenate(chance).mean()),
                beta=float(BETAS[i]), loglik_true=float(ll[i]), loglik_uniform=float(np.log(np.concatenate(chance).mean())),
                n_windows=len(sel))


KAPPAS = (1.0, 0.7, 0.5, 0.35, 0.25, 0.18, 0.125, 0.09, 0.06, 0.04)


def _kappa_job(task) -> dict:
    from .baselines import RandomPolicy
    from .env import Episode

    data, fcfg, kappa, seeds = task
    data = copy.copy(data)
    data.beta = data.beta * kappa
    ep = Episode(data, fcfg, "train")
    pol = RandomPolicy()
    n_conc = n_false = n_succ = 0
    steps = []
    for s in seeds:
        ep.reset(seed=int(s))
        pol.reset(ep, np.random.default_rng([int(s), 3]))
        while not ep.done:
            ep.step(pol.act(ep))
        n_conc += ep.last.concentrated
        n_false += ep.last.false_loc
        n_succ += ep.last.success
        steps.append(ep.t)
    return dict(kappa=kappa, n=len(seeds), concentrated=n_conc, false_loc=n_false, success=n_succ,
                mean_steps=float(np.mean(steps)))


def calibrate_kappa(data: ScenarioData, fcfg: dict, episodes: int, workers: int) -> tuple[float, list[dict]]:
    """Temperamento por sequência: escolhe kappa (beta = kappa * beta_1) que maximiza sucesso - localização falsa
    (a utilidade terminal da recompensa) de uma política aleatória em janelas de TREINO; empate -> maior kappa.
    O beta_1 de um passo ignora que observações sucessivas têm erros correlacionados, o que deixa
    o filtro confiante demais (muitas localizações falsas precoces)."""
    from multiprocessing import get_context

    seeds = 50_000 + np.arange(episodes)
    tasks = [(data, fcfg, k, seeds) for k in KAPPAS]
    with get_context("fork").Pool(min(workers, len(tasks))) as pool:
        rows = pool.map(_kappa_job, tasks)
    for r in rows:
        r["false_given_conc"] = r["false_loc"] / max(r["concentrated"], 1)
    for r in rows:
        r["utility"] = (r["success"] - r["false_loc"]) / r["n"]
    best = max(r["utility"] for r in rows)
    kappa = max(r["kappa"] for r in rows if r["utility"] >= best - 1e-9)
    return kappa, rows


def run(cfg: dict, descriptors: list[str], n_windows: int = 30, kappa_only: bool = False) -> pd.DataFrame:
    c = copy.deepcopy(cfg)
    c["filter"]["beta"] = 1.0  # valor provisório durante a calibração
    rows, betas = [], {}
    path = DATA / "beta.json"
    if path.exists():
        betas = json.loads(path.read_text())
    for desc in descriptors:
        for scen in cfg["scenarios"]:
            data = ScenarioData.load(desc, scen, c)
            prev = RESULTS / "retrieval.csv"
            if kappa_only and prev.exists():
                old = pd.read_csv(prev).set_index(["descriptor", "scenario"])
                st = old.loc[(desc, scen)].to_dict()
                st["beta"] = st.get("beta1", st["beta"])
            else:
                st = window_stats(data, cfg["filter"]["window"], n_windows, 0, cfg["filter"]["min_window_valid"])
            data.beta = st["beta"]
            kc = cfg.get("calibration", {})
            kappa, krows = calibrate_kappa(data, cfg["filter"], kc.get("episodes", 200), kc.get("workers", 8))
            for r in krows:
                print(f"    kappa {r['kappa']:.3f}: success {r['success'] / r['n']:.3f} false {r['false_loc'] / r['n']:.3f} "
                      f"false|conc {r['false_given_conc']:.3f} steps {r['mean_steps']:.1f}", flush=True)
            pd.DataFrame(krows).assign(descriptor=desc, scenario=scen).to_csv(
                RESULTS / f"kappa_{desc}_{scen}.csv", index=False)
            st.update(beta1=st["beta"], kappa=kappa, beta=st["beta"] * kappa)
            rows.append(dict(descriptor=desc, scenario=scen, **st))
            betas[f"{desc}_{scen}"] = dict(beta=st["beta"], beta1=st["beta1"], kappa=kappa,
                                           loglik_true=st["loglik_true"])
            print(f"[calib] {desc} {scen}: top1 {st['top1']:.3f} top5 {st['top5']:.3f} "
                  f"top1(<=1 cell) {st['top1_within1']:.3f} chance {st['chance_top1']:.4f} beta {st['beta']} "
                  f"ll {st['loglik_true']:.2f} (uniform {st['loglik_uniform']:.2f}) kappa {kappa} -> beta {st['beta']:.2f}",
                  flush=True)
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
    ap.add_argument("--kappa-only", action="store_true", help="reuse retrieval stats / beta_1 from retrieval.csv")
    a = ap.parse_args()
    run(load_config(a.config), a.descriptors, a.windows, a.kappa_only)


if __name__ == "__main__":
    main()

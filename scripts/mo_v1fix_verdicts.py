"""Vereditos v2 (DECISIONS_MO_V2.md §2) sobre os modelos MO v1, nos dois modos de avaliação.

Entradas (todas com os mesmos episódios, seed 2024 + i, e torch.manual_seed(2024 + i) no modo amostrado):
- results_mo_v1fix/episodes_presets.csv  (MO, PPOs fixos, PPO da PoC e reta)
- results_mo_v1fix/episodes_h6_sweep.csv (varreduras 1-D de w_risco e w_custo)
- results_mo_v1fix/poa_episodes_{argmax,sampled}.csv (H7)
- results_mo/stochastic/episodes_test.csv (grade de 37 pesos, amostrado; H5)

Saída: results_mo_v1fix/verdicts_v2.json e tabelas CSV ao lado.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from dal.config import ROOT, load_config
from dal.evaluation.statistics import percentile_ci

IN = ROOT / "results_mo_v1fix"
N_BOOT = 2000
ALPHA = 0.05
PRESETS = {"fast": (0.15, 0.55, 0.15, 0.15), "safe": (0.15, 0.15, 0.15, 0.55), "unif": (0.25, 0.25, 0.25, 0.25)}
# Normalização a priori (DECISIONS_MO_V2 §2): sucesso [0, 1], localização falsa [0, 1], distância [0, 4000] m.
MAX_DISTANCE_M = 4000.0
REWARD = load_config("configs/default.yaml")["env"]


def window_resampler(windows: np.ndarray, rng: np.random.Generator):
    """Gera pesos por episódio reamostrando janelas (episódios da mesma janela saem juntos)."""
    unique, inverse = np.unique(windows, return_inverse=True)

    def draw() -> np.ndarray:
        counts = np.bincount(rng.integers(len(unique), size=len(unique)), minlength=len(unique))
        return counts[inverse].astype(float)

    return draw


def by_episode(frame: pd.DataFrame, column: str) -> np.ndarray:
    return frame.sort_values("episode")[column].astype(float).to_numpy()


def windows_of(frame: pd.DataFrame) -> np.ndarray:
    first = frame.sort_values("episode")
    return (first.window_r0.astype(str) + "_" + first.window_c0.astype(str)).to_numpy()


def scalarized_return(frame: pd.DataFrame, weights) -> np.ndarray:
    """r·w por episódio, sem a entropia inicial (igual entre métodos no mesmo episódio; cancela no pareado)."""
    f = frame.sort_values("episode")
    info = -f.final_entropy.to_numpy(float)
    cost = -(REWARD["step_cost"] * f.steps.to_numpy(float) + REWARD["border_penalty"] * f.n_blocked.to_numpy(float))
    success = REWARD["success_reward"] * f.success.to_numpy(float)
    risk = -REWARD["false_loc_penalty"] * f.false_loc.to_numpy(float)
    return np.dot(np.stack([info, cost, success, risk], axis=1), np.asarray(weights, float))


def paired_ci(a: np.ndarray, b: np.ndarray, windows: np.ndarray, seed: int = 0) -> tuple[float, float, float, np.ndarray]:
    """Diferença média a − b com IC por bootstrap de janelas; devolve também as réplicas."""
    draw = window_resampler(windows, np.random.default_rng(seed))
    diff = a - b
    replicates = np.array([np.average(diff, weights=draw()) for _ in range(N_BOOT)])
    low, high = percentile_ci(replicates)
    return float(diff.mean()), low, high, replicates


# ============================================================================ H4
def verdict_h4(presets: pd.DataFrame) -> dict:
    """Descritivo: MO(w_k) − PPO-fixo(w_k) em r·w_k; o veredito é 'não avaliável' pela pré-condição."""
    rows = []
    for mode, group in presets.groupby("mode"):
        for preset in ("fast", "safe"):
            mo = group[(group.method == "mo_ppo") & (group.preset == preset)]
            fixed = group[group.method == f"ppo_fixed_{preset}"]
            estimate, low, high, _ = paired_ci(scalarized_return(mo, PRESETS[preset]),
                                               scalarized_return(fixed, PRESETS[preset]), windows_of(mo))
            rows.append({"mode": mode, "preset": preset, "diff_scalarized_return": estimate, "ci_low": low, "ci_high": high})
    pd.DataFrame(rows).to_csv(IN / "h4_scalarized_return.csv", index=False)
    return {"verdict": "não avaliável",
            "reason": "pré-condição v2 não atendida: gate de condicionamento reprovado (MO v1), orçamentos diferentes "
                      "(348k × 225k) e sem PPO fixo unif (3º preset)",
            "descriptive": rows}


# ============================================================================ H5
def holm(p_values: np.ndarray) -> np.ndarray:
    """p-valores ajustados por Holm (step-down)."""
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def verdict_h5(grid: pd.DataFrame, straight: pd.DataFrame) -> dict:
    """Existe w com Δsucesso ≥ −3 p.p., Δfalsa ≤ 0 e Δdistância < 0 (limites do IC)? Nominal e Holm."""
    windows = windows_of(straight)
    rows = []
    for (method, weight_idx), group in grid.groupby(["method", "weight_idx"]):
        tests = {}
        for column in ("success", "false_loc", "distance_m"):
            tests[column] = paired_ci(by_episode(group, column), by_episode(straight, column), windows, seed=weight_idx)
        s, f, d = tests["success"], tests["false_loc"], tests["distance_m"]
        nominal = s[1] >= -0.03 and f[2] <= 0 and d[2] < 0
        # p-valor de interseção-união: o maior dos três p unilaterais.
        p = max(float(np.mean(s[3] < -0.03)), float(np.mean(f[3] > 0)), float(np.mean(d[3] >= 0)))
        rows.append({"method": method, "weight_idx": weight_idx, "w_str": group.w_str.iloc[0], "d_success": s[0],
                     "d_success_lo": s[1], "d_false_loc": f[0], "d_false_loc_hi": f[2], "d_distance": d[0],
                     "d_distance_hi": d[2], "nominal_ok": nominal, "p_iut": p})
    table = pd.DataFrame(rows)
    table["p_holm"] = np.nan
    for method, part in table.groupby("method"):
        table.loc[part.index, "p_holm"] = holm(part.p_iut.to_numpy())
    table.to_csv(IN / "h5_grid_vs_straight.csv", index=False)
    mo = table[table.method == "mo_ppo"]
    return {"verdict_nominal": "suportada" if mo.nominal_ok.any() else "não suportada",
            "verdict_holm": "suportada" if (mo.p_holm < ALPHA).any() else "não suportada",
            "n_weights": int(len(mo)), "mode": "sampled (grade de results_mo/stochastic)"}


# ============================================================================ H6
def verdict_h6(sweep: pd.DataFrame) -> dict:
    """ρ de Spearman entre o peso e a taxa sobre os 11 pontos, com bootstrap de episódios pareado."""
    out = {}
    targets = {"risk": "false_loc", "cost": "distance_m"}
    rows = []
    for mode, by_mode in sweep.groupby("mode"):
        for sweep_name, target in targets.items():
            part = by_mode[by_mode.sweep == sweep_name]
            values = sorted(part.sweep_value.unique())
            matrix = np.stack([by_episode(part[part.sweep_value == v], target) for v in values])  # (11, n)
            rho = spearmanr(values, matrix.mean(axis=1)).statistic
            rng = np.random.default_rng(0)
            boot = []
            for _ in range(N_BOOT):
                idx = rng.integers(matrix.shape[1], size=matrix.shape[1])  # mesmos episódios em todos os pontos
                boot.append(spearmanr(values, matrix[:, idx].mean(axis=1)).statistic)
            low, high = percentile_ci(np.array(boot, dtype=float))
            rows.append({"mode": mode, "sweep": sweep_name, "target": target, "rho": rho, "ci_low": low, "ci_high": high,
                         **{f"rate_w{v:.1f}": r for v, r in zip(values, matrix.mean(axis=1))}})
    table = pd.DataFrame(rows)
    table.to_csv(IN / "h6_sweep.csv", index=False)
    for mode, part in table.groupby("mode"):
        ok = (part.ci_high < 0).all()
        contradicts = (part.ci_low >= 0).any()
        out[mode] = "suportada" if ok else "não suportada" if contradicts else "inconclusiva"
    return {"verdict": out.get("sampled"), "by_mode": out}


# ============================================================================ H7
def verdict_h7() -> dict:
    """Localização falsa do safe − PPO-S2 nas ilhas: IC inteiramente < 0?"""
    out = {}
    for mode in ("argmax", "sampled"):
        path = IN / f"poa_episodes_{mode}.csv"
        if not path.exists():
            out[mode] = None
            continue
        episodes = pd.read_csv(path)
        islands = episodes[episodes.region == "ilhas"]
        safe, ppo = islands[islands.method == "mo_safe"], islands[islands.method == "ppo_s2"]
        estimate, low, high, _ = paired_ci(by_episode(safe, "false_loc"), by_episode(ppo, "false_loc"), windows_of(safe))
        out[mode] = {"diff_false_loc": estimate, "ci_low": low, "ci_high": high,
                     "verdict": "suportada" if high < 0 else "não suportada"}
    return {"verdict": (out.get("sampled") or {}).get("verdict"), "by_mode": out}


# ============================================================================ EUM
def utilities(frame: pd.DataFrame) -> np.ndarray:
    """(n, 3) utilidades normalizadas a priori: sucesso, 1 − localização falsa, 1 − distância/4000."""
    f = frame.sort_values("episode")
    return np.stack([f.success.astype(float), 1 - f.false_loc.astype(float),
                     1 - np.clip(f.distance_m.astype(float) / MAX_DISTANCE_M, 0, 1)], axis=1)


def eum_with_ci(policy_sets: dict[str, list[pd.DataFrame]], windows: np.ndarray) -> pd.DataFrame:
    """EUM de cada conjunto (máximo sobre as políticas do conjunto de w·u, médio sobre w ~ Dirichlet(1))."""
    weights = np.random.default_rng(42).dirichlet(np.ones(3), size=1000)
    draw = window_resampler(windows, np.random.default_rng(1))
    episode_utilities = {name: [utilities(f) for f in frames] for name, frames in policy_sets.items()}

    def eum(name: str, episode_weights: np.ndarray) -> float:
        points = np.stack([np.average(u, axis=0, weights=episode_weights) for u in episode_utilities[name]])
        return float((weights @ points.T).max(axis=1).mean())

    ones = np.ones(len(windows))
    replicates = {name: [] for name in policy_sets}
    for _ in range(N_BOOT):
        w = draw()
        for name in policy_sets:
            replicates[name].append(eum(name, w))
    rows = []
    for name in policy_sets:
        low, high = percentile_ci(np.array(replicates[name]))
        rows.append({"set": name, "n_policies": len(policy_sets[name]), "eum": eum(name, ones), "ci_low": low,
                     "ci_high": high})
    return pd.DataFrame(rows)


def main() -> None:
    presets = pd.read_csv(IN / "episodes_presets.csv")
    sweep = pd.read_csv(IN / "episodes_h6_sweep.csv")
    straight = presets[presets.method == "straight"]
    result = {"protocol": "DECISIONS_MO_V2.md A1–A3; modo principal: amostrado; 300 episódios; Taquari S2 teste",
              "H4": verdict_h4(presets)}

    grid_path = ROOT / "results_mo/stochastic/episodes_test.csv"
    if grid_path.exists():
        grid = pd.read_csv(grid_path)
        grid = grid[grid.method.isin(["mo_ppo", "mo_ppo_nomem"])].assign(episode=lambda g: g.seed - 2024)
        result["H5"] = verdict_h5(grid, straight)
    result["H6"] = verdict_h6(sweep)
    result["H7"] = verdict_h7()

    eum_rows = []
    for mode, group in presets.groupby("mode"):
        sets = {
            "mo_ppo (3 presets)": [group[(group.method == "mo_ppo") & (group.preset == p)] for p in PRESETS],
            "mo_ppo_nomem (3 presets)": [group[(group.method == "mo_ppo_nomem") & (group.preset == p)] for p in PRESETS],
            "ppo_fixed (fast, safe)": [group[group.method == f"ppo_fixed_{p}"] for p in ("fast", "safe")],
            "ppo_poc (3 seeds)": [group[group.method == f"ppo_poc_s{s}"] for s in range(3)],
            "straight": [straight],
        }
        table = eum_with_ci(sets, windows_of(straight))
        table.insert(0, "mode", mode)
        eum_rows.append(table)
    eum = pd.concat(eum_rows, ignore_index=True)
    eum.to_csv(IN / "eum.csv", index=False)
    result["EUM"] = eum.round(4).to_dict(orient="records")
    result["hypervolume"] = "não calculado: exige conjuntos de mesmo tamanho (3 presets × 3 PPOs fixos); falta o PPO fixo unif na v1"
    (IN / "verdicts_v2.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=float))
    print(json.dumps({k: (v.get("verdict") if isinstance(v, dict) else v) for k, v in result.items()
                      if k in ("H4", "H5", "H6", "H7")}, ensure_ascii=False, indent=1))
    print(eum.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

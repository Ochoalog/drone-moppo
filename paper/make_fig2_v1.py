"""Fig. 2 do artigo (estudo v1): sucesso × quintil do índice de mudança (Taquari e Porto Alegre) + painel do M1.

Lê só CSVs de results_final/ e results_poa/. Saída: paper/figures/fig2_v1_motivation.pdf (vetorial,
3,5 in de largura, fonte ≥ 8 pt). Cores: as 4 primeiras da paleta categórica validada (ordem fixa por
método), com marcadores de formas diferentes como codificação secundária.

Uso: .venv/bin/python paper/make_fig2_v1.py
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dal.config import ROOT  # noqa: E402

OUT = ROOT / "paper" / "figures" / "fig2_v1_motivation.pdf"
QUINTILES = 5
N_BOOT = 2000
# Ordem fixa: a cor segue o método, nunca a posição.
METHODS = {
    "ppo": ("PPO (v1)", "#2a78d6", "o"),
    "straight": ("Straight", "#eb6834", "s"),
    "greedy_map": ("Greedy (map)", "#1baf7a", "^"),
    "lawnmower": ("Lawnmower", "#eda100", "D"),
}
POA_METHOD_ALIAS = {"ppo_S2": "ppo"}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    idx = rng.integers(len(values), size=(N_BOOT, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def quintile_curves(episodes: pd.DataFrame, unit_columns: list[str]) -> pd.DataFrame:
    """Sucesso médio e IC por quintil (quintis sobre as unidades de episódio do local)."""
    units = episodes.drop_duplicates(unit_columns)
    edges = np.quantile(units["change_window"].dropna(), np.linspace(0, 1, QUINTILES + 1))
    episodes = episodes.assign(quintile=np.clip(np.searchsorted(edges, episodes.change_window, side="right"),
                                                1, QUINTILES))
    rng = np.random.default_rng(0)
    rows = []
    for (method, quintile), part in episodes.groupby(["method", "quintile"]):
        values = part.success.astype(float).to_numpy()
        low, high = bootstrap_ci(values, rng)
        rows.append({"method": method, "quintile": quintile, "success": values.mean(), "lo": low, "hi": high})
    return pd.DataFrame(rows)


def style_axis(ax) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7.5)


def plot_quintiles(ax, curves: pd.DataFrame, title: str) -> None:
    offsets = np.linspace(-0.15, 0.15, len(METHODS))
    for (method, (label, color, marker)), dx in zip(METHODS.items(), offsets, strict=True):
        part = curves[curves.method == method].sort_values("quintile")
        if part.empty:
            continue
        x = part.quintile.to_numpy() + dx
        ax.errorbar(x, part.success, yerr=[part.success - part.lo, part.hi - part.success], color=color,
                    marker=marker, markersize=4, linewidth=1.4, elinewidth=0.8, capsize=0, label=label,
                    markeredgecolor="white", markeredgewidth=0.6)
    ax.set_xticks(range(1, QUINTILES + 1), [f"Q{q}" for q in range(1, QUINTILES + 1)])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Success rate", fontsize=8, color=TEXT)
    ax.set_title(title, fontsize=8.5, color=TEXT, loc="left")
    style_axis(ax)


def plot_m1(ax, visits: pd.DataFrame) -> None:
    """Índice de mudança médio das células visitadas: PPO sucesso, PPO localização falsa e reta loc. falsa."""
    groups = [("PPO, success", "ppo", "success", "#2a78d6", "o"),
              ("PPO, false loc.", "ppo", "false_loc", "#2a78d6", "x"),
              ("Straight, false loc.", "straight", "false_loc", "#eb6834", "s")]
    locations = [("taquari_S2", "Taquari S2"), ("poa_ilhas", "PoA islands"), ("poa_centro", "PoA downtown")]
    rng = np.random.default_rng(1)
    for j, (label, method, outcome, color, marker) in enumerate(groups):
        xs, ys, los, his = [], [], [], []
        for i, (loc, _) in enumerate(locations):
            part = visits[(visits.location == loc) & (visits.method == method) & visits[outcome]]
            values = part.visited_change.to_numpy()
            low, high = bootstrap_ci(values, rng)
            xs.append(i + (j - 1) * 0.2)
            ys.append(values.mean())
            los.append(values.mean() - low)
            his.append(high - values.mean())
        ax.errorbar(xs, ys, yerr=[los, his], color=color, marker=marker, linestyle="none", markersize=4.5,
                    elinewidth=0.8, label=label, markeredgewidth=1.0 if marker == "x" else 0.6,
                    markeredgecolor=color if marker == "x" else "white")
    ax.set_xticks(range(len(locations)), [name for _, name in locations])
    ax.set_ylabel("Change index\nof visited cells", fontsize=8, color=TEXT)
    ax.set_title("(c) Where failures happen (M1 data)", fontsize=8.5, color=TEXT, loc="left")
    style_axis(ax)
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3,
              handletextpad=0.2, columnspacing=0.8)


def main() -> None:
    taquari = pd.read_csv(ROOT / "results_final/change_index/taquari_episodes.csv")
    poa = pd.read_csv(ROOT / "results_poa/episodes.csv")
    poa = poa[poa.variant == "zeroshot"].assign(method=lambda f: f.method.replace(POA_METHOD_ALIAS))
    visits = pd.read_csv(ROOT / "results_final/mechanism/visits.csv")

    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, axes = plt.subplots(3, 1, figsize=(3.5, 5.6), gridspec_kw={"hspace": 0.62})
    plot_quintiles(axes[0], quintile_curves(taquari, ["scenario", "episode"]),
                   "(a) Taquari (S1+S2, test): success vs change quintile")
    plot_quintiles(axes[1], quintile_curves(poa, ["scenario", "region", "episode"]),
                   "(b) Porto Alegre (zero-shot): success vs change quintile")
    axes[1].set_xlabel("Map–observation change index quintile (Q5 = most changed)", fontsize=8, color=TEXT)
    axes[0].legend(fontsize=7, frameon=False, ncol=2, loc="lower left", handletextpad=0.3, columnspacing=0.8)
    plot_m1(axes[2], visits)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print("escrito", OUT)


if __name__ == "__main__":
    main()

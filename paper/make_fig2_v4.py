"""Fig. 2 do artigo v4 (RQ2): o que torna o problema difícil.

(a) Estudo v1, Taquari: sucesso × quintil do índice de mudança mapa–observação (mesma função da Fig. 2 da v1).
(b) v3, heurísticas no teste dos 5 alvos (`paper/tables/v3_baselines.csv`), ordenados pelo sucesso da aleatória:
    onde a percepção é ambígua, a ordem das heurísticas muda, e a gulosa com o mapa cai para perto da aleatória.
PDF vetorial, 3,5 in, fonte ≥ 8 pt. Cores na ordem fixa da paleta validada; a cor segue o método.

Uso: .venv/bin/python paper/make_fig2_v4.py
"""
from __future__ import annotations

import importlib.util

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dal.config import ROOT  # noqa: E402

_spec = importlib.util.spec_from_file_location("fig2_v1", ROOT / "paper" / "make_fig2_v1.py")
fig2_v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fig2_v1)

OUT = ROOT / "paper" / "figures" / "fig2_v4.pdf"
TEXT, MUTED = fig2_v1.TEXT, fig2_v1.MUTED
V3_METHODS = {  # cor e marcador seguem o método (mesmas da v1 quando o método existe lá)
    "straight": ("Straight", "#eb6834", "s"),
    "straight_150": ("Straight 150 m", "#6250d6", "v"),
    "lawnmower": ("Lawnmower", "#eda100", "D"),
    "greedy_map": ("Greedy (map)", "#1baf7a", "^"),
    "random": ("Random", "#52514e", "o"),
}
TARGETS = {("taquari", "S1"): "Taq. S1", ("taquari", "S2"): "Taq. S2", ("poa_ilhas", "P1"): "POA isl.",
           ("canoas", "C1"): "Canoas", ("poa_centro", "P1"): "POA ctr."}


def plot_v3(ax, table: pd.DataFrame) -> None:
    test = table[(table.split == "test") & table.method.isin(V3_METHODS)]
    keys = [k for k in TARGETS if ((test.region == k[0]) & (test.scenario == k[1])).any()]
    random = {k: float(test[(test.region == k[0]) & (test.scenario == k[1]) & (test.method == "random")].success.iloc[0])
              for k in keys}
    keys.sort(key=lambda k: -random[k])
    offsets = np.linspace(-0.24, 0.24, len(V3_METHODS))
    for (method, (label, color, marker)), dx in zip(V3_METHODS.items(), offsets, strict=True):
        ys = [float(test[(test.region == k[0]) & (test.scenario == k[1]) & (test.method == method)].success.iloc[0])
              for k in keys]
        ax.plot(np.arange(len(keys)) + dx, ys, linestyle="none", marker=marker, color=color, markersize=4.5,
                markeredgecolor="white", markeredgewidth=0.6, label=label)
    ax.set_xticks(range(len(keys)), [TARGETS[k] for k in keys])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Success rate", fontsize=8, color=TEXT)
    ax.set_title("(b) v3 heuristics (test), sorted by random success", fontsize=8, color=TEXT, loc="left")
    fig2_v1.style_axis(ax)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3,
              handletextpad=0.2, columnspacing=0.8)


def main() -> None:
    taquari = pd.read_csv(ROOT / "results_final/change_index/taquari_episodes.csv")
    table = pd.read_csv(ROOT / "paper/tables/v3_baselines.csv")
    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.4), gridspec_kw={"hspace": 0.55})
    fig2_v1.plot_quintiles(axes[0], fig2_v1.quintile_curves(taquari, ["scenario", "episode"]),
                           "(a) v1, Taquari (S1+S2): success vs change quintile")
    axes[0].set_xlabel("Map–observation change quintile (Q5 = most changed)", fontsize=8, color=TEXT)
    axes[0].tick_params(labelsize=8)
    axes[0].legend(fontsize=8, frameon=False, ncol=2, loc="lower left", handletextpad=0.3, columnspacing=0.8)
    plot_v3(axes[1], table)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print("escrito", OUT)


if __name__ == "__main__":
    main()

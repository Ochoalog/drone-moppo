"""Fig. 3 do artigo v4 (experimento E, V46): vantagem da PPO × ambiguidade perceptual.

Eixo x: regimes ordenados pelo sucesso da aleatória no treino (do mais fácil ao mais ambíguo). Eixo y: vantagem
de sucesso da PPO (amostrada) sobre a aleatória e sobre a melhor heurística (escolhida no treino), em p.p., com
IC 95% hierárquico. Acima de cada regime: execuções aprovadas no gate (critério original); abaixo do nome,
o sucesso da aleatória no treino. PDF vetorial, 3,5 in (a legenda do artigo explica as duas anotações).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SERIES = (("random", "over random", "#2a78d6", "o"), ("best", "over best heuristic", "#eb6834", "s"))
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _short(regime: str) -> str:
    """"Taquari L2 (σ=1)" → "L2\nσ=1"; "Taquari L0" → "Taq. L0"; os demais ficam iguais."""
    if regime.startswith("Taquari L") and "(" in regime:
        level, sigma = regime.removeprefix("Taquari ").split(" (")
        return f"{level}\n{sigma.rstrip(')')}"
    return regime.replace("Taquari", "Taq.")


def figure_3(curve: pd.DataFrame, path: Path, mode: str = "sampled") -> None:
    curve = curve.sort_values("random_train_success", ascending=False).reset_index(drop=True)
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "pdf.fonttype": 42, "axes.edgecolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK})
    figure, axis = plt.subplots(figsize=(3.5, 2.6))
    axis.axhline(0, color=MUTED, linewidth=0.8, zorder=1)
    offsets = (-0.12, 0.12)
    for (label, legend, color, marker), offset in zip(SERIES, offsets, strict=True):
        x = curve.index + offset
        y = 100 * curve[f"adv_{label}_{mode}"]
        low = 100 * curve[f"adv_{label}_{mode}_low"]
        high = 100 * curve[f"adv_{label}_{mode}_high"]
        axis.errorbar(x, y, yerr=[y - low, high - y], fmt=marker, color=color, markersize=4.5, elinewidth=1.2,
                      capsize=2, markeredgecolor="white", markeredgewidth=0.6, label=legend, zorder=3)
    names = [_short(r.regime) + f"\n{100 * r.random_train_success:.0f}%" for r in curve.itertuples()]
    axis.set_xticks(curve.index, names, fontsize=8)
    axis.set_xlim(-0.5, len(curve) - 0.5)
    axis.set_ylabel("PPO success advantage (p.p.)")
    axis.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    bottom, top = axis.get_ylim()
    axis.set_ylim(bottom, top + 0.12 * (top - bottom))  # faixa para a anotação do gate
    for r in curve.itertuples():
        axis.annotate(f"{r.gate_original_passed}/{r.n_runs}", (r.Index, 1.0), xycoords=("data", "axes fraction"),
                      ha="center", va="top", fontsize=8, color=MUTED)
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8, frameon=False,
                handletextpad=0.3, columnspacing=1.0)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)

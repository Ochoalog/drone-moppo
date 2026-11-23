"""Fig. 1 do artigo v4: visão geral do laço de localização ativa, com dados reais do estudo v1.

(a) Mapa pré-evento (02/10/2023, cenário S2) de uma janela de teste de 3,2 × 3,2 km (linha 381, coluna 50 da
    grade de 50 m), com a trajetória real de uma política PPO (`results/traj/test_S2_ppo0_ep0.npz`, sucesso em
    28 passos) e o footprint de 150 m da câmera no último passo.
(b) A mesma área de 1 km (o quadrado branco em (a), vindo da imagem de origem) como a câmera a vê (pós-evento,
    07/05/2024) e como está no mapa, com o footprint no último passo: o mapa está desatualizado.
(c) Crença do filtro de histograma em três passos do mesmo episódio.
(d) Política (PPO sobre a crença, como no ANL) ou heurística, que fecha o laço com a ação.

Imagens: recortes de `apresentacao_grupo/enchente_antes_depois.png` (gerada por `apresentacao_grupo/scene_flood.py`
a partir dos COGs Maxar/OpenAerialMap, CC BY-NC 4.0), guardados em `paper/figures/fig1_src/`:
- `map_s2_window.png`: 64 × 64 células da janela;
- `map_s2_zoom.png` e `post_zoom.png`: o recorte de 20 × 20 células (colunas 34–54, linhas 30–50).

PDF de largura dupla (7,16 in, `figure*`), fonte ≥ 7 pt. Uso: python paper/make_fig1.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import PowerNorm  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "paper" / "figures" / "fig1_src"
TRAJ = ROOT / "results" / "traj" / "test_S2_ppo0_ep0.npz"
OUT = ROOT / "paper" / "figures" / "fig1_overview.pdf"

WINDOW_CELLS = 64  # janela v1: 64 × 64 células de 50 m
CELL_M = 50.0
ZOOM_COL0, ZOOM_ROW0, ZOOM_CELLS = 34, 30, 20  # recorte ampliado da imagem de origem
FOOTPRINT_CELLS = 3  # 150 m
BELIEF_STEPS = (0, 13, 27)

TEXT, MUTED = "#0b0b0b", "#52514e"
PPO_BLUE = "#2a78d6"  # mesma cor da PPO nas Figs. 2 e 3
TRUTH = "#eb6834"
FS = 7.5


def load(name: str) -> np.ndarray:
    return np.asarray(Image.open(SRC / name).convert("RGB"))


def crop_zoom(image: np.ndarray, row: float, col: float, half_cells: float) -> np.ndarray:
    """Recorte quadrado da imagem ampliada centrado na célula (linha, coluna) da janela."""
    px_per_cell = image.shape[0] / ZOOM_CELLS
    cy, cx = (row + 0.5 - ZOOM_ROW0) * px_per_cell, (col + 0.5 - ZOOM_COL0) * px_per_cell
    h = half_cells * px_per_cell
    y0, y1, x0, x1 = (int(round(v)) for v in (cy - h, cy + h, cx - h, cx + h))
    if y0 < 0 or x0 < 0 or y1 > image.shape[0] or x1 > image.shape[1]:
        raise ValueError("o footprint sai do recorte ampliado; escolha outro passo")
    return image[y0:y1, x0:x1]


def clean(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(MUTED)
        spine.set_linewidth(0.5)


def arrow(fig, start, end, text: str = "", text_xy=None, rad: float = 0.0, ha: str = "center") -> None:
    fig.patches.append(FancyArrowPatch(start, end, transform=fig.transFigure, arrowstyle="-|>", mutation_scale=8,
                                       lw=0.9, color=TEXT, connectionstyle=f"arc3,rad={rad}"))
    if text:
        x, y = text_xy or ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.04)
        fig.text(x, y, text, fontsize=FS - 0.5, color=TEXT, ha=ha, va="center")


def main() -> None:
    plt.rcParams.update({"font.size": FS, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
    episode = np.load(TRAJ)
    traj, beliefs = episode["traj"].astype(float), episode["beliefs"]
    assert bool(episode["success"]) and not bool(episode["false_loc"])
    assert tuple(episode["window"]) == (381, 50), "as imagens são da janela (381, 50)"
    last = traj[-1]

    width, height = 7.16, 2.30
    fig = plt.figure(figsize=(width, height))

    def axes(x_in: float, y_in: float, w_in: float, h_in: float):
        return fig.add_axes([x_in / width, y_in / height, w_in / width, h_in / height])

    def draw_path(ax, offset_row: float = 0.0, offset_col: float = 0.0) -> None:
        cols, rows = traj[:, 1] + 0.5 - offset_col, traj[:, 0] + 0.5 - offset_row
        ax.plot(cols, rows, color="white", lw=2.2, solid_capstyle="round")
        ax.plot(cols, rows, color=PPO_BLUE, lw=1.2, solid_capstyle="round")

    def footprint(ax, offset_row: float = 0.0, offset_col: float = 0.0, lw: float = 1.1) -> None:
        ax.add_patch(Rectangle((last[1] + 0.5 - FOOTPRINT_CELLS / 2 - offset_col,
                                last[0] + 0.5 - FOOTPRINT_CELLS / 2 - offset_row),
                               FOOTPRINT_CELLS, FOOTPRINT_CELLS, fill=False, ec="#ffd84d", lw=lw))

    # (a) mapa pré-evento da janela + trajetória real; o quadrado branco (da imagem de origem) é a área de (b)
    size_a = 1.55
    ax = axes(0.04, 0.48, size_a, size_a)
    ax.imshow(load("map_s2_window.png"), extent=(0, WINDOW_CELLS, WINDOW_CELLS, 0), interpolation="lanczos")
    draw_path(ax)
    ax.plot(traj[0, 1] + 0.5, traj[0, 0] + 0.5, "o", ms=4, mfc="white", mec=PPO_BLUE, mew=1.0)
    ax.text(traj[0, 1] + 0.5, traj[0, 0] - 2.0, "start", color="white", fontsize=FS - 1, ha="center")
    footprint(ax, lw=0.9)
    ax.text(ZOOM_COL0 + ZOOM_CELLS - 1, ZOOM_ROW0 + ZOOM_CELLS + 3.2, "(b)", color="white", fontsize=FS - 1,
            ha="right")
    ax.plot([3, 23], [60, 60], color="white", lw=1.6)  # barra de 1 km (20 células de 50 m)
    ax.text(13, 58.6, "1 km", color="white", fontsize=FS - 1, ha="center", va="bottom")
    ax.set_xlim(0, WINDOW_CELLS)
    ax.set_ylim(WINDOW_CELLS, 0)
    clean(ax)
    ax.set_title("(a) Map: pre-event (Oct 2023)", fontsize=FS, color=TEXT, loc="left", pad=3)

    B_TOP = 1.36
    # (b) a mesma área de 1 km no mapa e na câmera (pós-evento), com o footprint de 150 m no último passo
    size_b = 0.68
    zoom_extent = (0, ZOOM_CELLS, ZOOM_CELLS, 0)
    for y_in, name, title, color in ((B_TOP, "post_zoom.png", "(b) Camera view: May 2024", TEXT),
                                     (0.48, "map_s2_zoom.png", "Map, same area", MUTED)):
        ax_b = axes(1.78, y_in, size_b, size_b)
        ax_b.imshow(load(name), extent=zoom_extent, interpolation="lanczos")
        draw_path(ax_b, ZOOM_ROW0, ZOOM_COL0)
        footprint(ax_b, ZOOM_ROW0, ZOOM_COL0)
        ax_b.set_xlim(0, ZOOM_CELLS)
        ax_b.set_ylim(ZOOM_CELLS, 0)
        clean(ax_b)
        ax_b.set_title(title, fontsize=FS, color=color, loc="left", pad=3)
    fig.text((1.78 + size_b / 2) / width, 0.40 / height, "yellow: 150 m camera footprint", fontsize=FS - 1.5,
             color=MUTED, ha="center", va="top")

    # (c) crença do filtro em três passos do mesmo episódio
    size_c, gap_c, x_c, y_c = 0.78, 0.07, 2.95, 1.17
    for i, step in enumerate(BELIEF_STEPS):
        ax_c = axes(x_c + i * (size_c + gap_c), y_c, size_c, size_c)
        belief = beliefs[step]
        ax_c.imshow(belief, cmap="magma", norm=PowerNorm(0.45, vmin=0, vmax=float(belief.max())),
                    interpolation="nearest")
        ax_c.plot(traj[step, 1], traj[step, 0], "o", ms=4.5, mfc="none", mec=TRUTH, mew=1.1)
        clean(ax_c)
        label = f"$t={step + 1}$" + (" (stop)" if step == len(traj) - 1 else "")
        ax_c.set_title(label, fontsize=FS, color=TEXT, pad=2)
    x_c_end = x_c + 3 * size_c + 2 * gap_c
    fig.text(x_c / width, (y_c + size_c + 0.21) / height,
             "(c) Belief $b_t$ of the histogram filter  (" + r"$\circ$" + " true position)",
             fontsize=FS, color=TEXT, ha="left", va="bottom")
    fig.text(((x_c + x_c_end) / 2) / width, (y_c - 0.08) / height,
             r"$b'(c)\propto b(c\ominus a)\,\exp(\beta\,z_t^{\top} m_c)$" + "\n"
             r"stop when $\geq 0.9$ of the mass lies within 100 m:" + "\n"
             "success if the mode is within 100 m of the truth,\nfalse localization otherwise",
             fontsize=FS - 0.5, color=TEXT, ha="center", va="top", linespacing=1.3)

    # (d) política ou heurística
    x_d, y_d, w_d, h_d = 5.62, 1.05, 1.48, 0.95
    fig.patches.append(FancyBboxPatch((x_d / width, y_d / height), w_d / width, h_d / height,
                                      boxstyle="round,pad=0.004,rounding_size=0.015", transform=fig.transFigure,
                                      fc="#eef4fc", ec=PPO_BLUE, lw=0.9))
    center = (x_d + w_d / 2) / width
    fig.text(center, (y_d + h_d - 0.09) / height, r"(d) Policy $\pi(a_t\,|\,b_t)$", fontsize=FS,
             color=TEXT, ha="center", va="top", weight="bold")
    fig.text(center, (y_d + h_d - 0.31) / height, "PPO on belief, map mask\nand distinctiveness (as in ANL)",
             fontsize=FS - 1, color=TEXT, ha="center", va="top", linespacing=1.2)
    fig.text(center, (y_d + 0.42) / height, "or a heuristic:\nstraight, lawnmower,\nrandom, greedy (map)",
             fontsize=FS - 1, color=MUTED, ha="center", va="top", linespacing=1.2)

    # setas do laço
    y_mid = (y_c + size_c / 2) / height
    arrow(fig, ((1.78 + size_b + 0.04) / width, (B_TOP + size_b / 2) / height), ((x_c - 0.05) / width, y_mid))
    fig.text((1.78 + size_b + 0.25) / width, (y_c + size_c / 2 - 0.20) / height, r"$z_t=\phi(o_t)$",
             fontsize=FS - 0.5, color=TEXT, ha="center", va="center")
    arrow(fig, ((x_c_end + 0.03) / width, y_mid), ((x_d - 0.03) / width, y_mid))
    y_loop = 0.22
    fig.add_artist(plt.Line2D([center, center], [y_d / height, y_loop / height], transform=fig.transFigure,
                              color=TEXT, lw=0.9))
    fig.add_artist(plt.Line2D([center, (0.04 + size_a / 2) / width], [y_loop / height, y_loop / height],
                              transform=fig.transFigure, color=TEXT, lw=0.9))
    arrow(fig, ((0.04 + size_a / 2) / width, y_loop / height), ((0.04 + size_a / 2) / width, 0.46 / height))
    fig.text(((x_c + x_c_end) / 2) / width, (y_loop - 0.10) / height,
             r"action $a_t$: move 50 m in one of 8 directions (v3 adds altitude changes); "
             r"the camera captures $o_{t+1}$",
             fontsize=FS - 0.5, color=TEXT, ha="center", va="center")

    # dpi fixo (as imagens raster entram no PDF a 200 dpi, independente do matplotlibrc) e sem data de criação,
    # para o PDF ser reproduzível byte a byte
    fig.savefig(OUT, bbox_inches=None, dpi=200, metadata={"CreationDate": None})
    fig.savefig(OUT.with_suffix(".png"), dpi=220)
    print(f"gravado {OUT}")


if __name__ == "__main__":
    main()

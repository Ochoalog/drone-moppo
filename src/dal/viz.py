"""Figuras que dependem só dos produtos de dados (visão geral, trajetórias sobre as miniaturas)."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import DATA, FIGS, load_config  # noqa: E402
from .data import Raster, load_date, load_masks  # noqa: E402
from .grid import Grid  # noqa: E402

DATE_LABEL = {"post": "post (2024-05-07)", "pre_mar23": "pre_mar23 (2023-03-02)", "pre_oct23": "pre_oct23 (2023-10-02)"}


def thumb_extent_km(t: Raster, grid: Grid) -> list[float]:
    """Extent do imshow de uma miniatura em km, com origem no canto superior esquerdo da grade (y = km ao sul)."""
    H, W = t.img.shape[:2]
    x0 = (t.x_origin - grid.x0) / 1000
    y0 = (grid.y1 - t.y_origin) / 1000
    return [x0, x0 + W * t.res / 1000, y0 + H * t.res / 1000, y0]


def data_overview(cfg: dict) -> None:
    d = cfg["data"]
    grid = Grid.from_aoi(**d["aoi"], cell=d["cell_m"])
    masks = load_masks()
    stats = json.loads((DATA / "masks_stats.json").read_text())
    align = json.loads((DATA / "align.json").read_text())
    train, test = grid.split_rows(d["split_lat"], d["split_margin_m"])
    dates = list(d["sources"])
    fig, axes = plt.subplots(1, 2 * len(dates), figsize=(4 * len(dates) * 1.1, 9), constrained_layout=True)
    W, H = grid.ncols * grid.cell / 1000, grid.nrows * grid.cell / 1000
    for i, date in enumerate(dates):
        t = load_date(date, thumb=True)
        ax = axes[2 * i]
        ax.imshow(t.img, extent=thumb_extent_km(t, grid))
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        a = align[date]
        ax.set_title(f"{DATE_LABEL[date]}\ncloud {100 * stats[date]['aoi_cloud_frac_of_valid']:.1f}%  "
                     f"offset {a['norm_m']:.1f} m", fontsize=9)
        ax.set_xlabel("km east")
        if i == 0:
            ax.set_ylabel("km south of AOI top")
        ax2 = axes[2 * i + 1]
        ok = masks[f"{date}_ok"].astype(float)
        ax2.imshow(ok, extent=[0, W, H, 0], cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
        ax2.set_title(f"valid cells: {100 * ok.mean():.0f}%\n(valid>=0.9, cloud<=0.1)", fontsize=9)
        ax2.set_xlabel("km east")
        for a_ in (ax, ax2):
            a_.axhline((train.max() + 1) * grid.cell / 1000, color="cyan", lw=1.5)
            a_.axhline(test.min() * grid.cell / 1000, color="magenta", lw=1.5)
    axes[0].text(0.1, (train.max() - 5) * grid.cell / 1000, "train (north)", color="cyan", fontsize=8)
    axes[0].text(0.1, (test.min() + 10) * grid.cell / 1000, "test (south)", color="magenta", fontsize=8)
    fig.suptitle(f"AOI {grid.ncols}x{grid.nrows} cells of {grid.cell:.0f} m; split at lat {d['split_lat']}")
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "data_overview.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    data_overview(load_config("configs/default.yaml"))

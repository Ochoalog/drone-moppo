"""Estima a translação entre cada data de mapa e `post` por correlação de fase sobre imagens de
magnitude do gradiente, usando blocos estruturados (gradiente alto) válidos nas duas datas.

Convenção: `offset_m = (dx, dy)` (leste, norte), tal que um detalhe do terreno no ponto UTM p de
`post` aparece em p + offset na data do mapa. Por isso os recortes da data do mapa são
amostrados em p + offset, para ficarem registrados com `post`.
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from .config import DATA, load_config
from .data import Raster, load_date, load_masks
from .grid import Grid


def grad_image(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 1.0)
    m = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    return (m - m.mean()) / (m.std() + 1e-6)


def phase_shift_px(ref: np.ndarray, mov: np.ndarray) -> tuple[float, float, float]:
    """(sx, sy, resposta) com mov(u) ~= ref(u - s), em pixels (x para a direita, y para baixo)."""
    win = cv2.createHanningWindow((ref.shape[1], ref.shape[0]), cv2.CV_32F)
    (sx, sy), resp = cv2.phaseCorrelate(ref.astype(np.float32), mov.astype(np.float32), win)
    return float(sx), float(sy), float(resp)


def estimate_offset(post: Raster, other: Raster, ok_both: np.ndarray, grid: Grid, tile_cells: int = 12,
                    n_tiles: int = 60, min_resp: float = 0.05, seed: int = 0) -> dict:
    """Deslocamento mediano (m) sobre os blocos mais estruturados e totalmente válidos."""
    cands = []
    for r in range(0, grid.nrows - tile_cells + 1, tile_cells):
        for c in range(0, grid.ncols - tile_cells + 1, tile_cells):
            if ok_both[r : r + tile_cells, c : c + tile_cells].mean() < 0.97:
                continue
            x0 = grid.x0 + c * grid.cell
            y1 = grid.y1 - r * grid.cell
            box = (x0, y1 - tile_cells * grid.cell, x0 + tile_cells * grid.cell, y1)
            a = post.window(*box)
            b = other.window(*box)
            if a.shape != b.shape or min(a.shape[:2]) < 64:
                continue
            g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
            score = float(np.mean(np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1))))
            cands.append((score, box, a, b))
    cands.sort(key=lambda t: -t[0])
    rows = []
    for score, box, a, b in cands[:n_tiles]:
        sx, sy, resp = phase_shift_px(grad_image(a), grad_image(b))
        # b(u) = a(u - s): o detalhe no pixel u de post aparece em u + s na outra data.
        rows.append(dict(box=box, score=score, dx_m=sx * post.res, dy_m=-sy * post.res, resp=resp))
    good = [r for r in rows if r["resp"] >= min_resp]
    use = good if len(good) >= 5 else rows
    d = np.array([[r["dx_m"], r["dy_m"]] for r in use])
    med = np.median(d, axis=0)
    mad = np.median(np.abs(d - med), axis=0)
    return dict(dx_m=float(med[0]), dy_m=float(med[1]), norm_m=float(np.hypot(*med)), mad_m=mad.tolist(),
                n_tiles=len(rows), n_used=len(use), resp_median=float(np.median([r["resp"] for r in rows])),
                tiles=rows)


def run(cfg: dict) -> dict:
    d = cfg["data"]
    grid = Grid.from_aoi(**d["aoi"], cell=d["cell_m"])
    masks = load_masks()
    post = load_date("post")
    out = {"post": dict(dx_m=0.0, dy_m=0.0, norm_m=0.0, applied=False)}
    for date in d["sources"]:
        if date == "post":
            continue
        other = load_date(date)
        est = estimate_offset(post, other, masks["post_ok"] & masks[f"{date}_ok"], grid, n_tiles=d["align_tiles"])
        est["applied"] = bool(est["norm_m"] > d["align_apply_min_m"])
        tiles = est.pop("tiles")
        out[date] = est
        print(f"[align] {date}: dx={est['dx_m']:.2f} m dy={est['dy_m']:.2f} m |d|={est['norm_m']:.2f} m "
              f"MAD={np.round(est['mad_m'], 2).tolist()} used {est['n_used']}/{est['n_tiles']} tiles "
              f"(median resp {est['resp_median']:.3f}) applied={est['applied']}")
        (DATA / f"align_tiles_{date}.json").write_text(json.dumps(tiles, indent=1))
        del other
    (DATA / "align.json").write_text(json.dumps(out, indent=1))
    return out


def load_offsets() -> dict[str, tuple[float, float]]:
    """Deslocamento a aplicar ao amostrar cada data (zero quando fica abaixo do limiar de aplicação)."""
    a = json.loads((DATA / "align.json").read_text())
    return {k: ((v["dx_m"], v["dy_m"]) if v.get("applied") else (0.0, 0.0)) for k, v in a.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    run(load_config(ap.parse_args().config))


if __name__ == "__main__":
    main()

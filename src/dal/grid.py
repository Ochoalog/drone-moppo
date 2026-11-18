"""Grade de células de 50 m sobre a área de estudo (UTM 22S) e a divisão norte/sul entre treino e teste."""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from pyproj import Transformer

CRS = "EPSG:32722"
_TO_UTM = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
_TO_LL = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)


def lonlat_to_utm(lon, lat):
    """Converte lon/lat (WGS84) para UTM 22S."""
    return _TO_UTM.transform(lon, lat)


def utm_to_lonlat(x, y):
    """Converte UTM 22S para lon/lat (WGS84)."""
    return _TO_LL.transform(x, y)


@dataclass(frozen=True)
class Grid:
    """Grade regular em UTM. A linha 0 é a mais ao norte; a célula (r, c) cobre
    x em [x0 + c*cell, x0 + (c+1)*cell] e y em [y1 - (r+1)*cell, y1 - r*cell]."""

    x0: float
    y1: float
    nrows: int
    ncols: int
    cell: float = 50.0

    @classmethod
    def from_aoi(cls, lon0: float, lon1: float, lat0: float, lat1: float, cell: float = 50.0) -> "Grid":
        """Maior retângulo UTM alinhado às células que cabe inteiro dentro da AOI em lon/lat."""
        xs, ys = lonlat_to_utm(np.array([lon0, lon1, lon0, lon1]), np.array([lat0, lat0, lat1, lat1]))
        xl = np.ceil(max(xs[0], xs[2]) / cell) * cell
        xr = np.floor(min(xs[1], xs[3]) / cell) * cell
        yb = np.ceil(max(ys[0], ys[1]) / cell) * cell
        yt = np.floor(min(ys[2], ys[3]) / cell) * cell
        return cls(float(xl), float(yt), int(round((yt - yb) / cell)), int(round((xr - xl) / cell)), cell)

    @property
    def x1(self) -> float:
        return self.x0 + self.ncols * self.cell

    @property
    def y0(self) -> float:
        return self.y1 - self.nrows * self.cell

    def center(self, r, c):
        """Centro UTM de uma ou mais células."""
        r = np.asarray(r, dtype=np.float64)
        c = np.asarray(c, dtype=np.float64)
        return self.x0 + (c + 0.5) * self.cell, self.y1 - (r + 0.5) * self.cell

    @cached_property
    def centers(self) -> tuple[np.ndarray, np.ndarray]:
        """Centros UTM de todas as células, como duas matrizes (nrows, ncols)."""
        rr, cc = np.mgrid[0 : self.nrows, 0 : self.ncols]
        return self.center(rr, cc)

    @cached_property
    def lat(self) -> np.ndarray:
        """Latitude do centro de cada célula, (nrows, ncols)."""
        x, y = self.centers
        return np.asarray(utm_to_lonlat(x, y)[1])

    def split_rows(self, split_lat: float, margin_m: float) -> tuple[np.ndarray, np.ndarray]:
        """Linhas de treino (célula inteira + margem estritamente ao norte de `split_lat`) e de teste
        (estritamente ao sul). As linhas no meio formam uma faixa de separação, contra vazamento."""
        xs = np.linspace(self.x0, self.x1, 64)
        lons, _ = utm_to_lonlat(xs, np.full_like(xs, (self.y0 + self.y1) / 2))
        _, ys = lonlat_to_utm(np.asarray(lons), np.full_like(xs, split_lat))
        y_max, y_min = float(np.max(ys)), float(np.min(ys))
        r = np.arange(self.nrows)
        top = self.y1 - r * self.cell
        bottom = top - self.cell
        train = r[bottom - margin_m > y_max]
        test = r[top + margin_m < y_min]
        return train, test

    def to_dict(self) -> dict:
        """Parâmetros da grade, para serializar em JSON."""
        return dict(x0=self.x0, y1=self.y1, nrows=self.nrows, ncols=self.ncols, cell=self.cell)

"""Histogram (grid) Bayes filter over a W x W episode window.

Belief is float64, always normalized. Motion: deterministic shift (moves that would leave the
window are blocked -> mass stays, mirroring the true dynamics) followed by a Gaussian blur
(odometry noise). Observation: p(z | c) ∝ exp(beta * cos(z, m_c)); cells whose map is invalid get
the mean likelihood of the valid cells (uninformative), and an invalid observation leaves the
belief unchanged.
"""
from __future__ import annotations

import numpy as np

# (d_row, d_col); row 0 is north.
ACTIONS: tuple[tuple[int, int], ...] = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))
ACTION_NAMES = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


def gaussian_kernel1d(sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0])
    r = max(1, int(np.ceil(3 * sigma)))
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def blur(b: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Separable convolution with symmetric reflection at the borders (mass preserving)."""
    if len(k) == 1:
        return b
    r = len(k) // 2
    p = np.pad(b, ((r, r), (0, 0)), mode="symmetric")
    out = sum(k[i] * p[i : i + b.shape[0]] for i in range(len(k)))
    p = np.pad(out, ((0, 0), (r, r)), mode="symmetric")
    return sum(k[i] * p[:, i : i + b.shape[1]] for i in range(len(k)))


def shift(b: np.ndarray, d: tuple[int, int]) -> np.ndarray:
    """Move mass by d; cells whose move would leave the grid keep their mass (blocked)."""
    dr, dc = d
    H, W = b.shape
    out = np.zeros_like(b)
    rs = slice(max(0, -dr), min(H, H - dr))  # sources that stay inside
    cs = slice(max(0, -dc), min(W, W - dc))
    rd = slice(rs.start + dr, rs.stop + dr)
    cd = slice(cs.start + dc, cs.stop + dc)
    out[rd, cd] += b[rs, cs]
    blocked = np.ones_like(b, dtype=bool)
    blocked[rs, cs] = False
    out[blocked] += b[blocked]
    return out


def entropy(b: np.ndarray) -> float:
    p = b[b > 0]
    return float(-(p * np.log(p)).sum())


def disk_offsets(radius: float) -> np.ndarray:
    r = int(np.floor(radius))
    dr, dc = np.mgrid[-r : r + 1, -r : r + 1]
    m = dr**2 + dc**2 <= radius**2 + 1e-9
    return np.stack([dr[m], dc[m]], axis=1)


class HistogramFilter:
    """Belief over a window given map descriptors `map_desc` (W, W, D) and validity mask."""

    def __init__(self, map_desc: np.ndarray, map_valid: np.ndarray, beta: float, blur_sigma: float = 0.5,
                 success_radius: float = 2.0):
        self.W = map_valid.shape[0]
        self.M = np.ascontiguousarray(map_desc.reshape(-1, map_desc.shape[-1]), dtype=np.float32)
        self.valid = map_valid.astype(bool)
        self.valid_flat = self.valid.ravel()
        self.beta = float(beta)
        self.kernel = gaussian_kernel1d(blur_sigma)
        self.disk = disk_offsets(success_radius)
        self.reset()

    # ------------------------------------------------------------------ core
    def reset(self) -> None:
        v = self.valid.astype(np.float64)
        self.b = v / v.sum() if v.sum() > 0 else np.full(v.shape, 1.0 / v.size)

    def predict(self, action: int, b: np.ndarray | None = None) -> np.ndarray:
        out = blur(shift(self.b if b is None else b, ACTIONS[action]), self.kernel)
        out = out / out.sum()
        if b is None:
            self.b = out
        return out

    def likelihood(self, z: np.ndarray) -> np.ndarray:
        """(W, W) likelihoods (max 1) for one or a batch (n, D) of descriptors -> (n, W, W)."""
        z = np.asarray(z, dtype=np.float32)
        single = z.ndim == 1
        s = (np.atleast_2d(z) @ self.M.T).astype(np.float64)  # (n, W*W) cosine similarities
        s[:, ~self.valid_flat] = -np.inf
        s -= s.max(axis=1, keepdims=True)
        L = np.exp(self.beta * s)
        if (~self.valid_flat).any():
            L[:, ~self.valid_flat] = L[:, self.valid_flat].mean(axis=1, keepdims=True)
        L = L.reshape(-1, self.W, self.W)
        return L[0] if single else L

    def update(self, z: np.ndarray | None, obs_valid: bool = True) -> None:
        if z is None or not obs_valid:
            return
        b = self.b * self.likelihood(z)
        s = b.sum()
        self.b = b / s if s > 0 and np.isfinite(s) else self.b

    # ------------------------------------------------------------------ summaries
    def entropy(self) -> float:
        return entropy(self.b)

    def map_estimate(self) -> tuple[int, int]:
        i = int(np.argmax(self.b))
        return i // self.W, i % self.W

    def mass_around(self, rc: tuple[int, int] | None = None) -> float:
        r, c = self.map_estimate() if rc is None else rc
        rr = self.disk[:, 0] + r
        cc = self.disk[:, 1] + c
        m = (rr >= 0) & (rr < self.W) & (cc >= 0) & (cc < self.W)
        return float(self.b[rr[m], cc[m]].sum())

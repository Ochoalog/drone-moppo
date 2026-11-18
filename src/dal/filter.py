"""Filtro de Bayes por histograma (grade) sobre a janela W x W de um episódio.

A crença é float64 e está sempre normalizada.
- Movimento: deslocamento determinístico seguido de um borramento gaussiano (ruído de odometria).
  Um movimento que sairia da janela é bloqueado e a massa fica parada, como na dinâmica real.
- Observação: p(z | c) ∝ exp(beta * cos(z, m_c)). Células com mapa inválido recebem a
  verossimilhança média das células válidas (não informativa), e uma observação inválida deixa a
  crença inalterada.
"""
from __future__ import annotations

import numpy as np

# (d_linha, d_coluna); a linha 0 é o norte.
ACTIONS: tuple[tuple[int, int], ...] = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))
ACTION_NAMES = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


def gaussian_kernel1d(sigma: float) -> np.ndarray:
    """Núcleo gaussiano 1D normalizado, truncado em 3 sigma."""
    if sigma <= 0:
        return np.array([1.0])
    radius = max(1, int(np.ceil(3 * sigma)))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    return kernel / kernel.sum()


def blur(belief: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolução separável com reflexão simétrica nas bordas (preserva a massa)."""
    if len(kernel) == 1:
        return belief
    radius = len(kernel) // 2
    padded = np.pad(belief, ((radius, radius), (0, 0)), mode="symmetric")
    out = sum(kernel[i] * padded[i : i + belief.shape[0]] for i in range(len(kernel)))
    padded = np.pad(out, ((0, 0), (radius, radius)), mode="symmetric")
    return sum(kernel[i] * padded[:, i : i + belief.shape[1]] for i in range(len(kernel)))


def shift(belief: np.ndarray, delta: tuple[int, int]) -> np.ndarray:
    """Desloca a massa por `delta`; células cujo movimento sairia da grade mantêm a massa (bloqueio)."""
    d_row, d_col = delta
    H, W = belief.shape
    out = np.zeros_like(belief)
    # Origens que continuam dentro da grade depois do deslocamento.
    src_rows = slice(max(0, -d_row), min(H, H - d_row))
    src_cols = slice(max(0, -d_col), min(W, W - d_col))
    dst_rows = slice(src_rows.start + d_row, src_rows.stop + d_row)
    dst_cols = slice(src_cols.start + d_col, src_cols.stop + d_col)
    out[dst_rows, dst_cols] += belief[src_rows, src_cols]
    blocked = np.ones_like(belief, dtype=bool)
    blocked[src_rows, src_cols] = False
    out[blocked] += belief[blocked]
    return out


def entropy(belief: np.ndarray) -> float:
    """Entropia de Shannon (nats) da crença."""
    positive = belief[belief > 0]
    return float(-(positive * np.log(positive)).sum())


def disk_offsets(radius: float) -> np.ndarray:
    """Deslocamentos (d_linha, d_coluna) de todas as células dentro de um disco de raio `radius`."""
    int_radius = int(np.floor(radius))
    d_rows, d_cols = np.mgrid[-int_radius : int_radius + 1, -int_radius : int_radius + 1]
    inside = d_rows**2 + d_cols**2 <= radius**2 + 1e-9
    return np.stack([d_rows[inside], d_cols[inside]], axis=1)


class HistogramFilter:
    """Crença sobre a janela, dados os descritores do mapa `map_desc` (W, W, D) e a máscara de validade.

    O atributo `b` guarda a crença (W, W); o nome curto é mantido porque outros módulos o acessam.
    """

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

    # ------------------------------------------------------------------ núcleo
    def reset(self) -> None:
        """Crença inicial uniforme sobre as células válidas do mapa."""
        valid = self.valid.astype(np.float64)
        self.b = valid / valid.sum() if valid.sum() > 0 else np.full(valid.shape, 1.0 / valid.size)

    def predict(self, action: int, b: np.ndarray | None = None) -> np.ndarray:
        """Passo de predição. Sem `b`, atualiza a crença interna; com `b`, só calcula (usado pela gulosa)."""
        out = blur(shift(self.b if b is None else b, ACTIONS[action]), self.kernel)
        out = out / out.sum()
        if b is None:
            self.b = out
        return out

    def likelihood(self, z: np.ndarray) -> np.ndarray:
        """Verossimilhanças (máximo 1) para um descritor (W, W) ou um lote (n, D) -> (n, W, W)."""
        z = np.asarray(z, dtype=np.float32)
        single = z.ndim == 1
        similarity = (np.atleast_2d(z) @ self.M.T).astype(np.float64)  # (n, W*W) similaridades de cosseno
        similarity[:, ~self.valid_flat] = -np.inf
        # Subtrai o máximo antes da exponencial para evitar overflow.
        similarity -= similarity.max(axis=1, keepdims=True)
        L = np.exp(self.beta * similarity)
        if (~self.valid_flat).any():
            L[:, ~self.valid_flat] = L[:, self.valid_flat].mean(axis=1, keepdims=True)
        L = L.reshape(-1, self.W, self.W)
        return L[0] if single else L

    def update(self, z: np.ndarray | None, obs_valid: bool = True) -> None:
        """Passo de correção bayesiana; observação ausente ou inválida não muda a crença."""
        if z is None or not obs_valid:
            return
        posterior = self.b * self.likelihood(z)
        total = posterior.sum()
        self.b = posterior / total if total > 0 and np.isfinite(total) else self.b

    # ------------------------------------------------------------------ resumos
    def entropy(self) -> float:
        """Entropia da crença atual."""
        return entropy(self.b)

    def map_estimate(self) -> tuple[int, int]:
        """Célula de maior probabilidade (estimativa MAP)."""
        index = int(np.argmax(self.b))
        return index // self.W, index % self.W

    def mass_around(self, rc: tuple[int, int] | None = None) -> float:
        """Massa da crença no disco de sucesso em torno de `rc` (padrão: a estimativa MAP)."""
        row, col = self.map_estimate() if rc is None else rc
        rows = self.disk[:, 0] + row
        cols = self.disk[:, 1] + col
        inside = (rows >= 0) & (rows < self.W) & (cols >= 0) & (cols < self.W)
        return float(self.b[rows[inside], cols[inside]].sum())

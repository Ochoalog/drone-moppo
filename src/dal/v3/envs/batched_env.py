"""Ambiente de localização ativa em lote (N episódios como tensores PyTorch; CPU ou GPU).

Modo legado (gate G2): reproduz `dal.env.Episode` da v1 — grade de 50 m, descritor da v1, 8 ações,
deslizamento com probabilidade `slip_prob`, filtro de histograma com borramento gaussiano e
verossimilhança temperada. Para as trajetórias saírem **idênticas**, cada episódio mantém o próprio
gerador numpy, consumido na mesma ordem da v1 (o sorteio do deslizamento só acontece quando o
movimento não é bloqueado, e a variante de observação é sorteada a cada passo). Só esses sorteios
ficam fora do lote; filtro, verossimilhança, predição, concentração e entropia são operações em lote.

A verossimilhança é calculada contra a grade inteira (N × células) e depois recolhida para as células
da janela de cada episódio: assim os descritores de cada janela não precisam ser copiados.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from dal.filter import ACTIONS, disk_offsets, gaussian_kernel1d
from dal.v3.localization.ca_filter import change_aware_likelihood

ACTION_DELTAS = torch.tensor(ACTIONS, dtype=torch.long)


@dataclass
class StepOutcome:
    """Resultado de um passo em lote (tensores (N,); episódios já terminados ficam inalterados)."""

    blocked: torch.Tensor
    success: torch.Tensor
    false_loc: torch.Tensor
    concentrated: torch.Tensor
    entropy: torch.Tensor
    error_m: torch.Tensor


class ScenarioTensors:
    """Dados de um cenário na grade inteira, como tensores no dispositivo de destino."""

    def __init__(self, data, device: str | torch.device, dtype: torch.dtype):
        self.device, self.dtype = torch.device(device), dtype
        rows, cols, dim = data.map_desc.shape
        self.shape = (rows, cols)
        # Descritores em float32, como a v1 (o produto z·M é float32 e só depois vira float64).
        self.map_desc = torch.as_tensor(np.asarray(data.map_desc, np.float32).reshape(rows * cols, dim), device=device)
        self.obs_desc = torch.as_tensor(np.asarray(data.obs_desc, np.float32).reshape(data.obs_desc.shape[0],
                                                                                      rows * cols, dim), device=device)
        self.map_ok = torch.as_tensor(np.asarray(data.map_ok), device=device)
        self.obs_ok = torch.as_tensor(np.asarray(data.obs_ok), device=device)
        self.distinct = torch.as_tensor(np.asarray(data.distinct, np.float32), device=device)
        self.pca = torch.as_tensor(np.asarray(data.pca, np.float32), device=device)
        self.beta = float(data.beta)
        self.n_variants = int(data.obs_desc.shape[0])


def _pad_symmetric(x: torch.Tensor, radius: int, dim: int) -> torch.Tensor:
    """Padding simétrico (repete a borda, como `np.pad(mode='symmetric')`) ao longo de `dim`."""
    size = x.shape[dim]
    head = x.narrow(dim, 0, radius).flip(dim)
    tail = x.narrow(dim, size - radius, radius).flip(dim)
    return torch.cat([head, x, tail], dim=dim)


class BatchedLocalizationEnv:
    """N episódios de localização em paralelo (modo legado: idêntico à v1)."""

    def __init__(self, data, filter_cfg: dict, split: str = "test", cell_m: float = 50.0,
                 device: str | torch.device = "cpu", dtype: torch.dtype = torch.float64,
                 change_prior=None, alpha: float = 0.0):
        """`change_prior` (grade inteira, valores em [0, 1]) e `alpha` ativam o filtro CA; α = 0 é o filtro padrão."""
        self.scenario = ScenarioTensors(data, device, dtype)
        self.alpha = float(alpha)
        self.change_prior = (None if change_prior is None
                             else torch.as_tensor(change_prior, dtype=dtype, device=device).reshape(-1))
        if self.alpha > 0 and self.change_prior is None:
            raise ValueError("alpha > 0 exige um change_prior")
        self.device, self.dtype = torch.device(device), dtype
        self.cfg = filter_cfg
        self.W = int(filter_cfg["window"])
        self.cell_m = float(cell_m)
        self.max_steps = int(filter_cfg["max_steps"])
        self.windows = data.windows(split, self.W, filter_cfg["min_window_valid"])
        if len(self.windows) == 0:
            raise ValueError(f"nenhuma janela válida {self.W}x{self.W} na divisão {split}")
        kernel = gaussian_kernel1d(filter_cfg["blur_sigma"])
        self.kernel = torch.as_tensor(kernel, dtype=dtype, device=device)
        self.disk = torch.as_tensor(disk_offsets(filter_cfg["success_radius"]), device=device)
        self.success_mass = float(filter_cfg["success_mass"])
        self.slip_prob = float(filter_cfg["slip_prob"])
        self.success_radius_m = float(filter_cfg["success_radius"]) * self.cell_m + 1e-6

    # ------------------------------------------------------------------ reinício
    def reset(self, seeds: list[int]) -> None:
        """Reinicia N episódios com os sorteios de `Episode.reset(seed=...)` da v1."""
        n, W = len(seeds), self.W
        windows, starts, self.rngs = [], [], []
        map_ok_np = self.scenario.map_ok.cpu().numpy()
        obs_ok_np = self.scenario.obs_ok.cpu().numpy()
        for seed in seeds:
            rng = np.random.default_rng(seed)
            init_rng = np.random.default_rng(rng.integers(2**63))
            self.rngs.append(np.random.default_rng(rng.integers(2**63)))
            r0, c0 = self.windows[init_rng.integers(len(self.windows))]
            candidates = np.argwhere(map_ok_np[r0:r0 + W, c0:c0 + W] & obs_ok_np[r0:r0 + W, c0:c0 + W])
            windows.append((int(r0), int(c0)))
            starts.append(tuple(int(v) for v in candidates[init_rng.integers(len(candidates))]))
        device = self.device
        self.n = n
        self.origin = torch.tensor(windows, dtype=torch.long, device=device)  # (N, 2)
        self.pos = torch.tensor(starts, dtype=torch.long, device=device)
        self.window_cells = self._window_cell_index()  # (N, W*W) índices na grade inteira
        rows, cols = self.scenario.shape
        flat_map_ok = self.scenario.map_ok.reshape(-1)
        flat_obs_ok = self.scenario.obs_ok.reshape(-1)
        self.map_ok = flat_map_ok[self.window_cells].view(n, W, W)
        self.obs_ok = flat_obs_ok[self.window_cells].view(n, W, W)
        valid = self.map_ok.to(self.dtype)
        totals = valid.sum(dim=(1, 2), keepdim=True)
        self.belief = torch.where(totals > 0, valid / totals.clamp(min=1), torch.full_like(valid, 1.0 / (W * W)))
        self.t = torch.zeros(n, dtype=torch.long, device=device)
        self.distance_m = torch.zeros(n, dtype=torch.float64, device=device)
        self.n_blocked = torch.zeros(n, dtype=torch.long, device=device)
        self.blocked_axes = torch.zeros(n, 2, dtype=torch.bool, device=device)
        self.done = torch.zeros(n, dtype=torch.bool, device=device)
        self.trajectories = [[tuple(s)] for s in starts]
        self._observe(torch.ones(n, dtype=torch.bool, device=device))
        self.last = StepOutcome(*(torch.zeros(n, dtype=torch.bool, device=device) for _ in range(4)),
                                self.entropy(), self.error_m())

    def _window_cell_index(self) -> torch.Tensor:
        W, (_, cols) = self.W, self.scenario.shape
        local = torch.arange(W, device=self.device)
        rows_idx = (self.origin[:, 0:1] + local.view(1, -1)).view(-1, W, 1)
        cols_idx = (self.origin[:, 1:2] + local.view(1, -1)).view(-1, 1, W)
        return (rows_idx * cols + cols_idx).view(-1, W * W)

    # ------------------------------------------------------------------ filtro
    def _likelihood(self, variant: torch.Tensor) -> torch.Tensor:
        """Verossimilhança (N, W, W) da observação na posição atual, como `HistogramFilter.likelihood`."""
        n, W = self.n, self.W
        cell = self.window_cells.gather(1, (self.pos[:, 0] * W + self.pos[:, 1]).view(-1, 1)).squeeze(1)
        z = self.scenario.obs_desc[variant, cell]  # (N, D) float32
        # Similaridade contra a grade inteira (N × células) e depois só as células da janela de cada episódio.
        sim = (z @ self.scenario.map_desc.T).gather(1, self.window_cells).to(self.dtype)
        valid = self.map_ok.view(n, -1)
        sim = sim.masked_fill(~valid, float("-inf"))
        sim = sim - sim.max(dim=1, keepdim=True).values
        likelihood = torch.exp(self.scenario.beta * sim)
        valid_count = valid.sum(dim=1, keepdim=True).clamp(min=1)
        mean_valid = torch.where(valid, likelihood, torch.zeros_like(likelihood)).sum(1, keepdim=True) / valid_count
        likelihood = torch.where(valid, likelihood, mean_valid.expand_as(likelihood))
        if self.alpha > 0:
            prior = self.change_prior[self.window_cells]
            likelihood = change_aware_likelihood(likelihood, prior, self.alpha, valid)
        return likelihood.view(n, W, W)

    def _observe(self, active: torch.Tensor) -> None:
        """Sorteia a variante (gerador de cada episódio) e aplica a correção bayesiana aos ativos."""
        variants = torch.zeros(self.n, dtype=torch.long, device=self.device)
        for i in torch.nonzero(active).flatten().tolist():
            variants[i] = int(self.rngs[i].integers(self.scenario.n_variants))
        r, c = self.pos[:, 0], self.pos[:, 1]
        obs_valid = self.obs_ok[torch.arange(self.n, device=self.device), r, c] & active
        if not bool(obs_valid.any()):
            return
        posterior = self.belief * self._likelihood(variants)
        total = posterior.sum(dim=(1, 2), keepdim=True)
        ok = obs_valid.view(-1, 1, 1) & (total > 0) & torch.isfinite(total)
        self.belief = torch.where(ok, posterior / torch.where(ok, total, torch.ones_like(total)), self.belief)

    def _predict(self, actions: torch.Tensor, active: torch.Tensor) -> None:
        """Deslocamento com bloqueio na borda + borramento gaussiano separável (simétrico) + normalização."""
        shifted = self.belief.clone()
        W = self.W
        for a, (d_row, d_col) in enumerate(ACTIONS):
            idx = torch.nonzero(active & (actions == a)).flatten()
            if idx.numel() == 0:
                continue
            b = self.belief[idx]
            out = torch.zeros_like(b)
            src_r = slice(max(0, -d_row), min(W, W - d_row))
            src_c = slice(max(0, -d_col), min(W, W - d_col))
            dst_r = slice(src_r.start + d_row, src_r.stop + d_row)
            dst_c = slice(src_c.start + d_col, src_c.stop + d_col)
            out[:, dst_r, dst_c] += b[:, src_r, src_c]
            stays = torch.ones(W, W, dtype=torch.bool, device=self.device)
            stays[src_r, src_c] = False
            out[:, stays] += b[:, stays]
            shifted[idx] = out
        blurred = self._blur(shifted)
        blurred = blurred / blurred.sum(dim=(1, 2), keepdim=True)
        self.belief = torch.where(active.view(-1, 1, 1), blurred, self.belief)

    def _blur(self, belief: torch.Tensor) -> torch.Tensor:
        k = self.kernel
        if k.numel() == 1:
            return belief
        radius = k.numel() // 2
        W = self.W
        padded = _pad_symmetric(belief, radius, 1)
        out = sum(k[i] * padded[:, i:i + W, :] for i in range(k.numel()))
        padded = _pad_symmetric(out, radius, 2)
        return sum(k[i] * padded[:, :, i:i + W] for i in range(k.numel()))

    # ------------------------------------------------------------------ resumos
    def entropy(self) -> torch.Tensor:
        b = self.belief.view(self.n, -1)
        return -(torch.where(b > 0, b * torch.log(b.clamp(min=1e-300)), torch.zeros_like(b))).sum(1)

    def map_estimate(self) -> torch.Tensor:
        """Célula MAP (N, 2); empate → primeiro índice, como `np.argmax`."""
        index = self.belief.view(self.n, -1).argmax(dim=1)
        return torch.stack([index // self.W, index % self.W], dim=1)

    def error_m(self) -> torch.Tensor:
        diff = (self.map_estimate() - self.pos).to(torch.float64)
        return torch.hypot(diff[:, 0], diff[:, 1]) * self.cell_m

    def mass_around_map(self) -> torch.Tensor:
        center = self.map_estimate()
        cells = center.view(-1, 1, 2) + self.disk.view(1, -1, 2)
        inside = (cells >= 0).all(-1) & (cells < self.W).all(-1)
        clamped = cells.clamp(0, self.W - 1)
        values = self.belief[torch.arange(self.n, device=self.device).view(-1, 1), clamped[..., 0], clamped[..., 1]]
        return (values * inside).sum(1)

    # ------------------------------------------------------------------ passo
    def step(self, actions: torch.Tensor) -> StepOutcome:
        """Um passo para todos os episódios ainda ativos (os terminados ficam parados)."""
        actions = actions.to(self.device).long()
        active = ~self.done
        W = self.W
        deltas = ACTION_DELTAS.to(self.device)[actions]
        target = self.pos + deltas
        blocked_axes = (target < 0) | (target >= W)
        blocked = blocked_axes.any(1) & active
        new_pos = torch.where(blocked.view(-1, 1), self.pos, target)
        for i in torch.nonzero(active & ~blocked).flatten().tolist():
            rng = self.rngs[i]
            if rng.random() < self.slip_prob:
                d_row, d_col = ACTIONS[int(rng.integers(8))]
                new_pos[i, 0] = min(max(int(new_pos[i, 0]) + d_row, 0), W - 1)
                new_pos[i, 1] = min(max(int(new_pos[i, 1]) + d_col, 0), W - 1)
        moved = (new_pos - self.pos).to(torch.float64)
        self.distance_m = torch.where(active, self.distance_m + torch.hypot(moved[:, 0], moved[:, 1]) * self.cell_m,
                                      self.distance_m)
        self.blocked_axes = torch.where(active.view(-1, 1), blocked_axes, self.blocked_axes)
        self.n_blocked += blocked.long()
        self.pos = torch.where(active.view(-1, 1), new_pos, self.pos)
        for i in torch.nonzero(active).flatten().tolist():
            self.trajectories[i].append((int(self.pos[i, 0]), int(self.pos[i, 1])))
        self._predict(actions, active)
        self._observe(active)
        self.t += active.long()
        concentrated = self.mass_around_map() >= self.success_mass
        error = self.error_m()
        success = concentrated & (error <= self.success_radius_m)
        outcome = StepOutcome(blocked, success & active, concentrated & ~success & active, concentrated & active,
                              self.entropy(), error)
        self.last = StepOutcome(*(torch.where(active, new, old) for new, old in
                                  zip(vars(outcome).values(), vars(self.last).values(), strict=True)))
        self.done = self.done | (active & (concentrated | (self.t >= self.max_steps)))
        return outcome

    # ------------------------------------------------------------------ observação da política
    def policy_obs(self) -> dict[str, np.ndarray]:
        """Mesma observação de `Episode.policy_obs` (crença normalizada, máscara, distintividade, PCA)."""
        n, W = self.n, self.W
        belief = self.belief / self.belief.view(n, -1).max(1).values.view(-1, 1, 1)
        distinct = self.scenario.distinct.reshape(-1)[self.window_cells].view(n, 1, W, W)
        pca = self.scenario.pca.reshape(self.scenario.pca.shape[0], -1)[:, self.window_cells]  # (P, N, W*W)
        pca = pca.permute(1, 0, 2).reshape(n, -1, W, W)
        maps = torch.cat([belief.float().unsqueeze(1), self.map_ok.float().unsqueeze(1), distinct, pca], dim=1)
        vec = torch.cat([(1.0 - self.t.double() / self.max_steps).float().unsqueeze(1),
                         self.blocked_axes.float()], dim=1)
        return {"map": maps.cpu().numpy(), "vec": vec.cpu().numpy()}

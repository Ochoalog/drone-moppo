"""Ambiente realista em lote (§3.3–3.5 do plano v3): grade de 25 m, posição contínua, deriva de odometria,
variantes de observação, altitude como ação (opcional), filtro padrão ou ciente da mudança, custo em tempo de voo.

Pareamento: todo o ruído de um episódio (janela, partida, deriva por passo e variante de observação por
passo) é sorteado no reset a partir da semente do episódio e indexado pelo passo. Assim, métodos diferentes
veem exatamente o mesmo ruído no episódio i, e o ruído não depende das ações (números aleatórios comuns).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from dal.filter import disk_offsets
from dal.v3.localization.ca_filter import change_aware_likelihood

HORIZONTAL_DIRECTIONS = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))  # (linha, coluna)
ACTION_UP, ACTION_DOWN = 8, 9
HORIZONTAL_STEP_TIME_S = 5.0  # 50 m a 10 m/s
CAPTURE_TIME_S = 1.0
ALTITUDE_CHANGE_TIME_S = 15.0


@dataclass
class RealisticConfig:
    """Parâmetros do ambiente realista (valores padrão do plano v3)."""

    cell_m: float = 25.0
    window_cells: int = 128  # 3,2 km
    step_m: float = 50.0
    drift_fraction: float = 0.02
    base_blur_cells: float = 0.5
    max_steps: int = 80
    success_mass: float = 0.9
    success_radius_m: float = 100.0
    min_window_valid: float = 0.7
    altitudes_m: tuple[int, ...] = (60, 100, 150)
    start_altitude_index: int = 1
    altitude_actions: bool = False
    alpha: float = 0.0  # filtro CA (0 = padrão)


@dataclass
class PerceptionTables:
    """Descritores e máscaras na grade inteira de 25 m, por altitude (L2-normalizados, float32)."""

    map_desc: dict[int, np.ndarray]  # altitude -> (células, D)
    obs_desc: dict[int, np.ndarray]  # altitude -> (K, células, D)
    map_ok: np.ndarray  # (linhas, colunas) bool
    obs_ok: np.ndarray
    beta: dict[int, float]  # altitude -> β (já com κ)
    grid_shape: tuple[int, int]
    rows_allowed: np.ndarray  # linhas da faixa (dobra) em que as janelas podem ficar
    change_prior: np.ndarray | None = None  # (linhas, colunas) em [0, 1]
    extra_channels: dict[str, np.ndarray] = field(default_factory=dict)  # canais do mapa para a política


def valid_windows(tables: PerceptionTables, window: int, min_valid: float) -> np.ndarray:
    """Cantos (r0, c0) das janelas inteiras na faixa permitida com fração mínima de células válidas."""
    rows, cols = tables.grid_shape
    ok = (tables.map_ok & tables.obs_ok).astype(np.float64)
    allowed = np.zeros(rows, bool)
    allowed[tables.rows_allowed] = True
    integral = np.pad(ok, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    out = []
    for r0 in range(0, rows - window + 1, max(1, window // 8)):
        if not allowed[r0:r0 + window].all():
            continue
        for c0 in range(0, cols - window + 1, max(1, window // 8)):
            r1, c1 = r0 + window, c0 + window
            total = integral[r1, c1] - integral[r0, c1] - integral[r1, c0] + integral[r0, c0]
            if total / window**2 >= min_valid:
                out.append((r0, c0))
    return np.array(out, dtype=np.int64).reshape(-1, 2)


def sample_drift(rng: np.random.Generator, n_steps: int, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Erro de escala ε e de rumo δθ (rad) por passo, ambos ~ N(0, fraction)."""
    return rng.normal(0.0, fraction, n_steps), rng.normal(0.0, fraction, n_steps)


def apply_drift(command_m: np.ndarray, scale_error: np.ndarray, heading_error: np.ndarray) -> np.ndarray:
    """Deslocamento real = R(δθ)·comando·(1 + ε); `command_m` (..., 2) em (linha, coluna) metros."""
    cos, sin = np.cos(heading_error), np.sin(heading_error)
    scaled = command_m * (1.0 + scale_error)[..., None]
    return np.stack([cos * scaled[..., 0] - sin * scaled[..., 1], sin * scaled[..., 0] + cos * scaled[..., 1]], -1)


class RealisticBatchedEnv:
    """N episódios do ambiente realista, como tensores PyTorch (CPU ou GPU)."""

    def __init__(self, tables: PerceptionTables, config: RealisticConfig, device: str = "cpu",
                 dtype: torch.dtype = torch.float32):
        self.tables, self.cfg = tables, config
        self.device, self.dtype = torch.device(device), dtype
        self.W = config.window_cells
        self.n_actions = 10 if config.altitude_actions else 8
        self.windows = valid_windows(tables, self.W, config.min_window_valid)
        if len(self.windows) == 0:
            raise ValueError("nenhuma janela válida na faixa permitida")
        self.map_desc = {h: torch.as_tensor(v, dtype=torch.float32, device=device) for h, v in tables.map_desc.items()}
        self.obs_desc = {h: torch.as_tensor(v, dtype=torch.float32, device=device) for h, v in tables.obs_desc.items()}
        self.map_ok = torch.as_tensor(tables.map_ok, device=device)
        self.obs_ok = torch.as_tensor(tables.obs_ok, device=device)
        self.prior = (None if tables.change_prior is None
                      else torch.as_tensor(tables.change_prior, dtype=dtype, device=device).reshape(-1))
        if config.alpha > 0 and self.prior is None:
            raise ValueError("alpha > 0 exige change_prior")
        self.disk = torch.as_tensor(disk_offsets(config.success_radius_m / config.cell_m), device=device)
        self.n_variants = next(iter(tables.obs_desc.values())).shape[0]

    # ------------------------------------------------------------------ reinício
    STATE_KEYS = ("origin", "pos_m", "scale_error", "heading_error", "variants", "altitude_index", "window_cells",
                  "win_map_ok", "win_obs_ok", "belief", "t", "flight_time_s", "distance_m", "done", "success",
                  "false_loc", "blocked_axes")

    def _fresh_state(self, seeds: list[int]) -> dict[str, torch.Tensor]:
        """Estado inicial de episódios novos, com todo o ruído do episódio sorteado a partir da semente."""
        cfg, W, n = self.cfg, self.W, len(seeds)
        windows, starts, scale, heading, variants = [], [], [], [], []
        ok_np = self.tables.map_ok & self.tables.obs_ok
        for seed in seeds:
            rng = np.random.default_rng([seed, 31])
            r0, c0 = self.windows[rng.integers(len(self.windows))]
            candidates = np.argwhere(ok_np[r0:r0 + W, c0:c0 + W])
            cell = candidates[rng.integers(len(candidates))]
            windows.append((r0, c0))
            # Posição contínua: um ponto uniforme dentro da célula de partida.
            starts.append((cell + rng.uniform(0, 1, 2)) * cfg.cell_m)
            eps, dtheta = sample_drift(rng, cfg.max_steps, cfg.drift_fraction)
            scale.append(eps)
            heading.append(dtheta)
            variants.append(rng.integers(self.n_variants, size=cfg.max_steps + 1))
        dev = self.device
        origin = torch.tensor(np.array(windows), dtype=torch.long, device=dev)
        local = torch.arange(W, device=dev)
        rows_idx = (origin[:, 0:1] + local.view(1, -1)).view(-1, W, 1)
        cols_idx = (origin[:, 1:2] + local.view(1, -1)).view(-1, 1, W)
        window_cells = (rows_idx * self.tables.grid_shape[1] + cols_idx).view(n, W * W)
        win_map_ok = self.map_ok.reshape(-1)[window_cells].view(n, W, W)
        valid = win_map_ok.to(self.dtype)
        zeros_f = torch.zeros(n, dtype=torch.float64, device=dev)
        zeros_b = torch.zeros(n, dtype=torch.bool, device=dev)
        return {
            "origin": origin, "pos_m": torch.tensor(np.array(starts), dtype=torch.float64, device=dev),
            "scale_error": torch.tensor(np.array(scale), dtype=torch.float64, device=dev),
            "heading_error": torch.tensor(np.array(heading), dtype=torch.float64, device=dev),
            "variants": torch.tensor(np.array(variants), dtype=torch.long, device=dev),
            "altitude_index": torch.full((n,), cfg.start_altitude_index, dtype=torch.long, device=dev),
            "window_cells": window_cells, "win_map_ok": win_map_ok,
            "win_obs_ok": self.obs_ok.reshape(-1)[window_cells].view(n, W, W),
            "belief": valid / valid.sum(dim=(1, 2), keepdim=True).clamp(min=1),
            "t": torch.zeros(n, dtype=torch.long, device=dev), "flight_time_s": zeros_f.clone(),
            "distance_m": zeros_f.clone(), "done": zeros_b.clone(), "success": zeros_b.clone(),
            "false_loc": zeros_b.clone(), "blocked_axes": torch.zeros(n, 2, dtype=torch.bool, device=dev),
        }

    def reset(self, seeds: list[int]) -> None:
        """Reinicia todos os episódios."""
        self.n = len(seeds)
        for key, value in self._fresh_state(seeds).items():
            setattr(self, key, value)
        self._observe(torch.ones(self.n, dtype=torch.bool, device=self.device))

    def reset_rows(self, rows: torch.Tensor, seeds: list[int]) -> None:
        """Reinicia só os episódios das linhas dadas (treino com episódios contínuos)."""
        rows = rows.to(self.device)
        for key, value in self._fresh_state(seeds).items():
            getattr(self, key)[rows] = value
        mask = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        mask[rows] = True
        self._observe(mask)

    # ------------------------------------------------------------------ estado derivado
    def cell(self) -> torch.Tensor:
        """Célula (linha, coluna) da janela que contém o drone."""
        return (self.pos_m / self.cfg.cell_m).floor().long().clamp(0, self.W - 1)

    def altitude_m(self) -> torch.Tensor:
        return torch.tensor(self.cfg.altitudes_m, device=self.device)[self.altitude_index]

    # ------------------------------------------------------------------ filtro
    def _likelihood(self, active: torch.Tensor) -> torch.Tensor:
        n, W = self.n, self.W
        cell = self.cell()
        flat_cell = self.window_cells.gather(1, (cell[:, 0] * W + cell[:, 1]).view(-1, 1)).squeeze(1)
        variant = self.variants.gather(1, self.t.clamp(max=self.cfg.max_steps).view(-1, 1)).squeeze(1)
        sim = torch.zeros(n, W * W, dtype=torch.float32, device=self.device)
        beta = torch.ones(n, 1, dtype=self.dtype, device=self.device)
        for index, altitude in enumerate(self.cfg.altitudes_m):
            rows = torch.nonzero(active & (self.altitude_index == index)).flatten()
            if rows.numel() == 0:
                continue
            z = self.obs_desc[altitude][variant[rows], flat_cell[rows]]
            sim[rows] = (z @ self.map_desc[altitude].T).gather(1, self.window_cells[rows])
            beta[rows] = self.tables.beta[altitude]
        valid = self.win_map_ok.view(n, -1)
        sim = sim.to(self.dtype).masked_fill(~valid, float("-inf"))
        sim = sim - sim.max(dim=1, keepdim=True).values
        likelihood = torch.exp(beta * sim)
        mean_valid = (likelihood * valid).sum(1, keepdim=True) / valid.sum(1, keepdim=True).clamp(min=1)
        likelihood = torch.where(valid, likelihood, mean_valid.expand_as(likelihood))
        if self.cfg.alpha > 0:
            likelihood = change_aware_likelihood(likelihood, self.prior[self.window_cells], self.cfg.alpha, valid)
        return likelihood.view(n, W, W)

    def _observe(self, active: torch.Tensor) -> None:
        cell = self.cell()
        obs_valid = self.win_obs_ok[torch.arange(self.n, device=self.device), cell[:, 0], cell[:, 1]] & active
        if not bool(obs_valid.any()):
            return
        posterior = self.belief * self._likelihood(obs_valid)
        total = posterior.sum(dim=(1, 2), keepdim=True)
        ok = obs_valid.view(-1, 1, 1) & (total > 0) & torch.isfinite(total)
        self.belief = torch.where(ok, posterior / torch.where(ok, total, torch.ones_like(total)), self.belief)

    def _predict(self, command_cells: torch.Tensor, moving: torch.Tensor) -> None:
        """Desloca a crença pelo comando (em células; inteiro) e borra com σ ∝ deslocamento comandado."""
        W = self.W
        shifted = self.belief.clone()
        for d_row, d_col in {tuple(v) for v in command_cells[moving].tolist()}:
            idx = torch.nonzero(moving & (command_cells[:, 0] == d_row) & (command_cells[:, 1] == d_col)).flatten()
            b = self.belief[idx]
            out = torch.zeros_like(b)
            src_r = slice(max(0, -d_row), min(W, W - d_row))
            src_c = slice(max(0, -d_col), min(W, W - d_col))
            out[:, src_r.start + d_row:src_r.stop + d_row, src_c.start + d_col:src_c.stop + d_col] += b[:, src_r, src_c]
            stays = torch.ones(W, W, dtype=torch.bool, device=self.device)
            stays[src_r, src_c] = False
            out[:, stays] += b[:, stays]
            shifted[idx] = out
        moved_m = command_cells.to(self.dtype).norm(dim=1) * self.cfg.cell_m
        sigma = torch.sqrt(self.cfg.base_blur_cells**2 + (self.cfg.drift_fraction * moved_m / self.cfg.cell_m) ** 2)
        blurred = gaussian_blur_batched(shifted, sigma)
        blurred = blurred / blurred.sum(dim=(1, 2), keepdim=True)
        self.belief = torch.where(moving.view(-1, 1, 1), blurred, self.belief)

    # ------------------------------------------------------------------ resumos
    def map_estimate(self) -> torch.Tensor:
        index = self.belief.view(self.n, -1).argmax(dim=1)
        return torch.stack([index // self.W, index % self.W], dim=1)

    def error_m(self) -> torch.Tensor:
        center = (self.map_estimate().to(torch.float64) + 0.5) * self.cfg.cell_m
        return (center - self.pos_m).norm(dim=1)

    def mass_around_map(self) -> torch.Tensor:
        center = self.map_estimate()
        cells = center.view(-1, 1, 2) + self.disk.view(1, -1, 2)
        inside = (cells >= 0).all(-1) & (cells < self.W).all(-1)
        clamped = cells.clamp(0, self.W - 1)
        values = self.belief[torch.arange(self.n, device=self.device).view(-1, 1), clamped[..., 0], clamped[..., 1]]
        return (values * inside).sum(1)

    def entropy(self) -> torch.Tensor:
        b = self.belief.view(self.n, -1)
        return -(b * torch.log(b.clamp(min=1e-30))).sum(1)

    # ------------------------------------------------------------------ passo
    def step(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Aplica as ações aos episódios ativos; devolve sucesso, localização falsa, bloqueio e custo do passo."""
        cfg, W = self.cfg, self.W
        actions = actions.to(self.device).long()
        active = ~self.done
        horizontal = active & (actions < 8)
        vertical = active & (actions >= 8)
        directions = torch.tensor(HORIZONTAL_DIRECTIONS, dtype=torch.long, device=self.device)
        command_cells = torch.zeros(self.n, 2, dtype=torch.long, device=self.device)
        command_cells[horizontal] = directions[actions[horizontal]] * int(round(cfg.step_m / cfg.cell_m))

        step_index = self.t.clamp(max=cfg.max_steps - 1)
        eps = self.scale_error.gather(1, step_index.view(-1, 1)).squeeze(1)
        dtheta = self.heading_error.gather(1, step_index.view(-1, 1)).squeeze(1)
        command_m = command_cells.to(torch.float64) * cfg.cell_m
        cos, sin = torch.cos(dtheta), torch.sin(dtheta)
        scaled = command_m * (1.0 + eps).view(-1, 1)
        actual = torch.stack([cos * scaled[:, 0] - sin * scaled[:, 1], sin * scaled[:, 0] + cos * scaled[:, 1]], 1)
        target = self.pos_m + actual
        limit = W * cfg.cell_m
        blocked_axes = ((target < 0) | (target >= limit)) & horizontal.view(-1, 1)
        blocked = blocked_axes.any(1)
        new_pos = torch.where(blocked.view(-1, 1), self.pos_m, target)
        moved = (new_pos - self.pos_m).norm(dim=1)
        self.distance_m += torch.where(horizontal & ~blocked, moved, torch.zeros_like(moved))
        self.pos_m = torch.where(active.view(-1, 1), new_pos, self.pos_m)
        self.blocked_axes = torch.where(active.view(-1, 1), blocked_axes, self.blocked_axes)

        if cfg.altitude_actions:
            delta = torch.where(actions == ACTION_UP, 1, torch.where(actions == ACTION_DOWN, -1, 0))
            new_level = (self.altitude_index + delta).clamp(0, len(cfg.altitudes_m) - 1)
            self.altitude_index = torch.where(vertical, new_level, self.altitude_index)
        step_time = torch.where(vertical, torch.full_like(self.flight_time_s, ALTITUDE_CHANGE_TIME_S),
                                torch.full_like(self.flight_time_s, HORIZONTAL_STEP_TIME_S)) + CAPTURE_TIME_S
        step_time = torch.where(active, step_time, torch.zeros_like(step_time))
        self.flight_time_s += step_time

        self._predict(torch.where((horizontal & ~blocked).view(-1, 1), command_cells, torch.zeros_like(command_cells)),
                      active)
        self.t += active.long()
        self._observe(active)
        concentrated = (self.mass_around_map() >= cfg.success_mass) & active
        error = self.error_m()
        success = concentrated & (error <= cfg.success_radius_m + 1e-6)
        self.success |= success
        self.false_loc |= concentrated & ~success
        self.done = self.done | concentrated | (active & (self.t >= cfg.max_steps))
        return {"success": success, "false_loc": concentrated & ~success, "blocked": blocked,
                "step_time_s": step_time, "concentrated": concentrated}


    # ------------------------------------------------------------------ observação da política
    def policy_obs(self, include_change_prior: bool = False) -> dict[str, torch.Tensor]:
        """Canais (crença/máx, máscara do mapa, canais extras, [prior m]) e vetor (tempo, bloqueio, altitude)."""
        n, W = self.n, self.W
        belief = self.belief / self.belief.view(n, -1).max(1).values.view(-1, 1, 1)
        channels = [belief.float().unsqueeze(1), self.win_map_ok.float().unsqueeze(1)]
        for values in self.tables.extra_channels.values():
            grid = torch.as_tensor(values, dtype=torch.float32, device=self.device).reshape(-1)
            channels.append(grid[self.window_cells].view(n, 1, W, W))
        if include_change_prior:
            channels.append(self.prior[self.window_cells].float().view(n, 1, W, W))
        altitude = torch.nn.functional.one_hot(self.altitude_index, len(self.cfg.altitudes_m)).float()
        remaining = (1.0 - self.t.double() / self.cfg.max_steps).float().unsqueeze(1)
        vec = torch.cat([remaining, self.blocked_axes.float(), altitude], dim=1)
        return {"map": torch.cat(channels, dim=1), "vec": vec}


def gaussian_blur_batched(belief: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Borramento gaussiano separável com σ por episódio (em células), padding simétrico (preserva a massa)."""
    radius = max(1, int(math.ceil(3 * float(sigma.max()))))
    offsets = torch.arange(-radius, radius + 1, dtype=belief.dtype, device=belief.device)
    kernels = torch.exp(-0.5 * (offsets.view(1, -1) / sigma.view(-1, 1).clamp(min=1e-6)) ** 2)
    kernels = kernels / kernels.sum(1, keepdim=True)
    n, W, _ = belief.shape
    padded = torch.cat([belief[:, :radius].flip(1), belief, belief[:, -radius:].flip(1)], dim=1)
    rows = sum(kernels[:, i].view(-1, 1, 1) * padded[:, i:i + W, :] for i in range(2 * radius + 1))
    padded = torch.cat([rows[:, :, :radius].flip(2), rows, rows[:, :, -radius:].flip(2)], dim=2)
    return sum(kernels[:, i].view(-1, 1, 1) * padded[:, :, i:i + W] for i in range(2 * radius + 1))

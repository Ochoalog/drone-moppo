"""Scenario data, episode simulator (shared by baselines and RL) and the gymnasium env."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np

from .config import DATA, default_config
from .filter import ACTIONS, HistogramFilter

CACHE = DATA / "cache"


# ============================================================================ scenario data
@dataclass
class ScenarioData:
    """Everything an episode needs, over the full grid (row-major cells, row 0 = north)."""

    name: str
    map_desc: np.ndarray  # (R, C, D) L2-normalized (float16/float32, may be memory-mapped)
    obs_desc: np.ndarray  # (K, R, C, D)
    map_ok: np.ndarray  # (R, C) bool
    obs_ok: np.ndarray  # (R, C) bool
    distinct: np.ndarray  # (R, C) float32 in [0, 1]
    pca: np.ndarray  # (P, R, C) float32 in [-1, 1]
    train_rows: np.ndarray
    test_rows: np.ndarray
    beta: float
    meta: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.map_ok.shape

    def rows(self, split: str) -> np.ndarray:
        return {"train": self.train_rows, "test": self.test_rows}[split]

    def windows(self, split: str, size: int, min_valid: float) -> np.ndarray:
        """Top-left corners (r0, c0) of windows fully inside the split with enough valid cells."""
        rows = np.sort(self.rows(split))
        R, C = self.shape
        ok = (self.map_ok & self.obs_ok).astype(np.float64)
        ii = np.pad(ok.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        out = []
        rowset = set(rows.tolist())
        for r0 in range(0, R - size + 1):
            if r0 not in rowset or (r0 + size - 1) not in rowset or rows.searchsorted(r0 + size - 1) - rows.searchsorted(r0) != size - 1:
                continue
            for c0 in range(0, C - size + 1):
                s = ii[r0 + size, c0 + size] - ii[r0, c0 + size] - ii[r0 + size, c0] + ii[r0, c0]
                if s / size**2 >= min_valid:
                    out.append((r0, c0))
        return np.array(out, dtype=np.int64).reshape(-1, 2)

    # ------------------------------------------------------------------ loading
    @classmethod
    def synthetic(cls, R: int = 90, C: int = 70, D: int = 16, K: int = 2, noise: float = 0.3,
                  invalid_frac: float = 0.05, beta: float = 20.0, seed: int = 0) -> "ScenarioData":
        """Small random world for unit tests: spatially smooth descriptors, noisy observations.
        Rows [0, R/2) are train, [R/2 + 4, R) are test."""
        from scipy.ndimage import gaussian_filter

        rng = np.random.default_rng(seed)
        m = gaussian_filter(rng.normal(size=(R, C, D)), sigma=(1.0, 1.0, 0))
        m /= np.linalg.norm(m, axis=-1, keepdims=True)
        o = m[None] + noise * rng.normal(size=(K, R, C, D)) / np.sqrt(D)
        o /= np.linalg.norm(o, axis=-1, keepdims=True)
        map_ok = rng.random((R, C)) > invalid_frac
        obs_ok = rng.random((R, C)) > invalid_frac / 2
        return cls(name="synthetic", map_desc=m.astype(np.float32), obs_desc=o.astype(np.float32),
                   map_ok=map_ok, obs_ok=obs_ok, distinct=distinctiveness(m, map_ok, 3, 5),
                   pca=np.clip(m[..., :4].transpose(2, 0, 1), -1, 1).astype(np.float32),
                   train_rows=np.arange(R // 2), test_rows=np.arange(R // 2 + 4, R), beta=beta,
                   meta=dict(descriptor="synthetic", scenario="SYN", map_date="syn"))

    @classmethod
    def load(cls, descriptor: str, scenario: str, cfg: dict | None = None) -> "ScenarioData":
        cfg = cfg or default_config()
        date = cfg["scenarios"][scenario]
        p = CACHE / descriptor
        if not (p / f"map_{date}.npy").exists():
            build_cache(cfg, descriptor)
        aux = np.load(p / f"aux_{date}.npz")
        masks = np.load(DATA / "masks.npz")
        meta = json.loads((p / "meta.json").read_text())
        beta = cfg["filter"].get("beta")
        if beta is None:
            beta = json.loads((DATA / "beta.json").read_text())[f"{descriptor}_{scenario}"]["beta"]
        return cls(
            name=f"{descriptor}_{scenario}",
            # float32 in RAM once: per-episode float16 -> float32 slicing dominated reset time
            map_desc=np.load(p / f"map_{date}.npy").astype(np.float32),
            obs_desc=np.load(p / "obs_post.npy").astype(np.float32),
            map_ok=masks[f"{date}_ok"],
            obs_ok=masks["post_ok"],
            distinct=aux["distinct"],
            pca=aux["pca"],
            train_rows=np.array(meta["train_rows"]),
            test_rows=np.array(meta["test_rows"]),
            beta=float(beta),
            meta=dict(descriptor=descriptor, scenario=scenario, map_date=date),
        )


def distinctiveness(m: np.ndarray, ok: np.ndarray, r_in: float, r_out: float) -> np.ndarray:
    """1 - max cosine similarity to valid cells in the annulus r_in <= dist <= r_out (cells)."""
    R, C, _ = m.shape
    m = m.astype(np.float32)
    best = np.full((R, C), -1.0, np.float32)
    k = int(np.ceil(r_out))
    for dr in range(-k, k + 1):
        for dc in range(-k, k + 1):
            d = np.hypot(dr, dc)
            if d < r_in or d > r_out:
                continue
            a = (slice(max(0, -dr), min(R, R - dr)), slice(max(0, -dc), min(C, C - dc)))
            b = (slice(max(0, dr), min(R, R + dr)), slice(max(0, dc), min(C, C + dc)))
            s = (m[a] * m[b]).sum(-1)
            s[~ok[b]] = -1.0
            best[a] = np.maximum(best[a], s)
    out = np.clip(1.0 - best, 0.0, 1.0)
    out[~ok] = 0.0
    out[best == -1.0] = 0.0
    return out


def build_cache(cfg: dict, descriptor: str) -> None:
    """Standardized float16 descriptors (converted to float32 on load) + distinctiveness/PCA channels."""
    from .features import load_descriptors
    from .grid import Grid

    d = cfg["data"]
    grid = Grid.from_aoi(**d["aoi"], cell=d["cell_m"])
    train, test = grid.split_rows(d["split_lat"], d["split_margin_m"])
    masks = np.load(DATA / "masks.npz")
    p = CACHE / descriptor
    p.mkdir(parents=True, exist_ok=True)
    r_in, r_out = cfg["env"]["distinct_radius"]
    P = cfg["env"]["pca_components"]
    for date in sorted(set(cfg["scenarios"].values())):
        m, o = load_descriptors(descriptor, date, grid, train)
        np.save(p / f"map_{date}.npy", m.astype(np.float16))
        if date == "post":
            np.save(p / "obs_post.npy", o.astype(np.float16))
        ok = masks[f"{date}_ok"]
        dist = distinctiveness(m, ok, r_in, r_out)
        # PCA fitted on valid TRAIN cells only.
        tr = np.zeros(ok.shape, bool)
        tr[train] = True
        X = m[tr & ok]
        mu = X.mean(0)
        _, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
        sd = S[:P] / np.sqrt(len(X))
        proj = ((m - mu) @ Vt[:P].T) / sd
        pca = np.clip(proj / 3.0, -1, 1).transpose(2, 0, 1).astype(np.float32)
        pca[:, ~ok] = 0.0
        np.savez(p / f"aux_{date}.npz", distinct=dist.astype(np.float32), pca=pca)
        print(f"[cache] {descriptor}/{date}: distinct mean {dist[ok].mean():.3f}, PCA var "
              f"{(S[:P] ** 2 / (S ** 2).sum()).round(3).tolist()}")
    (p / "meta.json").write_text(json.dumps(dict(train_rows=train.tolist(), test_rows=test.tolist())))


# ============================================================================ episode simulator
@dataclass
class StepResult:
    blocked: bool
    success: bool
    false_loc: bool
    concentrated: bool
    entropy: float
    error_m: float


class Episode:
    """One localization episode inside a W x W window: true state, noisy motion, observations
    (lookups into the precomputed table) and the belief filter."""

    def __init__(self, data: ScenarioData, fcfg: dict, split: str = "train", cell_m: float = 50.0):
        self.data = data
        self.cfg = fcfg
        self.W = int(fcfg["window"])
        self.split = split
        self.cell_m = cell_m
        self.windows = data.windows(split, self.W, fcfg["min_window_valid"])
        if len(self.windows) == 0:
            raise ValueError(f"no valid {self.W}x{self.W} windows in split {split}")
        self.max_steps = int(fcfg["max_steps"])

    # ------------------------------------------------------------------ lifecycle
    def reset(self, seed: int | None = None, rng: np.random.Generator | None = None,
              window: tuple[int, int] | None = None, start: tuple[int, int] | None = None) -> None:
        """`seed` fixes window, start and all dynamics noise (paired comparisons across policies)."""
        if rng is None:
            rng = np.random.default_rng(seed)
        init_rng = np.random.default_rng(rng.integers(2**63))
        self.rng = np.random.default_rng(rng.integers(2**63))  # slip / observation-variant noise
        self.r0, self.c0 = window if window is not None else self.windows[init_rng.integers(len(self.windows))]
        W, d = self.W, self.data
        sl = (slice(self.r0, self.r0 + W), slice(self.c0, self.c0 + W))
        self.map_ok = np.asarray(d.map_ok[sl])
        self.obs_ok = np.asarray(d.obs_ok[sl])
        self.map_desc = np.asarray(d.map_desc[sl], dtype=np.float32)
        self.obs_desc = np.asarray(d.obs_desc[:, sl[0], sl[1]], dtype=np.float32)
        self.distinct = np.asarray(d.distinct[sl])
        self.pca = np.asarray(d.pca[:, sl[0], sl[1]])
        if start is None:
            cand = np.argwhere(self.map_ok & self.obs_ok)
            start = tuple(cand[init_rng.integers(len(cand))])
        self.pos = (int(start[0]), int(start[1]))
        self.start = self.pos
        self.filter = HistogramFilter(self.map_desc, self.map_ok, d.beta, self.cfg["blur_sigma"],
                                      self.cfg["success_radius"])
        self.t = 0
        self.distance_m = 0.0
        self.done = False
        self.traj = [self.pos]
        self.n_blocked = 0
        self.blocked_axes = (False, False)
        self.observe()
        self.last = StepResult(False, False, False, False, self.filter.entropy(), self.error_m())

    def observe(self) -> None:
        k = self.rng.integers(self.obs_desc.shape[0])
        r, c = self.pos
        self.filter.update(self.obs_desc[k, r, c], bool(self.obs_ok[r, c]))

    def error_m(self) -> float:
        er, ec = self.filter.map_estimate()
        return float(np.hypot(er - self.pos[0], ec - self.pos[1]) * self.cell_m)

    def step(self, action: int) -> StepResult:
        assert not self.done, "episode finished"
        dr, dc = ACTIONS[int(action)]
        W = self.W
        tr, tc = self.pos[0] + dr, self.pos[1] + dc
        self.blocked_axes = (not 0 <= tr < W, not 0 <= tc < W)  # geofence reports which border
        blocked = any(self.blocked_axes)
        if blocked:
            tr, tc = self.pos
            self.n_blocked += 1
        elif self.rng.random() < self.cfg["slip_prob"]:
            sr, sc = ACTIONS[self.rng.integers(8)]
            tr, tc = int(np.clip(tr + sr, 0, W - 1)), int(np.clip(tc + sc, 0, W - 1))
        self.distance_m += float(np.hypot(tr - self.pos[0], tc - self.pos[1]) * self.cell_m)
        self.pos = (tr, tc)
        self.traj.append(self.pos)
        self.filter.predict(int(action))
        self.observe()
        self.t += 1
        conc = self.filter.mass_around() >= self.cfg["success_mass"]
        err = self.error_m()
        ok = conc and err <= self.cfg["success_radius"] * self.cell_m + 1e-6
        res = StepResult(blocked, ok, conc and not ok, conc, self.filter.entropy(), err)
        self.done = conc or self.t >= self.max_steps
        self.last = res
        return res

    # ------------------------------------------------------------------ policy observation
    def policy_obs(self) -> dict[str, np.ndarray]:
        b = self.filter.b
        bel = (b / b.max()).astype(np.float32)
        chans = [bel[None], self.map_ok[None].astype(np.float32), self.distinct[None].astype(np.float32),
                 self.pca.astype(np.float32)]
        vec = np.array([1.0 - self.t / self.max_steps, *map(float, self.blocked_axes)], np.float32)
        return {"map": np.concatenate(chans, 0), "vec": vec}


# ============================================================================ gymnasium env
class DroneLocEnv(gym.Env):
    """Active localization POMDP; the agent sees its belief + map channels (see `policy_obs`)."""

    metadata = {"render_modes": []}

    def __init__(self, data: ScenarioData, cfg: dict, split: str = "train"):
        super().__init__()
        self.cfg = cfg
        self.ecfg = cfg["env"]
        self.ep = Episode(data, cfg["filter"], split, cfg["data"]["cell_m"])
        W = self.ep.W
        C = 3 + data.pca.shape[0]
        self.observation_space = gym.spaces.Dict({
            "map": gym.spaces.Box(-1.0, 1.0, (C, W, W), np.float32),
            "vec": gym.spaces.Box(0.0, 1.0, (3,), np.float32),
        })
        self.action_space = gym.spaces.Discrete(len(ACTIONS))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        self.ep.reset(rng=self.np_random, window=options.get("window"), start=options.get("start"))
        self.H = self.ep.filter.entropy()
        return self.ep.policy_obs(), self._info()

    def step(self, action):
        e = self.ecfg
        res = self.ep.step(int(action))
        r = (self.H - res.entropy) - e["step_cost"] - e["border_penalty"] * res.blocked
        r += e["success_reward"] * res.success - e["false_loc_penalty"] * res.false_loc
        self.H = res.entropy
        terminated = res.concentrated
        truncated = (not terminated) and self.ep.t >= self.ep.max_steps
        return self.ep.policy_obs(), float(r), terminated, truncated, self._info()

    def _info(self) -> dict:
        ep = self.ep
        return dict(t=ep.t, success=ep.last.success, false_loc=ep.last.false_loc, error_m=ep.last.error_m,
                    entropy=ep.last.entropy, blocked=ep.last.blocked, distance_m=ep.distance_m)


def make_env(descriptor: str, scenario: str, cfg: dict, split: str = "train") -> DroneLocEnv:
    return DroneLocEnv(ScenarioData.load(descriptor, scenario, cfg), cfg, split)

"""Baseline policies. All share the interface `reset(ep, rng)` / `act(ep) -> action index`,
where `ep` is a `dal.env.Episode`. Policies only use what the drone knows: its belief, the map,
the step counter and the geofence signal (`ep.blocked_axes`), never `ep.pos`."""
from __future__ import annotations

import numpy as np

from .filter import ACTIONS, entropy

_IDX = {a: i for i, a in enumerate(ACTIONS)}


class Policy:
    name = "policy"

    def reset(self, ep, rng: np.random.Generator) -> None:
        self.rng = rng

    def act(self, ep) -> int:
        raise NotImplementedError


class RandomPolicy(Policy):
    name = "random"

    def act(self, ep) -> int:
        return int(self.rng.integers(len(ACTIONS)))


class StraightPolicy(Policy):
    """Random fixed heading; reflects the blocked component(s) at the window border."""

    name = "straight"

    def reset(self, ep, rng):
        super().reset(ep, rng)
        self.d = ACTIONS[rng.integers(len(ACTIONS))]

    def act(self, ep) -> int:
        br, bc = ep.blocked_axes
        dr, dc = self.d
        self.d = (-dr if br else dr, -dc if bc else dc)
        return _IDX[self.d]


class LawnmowerPolicy(Policy):
    """Boustrophedon: legs of `leg` cells along a random cardinal axis, `strip` cells between legs."""

    name = "lawnmower"

    def __init__(self, strip: int = 4, leg: int = 16):
        self.strip, self.leg = strip, leg

    def reset(self, ep, rng):
        super().reset(ep, rng)
        horizontal = rng.random() < 0.5
        s1, s2 = rng.choice([-1, 1], 2)
        self.along = (0, int(s1)) if horizontal else (int(s1), 0)
        self.across = (int(s2), 0) if horizontal else (0, int(s2))
        self.phase, self.k = "leg", 0

    def act(self, ep) -> int:
        if any(ep.blocked_axes) and ep.t > 0:
            if self.phase == "leg":  # hit the border along the leg: start the transverse move now
                self.phase, self.k = "across", 0
            else:  # hit the border across: sweep back the other way
                self.across = (-self.across[0], -self.across[1])
        if self.phase == "leg" and self.k >= self.leg:
            self.phase, self.k = "across", 0
        if self.phase == "across" and self.k >= self.strip:
            self.phase, self.k = "leg", 0
            self.along = (-self.along[0], -self.along[1])
        self.k += 1
        return _IDX[self.along if self.phase == "leg" else self.across]


class GreedyInfoGain(Policy):
    """One-step expected entropy reduction, Monte Carlo over M cells sampled from the predicted
    belief. `source="obs"` uses the precomputed `post` observation of each sampled cell (spec;
    privileged when the map is outdated). `source="map"` predicts it with the map descriptor
    (only what a real drone knows)."""

    def __init__(self, samples: int = 32, source: str = "obs"):
        self.M = samples
        self.source = source
        self.name = "greedy" if source == "obs" else "greedy_map"

    def act(self, ep) -> int:
        f = ep.filter
        W = f.W
        preds, zs, oks = [], [], []
        for a in range(len(ACTIONS)):
            bp = f.predict(a, f.b)
            preds.append(bp)
            cells = self.rng.choice(W * W, size=self.M, p=bp.ravel())
            r, c = cells // W, cells % W
            if self.source == "obs":
                k = self.rng.integers(ep.obs_desc.shape[0], size=self.M)
                zs.append(ep.obs_desc[k, r, c])
                oks.append(ep.obs_ok[r, c])
            else:
                zs.append(ep.map_desc[r, c])
                oks.append(ep.map_ok[r, c])
        L = f.likelihood(np.concatenate(zs)).reshape(len(ACTIONS), self.M, W, W)
        exp_h = np.empty(len(ACTIONS))
        for a, bp in enumerate(preds):
            post = bp[None] * L[a]
            post /= post.sum(axis=(1, 2), keepdims=True)
            h = np.array([entropy(p) for p in post])
            h[~oks[a]] = entropy(bp)  # uninformative observation
            exp_h[a] = h.mean()
        best = np.flatnonzero(exp_h <= exp_h.min() + 1e-9)
        return int(self.rng.choice(best))


def make_baselines(cfg: dict) -> dict[str, Policy]:
    m = cfg["greedy"]["samples"]
    return {p.name: p for p in (StraightPolicy(), LawnmowerPolicy(), RandomPolicy(),
                                GreedyInfoGain(m, "obs"), GreedyInfoGain(m, "map"))}

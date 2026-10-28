# DECISIONS_POA (Porto Alegre test battery) — in progress

| # | Decision | Reason | Discarded alternative |
|---|---|---|---|
| 1 | Paths configurable via `paths:` in config and `DAL_*_DIR` env vars; defaults unchanged | protect Taquari folders | hard-coded paths |
| 2 | `__replace__: true` in config dicts replaces instead of deep-merging | POA sources/scenarios must not inherit Taquari ones | separate non-inheriting config |
| 3 | POA-only cloud detector (bright OR smooth haze; open 41 px, dilate 101 px) | PoC detector flagged urban roofs and missed thin haze (checked visually, `results_poa/figs/maskcheck_*`) | PoC detector as is |
| 4 | Water = smooth (std<8 in 9 px) & (dark or brown hue 0–32); misses grey water of dec22 | simple, visually validated for post/feb24 | supervised classifier |
| 5 | Descriptor z-score per POA date (valid cells of whole box); Taquari PCA (recomputed read-only from data/, verified err<1e-4) applied without refit | standardization is unsupervised obs-model normalization; PCA = policy input transform | Taquari z-score stats (would not remove POA date radiometry) |
| 6 | Zero-shot β per scenario: P0←S0, P1←S1, P2←S2, shared by all methods | keeps episodes paired across methods | per-model β |
| 7 | Regions overlap 18.7 km²; recalibration on the other region is therefore not fully disjoint | 64×64 windows do not fit in the non-overlapping centro part | disjoint calibration |
| 8 | greedy (privileged) omitted; trajectories saved for zero-shot, PPO seed 0 | time/disk (1.9 GB free) | all variants/seeds |

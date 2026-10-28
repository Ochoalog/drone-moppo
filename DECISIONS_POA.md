# DECISIONS_POA (Porto Alegre test battery)

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
| 9 | Optional experiment (§6, PPO trained on ilhas P1) **not run** | user instruction on 2026-10-28: "sem treinar nas ilhas por enquanto" | train 3 seeds |
| 10 | 500 paired episodes per (region × scenario × variant × method); P2 kept in both regions (centro has 829 valid windows ≥ 20) | smoke: 1600 episodes in 62 s → full run ≈ 1 h, fits | 300 episodes |
| 11 | Recalibrated β₁ for P0 hits the top of the PoC β grid (300) in both regions | the grid (1–300) is the PoC one; P0 is the control and is at ceiling anyway | extend the grid only for POA |
| 12 | Long POA jobs always run under `systemd-inhibit --what=sleep:idle` | the machine suspended 15:46–17:32 during the β recalibration (it then finished normally) | rely on power settings |
| 13 | PPO channels: Taquari S1 models get the pre_mar23 PCA, S2 models the pre_oct23 PCA (each model's own training transform) | keeps each policy's input transform identical to training | one PCA for all models |
| 14 | Hypothesis rules fixed before the full run (smoke numbers not used to tune them): HP1 kept/broken/unclear per region with a −5 pp non-inferiority margin and paired Δdistance CI; HP2 (a) Spearman < 0 and Q1−Q5 CI > 0 for ≥ 75 % of methods, (b) centro − ilhas CI > 0 for ≥ 75 %; HP3 (a) every method better in centro, (b) active advantage over best simple baseline larger in ilhas (CI) for PPO S2 and greedy_map | pre-registered, explicit in REPORT_POA.md | ad-hoc reading |
| 15 | Change index quintiles pool P1 and P2 episodes of both regions (zero-shot); per-region quintiles also reported | widest range of change; region confounding shown via `share_ilhas` | P1 only |
| 16 | Descriptor standardization reference = cells with valid_frac ≥ 0.9 of that date (PoC used all train cells, ~97 % valid) | `poa_pre_dec22` covers only ~75 % of the box | all cells |
| 17 | `results_poa/traj/` (trajectory/belief npz) and `data_poa/` are git-ignored, kept on disk | ~2 GB free: committing them would duplicate them in .git | commit everything |
| 18 | `recal` β for ilhas P2 comes from centro P2 (β₁ 3.73 × κ 0.125 = 0.47): the filter almost never concentrates, so ilhas P2 recal success ≈ 0–7 % for every method; reported as is | κ chosen by the pre-set rule on the other region; not tuned on ilhas | override κ |

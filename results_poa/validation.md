Output of the mandatory validation (§8), run on 2026-10-28 before the full evaluation.

1. `pytest -q`: **45 passed** (36 PoC tests unchanged + 9 new `tests/test_poa_*.py`; the data-marked tests ran).
2. POA smoke (`scripts/smoke_poa.sh`: 20 paired episodes per method, P0 and P1, both β variants, all 6 PPO models
   + 4 baselines, 1600 episodes, through the report): **70 s** (limit 10 min). Results in `results_poa/smoke/`.
3. Gates: data gate **PASSED** (areas within −2 % of the hand-measured values, alignment 2.8 m / 0.7 m ≤ 10 m);
   sanity gate **PASSED** (P0 straight 0.99 ilhas, 1.00 centro; 200 episodes).
4. `sha256sum -c ../checkpoints/taquari_2026-10-28.sha256 --quiet`: **passed** (no protected file changed; 351 files).
5. Figures inspected: regions in place (ilhas over the Delta, centro over the Centro Histórico/4º Distrito), flood
   water visible and caught by the water mask in `poa_post`/`poa_pre_feb24` (grey water of `poa_pre_dec22` missed),
   cloud mask covers the cumulus over the NW islands and the Centro (shadows not masked), trajectories inside the
   64×64 windows.
6. Committed on branch `poa-tests` (no merge into `main`/`master`).

- Taquari manifest check **after the full run** (`sha256sum -c ../checkpoints/taquari_2026-10-28.sha256 --quiet`): passed (2026-10-28 18:33:37).

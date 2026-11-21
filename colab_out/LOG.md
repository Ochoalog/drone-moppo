| hora (UTC) | job_id | status | min GPU | número principal | próximo job |
|---|---|---|---|---|---|
| 2026-11-18 15:31 | g0_data_check | PASSED | 0.0 | Taquari EMSR720 19,09 km²; POA EMSN194 55,67 km²; Canoas 512,9 km² | g1_legacy_dinov2 |

| 2026-11-18T18:10:05+00:00 | _setup_check | OK | 0.0 | escrita ok |  |
| 2026-11-18T18:44:11+00:00 | g1_legacy_dinov2 | OK | 22.31 | crops/s 241.70110216 |  |
| 2026-11-18T22:08:44+00:00 | g1_legacy_grid4 | OK | 12.7 | crops/s 424.77104950 |  |
| 2026-11-19T03:47:35+00:00 | finetune_A | OK | 57.95 | loss 1.107913 |  |
| 2026-11-19T10:51:20+00:00 | g1_legacy_ftA | OK | 11.65 | crops/s 462.96811891 |  |
| 2026-11-19T14:04:18+00:00 | v3_extract_ft_taquari_100m | OK | 103.22 | gpu_min 103.22 |  |
| 2026-11-20T03:34:19+00:00 | v3_extract_ft_taquari_60m | OK | 100.97 | gpu_min 101.0 |  |
| 2026-11-20T05:15:16+00:00 | v3_extract_ft_taquari_150m | OK | 100.92 | gpu_min 100.9 |  |
| 2026-11-20T15:31:20+00:00 | calibrate_A_ftA | OK | 6.27 | kappa 0.04 | |
| 2026-11-20T14:32:35+00:00 | v3_extract_ft_taquari_100m | OK | 96.5 | gpu_min 96.5 | |
| 2026-11-20T17:50:10+00:00 | g3_A_ftA | OK | 1.29 | gpu_min 1.29 | |
| 2026-11-20T17:50:34+00:00 | v3_extract_ft_poa_100m | OK | 0.36 | gpu_min 0.4 |  |
| 2026-11-20T17:58:43+00:00 | v3_extract_ft_poa_60m | OK | 8.11 | gpu_min 8.1 |  |
| 2026-11-20T19:06:34+00:00 | v3_extract_ft_poa_150m | OK | 67.83 | gpu_min 67.8 |  |
| 2026-11-20T19:38:34+00:00 | calibrate_A_ftA_v2 | OK | 7.04 | kappa 0.040 alpha 0.00 | |
| 2026-11-20T19:44:27+00:00 | pilot_A_ppo_s0 | INVALID_GATE | 5.85 | fps 4785 | |
| 2026-11-20T20:47:06+00:00 | pilot_A_ppo_ca_s0 | INVALID_GATE | 6.12 | fps 4355 | |
| 2026-11-20T20:53:33+00:00 | pilot_A_ppo_ca_alt_s0 | INVALID_GATE | 6.31 | fps 4248 | |
| 2026-11-20T20:54:22+00:00 | h1_retrieval_ftA | OK | 0.67 | gpu_min 3.77 | |
| 2026-11-20T21:29:50+00:00 | v3_extract_ft_canoas_100m | OK | 29.69 | crops/s 318.4 | |
| 2026-11-20T23:10:14+00:00 | v3_extract_ft_canoas_60m | OK | 48.07 | crops/s 322.2 | |
| 2026-11-21T03:40:10+00:00 | v3_extract_ft_canoas_150m | OK | 50.7 | crops/s 304.1 | finetune_B |
| 2026-11-21T04:37:45+00:00 | finetune_B | OK | 57.41 | loss 1.066313 | finetune_C |
| 2026-11-21T05:35:38+00:00 | finetune_C | OK | 57.71 | loss 0.664034 | v3_extract_ftB_taquari_100m |
| 2026-11-21T15:45:40+00:00 | pilot_A_ppo_s0_v38 | INVALID_GATE | 6.54 | fps 4371 passed=False | pilot_A_ppo_ca_s0_v38 |
| 2026-11-21T15:52:20+00:00 | pilot_A_ppo_ca_s0_v38 | INVALID_GATE | 6.54 | fps 4028 passed=False | pilot_A_ppo_ca_alt_s0_v38 |
| 2026-11-21T15:59:06+00:00 | pilot_A_ppo_ca_alt_s0_v38 | INVALID_GATE | 6.61 | fps 3953 passed=False | v3_extract_ftB_taquari_100m |
| 2026-11-21T16:46:35+00:00 | diag_pilots_A | OK | 1.09 | episodes 5000 | v3_extract_ftB_taquari_100m |
| 2026-11-21T22:13:10+00:00 | eval_baselines_A | OK | 82.46 | episodes 60000 | mx_A_ppo_s0 |
| 2026-11-21T22:19:09+00:00 | mx_A_ppo_s0 | INVALID_GATE | 5.84 | fps 4589 pass=False adv=+0.010 | mx_A_ppo_ca_s0 |
| 2026-11-21T22:25:34+00:00 | mx_A_ppo_ca_s0 | INVALID_GATE | 6.28 | fps 4184 pass=False adv=+0.025 | mx_A_ppo_ca_alt_s0 |
| 2026-11-21T22:32:06+00:00 | mx_A_ppo_ca_alt_s0 | INVALID_GATE | 6.39 | fps 4121 pass=False adv=+0.020 | mx_A_ppo_s1 |
| 2026-11-21T22:37:58+00:00 | mx_A_ppo_s1 | INVALID_GATE | 5.74 | fps 4591 pass=False adv=+0.005 | mx_A_ppo_ca_s1 |
| 2026-11-21T22:44:15+00:00 | mx_A_ppo_ca_s1 | INVALID_GATE | 6.2 | fps 4208 pass=False adv=+0.000 | mx_A_ppo_ca_alt_s1 |
| 2026-11-21T22:50:36+00:00 | mx_A_ppo_ca_alt_s1 | INVALID_GATE | 6.27 | fps 4154 pass=False adv=-0.005 | mx_A_ppo_s2 |
| 2026-11-21T22:56:21+00:00 | mx_A_ppo_s2 | INVALID_GATE | 5.66 | fps 4601 pass=False adv=+0.020 | mx_A_ppo_ca_s2 |
| 2026-11-21T23:02:35+00:00 | mx_A_ppo_ca_s2 | INVALID_GATE | 6.15 | fps 4209 pass=False adv=+0.040 | mx_A_ppo_ca_alt_s2 |
| 2026-11-21T23:08:54+00:00 | mx_A_ppo_ca_alt_s2 | INVALID_GATE | 6.22 | fps 4131 pass=False adv=+0.035 | mx_A_ppo_s3 |
| 2026-11-21T23:14:40+00:00 | mx_A_ppo_s3 | INVALID_GATE | 5.67 | fps 4585 pass=False adv=+0.025 | mx_A_ppo_ca_s3 |
| 2026-11-21T23:20:53+00:00 | mx_A_ppo_ca_s3 | INVALID_GATE | 6.13 | fps 4215 pass=False adv=+0.025 | mx_A_ppo_ca_alt_s3 |
| 2026-11-21T23:27:14+00:00 | mx_A_ppo_ca_alt_s3 | INVALID_GATE | 6.27 | fps 4073 pass=False adv=+0.035 | mx_A_ppo_s4 |
| 2026-11-21T23:32:59+00:00 | mx_A_ppo_s4 | INVALID_GATE | 5.66 | fps 4579 pass=False adv=-0.015 | mx_A_ppo_ca_s4 |
| 2026-11-21T23:39:13+00:00 | mx_A_ppo_ca_s4 | INVALID_GATE | 6.14 | fps 4196 pass=False adv=+0.010 | mx_A_ppo_ca_alt_s4 |

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
| 2026-11-21T23:45:33+00:00 | mx_A_ppo_ca_alt_s4 | INVALID_GATE | 6.23 | fps 4116 pass=False adv=+0.025 | mx_A_ppo_s5 |
| 2026-11-21T23:51:15+00:00 | mx_A_ppo_s5 | INVALID_GATE | 5.61 | fps 4614 pass=False adv=+0.025 | mx_A_ppo_ca_s5 |
| 2026-11-21T23:57:27+00:00 | mx_A_ppo_ca_s5 | INVALID_GATE | 6.11 | fps 4218 pass=False adv=+0.030 | mx_A_ppo_ca_alt_s5 |
| 2026-11-22T00:03:44+00:00 | mx_A_ppo_ca_alt_s5 | INVALID_GATE | 6.18 | fps 4149 pass=False adv=+0.025 | mx_A_ppo_s6 |
| 2026-11-22T00:09:26+00:00 | mx_A_ppo_s6 | INVALID_GATE | 5.62 | fps 4591 pass=False adv=+0.030 | mx_A_ppo_ca_s6 |
| 2026-11-22T00:15:40+00:00 | mx_A_ppo_ca_s6 | INVALID_GATE | 6.14 | fps 4182 pass=False adv=+0.005 | mx_A_ppo_ca_alt_s6 |
| 2026-11-22T00:21:59+00:00 | mx_A_ppo_ca_alt_s6 | INVALID_GATE | 6.22 | fps 4135 pass=False adv=-0.005 | mx_A_ppo_s7 |
| 2026-11-22T00:27:41+00:00 | mx_A_ppo_s7 | INVALID_GATE | 5.62 | fps 4613 pass=False adv=+0.015 | mx_A_ppo_ca_s7 |
| 2026-11-22T00:33:51+00:00 | mx_A_ppo_ca_s7 | INVALID_GATE | 6.08 | fps 4223 pass=False adv=+0.035 | mx_A_ppo_ca_alt_s7 |
| 2026-11-22T00:40:07+00:00 | mx_A_ppo_ca_alt_s7 | INVALID_GATE | 6.18 | fps 4142 pass=False adv=+0.025 | mx_A_ppo_s8 |
| 2026-11-22T00:45:52+00:00 | mx_A_ppo_s8 | INVALID_GATE | 5.66 | fps 4584 pass=False adv=+0.020 | mx_A_ppo_ca_s8 |
| 2026-11-22T00:52:04+00:00 | mx_A_ppo_ca_s8 | INVALID_GATE | 6.11 | fps 4212 pass=False adv=+0.000 | mx_A_ppo_ca_alt_s8 |
| 2026-11-22T00:58:22+00:00 | mx_A_ppo_ca_alt_s8 | INVALID_GATE | 6.21 | fps 4137 pass=False adv=+0.015 | mx_A_ppo_s9 |
| 2026-11-22T01:04:07+00:00 | mx_A_ppo_s9 | INVALID_GATE | 5.66 | fps 4574 pass=False adv=+0.025 | mx_A_ppo_ca_s9 |
| 2026-11-22T01:10:20+00:00 | mx_A_ppo_ca_s9 | INVALID_GATE | 6.12 | fps 4204 pass=False adv=+0.040 | mx_A_ppo_ca_alt_s9 |
| 2026-11-22T01:16:41+00:00 | mx_A_ppo_ca_alt_s9 | INVALID_GATE | 6.27 | fps 4117 pass=False adv=+0.015 | matriz_concluida |
| 2026-11-22T01:34:41+00:00 | calibrate_canoas_ftA | FAILED | 0.26 | kappa = 0.0000 | E_canoas_ppo_s0 |
| 2026-11-22T04:56:23+00:00 | E_eval_taquari_L0 | OK | 19.72 | episodes 50000 | scan_noise_A |
| 2026-11-22T05:45:27+00:00 | scan_noise_A | OK | 48.98 | scan rows 8 | aguardando_selecao_niveis_ruido |
| 2026-11-22T15:17:45+00:00 | diag_canoas_split | FAILED | 0.72 | split_diagnostic | aguardando_decisao_vitor |
| 2026-11-22T16:06:07+00:00 | scan_noise_A_ext | OK | 27.71 | scan ext rows 4 | aguardando_selecao_niveis_ruido |
| 2026-11-22T16:14:02+00:00 | E_eval_canoas_zeroshot | OK | 7.79 | episodes 25000 | aguardando_jobs_E_synth |
| 2026-11-22T19:57:36+00:00 | E_synth_L3_ppo_s0 | INVALID_GATE | 6.12 | fps 4928 pass=False adv=+0.040 | E_synth_L3_ppo_s1 |
| 2026-11-22T20:03:16+00:00 | E_synth_L3_ppo_s1 | OK | 5.51 | fps 4922 pass=True adv=+0.135 | E_synth_L3_ppo_s2 |
| 2026-11-22T20:08:55+00:00 | E_synth_L3_ppo_s2 | OK | 5.52 | fps 4893 pass=True adv=+0.090 | E_synth_L3_ppo_s3 |
| 2026-11-22T20:14:31+00:00 | E_synth_L3_ppo_s3 | OK | 5.49 | fps 4910 pass=True adv=+0.120 | E_synth_L3_ppo_s4 |
| 2026-11-22T20:20:06+00:00 | E_synth_L3_ppo_s4 | OK | 5.48 | fps 4928 pass=True adv=+0.080 | E_eval_synth_L3 |
| 2026-11-22T20:36:43+00:00 | E_eval_synth_L3 | OK | 16.53 | episodes 30000 | L3_concluido |
| 2026-11-22T20:42:23+00:00 | E_synth_L1_ppo_s0 | INVALID_GATE | 5.56 | fps 4844 pass=False adv=+0.030 | E_synth_L1_ppo_s1 |
| 2026-11-22T20:48:04+00:00 | E_synth_L1_ppo_s1 | INVALID_GATE | 5.57 | fps 4789 pass=False adv=+0.010 | E_synth_L1_ppo_s2 |
| 2026-11-22T20:53:44+00:00 | E_synth_L1_ppo_s2 | INVALID_GATE | 5.58 | fps 4781 pass=False adv=+0.010 | E_synth_L1_ppo_s3 |
| 2026-11-22T20:59:21+00:00 | E_synth_L1_ppo_s3 | INVALID_GATE | 5.53 | fps 4814 pass=False adv=+0.030 | E_synth_L1_ppo_s4 |
| 2026-11-22T21:05:02+00:00 | E_synth_L1_ppo_s4 | INVALID_GATE | 5.58 | fps 4774 pass=False adv=+0.030 | E_eval_synth_L1 |
| 2026-11-22T21:21:31+00:00 | E_eval_synth_L1 | OK | 16.39 | episodes 30000 | L1_concluido |
| 2026-11-22T21:27:09+00:00 | E_synth_L2_ppo_s0 | INVALID_GATE | 5.53 | fps 4821 pass=False adv=+0.010 | E_synth_L2_ppo_s1 |
| 2026-11-22T21:32:47+00:00 | E_synth_L2_ppo_s1 | INVALID_GATE | 5.53 | fps 4823 pass=False adv=+0.035 | E_synth_L2_ppo_s2 |
| 2026-11-22T21:38:23+00:00 | E_synth_L2_ppo_s2 | INVALID_GATE | 5.49 | fps 4844 pass=False adv=+0.035 | E_synth_L2_ppo_s3 |
| 2026-11-22T21:43:55+00:00 | E_synth_L2_ppo_s3 | OK | 5.44 | fps 4885 pass=True adv=+0.050 | E_synth_L2_ppo_s4 |
| 2026-11-22T21:49:27+00:00 | E_synth_L2_ppo_s4 | OK | 5.44 | fps 4871 pass=True adv=+0.100 | E_eval_synth_L2 |
| 2026-11-22T22:05:47+00:00 | E_eval_synth_L2 | OK | 16.24 | episodes 30000 | L2_concluido |
| 2026-11-27T21:17:57+00:00 | _setup_check | OK | 0.0 | escrita ok |  |

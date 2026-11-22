# Colab (IA-2): fila de jobs e contrato de arquivos

A raiz do repositório GitHub **é** a pasta `drone_active_loc/`: os caminhos abaixo são relativos a ela.

## Sessão

1. Rode `00_setup.ipynb`. Ele monta o Drive, clona o repositório **no commit de `colab/CURRENT_COMMIT`** (lido da branch `feat/v3-change-aware`), instala as dependências e testa a escrita na branch `colab/results`.
2. Escolha o próximo job (abaixo) e rode o notebook dele com o `JOB_ID` certo na primeira célula.
3. No fim, o próprio notebook publica o resultado pequeno (`scripts/colab_publish_v3.py`).

## Como escolher o próximo job

1. `git fetch`; leia `colab/jobs_v3.csv` no commit de `CURRENT_COMMIT`.
2. Pegue o job de **maior prioridade** (menor número em `priority`) que **ainda não tem** `colab_out/<job_id>/DONE.json` na branch `colab/results` e cujas dependências (`depends_on`, separadas por `;`) já têm `DONE.json`.
3. **Só rode jobs com `status = queued`.** `on_hold` = em espera por decisão da IA-1 (não rodar); `done` = já feito. Não edite a fila: quem atualiza o `status` é a IA-1.

| Coluna | Significado |
|---|---|
| `job_id` | nome do job (também a pasta em `colab_out/`) |
| `notebook` | notebook a rodar |
| `config_path` | YAML do job (o hash dele vai no `DONE.json`) |
| `depends_on` | jobs que precisam terminar antes |
| `est_gpu_min` | estimativa de minutos de GPU |
| `priority` | 1 = mais urgente |
| `status` | `queued`, `running`, `done`, `invalid_gate`, `blocked` (escrito pela IA-1) |

## Contrato de arquivos

| O quê | Onde |
|---|---|
| Código | branch `feat/v3-change-aware`, commit de `colab/CURRENT_COMMIT` |
| Specs de recortes | `colab/specs/<nome>.parquet`: uma linha por recorte, com as colunas do contrato (`crop_id, site, date_id, cog_url, x, y, footprint_m, yaw_deg, roll_deg, pitch_deg, brightness, contrast, blur_sigma, noise_std, out_px`) e mais `src_res_m` (resolução da fonte a ler; ver `DECISIONS_V3.md` V4) |
| Renderização | `dal.v3.sensor.render.render_crops`: **não reimplemente** |
| Artefatos grandes | `MyDrive/drone_v3/<etapa>/` com `MANIFEST.sha256` |
| Resultados pequenos (≤ 5 MB) | branch `colab/results`, `colab_out/<job_id>/` + `DONE.json` |
| Diário | `colab_out/LOG.md` na branch `colab/results` |

`DONE.json`: `job_id`, `git_commit`, `config_hash`, `gpu_minutes`, `started_at`, `finished_at`, `status` (**só** `OK`, `INVALID_GATE` ou `FAILED`), `reason`, `metrics` (**com esse nome**). Use sempre `dal.v3.reporting.colab_io.write_done`, que já grava no formato certo. Tarefas fora da fila (como o G0) também usam esse formato, com `config_hash` do YAML da tarefa. A IA-1 confere se o commit e o hash da config batem com a fila (`dal.v3.reporting.colab_io.validate_done`).

## Retomar um job

Os jobs gravam blocos no Drive e pulam os que já existem. Se a sessão cair: rode o `00_setup.ipynb` e o mesmo notebook com o mesmo `JOB_ID`.

## Jobs disponíveis

| Job | Notebook | O que faz |
|---|---|---|
| `v3_extract_<local>_<h>m` | `01_data_and_crops.ipynb` | Gera as specs v3 do local e da altitude (confere o manifesto de hashes) e extrai DINOv2 zero-shot (GeM e CLS) + fração válida e de nuvem por recorte. Saída: `MyDrive/drone_v3/features/v3/dinov2_vits14/<spec>/` |
| `finetune_A` | `03_contrastive_finetune.ipynb` | Ajuste fino contrastivo (dobra A, só faixa de treino do Taquari). Saída: `MyDrive/drone_v3/finetune/finetune_A/backbone.pt` |
| `v3_extract_ft_<local>_<h>m` | `01_data_and_crops.ipynb` | Igual ao `v3_extract_*`, com os pesos do ajuste fino (depende de `finetune_A`). Saída: `features/v3/dinov2_vits14_ftA/<spec>/` |
| `calibrate_A_<descritor>` | `04_train_matrix.ipynb` | Calibração (β₁ por altitude, κ, α) na faixa de treino da dobra A. **Precisa do prior do Taquari refeito só com fontes ≤ 06/05/2024**, gravado como `MyDrive/drone_v3/prior/taquari_prior_emsr720_le0605_10m.tif` |
| `g1_legacy_grid4` | `02_dinov2_features.ipynb` | G1 oficial do DINOv2 com GeM espacial 4 × 4 (`grid4`, 6144 dim) nos recortes exatos da v1. A célula de G1 do notebook avalia `grid4` e `gem` |
| `g1_legacy_dinov2` | `02_dinov2_features.ipynb` | DINOv2 ViT-S/14 (GeM e CLS) e o hand da v1 nos mesmos pixels, sobre os recortes exatos da v1 (`specs/legacy_50m.parquet`, 323.595 recortes, overview 1,22 m/px). Saída: `MyDrive/drone_v3/features/legacy_50m/dinov2_vits14/{gem,cls,hand}.npy` |
| `calibrate_canoas_ftA` | `04_train_matrix.ipynb` | Experimento E (V46): escolhe a divisão de Canoas por colunas (teste a leste) e calibra β₁ e κ só na faixa de treino. Grava `calibration_canoas.json` na pasta da dobra A, ao lado da calibração do Taquari, que não muda. **Se terminar `FAILED` com "< 20 janelas de teste", pare e avise a IA-1** |
| `E_canoas_ppo_s<k>` | `04_train_matrix.ipynb` | PPO em Canoas (faixa de treino). O gate é registrado, mas **não** interrompe o treino |
| `E_eval_canoas`, `E_eval_taquari_L0` | `05_evaluate.ipynb` | Teste **e** treino, heurísticas + PPO nos dois modos. Entram também as execuções `_INVALID_GATE` (exceção da V46). Publica `episodes.parquet` (precisa do `pyarrow`, que já é dependência) |
| `scan_noise_A` | `04_train_matrix.ipynb` (`kind: scan_noise`) | Varre σ do ruído na observação e grava `scan.csv` e `calibration_noise_<σ>.json`, um por σ. Depois dele, a IA-1 escolhe os níveis e libera os treinos `E_synth_*` |

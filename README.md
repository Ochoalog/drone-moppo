# drone_active_loc: localização global ativa de drone com RL em mapas desatualizados

PoC para o projeto final de Aprendizado por Reforço (UFRGS). Um drone sem GNSS compara a imagem nadir
da câmera com uma ortoimagem de referência, mantendo um filtro de histograma sobre uma janela de
64 × 64 células de 50 m (3,2 km). Uma política escolhe para onde voar para reduzir a incerteza.
Comparamos PPO, uma heurística gulosa por ganho de informação e baselines (reta, zigue-zague,
aleatório) em três cenários de mapa: S0 = mesmo dia, S1 = mapa de mar/2023 limpo, S2 = out/2023
com lama e nuvens. Observações sempre de mai/2024, após a enchente.

Decisões e desvios da especificação: [`DECISIONS.md`](DECISIONS.md). Resultados: [`REPORT.md`](REPORT.md) (gerado).

## Instalação
```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch torchvision
uv pip install --python .venv/bin/python -e .        # versões exatas em requirements.txt
```
Os scripts fazem `unset PYTHONPATH`, porque o shell desta máquina exporta caminhos do ROS que quebram o pytest/numpy.

## Pipeline
| etapa | comando | tempo medido (CPU i7-10510U) |
|---|---|---|
| dados + descritores + calibração | `bash scripts/prepare_data.sh` | download 35 s; descritores ~66 min ("hand" e "cnn" em paralelo); calibração ~10 min |
| testes | `.venv/bin/pytest -q` | 5 s |
| gate de sanidade | `.venv/bin/python scripts/sanity_check.py` | ~8 min |
| smoke (treino 20k + avaliação + relatório) | `bash scripts/smoke.sh` | ~100 s |
| execução noturna | `nohup bash scripts/run_overnight.sh > runs/overnight_$(date +%F).log 2>&1 &` | ~6,5 h estimadas |

Módulos (`src/dal/`):
- `data.py`: leitura de janelas COG por `/vsicurl/` nos overviews, máscaras de validade/nuvem;
- `align.py`: correlação de fase em gradientes;
- `grid.py`: grade de 50 m e divisão norte/sul;
- `features.py`: recortes e descritores "hand"/"cnn";
- `calibrate.py`: retrieval, β₁ e temperamento κ;
- `filter.py`: filtro de histograma;
- `baselines.py`: baselines;
- `env.py`: simulador e ambiente gymnasium;
- `train.py`: PPO do SB3 com CNN própria;
- `evaluate.py`: avaliação pareada;
- `report.py`: estatística, figuras e `REPORT.md`;
- `viz.py`: figura dos dados.

Configs em `configs/{default,smoke,overnight}.yaml`.

## Acompanhar a execução noturna
```bash
tail -f runs/overnight_*.log                      # progresso: uma linha por execução de treino / etapa
tail -n 3 runs/overnight/S1_hand_s0/progress.csv  # métricas SB3 da execução corrente (CSV)
.venv/bin/tensorboard --logdir runs/overnight     # curvas no TensorBoard
ls runs/overnight/*/final_model.zip               # execuções concluídas
```
- O script é **retomável**: rodá-lo de novo pula as execuções com `final_model.zip`. Uma execução que falhar fica registrada no log e não interrompe as demais.
- Ordem: seed 0 de S0/S1/S2, depois seed 1 e seed 2.
- No fim, a avaliação pareada no **teste (sul)** usa 1000 episódios × método × cenário, e há também uma avaliação no norte (300 episódios, para H3).

**Onde saem os resultados:**
- `results/episodes_test.csv` e `results/episodes_train.csv` (um episódio por linha);
- `results/summary_*.csv`, `results/paired_tests_test.csv` e `results/hypotheses.json`;
- figuras em `results/figs/`;
- `REPORT.md` regenerado na raiz.

Para regenerar só o relatório: `.venv/bin/python -m dal.report --config configs/overnight.yaml`.

## Validação final (§7), saída resumida

Executada em 2026-10-28, antes de iniciar a execução noturna.

**1. `pytest -q`**
```
36 passed in 4.73s        (inclui os 9 testes @pytest.mark.data, que rodam com data/ presente)
```

**2. `bash scripts/smoke.sh`**
```
[smoke] descriptor: hand
[train] {'name': 'S0_hand_s0', 'seconds': 77.3, 'timesteps': 20480, 'fps': 264.8}
checkpoints: runs/smoke/S0_hand_s0/checkpoints/ppo_{2000..20000}_steps.zip; progress.csv + monitor.csv + TensorBoard
[eval] 300 episodes -> results/smoke/episodes_test.csv ; 150 episodes -> results/smoke/episodes_train.csv
[report] wrote REPORT.md
total_seconds=100   (limite: 600)
```
Também executei um ensaio completo do `run_overnight.sh` com 4k passos (9 treinos → avaliação de teste/treino → relatório), seguido de uma segunda chamada para conferir que a retomada pula os modelos prontos. Os artefatos do ensaio foram apagados.

**3. `python scripts/sanity_check.py`** (treino, 200 episódios por célula; taxa de sucesso)
```
method               greedy  lawnmower  random  straight
descriptor scenario
cnn        S0         1.000      0.995   0.995     0.995
           S1         0.800      0.830   0.655     0.900
           S2         0.555      0.590   0.545     0.630
hand       S0         1.000      1.000   1.000     1.000
           S1         0.900      0.960   0.860     0.980
           S2         0.565      0.645   0.500     0.730
GATE PASSED; best descriptor: hand
  hand: S0 greedy 1.000 (>= 0.6), lawnmower 1.000 (>= 0.4) -> pass; mean greedy S0-S2 0.822
  cnn:  S0 greedy 1.000 (>= 0.6), lawnmower 0.995 (>= 0.4) -> pass; mean greedy S0-S2 0.785
```

**4. Revisão do `REPORT.md` e das figuras**
- O relatório do smoke tem as seções de dados, retrieval, gate, teste, treino, curvas e hipóteses. Os números do PPO não significam nada (20k passos) e o relatório diz isso.
- `data_overview.png`: as três datas alinhadas, nuvens de out/2023 marcadas e linha de divisão norte/sul.
- `trajectories_*.png`: conferida também com episódios não triviais de S2 (do gate). A crença concentra ao longo do tempo, as trajetórias ficam dentro da janela e a localização falsa aparece rotulada.
- Curvas de sucesso × passos e curva de treino com eixos e legendas corretos.

Observações que valem para o artigo:
1. **S0 é degenerado**: todos os métodos localizam em 1 passo, então serve só como controle.
2. A **gulosa** tem muita localização falsa em S2 (27,5% no treino), porque persegue reduções de entropia vindas de casamentos espúrios. A reta é a baseline mais forte em S1/S2.
3. A ResNet18 pré-treinada casa pior entre datas do que o descritor "hand" (top-1 em S1: 10% vs 41%).

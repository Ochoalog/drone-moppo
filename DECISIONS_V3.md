# DECISIONS_V3: decisões da v3 (método ciente da mudança)

Cada linha: data, decisão, motivo, alternativa descartada. As hipóteses H1–H5, a matriz e as regras estatísticas entram na §2, e são commitadas **antes** de enfileirar a matriz completa. Vocabulário: "pré-especificado", nunca "pré-registrado".

## 1. Decisões de execução

| # | Data | Decisão | Motivo | Alternativa descartada |
|---|---|---|---|---|
| V1 | 18/11 | **Retreino P3 da MO cancelado** (`configs_mo_v2/` fica sem uso) | §0, item 9, do plano v3: a MO sai do texto principal e vira apêndice com resultado negativo | Rodar mesmo assim (consome CPU que a v3 precisa) |
| V2 | 18/11 | Fila de CPU local reordenada: avaliação amostrada (em andamento) → **reexecução do M1** → `results_mo_v1fix/` → H7 em dois modos. O benchmark de fps do MOPPO (P2) saiu da fila | O M1 fecha o estudo v1 hoje e é a motivação da v3. O P2 virou "medir só se sobrar CPU" (§7 do plano v3) | Manter a fila v2 (fps antes do M1) |
| V3 | 18/11 | `feat/v3-change-aware` criada **antes** do merge na `main`, a partir da ponta de `science/plan-v2-review` (commit 2453803). Depois do merge e da tag `v1-study-final`, a `main` é mergeada nela | A IA-2 precisa da spec `legacy_50m` hoje, e o merge depende do M1 (horas de CPU). O conteúdo da ponta da branch é o mesmo que vai para a `main` | Esperar o merge (deixa a IA-2 parada) |
| V4 | 18/11 | **Coluna extra `src_res_m`** nas specs (resolução da fonte a ler, m/px), além das colunas do contrato. Também: `kind`, `variant`, `cell_index`, `cell_row`, `cell_col`, `split` | Para reproduzir a v1, a renderização precisa ler o mesmo overview (1,2207 m/px, fator 4). Colunas extras não quebram o contrato, porque quem lê usa as colunas pelo nome | Codificar a resolução fora da spec |
| V5 | 18/11 | **Semântica fotométrica do `render.py`:** `clip(img·contrast + 128·brightness + noise_std·ruído)`, truncado para uint8; `brightness` é fração de 128 | É exatamente a fórmula da v1 (`photometric`: ganho `g` e viés `u·128`), o que permite reproduzir os recortes v1 | Brilho em DN absoluto |
| V6 | 18/11 | **Ruído por hash** de (crop_id, pixel, canal), sem gerador com estado | Mesmo resultado na CPU e na GPU, em qualquer ordem e tamanho de lote; "variantes determinísticas dada a semente" (§3.4 do prompt). Consequência: o ruído das variantes v1 não é reproduzido bit a bit pelo `render.py` (a v1 usava um gerador sequencial); a conferência injeta o ruído original | `torch.Generator` por recorte (lento na GPU e dependente do lote) |
| V7 | 18/11 | **Raio fixo do desfoque** (9 px; σ ≤ 3 px) | Um raio dependente do maior σ do lote mudava o recorte conforme o lote (o teste de paridade de lote pegou isso) | Raio por lote |
| V8 | 18/11 | **Inclinação** como raio da câmera (FOV 84°) girado por pitch (eixo leste) e roll (eixo sul) e intersectado com o solo; o yaw é aplicado depois, no plano da imagem, como na v1 | Com tilt 0 recai exatamente no afim da v1 (teste) | Homografia aproximada por cisalhamento |
| V9 | 18/11 | O job do G1 calcula **também o hand sobre os mesmos recortes** que o DINOv2 | A v1 recortou de rasters locais regravados em JPEG q95; a IA-2 lê os COGs originais. Comparar o DINOv2 (COG) com o hand da v1 (JPEG) misturaria descritor e fonte. O G1 compara os dois sobre os mesmos pixels e reporta também a referência v1 (0,409 / 0,269) | Mandar os rasters JPEG da v1 para o Drive (161 MB; depende do Vitor) |
| V10 | 18/11 | **Defeito na fila de CPU:** o invólucro esperava com `pgrep -f "<script>"`, e o padrão aparecia na própria linha de comando dele, então ele nunca saía da espera. A perda foi pequena: a avaliação terminou por volta de 12:24 e o M1 começou às 12:30, quando a fila foi refeita (a estimativa inicial de ~40 min estava errada). Corrigido: a fila agora roda os passos em sequência, sem espera por padrão de texto | A fila v2 tinha o mesmo defeito, ainda não manifestado | — |
| V11 | 18/11 | Desenvolvimento da v3 num **worktree separado** (`../dal_v3`, branch `feat/v3-change-aware`), com `data/`, `data_poa/`, `runs/` e `runs_mo/` como links só de leitura para a árvore principal | A árvore principal fica na `science/plan-v2-review` enquanto os processos do estudo v1 (M1, v1fix) rodam nela; trocar de branch sob processos em execução é arriscado | Trocar de branch na mesma árvore |
| V12 | 18/11 | **Ambiente em lote, modo legado:** cada episódio mantém o próprio gerador numpy, consumido na ordem da v1 (deslizamento só quando o movimento não é bloqueado; variante de observação a cada passo). Filtro, verossimilhança, predição e resumos ficam em lote no torch. A verossimilhança multiplica a observação pela grade inteira e recolhe as células da janela | Única forma de trajetórias **idênticas** (G2): os sorteios dependem dos dados e não têm equivalente no gerador do torch. No modo realista, o sorteio será em lote no torch | Sorteios em lote no torch também no modo legado (trajetórias divergiriam) |
| V13 | 18/11 | G2 no `pytest` padrão com 100 episódios; o gate oficial (1000 episódios, 3 seeds da PoC) roda em `scripts/gate_g2_parity_v3.py` e grava `results_v3/gates/G2.json` | A versão completa leva ~20 min de CPU e já há um teste longo na suíte | Só a versão completa no `pytest` |
| V14 | 18/11 | **Specs v3 geradas de forma determinística, com manifesto de hashes de conteúdo** (`colab/specs/v3_manifest.json`). A IA-1 commita o código e o manifesto; a IA-2 gera as specs no Colab (`python -m dal.v3.sensor.specs --site <local> --out <pasta> --verify`) e só as usa se todos os hashes baterem | As 24 specs v3 (Taquari e Porto Alegre, 3 altitudes) somam ~9,4 M de recortes e 425 MB em parquet: pesado demais para o git. O hash é do conteúdo (`pd.util.hash_pandas_object`), não dos bytes do parquet, que mudam com a versão do pyarrow | Commitar os parquets (git pesado) ou gravá-los direto no Drive (a IA-1 não tem acesso ao Drive) |

## 2. Hipóteses, matriz e regras estatísticas

*A escrever antes de enfileirar a matriz completa (Etapa C, §4.5 do prompt).*

## 3. Gates

| Gate | Critério | Situação |
|---|---|---|
| G0 | dados (prior, terceira área, evento externo) | aguardando `G0_report.md` da IA-2 |
| G1 preliminar | DINOv2 zero-shot × hand (S1 0,409; S2 0,269) na spec `legacy_50m` | spec publicada; caminho de avaliação conferido (o hand da v1 dá 0,409 e 0,269 pelo `scripts/eval_retrieval_v3.py`) |
| G2 | paridade do ambiente em lote | versão de 100 episódios **passou** (trajetórias idênticas para a reta e o zigue-zague; a PPO da PoC reproduz episódio a episódio o sucesso e os passos gravados); gate completo de 1000 episódios em execução |
| G3 | controle: reta ≥ 90% | pendente |
| G4 | ≥ 5.000 passos/s na T4 | pendente |

## 4. Conferência da spec `legacy_50m` (18/11)

`colab/specs/legacy_50m_check.json`: 20 recortes (10 de mapa, 10 de observação) renderizados pelo `render.py` a partir do raster local da v1, com o ruído original injetado nas observações:
- pixels idênticos em ≥ 99,5% (diferença máxima de 2 níveis, da interpolação em ponto fixo do OpenCV);
- descritor hand com cosseno ≥ 0,9999999 contra o gravado pela v1.

A spec tem 323.595 recortes: 64.719 células (43.287 de treino e 21.432 de teste) × (3 mapas + 2 variantes de observação de `post`).

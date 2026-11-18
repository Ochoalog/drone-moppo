# Plano de desenvolvimento: finalização do projeto (v2)

**Projeto:** localização global ativa de drone sem GNSS com RL sobre mapas desatualizados (enchentes do RS, 2024)
**Repositório:** https://github.com/Ochoalog/drone-moppo
**Versão:** v2, de 18/11/2026. Substitui a v1 (mesmo arquivo, histórico no git). As mudanças estão resumidas na §0.
**Situação:** PoC (PPO), bateria zero-shot em Porto Alegre e extensão multiobjetivo (MOPPO à la AMOR) com resultados. A revisão científica de 18/11 encontrou problemas que **invalidam a extensão MO do jeito que está** e afetam a forma de reportar a PoC. Antes do artigo, faltam as correções científicas, a tabela canônica e o congelamento dos resultados. A refatoração vem depois.
**Prazo:** apresentação na semana de 23/11 (**confirmar o dia com o professor hoje**: o cronograma da §5 muda conforme o dia).

---

## 0. O que mudou da v1 para a v2

| # | Mudança | Motivo |
|---|---|---|
| 1 | Ordem: **ciência → congelamento → artigo**. A refatoração roda em paralelo numa branch e só entra depois do congelamento | Na v1, a refatoração terminava no mesmo dia do congelamento, e um erro dela mudaria números já citados no artigo |
| 2 | A pendência MO deixa de ser "presets idênticos" e vira um pacote de 5 problemas (§2.1–2.4) | Gate de condicionamento que falhou e foi ignorado, orçamento desigual, throughput 11× menor, desenho inválido do H6 e vereditos fora do critério. A troca para avaliação estocástica, feita no diagnóstico de 18/11, ganha regras (§2.1) |
| 3 | Bootstrap hierárquico (seed → janela) em todos os ICs (§2.5) | Os ICs atuais ignoram a variação entre seeds (7,3 p.p. em S2) e a dependência espacial das janelas |
| 4 | Tabela canônica e glossário de métricas (§2.6) | Há números divergentes entre `REPORT.md`, `REPORT_POA.md` e `REPORT_MO.md` |
| 5 | Análise de mecanismo e índice de mudança no Taquari (§2.7) | É o que explica os resultados negativos; estava nos planos antigos e sumiu da v1 |
| 6 | Correção da apresentação e da errata da MO (§2.8) | `PRESENTATION_MO.md` afirma que "o MOPPO domina", o que os dados contradizem |
| 7 | Regra de segurança da refatoração ganha teste de carga dos modelos e determinismo (§4) | Mover módulos quebra o `PPO.load` do SB3 (classe do extrator salva por caminho de módulo) |
| 8 | Plano B explícito para a MO e cronograma por data da apresentação (§5) | O dia da apresentação ainda não está confirmado |

---

## 1. Princípios

1. **Nenhum número muda sem registro.** Os resultados protegidos (`results/`, `results_poa/`, `runs/overnight/`, `data*/`, `configs*/`, manifestos em `../checkpoints/`) **não** são alterados. Tudo que é novo vai para diretórios novos (§3).
2. **Critérios antes da avaliação.** Todo critério novo ou alterado é escrito em `DECISIONS_*.md` e commitado **antes** de rodar a avaliação correspondente. Critérios originais nunca são apagados: o novo entra com data e motivo, e os vereditos são reportados **pelos dois**.
3. **Usamos "pré-especificado", não "pré-registrado".** Não houve registro público, e o critério do HP1 mudou depois do smoke.
4. **Nenhuma afirmação sem número.** Todo número do artigo, da apresentação e dos relatórios vem de um CSV versionado na tag `results-final`.
5. **Resultado negativo é resultado.** Se uma correção não couber no prazo, o item é reportado como limitação, com o mecanismo explicado. Nunca com o veredito antigo.

---

## 2. Pendências científicas (prioridade máxima, nesta ordem)

> Responsáveis: preencher na reunião de hoje. A coluna "Saída" diz onde o resultado fica.

### 2.1 P1: condicionamento da MO (diagnóstico feito em 18/11; falta a leitura correta)

**Evidência inicial:** no `REPORT_MO.md`, `safe` dá **exatamente** os mesmos números que `unif` no Taquari (60,0% / 25,0% / 1019,7 m), no Centro e nas ilhas.

**Diagnóstico de 18/11** (`scripts/diagnose_mo_conditioning.py`):
- **O gate de condicionamento falhou aos 25% do treino** (`fast` = `safe` = 1096,3 m, IC [0, 0]; `runs_mo/moppo_S2_hand_s0/gate_check_25pct.json`) e o pipeline **seguiu mesmo assim**.
- O `w` chega à rede corretamente (`mo_eval.run_episode` → `reset(options={"w": w})`). A hipótese de bug no preset está descartada.
- A escala dos objetivos não explica o problema (desvio do retorno de custo 2,55; os outros ~0,4–0,5).
- A distribuição de ações muda com `w` (variação total média de até 0,26), **mas o argmax é o mesmo em 100% dos estados** entre `unif`, `fast` e `safe` (89% para `only_cost`).
- Correção adotada: avaliação **amostrada** (`deterministic=False`), com semente do torch fixa por episódio, em `results_mo/stochastic/`.

**Leitura científica (o que muda na v2):**
1. **O condicionamento é fraco, e isso é um achado do treino, não só da avaliação.** O próprio gate pré-especificado falhou. A avaliação amostrada pode revelar diferenças, mas elas vêm de deslocamentos pequenos na distribuição de ações. A frase "a avaliação com argmax apagou o condicionamento" só pode ser usada junto com "o gate de condicionamento falhou no treino".
2. **A troca de protocolo é post hoc.** Ela foi decidida depois de ver os resultados. Regras:
   - registrar em `DECISIONS_MO_V2.md`, com data, como alteração de protocolo;
   - aplicar o **mesmo** modo a **todas** as políticas aprendidas da comparação (MO, PPOs fixos **e** PPO da PoC). Comparar a MO amostrada com a PPO da PoC em argmax não é justo: a amostragem muda sucesso, passos e localização falsa;
   - reportar os dois modos (argmax e amostrado) em todas as tabelas da MO; o modo principal é o amostrado, por ser o do treino, e isso é dito no texto;
   - manter a semente comum por episódio entre todos os `w` e métodos (números aleatórios comuns).
3. **Regra nova para todo treino:** se um gate falhar, o script marca a execução como inválida (`INVALID_GATE` no log e no nome do diretório de resultados) e não gera vereditos a partir dela.
4. O teste D continua valendo para o retreino: a normalização `(w − 0,25) × 4` leva os vértices do simplex a 3, região pouco vista no treino com Dirichlet(1). Registrar a distribuição de `w` vista no treino.

Saída: `docs/diagnostico_mo.md` (diagnóstico + leitura acima). Os resultados amostrados dos modelos v1 vão para `results_mo_v1fix/` (ou ficam em `results_mo/stochastic/`, com nota no `docs/conciliacao_numeros.md`). **Prazo: hoje.**

### 2.2 P2: throughput do MOPPO

**Evidência:** o MOPPO rodou a ~43 fps; o PPO da PoC (SB3, mesma CPU) rodou a ~500 fps. Por isso a MO treinou só 348k passos e os PPOs fixos 225k, contra 1,4M da PoC.

1. Perfilar 20k passos com `py-spy record` ou `cProfile`.
2. Suspeitos, em ordem: serialização de observações grandes no `AsyncVectorEnv` (testar `SyncVectorEnv` com o mesmo n_envs, e `float32` em vez de `float64`); conversões `numpy ↔ torch` e `.cpu()` passo a passo; avaliação ou logging dentro do loop de coleta; `torch.set_num_threads` brigando com os processos dos envs.
3. **Meta:** ≥ 300 fps. Registrar antes/depois em `DECISIONS_MO_V2.md`.
4. A otimização não pode mudar a matemática: os testes (a), (b) e (c) do MOPPO e a propriedade de soma da recompensa continuam passando.

Saída: commit `perf:` e linha no `DECISIONS_MO_V2.md`. **Prazo: hoje/amanhã cedo.** Se não chegar a 150 fps, ir para o plano B (§2.9).

### 2.3 P3: retreino justo da MO (noite de 19→20)

Todos com **1,4M passos** (o mesmo da PoC), mesmos hiperparâmetros, mesma arquitetura, mesma memória de 8 ações, ≤ 3 checkpoints por execução. Saídas em `runs_mo_v2/` e `results_mo_v2/`. Os resultados v1 ficam como histórico.

| Prioridade | Execução | Para quê |
|---|---|---|
| 1 | MO com memória, seed 0 | política principal |
| 2 | PPO fixo `unif` (mesma implementação, w fixo), seed 0 | H4 e controle contra a PPO da PoC |
| 3 | PPO fixo `safe`, seed 0 | H4 |
| 4 | PPO fixo `fast`, seed 0 | H4 |
| 5 | MO sem memória, seed 0 | ablação da memória |
| 6 | MO com memória, seed 1 | variação entre seeds |

Orçamento: 6 × 1,4M = 8,4M passos. A 400 fps, ~5,8 h; a 300 fps, ~7,8 h. Se não couber, corte de baixo para cima e registre. **Observação para o texto:** cada execução recebe o mesmo número de passos. Isso é conservador para a MO (o AMOR relata que o MOPPO converge mais devagar); dizer isso no artigo.

Gates do treino (mantidos da v1, agora nos 25% do orçamento novo): sucesso ≥ 0,5 em `unif` no treino; e distância de `fast` × `safe` com IC da diferença que exclui 0. Os gates são medidos **nos dois modos** (argmax e amostrado); o de condicionamento passa se passar no modo amostrado, e o resultado em argmax é registrado. Gate que falha marca a execução como `INVALID_GATE` (§2.1, item 3). Se o gate de condicionamento falhar, **não** gastar a noite toda: interromper as execuções 1, 5 e 6, terminar 2–4 e ir para o plano B.

**Critérios v2, escritos em `DECISIONS_MO_V2.md` e commitados antes da avaliação:**
- **H4:** para cada um dos 3 presets, IC 95% (bootstrap hierárquico, §2.5) da diferença do retorno escalarizado médio r·w_k entre MO(w_k) e PPO-fixo(w_k). Suportada se todos contêm 0 ou favorecem a MO; não suportada se algum for inteiramente desfavorável; inconclusiva nos demais casos. Reportar também pelo critério original (2 presets), com nota de que o 3º foi cortado na v1.
- **H5:** inalterado (existe w na grade com sucesso ≥ −3 p.p., localização falsa ≤ 0 e distância < 0 pelos limites do IC), com Holm sobre os 38 pesos.
- **H6:** **varredura 1-D dedicada**: w_risco ∈ {0; 0,1; …; 1} com os outros três pesos = (1 − w_risco)/3, e o mesmo para w_custo. São 11 pontos × 300 episódios pareados. Estatística: ρ de Spearman entre o peso e a taxa (localização falsa ou distância) sobre os 11 pontos, com IC por bootstrap de episódios dentro de cada ponto. Critério original mantido: os dois ρ < 0 com IC que exclui 0. Motivo da mudança: com a grade de passo ¼, só existem 2 pontos com "os outros pesos iguais".
- **H7:** inalterado.
- **Hipervolume e EUM:** o **EUM** é a métrica principal, com IC por bootstrap. O hipervolume só compara conjuntos do **mesmo tamanho** (a MO avaliada nos 3 presets × os 3 PPOs fixos), com o ponto de referência fixado a priori: sucesso = 0, localização falsa = 1, distância = 4000 m (80 passos × 50 m).

### 2.4 P4: errata dos vereditos v1 (sem retreino, hoje)

Recalcular os vereditos v1 **pelos critérios que estavam no `DECISIONS_MO.md`** e publicar em `REPORT_MO_ERRATA.md` (não reescrever o `REPORT_MO.md`, que é histórico):

| Hipótese | v1 dizia | Pelo critério | Motivo |
|---|---|---|---|
| H4 | suportada | **não avaliável** | o gate de condicionamento falhou aos 25% e o treino seguiu; orçamentos diferentes (348k × 225k × 1,4M da PoC); o veredito usou sucesso/falsa/distância, não o IC de r·w_k; critério reduzido de 3 para 2 presets |
| H5 | não suportada | não suportada | — |
| H6 | parcial | **não suportada** | ρ(w_custo, distância) = +0,87, sinal oposto ao previsto; ρ(w_risco) sem variância; a grade não permite o desenho |
| H7 | parcialmente suportada | **não suportada** | IC da diferença inclui 0; o critério exige o IC inteiramente abaixo de 0 |

Remover também a atribuição causal "disparidade radiométrica insular": nos mesmos episódios, a PPO-S2 da PoC faz 71,3% nas ilhas e a MO faz 49,0%. A queda é da política MO, não do domínio.

### 2.5 P5: estatística (todos os relatórios)

1. **Bootstrap hierárquico** em `src/dal/evaluation/statistics.py` (função nova; a antiga continua existindo): B = 2000; reamostrar seeds com reposição (quando houver ≥ 2) e, dentro delas, **janelas** com reposição (episódios da mesma janela saem juntos). Nas diferenças pareadas, os dois métodos usam as mesmas reamostragens.
2. Se o CSV de episódios não tiver o id da janela, reconstruí-lo a partir da semente do episódio (2024 + i) reproduzindo o sorteio; testar que bate com as trajetórias salvas.
3. Tabelas com **resultado por seed** ao lado do agregado.
4. **Seeds extras (opcional, noite de 20→21):** PPO S1 e S2, seeds 3 e 4, em `runs/overnight_extra/` (nunca em `runs/overnight/`). ~4 × 47 min.
5. Recalcular os ICs da PoC, de Porto Alegre e da MO v2 em `results_final/stats/`. Os vereditos H1–H3 e HP1–HP3 são recalculados com os ICs novos e reportados **ao lado** dos originais.

### 2.6 P6: tabela canônica e glossário de métricas

Script `scripts/make_canonical_tables.py` → `paper/tables/*.csv`, com as colunas `source_file`, `protocol`, `n_episodes`, `seeds`, `region`, `scenario` em toda linha.

Glossário (vale para tudo que for novo; registrar as definições antigas onde diferirem):

| Métrica | Definição |
|---|---|
| `success` | massa da crença ≥ 0,9 num raio de 100 m **e** erro ≤ 100 m, no passo em que a crença concentra |
| `false_localization` | a crença concentrou (massa ≥ 0,9 num raio de 100 m) a mais de 100 m da posição real |
| `timeout` | não concentrou em 80 passos |
| `steps_success_median` | mediana de passos só nos episódios com sucesso (definição do `REPORT.md`) |
| `steps_penalized` | passos, com falha = 81 (usado no Wilcoxon) |
| `distance_mean_all` | distância média em todos os episódios |
| `distance_mean_success` | distância média só nos episódios com sucesso (**nova**; é a que sustenta a frase "metade da distância") |

Conciliar explicitamente: PPO-S2 nas ilhas P1 = 62,1% (`REPORT_POA`) × 71,3% (`REPORT_MO`); mediana de passos da reta em S2 = 26 (`REPORT.md`) × 34 (`REPORT_MO`). Dizer de onde vem cada diferença (seeds, n, definição) em `docs/conciliacao_numeros.md`.

### 2.7 P7: mecanismo e índice de mudança no Taquari

1. **Índice de mudança no Taquari**, com a mesma função de Porto Alegre, em modo somente leitura sobre `data/`. Figura única "sucesso × índice de mudança (quintis)" para os dois locais.
2. **Covariável independente do descritor:** fração de água (já existe em Porto Alegre) e, no Taquari, a diferença de cor média por célula depois de igualar os histogramas das duas datas. Serve para mostrar que a queda de sucesso não é só tautologia do descritor.
3. **Mecanismo (exploratório, critério escrito antes):** reavaliar PPO-S2 (3 seeds) e reta em S2, e nas ilhas e no Centro P1, registrando a célula de cada passo. Para cada episódio: média da **distintividade do mapa** e do **índice de mudança** das células visitadas. Hipótese M1: nos episódios de localização falsa, a PPO visita células de distintividade mais alta **e** mudança mais alta do que nos episódios de sucesso, e mais do que a reta. Saída em `results_final/mechanism/`.

### 2.8 P8: apresentação e textos já escritos

1. Corrigir `PRESENTATION_MO.md`: remover "Por que o MOPPO domina os outros métodos?", "validando a Hipótese H4" e "superioridade"; usar os vereditos da errata (§2.4).
2. Corrigir a decisão nº 12 do `DECISIONS_MO.md` acrescentando uma linha de errata (não apagar a original).
3. Conferir o texto dos vídeos (`poa_ilhas_presentation_deck.mp4` e similares). Se algum quadro afirmar superioridade, regenerar com o texto corrigido ou não usar o vídeo.

### 2.9 Plano B da MO

Se P2 não chegar a 150 fps, ou se o gate de condicionamento falhar no retreino:
- reportar a MO como **resultado negativo preliminar**: "o condicionamento em preferências não foi demonstrado com o orçamento disponível";
- usar só a errata (§2.4), o diagnóstico (§2.1) e a varredura H6 feita sobre o modelo v1 em avaliação amostrada (avaliação apenas, barata), com a PPO da PoC também amostrada para comparação;
- H4–H7 **não** aparecem como suportadas em lugar nenhum.

### 2.10 P9: ameaças à validade a declarar (sem experimento novo)

- Em Porto Alegre, a padronização dos descritores usa a caixa inteira, incluindo as regiões avaliadas (normalização transdutiva, não supervisionada). No Taquari, só as células de treino.
- As regiões `ilhas` e `centro` se sobrepõem em 18,7 km², e a recalibração do β "na outra região" não é totalmente disjunta.
- Erro de alinhamento global de 2,5–5 m (5–10% de uma célula de 50 m). O risco maior é o desalinhamento **local**, como os prédios inclinados no Centro.
- O critério do HP1 mudou depois do smoke (registrado no checkpoint de 28/10), o que contradiz a decisão nº 14 do `DECISIONS_POA.md`. Conferir no `git log` e reportar os dois critérios.

### 2.11 P10: validação com o professor

O escopo foi validado em 18/11. Ainda confirmar, se não estiver confirmado: o template (o enunciado sugere arXiv/Prime AI; propomos IEEE), o idioma e o dia exato da apresentação. Avisar que a organização do artigo mudou para 3 perguntas de pesquisa (ver `PLANO_ARTIGO.md`), sem mudar o escopo.

---

## 3. Onde fica cada saída nova

| Conteúdo | Diretório |
|---|---|
| Retreino MO v2 | `runs_mo_v2/`, `results_mo_v2/`, `configs_mo_v2/` |
| Reavaliação amostrada dos modelos v1 (MO, PPOs fixos e PPO da PoC no mesmo modo) | `results_mo_v1fix/` (ou `results_mo/stochastic/`, já em andamento) |
| Seeds extras da PPO | `runs/overnight_extra/` |
| Estatística, mecanismo, índice no Taquari | `results_final/` |
| Tabelas e figuras do artigo | `paper/tables/`, `paper/figures/` |
| Decisões | `DECISIONS_MO_V2.md`, `DECISIONS_FINAL.md` |
| Diagnósticos e conciliação | `docs/diagnostico_mo.md`, `docs/conciliacao_numeros.md` |
| Errata | `REPORT_MO_ERRATA.md` |

Disco: manter `runs_mo_v2/` + `results_mo_v2/` abaixo de 1 GB; rodar `df -h .` antes de cada noite.

---

## 4. Refatoração (em paralelo, sem mudar nenhum número)

### 4.1 Diagnóstico do código atual

| Tema | Situação | O que muda |
|---|---|---|
| Idioma dos identificadores | Quase tudo em inglês, com mistura de português (`mo_poa_nuvem_video.py`, `generate_taquari_nuvem_comparison_video`, `success_ilhas`, `centro_minus_ilhas`, `adv_ilhas`) | Tudo em inglês: `nuvem` → `cloud`, `ilhas` → `islands`, `centro` → `downtown`. Os nomes das regiões continuam em português só como **valores** (rótulos nos CSVs) |
| Idioma dos comentários | ~776 comentários e as docstrings em inglês | Comentários e docstrings em **português** |
| Organização | 26 módulos soltos em `src/dal/`; ~3.000 linhas de vídeo e PPTX misturadas com o núcleo | Subpacotes (§4.3) |
| Duplicação | Cinco geradores de vídeo repetem quadro, legenda e mapa | `presentation/rendering.py` comum |
| Funções longas | 15 funções com mais de 100 linhas (`train_moppo` 254, `run_pipeline` 190, …) | Quebrar em funções de ~40 linhas; prioridade para o núcleo |
| Nomes curtos | `dados_enchente/analyze_v2.py`, `apresentacao_grupo/*.py` (`m`, `pa`, `pb`, `cb`, `inl`, `d`) | Mover para `legacy/` e documentar que estão fora do pipeline |
| Commits | Português e inglês misturados | Conventional Commits em inglês: `feat:`, `fix:`, `perf:`, `refactor:`, `docs:`, `test:` |
| Lint | Não há | `ruff` com regras `E,F,I,N,B,UP` |

### 4.2 Convenções (valem para todo o grupo)

1. **Identificadores** (módulos, classes, funções, variáveis, constantes, chaves de config, colunas novas de CSV): **inglês**, `snake_case`, `PascalCase` para classes, `UPPER_CASE` para constantes.
2. **Comentários e docstrings:** **português**. A docstring diz *o que* a função faz e *por quê*. Referências: `# Ver AMOR (Alegre et al., 2025), §5`.
3. **Clean code:** uma função faz uma coisa (~40 linhas, ~5 parâmetros; mais que isso, `dataclass` de configuração); nada de números mágicos; nomes que revelam intenção; type hints nas funções públicas; sem código morto.
4. **Glossário do código:**

| Conceito | Nome no código |
|---|---|
| crença / histograma | `belief` |
| localização falsa | `false_localization` |
| sucesso | `success` |
| passo | `step` |
| vetor de pesos (preferência) | `preference_weights` (`w` só dentro de fórmulas) |
| nuvem | `cloud` |
| ilhas / centro | `islands` / `downtown` (rótulo de dado: `"ilhas"` / `"centro"`) |
| linha reta / zigue-zague | `straight` / `lawnmower` |

5. **Exemplo:**

```python
def compute_false_localization_rate(episodes: pd.DataFrame) -> float:
    """Calcula a fração de episódios em que a crença convergiu para o lugar errado.

    Um episódio conta como localização falsa quando a massa da crença passa de 0,9
    num raio de 100 m em torno de um ponto a mais de 100 m da posição real.
    """
    # Episódios truncados por limite de passos não contam como localização falsa.
    finished = episodes[episodes["terminated"]]
    return float(finished["false_localization"].mean())
```

### 4.3 Estrutura alvo

```
drone-moppo/
├── README.md, CONTRIBUTING.md, pyproject.toml
├── configs/                     # taquari/, poa/, moppo/, moppo_v2/
├── src/dal/
│   ├── geo/                     # data.py, align.py, grid.py
│   ├── localization/            # features.py, calibrate.py, filter.py
│   ├── envs/                    # env.py, mo_env.py
│   ├── agents/                  # train.py (PPO), mo_ppo.py (MOPPO), baselines.py
│   ├── evaluation/              # evaluate.py, mo_eval.py, poa.py, mo_poa_eval.py, statistics.py
│   ├── reporting/               # report.py, poa_report.py, mo_report.py, figures.py
│   ├── pipelines/               # mo_pipeline.py e CLIs
│   └── _compat.py               # aliases dos caminhos antigos (§4.4, item 3)
├── presentation/                # vídeos e PPTX; rendering.py comum
├── scripts/, tests/, docs/
├── legacy/                      # dados_enchente/, apresentacao_grupo/
└── paper/                       # LaTeX, tables/, figures/, make_figures.py
```

### 4.4 Regras de segurança da refatoração

1. Tag `v0-pre-refactor` (já existe) e os manifestos sha256 em `../checkpoints/` passam antes e depois de cada PR.
2. **Golden determinístico:** `scripts/smoke.sh`, `scripts/smoke_poa.sh` e `scripts/run_mo_smoke.sh` com seed fixa, `torch.set_num_threads(1)` e `torch.use_deterministic_algorithms(True)`; CSVs de episódios em `tests/golden/`; `tests/test_regression_golden.py` compara com `pd.testing.assert_frame_equal` (tolerância só nas colunas de tempo). Rodar o golden 2× seguidas antes de confiar nele.
3. **Teste de carga dos modelos** (`tests/test_model_loading.py`, marcado `@pytest.mark.data`): carregar os 9 modelos de `runs/overnight/`, os da MO (v1 e v2) e reproduzir 20 episódios com resultado idêntico ao gravado. O SB3 guarda a classe do extrator CNN pelo caminho do módulo; mover o módulo quebra o `PPO.load`. Solução: `src/dal/_compat.py` registra os caminhos antigos em `sys.modules`. **Nunca retreinar por causa da refatoração.**
4. Cada PR precisa de `pytest -q`, golden, teste de carga e `ruff check` verdes.
5. **Nenhum PR de refatoração entra na `main` antes da tag `results-final`.** Até lá, a refatoração vive na branch `refactor/integration`.

---

## 5. Cronograma

> Hoje é quarta, 18/11. D = dia da apresentação. A tabela assume **D ≥ quarta 25/11**. Para D mais cedo, ver §5.1.

| Quando | Entrega | Responsáveis |
|---|---|---|
| **Qua 18/11** | ✅ Escopo validado. ✅ Diagnóstico P1. Hoje ainda: regras da avaliação amostrada (§2.1) e reavaliação da PPO da PoC no mesmo modo; P2 (throughput); P4 (errata); confirmar template, idioma e dia (P10) | Vitor (P10), A (P1, P4), B (P2) |
| **Qui 19/11** | P5 (bootstrap hierárquico), P6 (tabela canônica), P8 (apresentação). `DECISIONS_MO_V2.md` com critérios v2 **commitado antes** de qualquer avaliação. Varredura H6 sobre o modelo v1 corrigido (só avaliação). **Noite 19→20:** P3 (retreino) | A, B; C começa `refactor/integration` |
| **Sex 20/11** | Avaliação v2 (§5.4 do prompt MO + varredura H6), vereditos H4–H7 v2, P7 (mecanismo e índice no Taquari). **Noite 20→21 (opcional):** seeds extras da PPO | A, B |
| **Sáb 21/11** | Tabelas recalculadas com as seeds extras (se houver). **Congelamento:** tag `results-final`, `paper/make_figures.py` gera tudo | A |
| **Sáb 21 – Seg 23/11** | Rascunho completo do artigo (ver `PLANO_ARTIGO.md`). Refatoração: PRs (1) renomeações e (2) comentários entram na `main` depois do congelamento, com todos os testes verdes | todos; C na refatoração |
| **Seg 23 – Ter 24/11** | Revisão cruzada do artigo, slides, ensaio | todos |
| **D − 1** | PDF final e release `v1.0` com modelos e features como assets | Vitor |

**Regra de corte:** artigo e números corretos vêm antes da estética do código. Os PRs (3) mover `presentation/` e `legacy/`, (4) subpacotes e (5) funções longas ficam para depois da entrega se atrasarem. Os PRs (1) e (2) são obrigatórios antes da release `v1.0`, mas nunca antes do congelamento.

### 5.1 Se a apresentação for antes de quarta 25/11

| D | O que cai |
|---|---|
| Ter 24/11 | Seeds extras da PPO; P7 fica só com o índice no Taquari (sem mecanismo). Congelamento na sexta 20 à noite |
| Seg 23/11 | Também o retreino P3: a MO vai pelo plano B (§2.9). Congelamento na sexta 20 à tarde |

---

## 6. Fluxo de trabalho no GitHub

- `main` protegida. Branches `fix/...`, `perf/...`, `science/...`, `docs/...`, `refactor/...`; um revisor do grupo por PR.
- Uma issue por item das §2 e §4, com os rótulos `science`, `refactor`, `paper`, `presentation`.
- Dados (`data*/`, ~1,4 GB) e modelos (`runs*/`) **fora** do git. São reproduzidos por `scripts/prepare_data.sh`.
- **Risco de reprodutibilidade:** o bucket de origem se chama `oin-hotosm-temp` (temporário). Na release `v1.0`, anexar como assets os modelos finais, as features derivadas (`data/`, `data_poa/`) e um arquivo com o sha256 de cada TIF de origem.
- Dados Maxar/OpenAerialMap sob **CC BY-NC 4.0**: citar no README e no artigo.

## 7. Definição de pronto

Ciência:
- [ ] `docs/diagnostico_mo.md` com o diagnóstico do P1 e a leitura da §2.1 (gate que falhou + regras da avaliação amostrada)
- [ ] Todas as políticas aprendidas da comparação MO avaliadas no mesmo modo (argmax e amostrado)
- [ ] Scripts de treino marcam a execução como inválida quando um gate falha
- [ ] `REPORT_MO_ERRATA.md` publicado e vereditos v1 recalculados pelo critério original
- [ ] MO v2 avaliada com critérios v2 commitados antes, **ou** plano B documentado
- [ ] ICs por bootstrap hierárquico e resultados por seed em todas as tabelas
- [ ] `paper/tables/` gerado por script, com `source_file`, `protocol`, `n_episodes` e `seeds` em toda linha
- [ ] `docs/conciliacao_numeros.md` explica toda divergência entre relatórios
- [ ] `PRESENTATION_MO.md` e vídeos sem afirmações que os dados contradizem
- [ ] Manifestos do Taquari e de Porto Alegre passando
- [ ] Tag `results-final`

Código:
- [ ] `ruff check` sem erros; `pytest -q` verde, incluindo golden e teste de carga dos modelos
- [ ] Nenhum identificador em português (regra N do `ruff` e grep pelo glossário)
- [ ] Comentários e docstrings em português
- [ ] README em português, com reprodução passo a passo
- [ ] Release `v1.0` com modelos, features e hashes das fontes

Artigo:
- [ ] PDF no template aprovado pelo professor, em `paper/`

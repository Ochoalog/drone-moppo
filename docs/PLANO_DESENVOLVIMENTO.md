# Plano de desenvolvimento v3: versão final para a disciplina (entrega 25/11/2026)

**Projeto:** localização global ativa de drone sem GNSS sobre mapas desatualizados por enchentes (RS, 2023–2024)
**Repositório:** https://github.com/Ochoalog/drone-moppo
**Versão:** v3, de 18/11/2026. Substitui a v2 (preservada em `docs/PLANO_DESENVOLVIMENTO_v2.md`). A v1 editada está em `docs/PLANO_DESENVOLVIMENTO_v1.md`.
**Prazo:** **quarta, 25/11/2026**: artigo final (PDF), código com release `v1.0`, apresentação. Não depende de aprovação do orientador.
**Equipe:** Vitor (tempo integral), **IA-1** (Claude Code, 24/7, máquina local + GitHub), **IA-2** (Gemini no Google Colab, GPU T4). Colegas do grupo, se houver: revisão cruzada e seções do artigo.

---

## 0. O que muda da v2 para a v3

A v2 corrigia o estudo existente. A v3 transforma o estudo num **artigo de método**, com o roteiro que estava previsto para o ICUAS 2027 comprimido em 7 dias. O desenvolvimento não é o gargalo: o gargalo é **tempo de máquina** e **pontos de decisão**. Por isso o plano é organizado por gates diários, com uma ordem de corte definida.

| # | Mudança | Motivo |
|---|---|---|
| 1 | O estudo atual (PoC, Porto Alegre, MO) vira o **"estudo v1"**: congelado hoje com a tag `v1-study-final` e usado como linha de base e motivação | Já está corrigido pela v2 (errata, conciliação, ICs hierárquicos) e explica por que o método novo é necessário |
| 2 | **Descritor aprendido** (DINOv2, depois ajuste fino contrastivo) no lugar do "hand" | Responder à crítica "a reta só é difícil de bater porque a percepção é fraca" |
| 3 | **Ambiente realista**: grade de 25 m, footprint de câmera derivado da altitude e do FOV, deslocamento dentro da célula, erro de yaw e inclinação, ruído fotométrico, **deriva de odometria** | Avaliadores de UAV cobram sensor e movimento plausíveis |
| 4 | **Filtro e política cientes da mudança** (CA, de *change-aware*), usando a delimitação de inundação do Copernicus EMS | Ataca o mecanismo que o estudo v1 identificou: o planejamento falha onde o mapa mudou |
| 5 | **Altitude como ação** (60, 100 e 150 m) | Contribuição nova: subir mostra estruturas grandes que sobrevivem à enchente |
| 6 | **Ambiente vetorizado em GPU** (PyTorch) no Colab | Torna viáveis 10 seeds, validação cruzada espacial e as ablações em dias, não semanas |
| 7 | **10 seeds** nos resultados principais, com rliable (IQM, bootstrap estratificado, probabilidade de melhora), e **validação cruzada espacial em 3 faixas** | Variação entre seeds grande (IC da PPO − reta em S2 passou de [−2,7; +4,0] para [−6,1; +9,8] p.p. com o bootstrap hierárquico) e um único corte geográfico |
| 8 | **Terceira área** da enchente de 2024 no RS e, se os dados existirem, **um evento externo** | Generalização entre domínios e eventos |
| 9 | A **extensão multiobjetivo sai do texto principal**: vai para um apêndice curto como resultado negativo, com a errata. O retreino P3 da v2 está **cancelado** | É a parte mais fraca e não ataca o mecanismo encontrado |
| 10 | A refatoração dos módulos antigos (renomeações, subpacotes) fica **para depois de 25/11**. O código novo nasce no padrão do `CONTRIBUTING.md` | Não gastar a semana em estética de código antigo |

---

## 1. Princípios (valem para as duas IAs)

1. **Nada protegido muda.** `results/`, `results_poa/`, `runs/overnight/`, `data*/`, `configs*/` e os manifestos de `../checkpoints/` continuam intocados. Tudo da v3 vai para `*_v3/` (§6).
2. **Critérios antes da avaliação.** As hipóteses v3 (§4) são escritas em `DECISIONS_V3.md` e commitadas **antes** de rodar a matriz completa. Alterações posteriores entram com data e motivo, e os vereditos são reportados pelos dois critérios.
3. **"Pré-especificado", nunca "pré-registrado".**
4. **Todo número vem de um CSV** com `source_file`, `protocol`, `n_episodes`, `seeds`, `fold`, `region`, `git_commit` e `config_hash`.
5. **O prior de mudança não pode vir da imagem que a câmera vê.** A delimitação de inundação vem de uma fonte independente (Copernicus EMS ou Sentinel-1) com data **anterior ou igual** à do voo simulado. Usar a imagem Maxar pós-evento para construir o prior seria vazamento.
6. **Paridade antes de confiar.** O ambiente em GPU só é usado depois de reproduzir o ambiente em CPU no modo legado (§3.2, gate G2).
7. **Um gate que falha marca a execução como `INVALID_GATE`** e nenhum veredito sai dela.
8. **Ordem de corte definida** (§8). Ninguém corta fora da ordem sem registrar em `DECISIONS_V3.md`.

---

## 2. Divisão de trabalho e protocolo entre as IAs

| Quem | Onde roda | Responsável por |
|---|---|---|
| **IA-1** (Claude Code, 24/7) | Máquina local (CPU) + GitHub | Todo o código em `src/dal/v3/` com testes; ambiente GPU (desenvolvido e testado em CPU com PyTorch); testes de paridade; filtro CA; altitude; scripts de orquestração; notebooks em `colab/` para a IA-2 rodar; avaliação, estatística, figuras, tabelas LaTeX e relatórios gerados |
| **IA-2** (Gemini no Colab, T4) | Google Colab + Google Drive | Rodar os notebooks de `colab/`: extração de recortes direto dos COGs no S3, descritores DINOv2, ajuste fino contrastivo, treino da matriz de PPO, avaliação em GPU. Em paralelo: conferir referências e rascunhar os trabalhos relacionados |
| **Vitor** | — | Decisões nos gates, revisão dos PRs, introdução e discussão, conferência visual de figuras e dados, apresentação |

**Protocolo de passagem:**
- O GitHub é a fonte da verdade do código. Todo notebook começa com `git clone` + `git checkout <commit>` e grava o commit no resultado.
- Artefatos grandes (recortes, descritores, modelos) vão para `Drive/drone_v3/<etapa>/` com um `MANIFEST.sha256`.
- Todo job termina gravando `DONE.json` (commit, hash do config, tempo de GPU, métricas resumidas) e copiando os CSVs pequenos para `results_v3/` via commit da IA-1.
- **Limites do Colab:** sessões caem (tempo máximo de sessão, inatividade, cota de GPU). Todo job precisa ser **retomável**, com checkpoint no Drive a cada ≤ 20 min e jobs de ≤ 3 h. Se a cota de GPU acabar, o fallback é a CPU local com a matriz reduzida (§8).

---

## 3. Componentes técnicos

### 3.1 Descritor aprendido (IA-2 roda, IA-1 escreve)

1. **DINOv2 zero-shot:** ViT-S/14 (ViT-B/14 se couber), características densas agregadas por GeM (ou VLAD, como no AnyLoc), redução por PCA para 128 dimensões ajustada **só nas células de treino**.
2. **Ajuste fino contrastivo** (InfoNCE, 2–4 h na T4): positivos = a mesma célula em datas diferentes (mapa × pós-evento), com aumentos realistas (§3.3); negativos = outras células do lote. **Só a faixa de treino da dobra correspondente** (§3.6). Sem nenhuma imagem de Porto Alegre ou da terceira área.
3. **Gate G1 (percepção):** top-1 de recuperação nas células de treino de S1 e S2. O descritor escolhido precisa superar o "hand" (S1: 0,409; S2: 0,269). Ordem de preferência: ajuste fino > zero-shot > hand. Se nenhum superar o hand, o artigo usa o hand e reporta o resultado negativo da percepção.
4. Os descritores são calculados para **cada altitude** (footprint diferente) e para **cada variante de observação** (§3.3), e salvos em float16.

Estimativa: ~260 mil células de 25 m no Taquari × 3 altitudes × (2 datas de mapa + 4 variantes de observação) ≈ 4,7 M recortes. Na T4, em lote e fp16, é ~1 h de GPU mais a leitura. Ler os COGs uma vez em resolução reduzida (~0,6 m/px), por data, e recortar em memória.

### 3.2 Ambiente vetorizado em GPU (IA-1)

- `src/dal/v3/envs/batched_env.py`: N ambientes em paralelo (N = 512–2048) como tensores PyTorch. Crença, máscara, verossimilhança, predição de movimento e recompensa são operações em lote.
- A verossimilhança é o produto do descritor da observação (128-d) pelos descritores das células da janela, já em memória da GPU.
- **Modo legado:** grade de 50 m, descritor hand, sem deriva, altitude fixa, mesmas sementes de episódio (2024 + i).
- **Gate G2 (paridade):** no modo legado, reta e zigue-zague (determinísticas) produzem **as mesmas trajetórias** que o ambiente em CPU nos 1000 episódios de teste de S2, e as crenças batem com tolerância de 1e-4. A PPO da PoC (argmax), carregada no ambiente GPU, reproduz o sucesso do `REPORT.md` dentro do IC.
- **Gate G4 (throughput):** ≥ 5.000 passos/s de treino na T4. Abaixo disso, reduzir a matriz (§8).
- O PPO é o mesmo código PyTorch do MOPPO com m = 1, já validado pelo teste de redução (b) da v2. Ele ganha a opção de rodar com o ambiente em lote.

### 3.3 Sensor e movimento realistas (IA-1)

| Elemento | Modelo |
|---|---|
| Footprint | largura = 2·h·tan(FOV/2), FOV horizontal de 84°: 108 m a 60 m, 180 m a 100 m, 270 m a 150 m. Recorte reamostrado para 224 px |
| Posição | contínua; a célula é a de 25 m que contém o drone. Cada variante de observação tem um deslocamento uniforme dentro da célula (±12,5 m) |
| Orientação | norte para cima pela bússola, com erro de yaw ~ N(0, 5°) |
| Inclinação | roll e pitch ≤ 5°, como homografia |
| Fotometria | brilho e contraste ±15%, desfoque gaussiano σ ∈ [0; 1,5] px, ruído |
| Variantes | K = 4 por célula e por altitude, pré-calculadas; o ambiente sorteia uma por passo |
| Movimento | passo horizontal de 50 m (2 células) em 8 direções; deslocamento real = comandado × (1 + ε_escala) + erro de rumo, com deriva de **2% da distância** (ablação: 0% e 5%) |
| Predição do filtro | convolução da crença com uma gaussiana de σ proporcional ao deslocamento comandado |
| Tempo de voo | horizontal 50 m a 10 m/s (5 s) + 1 s de captura; subir ou descer um nível ~15 s. É a métrica de custo principal |

O β (com κ) é recalibrado **por altitude** e **por descritor**, só no treino, com a regra da PoC (maximizar sucesso − localização falsa de uma política aleatória).

### 3.4 Filtro e política cientes da mudança (IA-1)

- **Prior de mudança m_x ∈ [0, 1]:** fração da célula marcada como inundada.
  - Fonte principal: delimitação do Copernicus EMS (ativação **EMSR720**, rapid mapping do RS, e **EMSN194**, Porto Alegre).
  - Fallback: máscara de água de Sentinel-1 (por exemplo, o produto GFM do GloFAS) ou a mancha de inundação mapeada pela UFRGS/IPH, se disponível.
  - **Gate G0 (dados, hoje):** conferir a cobertura de cada fonte sobre a AOI do Taquari (Encantado, Roca Sales, Muçum), sobre as ilhas e o Centro de Porto Alegre e sobre a terceira área, e a data de cada produto (≤ 07/05/2024 no Taquari, ≤ 09/05/2024 em Porto Alegre).
- **Filtro CA:** verossimilhança com mistura robusta, L_x ∝ (1 − ρ_x)·exp(β·s_x) + ρ_x·ū, com ρ_x = α·m_x e ū = média de exp(β·s) na janela. O α vem de {0; 0,25; 0,5; 0,75; 1}, calibrado no treino com a mesma regra do κ.
- **Política CA:** a entrada da PPO ganha o canal m. Isso permite separar o efeito do **filtro** CA (vale para todos os métodos, inclusive a reta) do efeito da **política** CA.
- `greedy_map` também ganha versão CA: o ganho de informação previsto usa a verossimilhança CA.

### 3.5 Altitude como ação (IA-1)

- Espaço de ações: 8 direções horizontais + subir + descer (10 ações), com h ∈ {60, 100, 150} m e início a 100 m.
- A verossimilhança em cada passo usa os descritores do mapa **da mesma altitude**.
- A recompensa de custo usa o **tempo de voo** (§3.3), para que subir tenha preço.
- Métricas de comportamento: fração de passos em cada altitude; correlação entre altitude escolhida e índice de mudança local.

### 3.6 Dados e divisões

| Local | Papel | Situação |
|---|---|---|
| Taquari (S1: mar/23 → mai/24; S2: out/23 → mai/24) | treino e teste | dados prontos; refazer recortes no Colab a partir dos COGs |
| Porto Alegre P1 (fev/24 → 09/05/24), ilhas e Centro | teste zero-shot | dados prontos |
| **Terceira área do RS** (Canoas, São Leopoldo ou Eldorado do Sul) | teste zero-shot | **Gate G0:** procurar no catálogo do OpenAerialMap um par antes/depois Maxar a ~0,3 m/px; escolher a área com melhor cobertura |
| **Evento externo** (furacão Helene 2024, Maxar Open Data; ou Valência 2024, se estiver no programa aberto) | teste zero-shot | **Gate G0:** existe par antes/depois ≤ 0,5 m/px com sobreposição ≥ 30 km² e prior de mudança independente? Se não, sai |

**Validação cruzada espacial no Taquari:** 3 faixas de latitude com largura igual (≥ 3,2 km cada, para caber a janela) e uma margem de 200 m entre elas. Cada dobra treina em 2 faixas e testa na terceira. A dobra A (teste na faixa sul) reproduz a divisão do estudo v1.

### 3.7 Matriz de experimentos (`DECISIONS_V3.md` antes de rodar)

Todos no ambiente realista, com o descritor escolhido no G1, deriva de 2%, cenário de treino S2 (S1 avaliado também).

| Bloco | Métodos | Seeds | Dobras | Para quê |
|---|---|---|---|---|
| Principal | PPO, PPO-CA, PPO-CA-ALT | 10 | A | H3, H4 |
| Validação espacial | PPO-CA, PPO-CA-ALT | 3 | B, C | robustez espacial |
| Ablação de percepção | PPO-CA com hand | 3 | A | H1 no nível da tarefa |
| Ablação de deriva | PPO-CA com deriva 0% e 5% | 3 | A | sensibilidade |
| Baselines | reta, zigue-zague, aleatória, greedy_map, cada uma com filtro padrão e filtro CA (e ALT para a reta: "sobe para 150 m e voa reto") | — | A, B, C | H2, H3 |
| Zero-shot | os modelos da dobra A, sem retreino | os mesmos | — | H5 |

Orçamento: 1,4M passos por execução (aumentar para 3M se o G4 der folga). Avaliação: 1000 episódios pareados por (método × local × dobra). Número total de execuções de treino: 30 + 12 + 3 + 6 = 51.

---

## 4. Hipóteses v3 (escrever em `DECISIONS_V3.md` e commitar antes da matriz completa)

| Id | Hipótese | Critério |
|---|---|---|
| **H1** | O descritor aprendido supera o hand | top-1 de recuperação no teste (S1 e S2, dobra A), IC por bootstrap de células com limite inferior da diferença > 0 |
| **H2** | O filtro CA reduz a localização falsa | para a reta **e** para a PPO, no Taquari S2 e nas ilhas: IC da diferença de localização falsa (CA − padrão) < 0, com perda de sucesso ≥ −2 p.p. |
| **H3** (principal) | Com percepção forte e modelo realista, a PPO-CA paga a pena contra a melhor heurística CA | **superioridade:** probabilidade de melhora em sucesso > 0,5 com IC que exclui 0,5 (rliable, 10 seeds); ou **não inferioridade com eficiência:** limite inferior da diferença de sucesso ≥ −3 p.p. **e** IC da diferença de tempo de voo < 0. Reportar qual das duas, por local, com Holm entre locais |
| **H4** | A altitude ajuda onde o mapa mais mudou | PPO-CA-ALT × PPO-CA: (a) sucesso não inferior (−3 p.p.) e tempo de voo menor, **ou** sucesso superior; (b) a altitude média escolhida cresce com o índice de mudança da janela (Spearman > 0 com IC que exclui 0) |
| **H5** | A relação do H3 se mantém zero-shot | vale em ≥ 2 de 3 regiões (ilhas, Centro, terceira área); o evento externo é exploratório |
| **M1** | Mecanismo (estudo v1): as localizações falsas da PPO v1 acontecem em células distintivas no mapa antigo e mudadas | critério já fixado em `DECISIONS_FINAL.md` (F3) |

Estatística: IQM e probabilidade de melhora entre seeds (rliable), bootstrap estratificado por dobra e região, e bootstrap hierárquico seeds → janelas nas diferenças pareadas. Os ICs antigos nunca aparecem sozinhos.

---

## 5. Cronograma (hoje, quarta 18/11, até quarta 25/11)

> Cada dia termina com um gate. Se o gate falhar, aplica-se a §8 **no mesmo dia**, para não acumular atraso.

| Dia | IA-1 (local, 24/7) | IA-2 (Colab T4) | Vitor | Gate do dia |
|---|---|---|---|---|
| **Qua 18** (tarde e noite) | Fechar o estudo v1: M1 e índice de mudança no Taquari (F1–F3), tabelas canônicas, `REPORT_MO_ERRATA` já pronto; merge de `chore/conventions` e `science/plan-v2-review` na `main`; tag **`v1-study-final`**. Esqueleto de `src/dal/v3/` e do ambiente em lote | **G0:** conferir dados (terceira área, evento externo, cobertura do EMSR720/EMSN194, datas). DINOv2 zero-shot na grade de 50 m atual (top-1 em S1 e S2) | Confirmar o dia da apresentação e o template; decidir a terceira área e o evento externo | **G0** (dados) |
| **Qui 19** | Ambiente em lote + modo legado + testes; teste de paridade em CPU; sensor e movimento realistas; filtro CA; recortes e variantes (código) | Recortes multialtitude a 25 m e variantes; descritores zero-shot; ajuste fino contrastivo (dobra A primeiro) | Revisar `DECISIONS_V3.md` (hipóteses, matriz) e aprovar | **G1** (percepção) e **G2** (paridade) |
| **Sex 20** | Altitude como ação; baselines no ambiente realista (CPU e GPU); calibração de β, κ e α; **commit de `DECISIONS_V3.md` antes da matriz** | Medir throughput (G4); treinos piloto (1 seed por método) para conferir os gates de aprendizado; ajuste fino das dobras B e C | Conferir visualmente recortes, variantes e prior de mudança | **G3** (sanidade: no controle S0, a reta ≥ 90%) e **G4** (throughput) |
| **Sáb 21** | Avaliação dos pilotos; scripts de avaliação em lote e rliable; figuras automáticas | **Bloco principal** (30 execuções) e ablações | Rascunho de Método e Formulação | pilotos passam no gate de aprendizado |
| **Dom 22** | Avaliação completa (Taquari, dobras, Porto Alegre, terceira área, evento externo); estatística; mecanismo e comportamento de altitude | **Validação espacial** (12 execuções) + zero-shot | Rascunho de Resultados com os números que forem saindo | matriz completa avaliada |
| **Seg 23** | **Congelamento às 12:00: tag `results-final`.** `paper/make_figures.py` e tabelas LaTeX; release candidata | Reexecuções só para bug confirmado | Introdução, Discussão, Conclusão | artigo completo em rascunho |
| **Ter 24** | Conferência número a número (registro de afirmações); README v3; vídeo curto de trajetórias | Revisão do texto (segunda leitura); conferência de referências | Revisão final, slides, ensaio | checklist da §9 |
| **Qua 25** | Release **`v1.0`** com modelos, descritores e manifestos | — | PDF final e apresentação | entrega |

Se a apresentação for **antes** de 25/11, apresentar o estado congelado mais recente (o estudo v1 como resultado e a v3 como andamento) e manter o 25/11 para o artigo.

---

## 6. Repositório e saídas

- Branch `feat/v3-change-aware` a partir da `main` depois do merge de hoje. PRs pequenos com revisão do Vitor (ou de um colega).
- Código novo em `src/dal/v3/`: `descriptors/` (DINOv2, ajuste fino), `envs/` (ambiente em lote, sensor, movimento), `localization/` (filtro padrão e CA), `agents/` (PPO em lote), `evaluation/` (rliable, bootstrap hierárquico), `reporting/`. Identificadores em inglês; comentários e docstrings em português (`CONTRIBUTING.md`).
- Notebooks em `colab/` numerados: `01_data_and_crops.ipynb`, `02_dinov2_features.ipynb`, `03_contrastive_finetune.ipynb`, `04_train_matrix.ipynb`, `05_evaluate.ipynb`.
- Saídas: `configs_v3/`, `runs_v3/` (só os modelos finais no disco local; o resto no Drive), `results_v3/`, `paper/`.
- Decisões: `DECISIONS_V3.md` (decisão, motivo, alternativa descartada).
- Dados: Maxar/OpenAerialMap sob **CC BY-NC 4.0**; produtos do Copernicus EMS com atribuição. Citar no README e no artigo. Os recortes derivados vão como asset da release com os hashes das fontes, porque o bucket de origem (`oin-hotosm-temp`) é temporário.

### 6.1 Testes obrigatórios (`tests/test_v3_*.py`)

- paridade do ambiente em lote com o de CPU no modo legado (G2);
- a verossimilhança CA com α = 0 é idêntica à padrão;
- o prior de mudança não lê nenhum arquivo da imagem pós-evento (teste de proveniência);
- o ajuste fino e o PCA só usam células da faixa de treino da dobra (teste de vazamento);
- footprint correto para cada altitude; deriva com a média e o desvio configurados;
- as variantes são determinísticas dada a semente;
- teste de proteção: nada em `results/`, `results_poa/`, `data*/`, `runs/overnight/` ou `configs*/` muda;
- golden e teste de carga dos modelos v1 continuam passando.

---

## 7. O que acontece com o trabalho da v2

| Item da v2 | Situação | Na v3 |
|---|---|---|
| Errata da MO, diagnóstico, conciliação, critérios v2 | feitos | entram no apêndice e no histórico; nada a refazer |
| Reavaliação amostrada (`results_mo_v1fix/`) | feita | tabela do apêndice da MO |
| Retreino P3 da MO (`configs_mo_v2/`) | não iniciado | **cancelado** (registrar em `DECISIONS_V3.md`) |
| Throughput do MOPPO (P2) | otimizações commitadas, medição pendente | medir só se sobrar CPU; não bloqueia nada |
| ICs hierárquicos, tabela canônica (P5, P6) | feitos | o mesmo código serve para a v3 |
| Índice de mudança no Taquari e M1 (P7, F1–F3) | critérios commitados | **terminar hoje**: é a motivação do método |
| Refatoração dos módulos antigos | comentários feitos | renomeações e subpacotes ficam para depois de 25/11 |

---

## 8. Ordem de corte e plano B

Cortar nesta ordem, só o necessário, registrando em `DECISIONS_V3.md`:

1. evento externo;
2. ablação de deriva;
3. ajuste fino contrastivo (fica o DINOv2 zero-shot);
4. validação espacial com 3 seeds (fica só a dobra A);
5. altitude como ação (H4 vira trabalho futuro);
6. terceira área do RS;
7. 10 seeds → 5 seeds.

**Se o G2 (paridade) falhar até sexta ao meio-dia:** abandonar o ambiente em GPU e rodar o ambiente realista na CPU local com a matriz mínima: PPO e PPO-CA, 3 seeds, dobra A, sem altitude.

**Mínimo para a entrega de 25/11:** descritor aprendido (pelo menos zero-shot) + ambiente realista com deriva + filtro CA para todos os métodos + PPO-CA × baselines CA + Porto Alegre zero-shot + estatística com rliable. Com isso o artigo já tem método, ablação e transferência.

---

## 9. Definição de pronto (25/11)

Ciência:
- [ ] Tag `v1-study-final` (estudo v1 fechado com errata, conciliação e M1)
- [ ] G0–G4 registrados em `DECISIONS_V3.md`, com o desfecho de cada um
- [ ] `DECISIONS_V3.md` com as hipóteses commitado antes da matriz completa
- [ ] Todos os testes v3 verdes; manifestos protegidos passando
- [ ] Tag `results-final` (congelamento de segunda, 23/11)
- [ ] `paper/tables/` e `paper/figures/` gerados por script a partir de `results_v3/`

Código:
- [ ] Código novo no padrão do `CONTRIBUTING.md`, `ruff check` sem erros
- [ ] Notebooks do Colab reproduzíveis a partir de um commit
- [ ] README com reprodução da v3 (local + Colab)
- [ ] Release `v1.0` com modelos, descritores, recortes derivados e hashes das fontes

Artigo e apresentação:
- [ ] PDF final no template confirmado
- [ ] Toda afirmação conferida contra o registro (`PLANO_ARTIGO.md` §4)
- [ ] Slides e vídeo curto de trajetórias

# DECISIONS_MO_V2: protocolo e critérios da extensão multiobjetivo (v2)

Complementa o `DECISIONS_MO.md` (v1), que **não é alterado**. Cada entrada tem data, motivo e alternativa descartada.
Os critérios abaixo foram escritos e commitados **antes** das avaliações a que se referem. Nos casos em que foram decididos depois de ver resultados, isso está dito explicitamente.

Vocabulário: "pré-especificado" = escrito e commitado antes da avaliação correspondente. Não houve registro público.

---

## 1. Alterações de protocolo

| # | Data | Alteração | Motivo | Quando foi decidida | Alternativa descartada |
|---|---|---|---|---|---|
| A1 | 18/11/2026 | **Avaliação das políticas aprendidas passa de argmax (`deterministic=True`) para ações amostradas de π(·\|s, w)**. É o modo principal; o argmax continua sendo reportado ao lado, em todas as tabelas da MO | No `REPORT_MO.md`, 32 dos 37 pesos da grade deram números idênticos. O diagnóstico (`scripts/diagnose_mo_conditioning.py`, `docs/diagnostico_mo.md`) mostrou que π(·\|s, w) muda pouco com w e que o argmax é o mesmo em 100% dos estados entre `unif`, `fast` e `safe`. A política foi treinada amostrando ações (entropia ~1,4 nats de no máximo 2,08), então o modo amostrado é o do treino | **Post hoc**: depois de ver os resultados v1. **O gate de condicionamento já tinha falhado aos 25% do treino** (`runs_mo/moppo_S2_hand_s0/gate_check_25pct.json`): o condicionamento é fraco no treino, não só invisível na avaliação | Manter só o argmax (esconde qualquer efeito de w); usar o modo amostrado só para a MO (injusto: a amostragem muda sucesso, passos e localização falsa) |
| A2 | 18/11/2026 | O modo amostrado vale para **todas** as políticas aprendidas da comparação: MO com memória, MO sem memória, PPOs fixos `fast`/`safe` e PPO da PoC (seeds 0, 1 e 2). As baselines não aprendidas (reta etc.) não mudam | Comparação justa (regra 2 da §2.1 do `PLANO_DESENVOLVIMENTO.md` v2) | Junto com A1 | Comparar a MO amostrada com a PPO da PoC em argmax |
| A3 | 18/11/2026 | **Números aleatórios comuns:** antes de cada episódio, `torch.manual_seed(2024 + i)`, com o mesmo `i` do seed do ambiente. Todos os w e todos os métodos veem a mesma sequência aleatória no episódio `i`. Um processo por tarefa e `torch.set_num_threads(1)` | Mantém o pareamento entre pesos e métodos (decisão nº 10 da v1), agora também para o sorteio das ações | Junto com A1 | Semente global única (o pareamento dependeria da ordem de execução) |
| A4 | 18/11/2026 | **Gate que falha marca a execução como inválida.** O treino grava `INVALID_GATE` em `gates.log`, cria o marcador `INVALID_GATE` no diretório da execução e emite um aviso. Os scripts de avaliação e relatório não geram vereditos a partir de execuções marcadas. Os gates passam a ser medidos nos dois modos; o de condicionamento vale no modo amostrado, e o argmax é registrado | Na v1, o gate de condicionamento falhou aos 25% e o pipeline seguiu sem aviso | 18/11, depois do diagnóstico | Interromper o treino automaticamente (perderia a execução inteira; a marcação é suficiente para impedir o uso) |
| A5 | 18/11/2026 | **H6 passa a ser medido por uma varredura 1-D dedicada** (critério na §2). O critério da v1 continua sendo reportado | Na grade de passo ¼ só existem 2 pontos com "os outros pesos iguais", então o desenho da v1 não permite estimar ρ | Depois de ver os resultados v1 | Manter a grade (desenho inválido) |
| A6 | 18/11/2026 | **H4 passa a usar os 3 presets** (`fast`, `safe`, `unif`), como na especificação original. O critério v1 (2 presets) continua sendo reportado | A v1 cortou o PPO fixo `unif` por orçamento (decisão nº 7 da v1) | Depois de ver os resultados v1 | — |

## 2. Critérios v2 (pré-especificados; valem para as avaliações abaixo)

Estatística comum: IC 95% por bootstrap com B = 2000. Com ≥ 2 seeds, bootstrap **hierárquico** (seeds com reposição e, dentro delas, janelas com reposição); com 1 seed, bootstrap por **janela** (episódios da mesma janela saem juntos). Diferenças pareadas usam as mesmas reamostragens nos dois métodos.

- **H4 (uma política no lugar de várias):** para cada preset w_k ∈ {fast, safe, unif}, IC 95% da diferença do retorno escalarizado médio r·w_k entre MO(w_k) e PPO-fixo(w_k), nos mesmos episódios. **Suportada** se todos os ICs contêm 0 ou favorecem a MO; **não suportada** se algum for inteiramente desfavorável; **inconclusiva** nos demais casos. Também pelo critério v1 (2 presets). **Pré-condição:** MO e PPOs fixos com o mesmo orçamento de passos e com os gates aprovados. Sem isso, o veredito é **não avaliável**.
- **H5 (dominar a reta em S2):** igual à v1 (existe w com limite inferior de Δsucesso ≥ −3 p.p., limite superior de Δlocalização falsa ≤ 0 e limite superior de Δdistância < 0), reportado com IC nominal e com Holm sobre os 37 pesos da grade.
- **H6 (resposta a w):** duas varreduras 1-D, cada uma com 11 pontos e 300 episódios pareados por ponto:
  - w_risco ∈ {0; 0,1; …; 1}, com os outros três pesos = (1 − w_risco)/3;
  - w_custo ∈ {0; 0,1; …; 1}, com os outros três pesos = (1 − w_custo)/3.

  Estatística: ρ de Spearman entre o peso e a taxa (localização falsa para w_risco; distância média para w_custo) sobre os 11 pontos. O IC vem de bootstrap de episódios dentro de cada ponto, com os mesmos índices de episódio em todos os pontos (pareamento), B = 2000. **Suportada** se os dois ρ < 0 com IC que exclui 0; **não suportada** se algum ρ tiver IC inteiramente ≥ 0; **inconclusiva** nos demais casos. A varredura roda nos dois modos; o veredito usa o modo amostrado.
- **H7 (Porto Alegre, ilhas):** igual à v1 (IC da diferença pareada da localização falsa de `safe` contra a PPO-S2 inteiramente abaixo de 0). A PPO-S2 é avaliada no mesmo modo que a MO (A2).
- **Fronteira (EUM e hipervolume):** o **EUM** é a métrica principal, com IC por bootstrap de episódios. Os pesos de utilidade são 1000 amostras Dirichlet(1) com seed 42. A normalização usa limites fixados a priori: sucesso em [0, 1], localização falsa em [0, 1] (invertida), distância em [0, 4000] m (invertida; 80 passos × 50 m). O hipervolume só compara conjuntos de **mesmo tamanho** (MO nos 3 presets × os 3 PPOs fixos), com ponto de referência fixo: sucesso = 0, localização falsa = 1, distância = 4000 m.

## 3. Distribuição de w no treino (teste D)

O treino sorteia w ~ Dirichlet(α = 1) por episódio (`mo_env.reset`). Valores por Monte Carlo (10⁶ amostras, seed 0):

| Evento | Probabilidade |
|---|---|
| algum componente ≥ 0,5 | 50,0% |
| algum componente ≥ 0,75 | 6,3% |
| um componente específico ≥ 0,55 (como no preset `fast` ou `safe`) | 9,1% |
| um componente específico ≥ 0,9 | 0,1% |

A rede vê w normalizado como (w − 0,25) × 4. Os vértices do simplex viram o valor 3, numa região quase nunca vista no treino. Por isso, os extremos das varreduras de H6 (peso ≥ 0,8) são **extrapolação**, e isso será dito junto com os resultados.

## 4. Condição para o retreino v2 (§2.3 do plano)

Decidido em 18/11 com o responsável pelo projeto. O retreino justo (6 execuções × 1,4M passos) **só roda se**:
1. o P2 (velocidade do MOPPO) chegar a **≥ 150 fps** (meta 300); **e**
2. a apresentação for **depois de terça, 24/11**.

Se qualquer condição falhar: **plano B** (§2.9). H4 fica **não avaliável**, sem retreino. H6 é avaliada pela varredura 1-D sobre o modelo v1 em modo amostrado, com a PPO da PoC também amostrada para comparação.

## 5. Saídas

- Reavaliação amostrada da grade v1 (MO com e sem memória, PPOs fixos): `results_mo/stochastic/` (iniciada antes desta revisão; ver `docs/conciliacao_numeros.md`).
- PPO da PoC e PPOs fixos nos dois modos, varredura H6 e EUM com IC: `results_mo_v1fix/`.
- Retreino, se as condições da §4 forem atendidas: `runs_mo_v2/`, `results_mo_v2/`, `configs_mo_v2/`.

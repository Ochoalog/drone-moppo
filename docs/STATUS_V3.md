# STATUS v3 (IA-1)

**Última atualização:** 21/11/2026, ≈ 14:30 (diagnóstico dos pilotos, V44; decisão: só bloco principal p/ o gate + avaliação das baselines, V45; cortes de deriva e validação espacial). Antes, ≈ 16:00 (vereditos H2–H5 prontos). Antes, 13:00 (patches da IA-2 aceitos, V39). Antes, 12:30. Antes, 09:00 (G1 aprovado com o ajuste fino, V29). Antes, 18/11 ≈ 21:30 (`finetune_A` iniciado; extração local reiniciada com autorização; Valência com alerta de proveniência, V28). Antes, 18/11 ≈ 21:00 (G1 do grid4 conferido: não passa em S1; `finetune_A` liberado com G1 pré-especificado na faixa de teste; Canoas e Valência registradas). Antes, 17:45 (extração local parada por falta de memória). Antes, 17:05 (descritor em grade e ajuste fino revistos; hand v3 local em andamento). Antes, 14:45 (fila do Colab: só `g1_legacy_grid4`; demais jobs `on_hold`). Antes, 14:10 (G1 preliminar recebido; Canoas e Valência decididas). Antes, 13:55: Prior rasterizado na grade de 25 m. Antes: Fim da etapa B (parte local): `pytest` 123/123 na `feat/v3-change-aware`, `ruff` sem erros, manifestos OK. **Etapa:** A concluída (tag `v1-study-final`); B quase toda pronta (falta só o que depende dos descritores da IA-2).

## [AÇÃO DO VITOR]

1. **Repassar à IA-2:** novo `CURRENT_COMMIT`; rodar `eval_baselines_A` (prioridade 1, ~40 min) e os 30 treinos `mx_A_*` (prioridade 2, ~8 min cada). Extrações/calibrações B/C e o resto da matriz ficam `on_hold` (cortados, V45).

## Gates

| Gate | Situação | Número |
|---|---|---|
| G0 (dados) | **aprovado** para o prior (linhagem autodeclarada pela IA-2; ver `DECISIONS_V3.md` §5) | Taquari: EMSR720; Porto Alegre: EMSN194 ("otimista"); Canoas: EMSN194 recortado; Valência: EMSR773 (falta conferir a data das fontes) |
| G1 | **PASSOU com o ajuste fino** (V29); zero-shot não passou | top-1 nas janelas de teste, ft / hand / grid4: S1 0,460 / 0,368 / 0,401; S2 0,332 / 0,239 / 0,283. Atenção: S0 0,906 / 0,996 / 0,956 (risco para o G3) |
| G2 (paridade) | **PASSOU** | reta e zigue-zague: 1000/1000 trajetórias idênticas; PPO da PoC: 1000/1000 resultados idênticos em cada uma das 3 seeds (média 0,707, dentro do IC do REPORT.md) |
| G3 (controle ≥ 90%) | **PASSOU** | reta em S0: 100%; S2: reta 99,8% (teto, V34) |
| G4 (≥ 5.000 passos/s na T4) | **NÃO PASSA** | 4.248–4.785 passos/s; orçamento fica em 1,4 M passos (~5 min por execução) |

## Rodando agora

- **Colab:** `eval_baselines_A` → 30 treinos do bloco principal (só para o gate).
- **Local:** resumo do gate nos 30 treinos; tabelas e figuras do artigo a partir de H1, baselines e diagnóstico.

## Feito hoje

- **Merge do estudo v1 na `main` e tag `v1-study-final`** (`pytest` 69/69, manifestos OK).
- **v3, parte 2:** ajuste fino contrastivo (notebook 03), jobs de calibração, treino e avaliação (notebooks 04 e 05), PCA do Taquari reaplicada no zero-shot, G1 calculado no próprio Colab.

- **Estudo v1 fechado:**
  - F1: índice de mudança no Taquari;
  - F2: covariável independente; a diferença de cor Lab também cai com o sucesso, ρ de −0,15 a −0,25;
  - **M1: não suportada**, o que muda a tese do artigo (ver abaixo);
  - tabelas `paper/tables/v1_*.csv` e Fig. 2;
  - apêndice da MO nos dois modos: H4 não avaliável; H5, H6 e H7 não suportadas.
- **v3:**
  - `render.py`, que reproduz os recortes v1;
  - spec `legacy_50m` e avaliação de recuperação;
  - contrato com o Colab e notebooks 00 e 02;
  - ambiente em lote legado (G2) e ambiente realista: 25 m, deriva, altitude, tempo de voo, filtro CA;
  - baselines em lote, calibração (β, κ, α), treinador PPO retomável com gate de aprendizado;
  - dobras espaciais, PCA só no treino;
  - specs v3 (24, por manifesto de hashes) e prior de mudança com teste de proveniência.

## Atenção para o artigo (tese)

O M1 refutou a parte "a política v1 é atraída por células distintivas no mapa antigo": as localizações falsas da PPO acontecem em células **mais mudadas** (nos 3 locais), mas não mais distintivas, nem mais mudadas que as da reta. Continua valendo que **a falha acontece onde o mapa mudou**, o que motiva o filtro CA. Por quintil de mudança, a PPO v1 é o método que **menos** perde sucesso (Fig. 2).

## Cortes

1. **Evento externo (Valência)**, item 1 da §8 (V30).
2. **Ablação de percepção (hand)** (V41).
3. **Ablação de deriva** (§8 item 2) e **validação espacial** (§8 item 4) (V45).
4. **Vereditos de RL (H3–H5)**: não avaliáveis pelo gate (V45).

## Próximos passos (IA-1)

1. Conferir cada extração `ft` publicada (DONE.json, tamanhos, frações válidas).
2. Pilotos (G3, G4) assim que `calibrate_A_ftA` sair; depois a matriz principal.
3. Job do hand nas specs v3 no Colab (ablação de percepção, 3 execuções).
4. Código de vereditos do H1 (IC por bootstrap de células consulta) e dos demais a partir da §2.

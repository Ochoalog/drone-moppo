# STATUS v3 (IA-1)

**Última atualização:** 22/11/2026, madrugada (V48: Canoas FAILED; 30/30 `mx_A` INVALID_GATE). Antes, 21/11/2026, 21:35 (prompt v4: artigo centrado em RL; experimento E pré-especificado na V46, commit `9ff0248`; código e fila de E prontos). Antes, ≈ 14:30 (V44/V45).

## [AÇÃO DO VITOR]

1. **Canoas (V51, decidido):** o E-real foi cortado; Canoas fica no artigo pelas baselines. Item descritivo novo na fila: `E_eval_canoas_zeroshot` (as 10 `mx_A_ppo` nas 31 janelas de Canoas, ~15 min). **Repassar à IA-2:** atualizar para o novo `CURRENT_COMMIT` depois do `scan_noise_A_ext` e rodar o `E_eval_canoas_zeroshot`.
2. **Ruído (V50, decidido):** `scan_noise_A_ext` (σ ∈ {4; 6; 8; 12}) rodando. Quando publicar, eu escolho L1–L3 pela regra da V46 sobre as 12 σ, commito o `noise_levels.json` e libero os 15 treinos e as 3 avaliações do E-synth (L3 primeiro). Se ainda não houver σ < 75%, paro e pergunto.
3. **Referência L0 (Taquari, 10 seeds, amostrado):** PPO 97,8%; PPO − aleatória **+3,8 p.p. [+2,3; +5,3]**; PPO − melhor heurística (reta a 150 m, escolhida no treino) **−1,8 p.p. [−2,5; −1,0]**. Argmax: +3,3 [+1,3; +5,2] e −2,3 [−3,8; −0,9].

4. **Correção para o `PLANO_ARTIGO.md` v4 (§5 e §6):** "gulosa < aleatória" vale só em **Canoas** (36,9 × 40,8%). Nas **ilhas** é empate (43,3 × 44,4%) e no **Centro** a gulosa fica **acima** da aleatória (35,8 × 27,8%). O texto do artigo (VI-B) já diz assim: "a gulosa é a pior estratégia informada em todo lugar; fica abaixo da aleatória em Canoas e empata nas ilhas". Não mexi no plano, porque o arquivo é seu.
5. **Pedir à IA-2 (opcional, quando houver folga):** publicar os episódios do `eval_baselines_A` (`MyDrive/drone_v3/v3/dinov2_vits14_ftA/fold_A/eval/eval_baselines_A/episodes.csv`) em parquet na `colab/results`, para dar IC às baselines da v3 no apêndice.
6. **Artigo:** a LaTeX não compila nesta máquina (sem TeX). Compile no Overleaf para conferir o tamanho da Tab. II (está com `\resizebox`).

## Experimento E: previsão (T4, um job por vez)

| Bloco | Jobs | GPU (min) | Término previsto |
|---|---|---|---|
| resto do bloco principal | `mx_A_*_s7–s9` (9) | ~60 | 21/11 ~22:30 |
| E-real | `calibrate_canoas_ftA`, `E_canoas_ppo_s0–4`, `E_eval_canoas` | ~10 + 40 + 45 | 22/11 ~00:10 |
| L0 | `E_eval_taquari_L0` (10 `mx_A_ppo` × 2 modos × teste/treino + heurísticas) | ~50 | ~01:00 |
| E-synth, varredura | `scan_noise_A` (8 σ) | ~45 | ~01:45 |
| E-synth, níveis | `E_synth_L{1,2,3}_ppo_s0–4` (15) + 3 avaliações | ~105 + 135 | ~06:00 (+ a latência da seleção dos níveis) |
| E-alt (opcional) | `E_canoas_ppo_alt_s0–2` + avaliação | ~45 | ~07:00, se liberado |

Folga grande até o limite de domingo às 20:00 (V46). A ordem de corte (E-alt → E-synth parcial → E-synth inteiro) só entra se a fila atrasar mais de ~12 h.

## Resultados que já chegaram (21/11)

- `eval_baselines_A` (OK, 82 min de GPU), sucesso de aleatória / reta / reta a 150 m / gulosa no teste:
  - Taquari S2: 94,0 / 99,8 / 99,6 / 83,4%;
  - Canoas C1: 40,8 / 64,5 / 70,6 / 36,9%;
  - ilhas P1: 44,4 / 79,1 / 82,8 / 43,3%;
  - Centro P1: 27,8 / 67,9 / 67,4 / 35,8%.
- Gate do bloco principal: **22 de 22** execuções terminadas são `INVALID_GATE` pelos dois critérios (s0–s6 dos três métodos e `mx_A_ppo_s7`). Vantagem sobre a aleatória de −1,5 a +4,0 p.p.; razão de tempo de voo de 0,87 a 1,08.

## Gates

| Gate | Situação | Número |
|---|---|---|
| G0 (dados) | **aprovado** para o prior (linhagem autodeclarada pela IA-2; ver `DECISIONS_V3.md` §5) | Taquari: EMSR720; Porto Alegre: EMSN194 ("otimista"); Canoas: EMSN194 recortado; Valência: EMSR773 (falta conferir a data das fontes) |
| G1 | **PASSOU com o ajuste fino** (V29); zero-shot não passou | top-1 nas janelas de teste, ft / hand / grid4: S1 0,460 / 0,368 / 0,401; S2 0,332 / 0,239 / 0,283. Atenção: S0 0,906 / 0,996 / 0,956 (risco para o G3) |
| G2 (paridade) | **PASSOU** | reta e zigue-zague: 1000/1000 trajetórias idênticas; PPO da PoC: 1000/1000 resultados idênticos em cada uma das 3 seeds (média 0,707, dentro do IC do REPORT.md) |
| G3 (controle ≥ 90%) | **PASSOU** | reta em S0: 100%; S2: reta 99,8% (teto, V34) |
| G4 (≥ 5.000 passos/s na T4) | **NÃO PASSA** | 4.248–4.785 passos/s; orçamento fica em 1,4 M passos (~5 min por execução) |

## Rodando agora

- **Colab:** `mx_A_*_s7–s9`, depois a fila do experimento E (acima).
- **Local:** análise do E (`scripts/verdicts_E_v3.py`), tabelas e figuras do artigo v4 (Tab. I–II, Fig. 2), seções II–VI-C em LaTeX.

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

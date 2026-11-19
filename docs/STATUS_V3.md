# STATUS v3 (IA-1)

**Última atualização:** 19/11/2026, ≈ 00:30 (`finetune_A` rodando no Colab; extração local do hand reiniciada com autorização; Valência com alerta de proveniência, V28). Antes, 18/11 ≈ 23:00 (G1 do grid4 conferido: não passa em S1; `finetune_A` liberado com G1 pré-especificado na faixa de teste; Canoas e Valência registradas). Antes, 17:45 (extração local parada por falta de memória). Antes, 17:05 (descritor em grade e ajuste fino revistos; hand v3 local em andamento). Antes, 14:45 (fila do Colab: só `g1_legacy_grid4`; demais jobs `on_hold`). Antes, 14:10 (G1 preliminar recebido; Canoas e Valência decididas). Antes, 13:55: Prior rasterizado na grade de 25 m. Antes: Fim da etapa B (parte local): `pytest` 123/123 na `feat/v3-change-aware`, `ruff` sem erros, manifestos OK. **Etapa:** A concluída (tag `v1-study-final`); B quase toda pronta (falta só o que depende dos descritores da IA-2).

## [AÇÃO DO VITOR]

1. **Valência (V28):** o prior do EMSR773 parece ter sido desenhado sobre a **mesma** imagem WorldView-2 de 31/10 10:22 que usaríamos como pós-evento, ou seja, não é independente. Decidir: (a) usar Valência com o prior rotulado "otimista" (como Porto Alegre); (b) pedir à IA-2 um prior do Sentinel-1 (29–30/10); (c) cortar Valência (é o primeiro corte da §8). Confirmar também com a IA-2: AOI01 ou AOI03; pré `103001010890C000` ou `105001003E84F500`; AOI de 225 km² ou a caixa de ~12 km².
4. **Revisar `docs/DECISIONS_V3_hipoteses_RASCUNHO.md`** (quinta).
5. **Dia da apresentação e template** (IEEE × arXiv/Prime AI; idioma).

Resolvido: figuras do prior aprovadas; Canoas (AOI e prior) e Valência (par, AOI, prior) recebidos e registrados em V27.

## Gates

| Gate | Situação | Número |
|---|---|---|
| G0 (dados) | **aprovado** para o prior (linhagem autodeclarada pela IA-2; ver `DECISIONS_V3.md` §5) | Taquari: EMSR720; Porto Alegre: EMSN194 ("otimista"); Canoas: EMSN194 recortado; Valência: EMSR773 (falta conferir a data das fontes) |
| G1 zero-shot | **Nenhum DINOv2 zero-shot supera o hand.** O grid4 passa só em S2 | top-1 no treino (mesmos pixels), GeM / grid4 / hand: S1 0,139 / **0,377** / 0,411; S2 0,063 / **0,300** / 0,270 (V26) |
| G1 ajuste fino | pendente; critério pré-especificado em V26: top-1 da saída `ft` acima do hand nas **janelas de teste** da v1 em S1 e S2 | referência local (hand v1, pixels JPEG, janelas de teste): S1 0,366; S2 0,239. A do G1 sai do hand nos pixels do COG, calculada no mesmo job |
| G2 (paridade) | **PASSOU** | reta e zigue-zague: 1000/1000 trajetórias idênticas; PPO da PoC: 1000/1000 resultados idênticos em cada uma das 3 seeds (média 0,707, dentro do IC do REPORT.md) |
| G3 (controle ≥ 90%) | não iniciado (precisa dos descritores v3) | — |
| G4 (≥ 5.000 passos/s na T4) | treinador pronto; medida no primeiro piloto da IA-2 | — |

## Rodando agora

- **Colab:** `finetune_A` (iniciado pela IA-2 no commit `efe1aad`) → `g1_legacy_ftA`. O resto está `on_hold`.
- **Local:** extração do hand nas specs v3 (Taquari, 100 m, 2 processos), reiniciada em 19/11 com autorização do Vitor; log em `runs_v3/hand_extract_taquari_100m.log`.

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

Nenhum.

## Próximos passos (IA-1)

1. Conferir o `g1_legacy_ftA` quando a IA-2 publicar e decidir o descritor pelo critério de V26.
2. Se o ajuste fino passar: liberar `v3_extract_ft_*` (Taquari primeiro) e `calibrate_A_ftA`. Se não: hand nas specs v3 (local) e calibração com ele.
3. Sites Canoas e Valência em `specs.py` (Valência precisa de CRS por local).
4. Depois da sua revisão: §2 do `DECISIONS_V3.md` commitada; pilotos (G3, G4) e matriz.

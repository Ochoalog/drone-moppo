# STATUS v3 (IA-1)

**Última atualização:** 19/11/2026, 09:00 (**G1 aprovado com o ajuste fino**, V29; fila liberada para as extrações `ft`, a calibração da dobra A e o ajuste fino das dobras B e C). Antes, 18/11 ≈ 21:30 (`finetune_A` iniciado; extração local reiniciada com autorização; Valência com alerta de proveniência, V28). Antes, 18/11 ≈ 21:00 (G1 do grid4 conferido: não passa em S1; `finetune_A` liberado com G1 pré-especificado na faixa de teste; Canoas e Valência registradas). Antes, 17:45 (extração local parada por falta de memória). Antes, 17:05 (descritor em grade e ajuste fino revistos; hand v3 local em andamento). Antes, 14:45 (fila do Colab: só `g1_legacy_grid4`; demais jobs `on_hold`). Antes, 14:10 (G1 preliminar recebido; Canoas e Valência decididas). Antes, 13:55: Prior rasterizado na grade de 25 m. Antes: Fim da etapa B (parte local): `pytest` 123/123 na `feat/v3-change-aware`, `ruff` sem erros, manifestos OK. **Etapa:** A concluída (tag `v1-study-final`); B quase toda pronta (falta só o que depende dos descritores da IA-2).

## [AÇÃO DO VITOR]

1. **Avisar a IA-2:** a fila está liberada (ver "Rodando agora"); o `CURRENT_COMMIT` mudou (configs novas das dobras B e C).
2. **Espaço no Drive:** as saídas `ft` somam ~11 GB (3 dobras × 3 altitudes no Taquari + POA). Se faltar espaço, os `.npy` de 6144 dim do G1 (`features/legacy_50m/dinov2_vits14_grid4/grid4.npy`, ~4 GB) podem sair, mas só com a sua autorização (é apagar resultado).
3. **Valência (V28):** o prior do EMSR773 parece ter sido desenhado sobre a **mesma** imagem WorldView-2 de 31/10 10:22 que usaríamos como pós-evento, ou seja, não é independente. Decidir: (a) usar Valência com o prior rotulado "otimista" (como Porto Alegre); (b) pedir à IA-2 um prior do Sentinel-1 (29–30/10); (c) cortar Valência (é o primeiro corte da §8). Confirmar também com a IA-2: AOI01 ou AOI03; pré `103001010890C000` ou `105001003E84F500`; AOI de 225 km² ou a caixa de ~12 km².
4. **Revisar `docs/DECISIONS_V3_hipoteses_RASCUNHO.md`** (quinta).
5. **Dia da apresentação e template** (IEEE × arXiv/Prime AI; idioma).

Resolvido: figuras do prior aprovadas; Canoas (AOI e prior) e Valência (par, AOI, prior) recebidos e registrados em V27.

## Gates

| Gate | Situação | Número |
|---|---|---|
| G0 (dados) | **aprovado** para o prior (linhagem autodeclarada pela IA-2; ver `DECISIONS_V3.md` §5) | Taquari: EMSR720; Porto Alegre: EMSN194 ("otimista"); Canoas: EMSN194 recortado; Valência: EMSR773 (falta conferir a data das fontes) |
| G1 | **PASSOU com o ajuste fino** (V29); zero-shot não passou | top-1 nas janelas de teste, ft / hand / grid4: S1 0,460 / 0,368 / 0,401; S2 0,332 / 0,239 / 0,283. Atenção: S0 0,906 / 0,996 / 0,956 (risco para o G3) |
| G2 (paridade) | **PASSOU** | reta e zigue-zague: 1000/1000 trajetórias idênticas; PPO da PoC: 1000/1000 resultados idênticos em cada uma das 3 seeds (média 0,707, dentro do IC do REPORT.md) |
| G3 (controle ≥ 90%) | não iniciado (precisa dos descritores v3) | — |
| G4 (≥ 5.000 passos/s na T4) | treinador pronto; medida no primeiro piloto da IA-2 | — |

## Rodando agora

- **Colab (`queued`, nesta ordem):** `v3_extract_ft_taquari_{100,60,150}m` (~70 min cada) → `v3_extract_ft_poa_*` → `calibrate_A_ftA` → `finetune_B`, `finetune_C` (~60 min cada) → extrações das dobras B e C no Taquari. Total ≈ 13 h de GPU.
- **Local:** parada (proteção de memória, 04:44). Não é mais caminho crítico: o hand fica para a ablação.

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

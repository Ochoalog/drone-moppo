# STATUS v3 (IA-1)

**Última atualização:** 18/11/2026, 14:10 (G1 preliminar recebido; Canoas e Valência decididas). Antes, 13:55: Prior rasterizado na grade de 25 m. Antes: Fim da etapa B (parte local): `pytest` 123/123 na `feat/v3-change-aware`, `ruff` sem erros, manifestos OK. **Etapa:** A concluída (tag `v1-study-final`); B quase toda pronta (falta só o que depende dos descritores da IA-2).

## [AÇÃO DO VITOR]

1. **Repassar à IA-2 (dados de Canoas e Valência, ver `DECISIONS_V3.md` V21):**
   - Canoas: a **caixa da AOI** (lon/lat) usada para contar as 42 janelas, e o **prior recortado** `canoas_prior_emsn194_10m.tif` em `MyDrive/drone_v3/prior/` (o raster de Porto Alegre não chega a Mathias Velho);
   - Valência: **um par de COGs** pré/pós (ou o mosaico) com as URLs, a AOI e o prior do EMSR773 em raster.
2. **Revisar `docs/DECISIONS_V3_hipoteses_RASCUNHO.md`** (quinta). Ponto novo: o G1 preliminar reprovou o DINOv2 zero-shot (ver Gates); o rascunho já prevê usar o hand se nenhum descritor aprendido o superar.
3. **Dia da apresentação e template** (IEEE × arXiv/Prime AI; idioma).

Resolvido: figuras do prior aprovadas; terceira área = **Canoas**; evento externo = **Valência**.

## Gates

| Gate | Situação | Número |
|---|---|---|
| G0 (dados) | **aprovado** para o prior (linhagem autodeclarada pela IA-2; ver `DECISIONS_V3.md` §5) | Taquari: EMSR720; Porto Alegre: EMSN194 ("otimista"); 3ª área e evento externo: decisão do Vitor |
| G1 preliminar | **DINOv2 zero-shot não supera o hand** | top-1 no treino (GeM / hand nos mesmos pixels): S0 0,816 / 0,998; S1 0,139 / 0,411; S2 0,063 / 0,270. O hand reproduz a v1. Diagnóstico em andamento: o pooling global não separa células vizinhas (até o S0, com a mesma imagem, fica em 0,82) |
| G2 (paridade) | **PASSOU** | reta e zigue-zague: 1000/1000 trajetórias idênticas; PPO da PoC: 1000/1000 resultados idênticos em cada uma das 3 seeds (média 0,707, dentro do IC do REPORT.md) |
| G3 (controle ≥ 90%) | não iniciado (precisa dos descritores v3) | — |
| G4 (≥ 5.000 passos/s na T4) | treinador pronto; medida no primeiro piloto da IA-2 | — |

## Rodando agora

- **Local:** nada pesado.
- **Colab (fila):** `g1_legacy_dinov2` → 6 × `v3_extract_*` → `finetune_A` → 6 × `v3_extract_ft_*` → `calibrate_A_*` (precisa do prior refeito).

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

1. Conferir o G1 preliminar quando a IA-2 publicar (`colab_out/g1_legacy_dinov2/`).
2. Esqueleto LaTeX em `paper/` com macros de números lidas dos CSVs; Fig. 1 em rascunho.
3. Avaliação, estatística (rliable) e geração de vereditos e figuras da v3 a partir dos episódios.
4. Depois da sua revisão: §2 do `DECISIONS_V3.md` commitada; pilotos (G3, G4) e matriz.

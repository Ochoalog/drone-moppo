# STATUS v3 (IA-1)

**Última atualização:** 18/11/2026, 13:25. **Etapa:** A concluída (tag `v1-study-final`); B quase toda pronta (falta só o que depende dos descritores da IA-2).

## [AÇÃO DO VITOR]

1. **Prior do Taquari: possível vazamento (urgente, repassar à IA-2).** O relatório G0 lista, na AOI 01 do EMSR720, um produto de *grading* feito a partir de **GeoEye-1 de 07/05/2024**. O GeoEye-1 é da Maxar, e a imagem que a câmera do drone vê também é Maxar de 07/05/2024: pode ser a mesma aquisição. **Pedir à IA-2 que refaça `taquari_prior_emsr720_10m.tif` só com produtos de fontes ≤ 06/05/2024.** Detalhes: `DECISIONS_V3.md` §5.
2. **Baixar os rasters do prior** de `MyDrive/drone_v3/prior/` para `drone_active_loc/data_v3/prior/` (não há `rclone` nesta máquina). Conferir visualmente as duas figuras `*_overlay_preevent.png`.
3. **Decidir a terceira área e o evento externo.** A IA-2 recomenda **Canoas** (par Maxar 09/05/2024 × 02/02/2024; prior do EMSN194) e **Valência** (Maxar Open Data, 26/10 × 31/10/2024; prior do EMSR773). O furacão Helene foi descartado (sem ativação do Copernicus EMS).
4. **Token do GitHub** nos *secrets* do Colab (`GITHUB_TOKEN`), se ainda não estiver: o commit do G0 saiu com a sua conta, então parece que já está.
5. **Dia da apresentação e template** (IEEE × arXiv/Prime AI; idioma): confirmar com o professor.
6. **Revisar o rascunho das hipóteses e da matriz** em `docs/DECISIONS_V3_hipoteses_RASCUNHO.md` (quinta). Os pontos marcados com "[definido aqui]" foram operacionalizados pela IA-1 e precisam do seu aval. Depois da revisão, o texto vai para a §2 do `DECISIONS_V3.md` e fica pré-especificado.
7. **Nome do prior refeito do Taquari:** pedir à IA-2 que grave o raster refeito (só fontes ≤ 06/05/2024) como `MyDrive/drone_v3/prior/taquari_prior_emsr720_le0605_10m.tif`; os jobs de calibração usam esse nome.

## Gates

| Gate | Situação | Número |
|---|---|---|
| G0 (dados) | relatório recebido; **aprovado com condições** (prior do Taquari a refazer) | Taquari: EMSR720; Porto Alegre: EMSN194 ("otimista"); 3ª área e evento externo: decisão do Vitor |
| G1 preliminar | spec e job prontos; aguardando a IA-2 rodar `g1_legacy_dinov2` | referência hand: S1 0,409, S2 0,269 |
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

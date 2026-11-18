# STATUS v3 (IA-1)

**Última atualização:** 18/11/2026, 12:30 (horário local). **Etapa:** A (fechar o estudo v1) em paralelo com o começo da B (para não deixar a IA-2 parada).

## [AÇÃO DO VITOR]

1. **Token do GitHub para a IA-2:** criar um *fine-grained token* com permissão de escrita em `Ochoalog/drone-moppo` (*Contents: read and write*) e colocá-lo nos *secrets* do Colab com o nome `GITHUB_TOKEN`. Sem ele, a IA-2 não publica resultados na branch `colab/results`.
2. **Dia da apresentação e template** (IEEE ou arXiv/Prime AI, e o idioma): confirmar com o professor. O cronograma da §5 do plano muda se a apresentação for antes de 25/11.
3. **Terceira área e evento externo:** decidir quando a IA-2 entregar o `G0_report.md` (branch `colab/results`, `colab_out/g0_data_check/`).
4. **Passar à IA-2:** o código está na branch `feat/v3-change-aware`; o primeiro job é `g1_legacy_dinov2` (`colab/README.md`).

## Gates

| Gate | Situação | Número |
|---|---|---|
| G0 (dados) | aguardando a IA-2 | — |
| G1 preliminar | spec publicada; job `g1_legacy_dinov2` na fila | referência hand: S1 0,409, S2 0,269 (caminho de avaliação conferido) |
| G2 (paridade) | não iniciado | — |
| G3 (controle ≥ 90%) | não iniciado | — |
| G4 (≥ 5.000 passos/s) | não iniciado | — |

## Rodando agora

- **Local (CPU):** reexecução do M1 (F3) → reavaliação `results_mo_v1fix/` (apêndice da MO) → H7 nos dois modos. A reavaliação amostrada da grade (`results_mo/stochastic/`) terminou (76/76).
- **Colab:** nada ainda (esperando a IA-2 pegar o `g1_legacy_dinov2`).

## Feito hoje (v3)

- `render.py` (renderização única CPU/GPU), que reproduz os recortes v1 (≥ 99,5% dos pixels idênticos; descritor hand com cosseno ≥ 0,9999999).
- `colab/specs/legacy_50m.parquet` (323.595 recortes exatos da v1).
- `scripts/eval_retrieval_v3.py` (mesma função do `REPORT.md` §2; reproduz 0,409 e 0,269).
- Contrato com a IA-2: `DONE.json`, fila, notebooks `00_setup` e `02_dinov2_features`, branch `colab/results`, `CURRENT_COMMIT`.

## Cortes

Nenhum.

## Próximos passos (IA-1)

1. Etapa A: F2 (covariável independente), veredito do M1, tabelas `paper/tables/v1_*.csv`, Fig. 2; `pytest -q` + manifestos; merge na `main`; tag `v1-study-final`.
2. Etapa B: ambiente em lote + modo legado + G2; specs v3 (25 m, 3 altitudes, K = 4); notebooks 01 e 03.

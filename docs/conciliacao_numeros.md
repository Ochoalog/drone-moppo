# Conciliação dos números entre relatórios

Cada divergência entre `REPORT.md`, `REPORT_POA.md`, `REPORT_MO.md` e os planos tem uma explicação. Todos os valores abaixo foram recalculados a partir dos CSVs de episódios em 18/11/2026. Os números do artigo vêm de `paper/tables/` (tag `results-final`), que traz `source_file`, `protocol`, `n_episodes` e `seeds` em toda linha.

| # | Número A | Número B | De onde vem a diferença |
|---|---|---|---|
| 1 | PPO-S2, ilhas P1 = **62,1%** (`REPORT_POA.md`) | **71,3%** (`REPORT_MO.md`) | **Seeds e n.** 62,1% é a média das **3 seeds** em **500** episódios (seed 0: 70,6%; seed 1: 61,8%; seed 2: 54,0%). 71,3% é **só a seed 0** nos **300 primeiros** episódios (`results_poa/episodes.csv` dá exatamente 0,713 nesse recorte). Mesmo modelo e mesmos episódios: não há erro, só recortes diferentes |
| 2 | Mediana de passos da reta em S2 = **26** (`REPORT.md`) | **34** (`REPORT_MO.md`, 34,5 na tabela) | **Definição e n.** 26 é a mediana **só nos episódios com sucesso**, em 1000 episódios (`steps_success_median` no glossário). 34,5 é a mediana de **todos** os episódios (falhas contam com os passos que tiveram), nos 300 primeiros. Nos 1000 episódios, a mediana de todos é 33 |
| 3 | Sucesso da reta em S2 = **70,0%** (`REPORT.md`) | **73,0%** (`REPORT_MO.md`) | **n.** 1000 episódios × os 300 primeiros. Os 300 primeiros são um subconjunto um pouco mais fácil para a reta |
| 4 | PPO da PoC em S2 = **70,7%** (`REPORT.md`) | 81,0% / 68,3% / 72,3% (`REPORT_MO.md`) | **Seeds e n.** 70,7% é a média das 3 seeds em 1000 episódios. O `REPORT_MO.md` mostra cada seed nos 300 primeiros episódios. Os valores por seed batem exatamente com `results/episodes_test.csv` restrito a `episode < 300`, o que confirma que a avaliação da MO usou os mesmos episódios da PoC |
| 5 | Throughput do MOPPO = **~43 fps** (`DECISIONS_MO.md` nº 6; plano v2 §2.2) | **88 fps** (`runs_mo/*/progress.csv`) | 43 fps foi uma medição preliminar, usada para dimensionar o orçamento. As 4 execuções completas rodaram a 88,1–88,9 fps. A PPO da PoC (SB3) rodou a 501–510 fps: a razão é ~5,8×, não 11× |
| 6 | IC de Δsucesso PPO − reta em S2 = **[−2,7; +4,0] p.p.** (`REPORT.md`) | **[−6,1; +9,8] p.p.** (`paper/tables/rq1_taquari.csv`) | **Método do IC.** O `REPORT.md` reamostra episódios, com a média das seeds por episódio. A tabela canônica usa bootstrap hierárquico (seeds → janelas, §2.5): a variação entre as 3 seeds (80,9%, 64,4% e 66,7%; média 70,7%) alarga o IC. Em 1000 episódios há 925 janelas distintas, então o agrupamento por janela muda pouco; o efeito é quase todo das seeds |
| 7 | "A PPO voa metade da distância da reta em S2" (1,24 × 2,37 km, `REPORT.md`) | Nos episódios com sucesso: **1,25 × 1,89 km** | **Definição.** A média sobre todos os episódios mistura episódios que terminam cedo por localização falsa (mais comuns na PPO: 20,2% × 13,0%). Nos episódios com sucesso (`distance_mean_success`), a PPO voa ≈ 2/3 da distância da reta. A frase "metade" não deve ser usada |
| 8 | MO v1: presets idênticos (argmax, `results_mo/summary_test.csv`) | MO v1 amostrada (`results_mo/stochastic/`, `results_mo_v1fix/`) | **Modo de avaliação** (decisão A1 do `DECISIONS_MO_V2.md`). As duas versões são válidas e são reportadas lado a lado; o modo principal é o amostrado. `results_mo/stochastic/` foi gerado antes das regras v2 e cobre só a MO e os PPOs fixos; `results_mo_v1fix/` aplica o mesmo modo também à PPO da PoC e guarda os episódios um a um. Para o artigo, usar `results_mo_v1fix/` |

## Regras para não criar novas divergências

1. Toda tabela diz **n**, **seeds** e **protocolo** (modo de avaliação, região, divisão).
2. Usar os nomes do glossário (`docs/PLANO_DESENVOLVIMENTO.md` §2.6): `steps_success_median` ≠ `steps_penalized_median`; `distance_mean_all` ≠ `distance_mean_success`.
3. Comparações entre políticas aprendidas, só no **mesmo modo** de avaliação.
4. ICs do artigo: bootstrap hierárquico. Os ICs antigos podem aparecer só ao lado dos novos, identificados.

# Diagnóstico do condicionamento da MO v1 (P1)

**Data:** 18/11/2026. **Modelo:** `runs_mo/moppo_S2_hand_s0/final_model.pt` (MO com memória, seed 0, 348k passos).
**Resumo:** o condicionamento em w é **fraco no treino** (o gate pré-especificado falhou aos 25%), e a avaliação por argmax o torna **invisível**. Valem as duas afirmações juntas; nenhuma delas deve aparecer sozinha.

---

## 1. Sintoma

No `results_mo/summary_test.csv` (argmax, Taquari S2, 300 episódios), a MO com memória dá **exatamente** 60,0% de sucesso, 25,0% de localização falsa e 1019,7 m em **32 dos 37** pesos da grade. Só os pesos com w_custo ≥ 0,5 e w_info = 0 mudam (o maior desvio é w = [0, 1, 0, 0]: 71,0%, 20,7%, 1128,8 m). Em Porto Alegre, `safe` e `unif` também dão números idênticos nas duas regiões.

## 2. O que foi verificado

| Verificação | Resultado | Evidência |
|---|---|---|
| O gate de condicionamento aos 25% do treino | **falhou**: `fast` = `safe` = 1096,3 m, IC da diferença [0,0; 0,0] | `runs_mo/moppo_S2_hand_s0/gate_check_25pct.json`, `gates.log` |
| Por que o treino não parou | `train_moppo` gravava o JSON e seguia; nenhum código lia o resultado | corrigido: marcador `INVALID_GATE` (A4 do `DECISIONS_MO_V2.md`) |
| O w chega à rede na avaliação? | sim: `run_episode` → `env.reset(options={"w": w})` → `vec[:4]` | `src/dal/mo_eval.py`, `src/dal/mo_env.py` |
| Escala dos objetivos (60 episódios de treino, w uniforme) | desvio do retorno: info 0,42; custo 2,55; sucesso 0,50; risco 0,48. O custo não é desprezível; a escala não explica o colapso | `results_mo/diagnostics/objective_scales.json` |
| A distribuição π(·\|s, w) muda com w? (2023 estados) | pouco: variação total média de 0,027 (`unif`×`safe`) a 0,051 (`unif`×`fast`); 0,26 entre os vértices `only_info`×`only_cost` (máximo 0,33) | `results_mo/diagnostics/weight_sensitivity.csv` |
| A ação mais provável muda com w? | **não** entre `fast`, `safe` e `unif` (100% de concordância); 89% de concordância entre `only_cost` e os demais | idem |
| Entropia da política no treino | ~1,4 nats (máximo para 8 ações: 2,08): a política treinada é bem estocástica | `runs_mo/moppo_S2_hand_s0/progress.csv` |
| Distribuição de w vista no treino (teste D) | Dirichlet(1): algum componente ≥ 0,75 em só 6,3% dos episódios; ≥ 0,9 em 0,1%. Os vértices do simplex são extrapolação | `DECISIONS_MO_V2.md` §3 |
| Coleta vetorizada | autoreset `NEXT_STEP`: ≈ 1 transição espúria por episódio no buffer | `tests/test_mo_autoreset.py`; A7 |

### Comandos

```bash
unset PYTHONPATH
.venv/bin/python scripts/diagnose_mo_conditioning.py --episodes 60      # saída em results_mo/diagnostics/
.venv/bin/python scripts/evaluate_mo_stochastic.py --workers 8          # grade amostrada (MO e PPOs fixos)
.venv/bin/python scripts/evaluate_v1fix.py --workers 8                  # todas as políticas nos dois modos + H6 1-D
.venv/bin/python -m dal.mo_poa_eval --mode sampled --out results_mo_v1fix   # H7 nos dois modos
.venv/bin/python -m dal.mo_poa_eval --mode argmax  --out results_mo_v1fix
```

Saída resumida do diagnóstico (60 episódios na região de treino, seed base 7000):

```
estados analisados: 2023
std:  info 0.424 | cost 2.548 | success 0.504 | risk 0.481
   weights_a    weights_b  mean_total_variation  max_total_variation  argmax_agreement
        unif         fast              0.051012             0.062976          1.000000
        unif         safe              0.026884             0.039351          1.000000
        fast         safe              0.025914             0.030564          1.000000
   only_info    only_cost              0.258996             0.330341          0.892734
```

## 3. Leitura

1. **O condicionamento é fraco, e isso é um achado do treino.** O próprio gate pré-especificado falhou. Entre os presets usados nas hipóteses (`fast`, `safe`, `unif`), a distribuição de ações muda em média 3–5% de massa, e a ação mais provável não muda em nenhum estado. Os 33% são a variação **máxima** entre pesos **extremos** (`only_info` × `only_cost`), que quase não aparecem no treino.
2. **A avaliação por argmax torna o efeito invisível.** Com mudanças tão pequenas em π, o argmax é constante. A avaliação amostrada (o modo do treino) pode revelar diferenças, mas elas vêm desses deslocamentos pequenos. Forma correta no texto: "o gate de condicionamento falhou no treino; a distribuição de ações muda pouco com w e o argmax não muda. Com avaliação amostrada, aplicada a todas as políticas aprendidas, …".
3. **A troca de modo é post hoc** (A1–A3 do `DECISIONS_MO_V2.md`): vale para todas as políticas aprendidas, com números aleatórios comuns por episódio, e os dois modos aparecem em todas as tabelas.
4. **Fatores que podem ter contribuído** (não testados isoladamente): orçamento de 348k passos (¼ da PoC); vértices do simplex quase nunca vistos no treino; transições espúrias do `NEXT_STEP`; e o sinal da vantagem escalarizada, dominado pelo custo, que é o objetivo de maior variância.

## 4. Resultados nos dois modos

*Pendente: preenchido com `results_mo_v1fix/` quando a avaliação terminar (ver `docs/REVISAO_TRABALHO_v1.md` §5).*

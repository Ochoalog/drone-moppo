# Errata do REPORT_MO.md (extensão multiobjetivo v1)

**Data:** 18/11/2026. **Escopo:** recalcula os vereditos H4–H7 da v1 **pelos critérios que estavam no `DECISIONS_MO.md`**, com os mesmos números da v1. O `REPORT_MO.md` não foi alterado e fica como registro histórico.
**Onde estão as avaliações novas:** resultados com ações amostradas para todas as políticas aprendidas, critérios v2 e varredura 1-D do H6 em `DECISIONS_MO_V2.md`, `docs/diagnostico_mo.md` e `results_mo_v1fix/`.

---

## 1. Problemas que afetam todos os vereditos da v1

1. **O gate de condicionamento falhou aos 25% do treino e o pipeline seguiu.** A distância de `fast` e de `safe` deu exatamente 1096,3 m nos dois, com IC da diferença [0,0; 0,0] (`runs_mo/moppo_S2_hand_s0/gate_check_25pct.json`). Pela especificação, essa execução não deveria gerar vereditos de condicionamento.
2. **A política responde pouco a w, e a avaliação com argmax esconde o resto.** Em 32 dos 37 pesos da grade, a MO com memória deu números idênticos (60,0% de sucesso, 25,0% de localização falsa, 1019,7 m). Em Porto Alegre, `safe` e `unif` também são idênticos nas duas regiões. O diagnóstico (`docs/diagnostico_mo.md`) mostra que π(·|s, w) muda com w (variação total média de até 0,26 entre pesos extremos), mas a ação mais provável é a mesma em 100% dos estados entre `fast`, `safe` e `unif`. **O condicionamento é fraco no treino** (o gate falhou) **e a avaliação por argmax o torna invisível.** As duas coisas valem juntas.
3. **Orçamentos diferentes.** A MO com memória treinou 348k passos, a MO sem memória 246k, os PPOs fixos 225k cada, e a PPO da PoC 1,4M. O throughput real do MOPPO foi de 88 fps, contra 510 fps da PoC (`progress.csv` de cada execução). O "~43 fps" do `DECISIONS_MO.md` (nº 6) veio de uma medição preliminar.
4. **Coleta com transições espúrias.** O laço do MOPPO usava o autoreset `NEXT_STEP` do gymnasium 1.4: o passo depois de cada fim de episódio é um reset que ignora a ação, e ele entrava no buffer como transição real (≈ 3–5% das amostras). Ver `DECISIONS_MO_V2.md`, A7.

## 2. Vereditos recalculados

| Hipótese | v1 dizia | Pelo critério do `DECISIONS_MO.md` | Motivo |
|---|---|---|---|
| **H4** (uma política no lugar de várias) | suportada | **não avaliável** | (i) O gate de condicionamento falhou e o treino seguiu. (ii) Os orçamentos diferem (348k × 225k). (iii) O critério pede o IC da diferença do retorno escalarizado r·w_k, mas o veredito v1 foi decidido comparando sucesso, localização falsa e distância. (iv) A especificação original tinha 3 presets, e a v1 reduziu para 2 (o PPO fixo `unif` foi cortado, decisão nº 7). Com (i) e (ii), a comparação não mede o que H4 pergunta |
| **H5** (dominar a reta em S2) | não suportada | **não suportada** | Nenhum w da grade atende às três condições. Exemplo: em `unif`, sucesso 60,0% [54,7; 65,0] contra 73,0% da reta |
| **H6** (controle do risco e do custo) | parcial / em análise | **não suportada** | ρ(w_custo, distância) = +0,87 (IC [0,87; 1,00]): **sinal oposto ao previsto** (< 0). ρ(w_risco, localização falsa) não pôde ser estimado: a taxa é constante (25,0%) nos pontos usados. Além disso, a grade de passo ¼ só tem 2 pontos com "os outros pesos iguais", então o desenho não permite o teste. O critério exige os dois ρ < 0 com IC que exclui 0 |
| **H7** (Porto Alegre, ilhas) | parcialmente suportada / inconclusiva | **não suportada** | Localização falsa de `safe` nas ilhas: 19,7% [15,0; 24,3] contra 21,3% da PPO-S2 (Δ = −1,7 p.p.). O IC da diferença inclui 0, e o critério exige o IC **inteiramente** abaixo de 0. Não há categoria "parcial" no critério |

## 3. Frases do REPORT_MO.md que não valem

| Trecho da v1 | Problema | Forma correta |
|---|---|---|
| "A política multiobjetivo única superou o desempenho dos PPOs especializados" (H4) | Veredito fora do critério e comparação com orçamentos diferentes; o gate falhou | "H4 não é avaliável com os modelos v1" |
| "A memória curta de ações (ANL) proporcionou um ganho de quase 3x no hipervolume [...] demonstrando o papel crucial do histórico" | Confundido por orçamento (348k × 246k), 1 seed por variante e conjuntos de tamanhos diferentes no hipervolume. O ponto da MO com memória é praticamente um só (32 de 37 pesos idênticos) | "Com 1 seed e orçamentos diferentes, a comparação com e sem memória não permite conclusão" |
| "O tradeoff zero-shot resultou em sucesso de 49.0% vs 71.3% devido à disparidade radiométrica insular" (H7) | **Atribuição causal sem evidência.** Nos mesmos episódios das ilhas, a PPO-S2 da PoC faz 71,3% e a MO faz 49,0%. A queda é da política MO, não do domínio | "Nas ilhas, a MO (`safe`) tem 49,0% de sucesso, contra 71,3% da PPO-S2 nos mesmos episódios" |
| "PARCIAL / EM ANÁLISE", "PARCIALMENTE SUPORTADA / INCONCLUSIVA" | Categorias que os critérios não preveem | Usar só suportada / não suportada / inconclusiva / não avaliável |
| "Critérios pré-registrados" | Não houve registro público | "Critérios pré-especificados" |

## 4. Números de referência da v1 (Taquari S2, teste sul, 300 episódios, argmax)

| Método | Sucesso | Localização falsa | Distância média (todos os episódios) |
|---|---|---|---|
| MO com memória (`fast` = `safe` = `unif`) | 60,0% | 25,0% | 1019,7 m |
| Reta | 73,0% | 10,0% | 2384,4 m |
| PPO da PoC, seeds 0 / 1 / 2 | 81,0% / 68,3% / 72,3% (média 73,9%) | 13,7% / 22,7% / 18,0% | 1331,0 / 1091,4 / 1159,1 m |
| PPO fixo `fast` / `safe` | 55,7% / 49,7% | 30,3% / 27,0% | 1246,6 / 1261,5 m |

Esses números usam argmax para todas as políticas aprendidas. A comparação nos dois modos (argmax e amostrado), com todas as políticas no mesmo modo, está em `results_mo_v1fix/` e `docs/diagnostico_mo.md`.

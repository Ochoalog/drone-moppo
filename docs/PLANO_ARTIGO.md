# Plano de escrita do artigo v4 (projeto final de Aprendizado por Reforço, UFRGS)

**Versão:** v4, de 21/11/2026. Substitui a v3 (preservada em `docs/PLANO_ARTIGO_v3.md`).
**Por que mudou:** a v3 vendia um método de percepção ("change-aware"), mas os dados mostram outra coisa:
- com o descritor ajustado, as baselines chegaram ao teto em Taquari (aleatória 94%, reta 99,8% em S2 teste);
- nenhum dos treinos de PPO passou no gate de aprendizado;
- a calibração desligou o filtro ciente da mudança (α = 0).

O artigo de uma disciplina de RL precisa ter o RL no centro. A v4 reorganiza o texto em torno de uma pergunta de RL que os dados respondem, e acrescenta um experimento curto, exploratório e pré-especificado (experimento E) para responder a ela diretamente.

**Formato:** IEEE Conference, duas colunas (decisão do Vitor, V35), **8 páginas de texto + referências**.
**Datas:** congelamento dos resultados na segunda, 23/11, 12:00 (tag `results-final`); artigo em 25/11; apresentação em 27/11.
**Fonte da verdade:** só números de `paper/tables/*.csv` na tag `results-final`, com `source_file` e commit.
**Relação com os seminários:** o método é o do Active Neural Localization (política sobre a crença de um filtro de histograma), o seminário do grupo. A extensão multiobjetivo vem do AMOR (seminário de outro grupo). As duas relações precisam estar explícitas.

---

## 1. Tese, pergunta e título

**Pergunta central:** quando aprender a agir compensa na localização ativa?

**Tese:** o valor de uma política aprendida sobre a crença depende da **ambiguidade perceptual**.
- Com percepção ambígua (estudo v1; Canoas e Porto Alegre na v3), decidir para onde voar importa: a aleatória fica bem abaixo da reta. A PPO aprende a ser mais eficiente que as heurísticas, mas assume mais risco de localização falsa, e o condicionamento em preferências (MOPPO) não conseguiu controlar esse risco.
- Quando a percepção remove a ambiguidade (Taquari na v3), qualquer movimento localiza: sobra pouco valor de informação para a política explorar, e ela não passa no gate.
- Planejar com as previsões do mapa desatualizado (gulosa por ganho de informação) é a pior estratégia nos locais difíceis, pior até que a aleatória.

O experimento E testa a tese diretamente. Se ele não mostrar a vantagem da PPO crescendo com a ambiguidade, a tese recua para a parte descritiva: "a política aprendida supera o movimento sem informação onde há ambiguidade, mas uma heurística de cobertura simples continua difícil de bater". O título 1 funciona nos dois casos.

**Títulos:**
1. *When Does Learning to Act Pay Off? Belief-Space Reinforcement Learning for UAV Localization over Outdated Post-Flood Maps*
2. *Learning Where to Fly When the Map Is Wrong: Active UAV Localization after Floods*

"Change-aware" sai do título e da tese: com α = 0, o filtro CA não foi exercido.

---

## 2. Perguntas de pesquisa e de onde vêm as respostas

| Pergunta | Resposta | Origem | Situação |
|---|---|---|---|
| **RQ1:** com percepção ambígua, uma política aprendida sobre a crença supera heurísticas simples? | PPO ≈ reta em sucesso, com menos passos e menos distância, mas mais localizações falsas; generaliza norte→sul | estudo v1 (H1–H3 v1) | pronta |
| **RQ2:** o que torna o problema difícil, e a política transfere? | o sucesso de todos cai com o descasamento mapa–observação; transferência mantida no Centro, quebrada nas ilhas; gulosa com o mapa é a pior nos locais difíceis | estudo v1 (HP1–HP3, F1–F2, M1 não suportada) + baselines v3 nos 6 alvos | pronta |
| **RQ3:** dá para controlar o equilíbrio eficiência × risco com uma política condicionada em preferências? | não neste orçamento: o gate de condicionamento falhou, o argmax era idêntico entre presets, e a avaliação amostrada só mostra diferenças pequenas | estudo v1 (MOPPO, errata, `results_mo_v1fix/`) | pronta |
| **RQ4:** quando a percepção melhora, o que acontece com o valor da política? E a vantagem dela cresce com a ambiguidade? | Taquari v3: teto das baselines, 0 de 30 treinos no gate; **experimento E**: Canoas (ambiguidade real) e ruído controlado no descritor | v3 (H1, G3, V44, matriz) + **E** | E em andamento |

Os vereditos das hipóteses pré-especificadas (v1: H1–H7, HP1–HP3; v3: H1–H5) vão para uma tabela do apêndice, com a tabela de alterações de critério e de protocolo. No texto principal, cada resultado aparece como resposta a uma RQ.

---

## 3. Experimento E: quando aprender a agir compensa (exploratório, pré-especificado em 21/11)

**Por que existe:** os dois regimes do artigo (v1 e v3) diferem em muitas coisas ao mesmo tempo (grade, sensor, descritor). O experimento E isola a ambiguidade perceptual **dentro do mesmo ambiente v3**, com a mesma PPO, o mesmo orçamento e as mesmas heurísticas.

**Status:** exploratório. Surgiu depois de ver o teto em Taquari, então é declarado como tal no texto. Os critérios abaixo são escritos no `DECISIONS_V3.md` (V46) e commitados **antes** de qualquer treino de E, e o resultado é reportado qualquer que seja.

**Exceção ao gate (decisão do Vitor, V46):** no experimento E, todas as execuções são avaliadas no teste, passem ou não no gate de aprendizado. A taxa de aprovação no gate é ela mesma um resultado (deve subir com a ambiguidade). As hipóteses H3–H5 da v3 continuam sem veredito, como na V45.

### E-real: ambiguidade real (prioridade 1)
- **Canoas C1** (mapa fev/24 → pós-evento 09/05/24), ambiente realista v3, descritor ftA (zero-shot em Canoas), sem altitude, filtro padrão.
- Divisão espacial dentro de Canoas: ao longo do eixo mais longo da caixa, com margem de 200 m e uma faixa de teste que caiba ≥ 1,4 janela. Calibração de β₁ e κ só na faixa de treino, com a regra da PoC e a grade estendida da V33.
- PPO (sem canal de prior), **5 seeds**, 1,4 M passos, os mesmos hiperparâmetros da matriz.
- Heurísticas nos mesmos episódios: reta, zigue-zague, aleatória, greedy_map, reta a 150 m. Teste e treino (o treino serve para escolher a melhor heurística).
- Referência fácil: Taquari S2, com as 10 execuções `mx_A_ppo_s0–9` já treinadas, avaliadas no teste da dobra A (exceção da V46).

### E-synth: ambiguidade controlada (prioridade 2)
- Taquari S2, dobra A. Ruído gaussiano no descritor da **observação** (não no mapa): z' = normalizar(z + σ·g), g ~ N(0, I/D), determinístico por semente, igual para todos os métodos.
- **Escolha dos níveis (regra fixada antes):** varrer σ ∈ {0,25; 0,5; 0,75; 1,0; 1,25; 1,5; 2,0; 3,0}; para cada σ, recalibrar β₁ e κ no treino e medir o sucesso da aleatória no treino. Escolher L1, L2 e L3 como os σ com sucesso da aleatória mais perto de **75%, 55% e 35%**. Nível inválido se a calibração escolher um κ em que a aleatória acerta < 5% (filtro que não concentra); nesse caso, usar o σ válido mais próximo.
- L0 = σ 0 (as execuções `mx_A_ppo` já existentes).
- PPO, 5 seeds por nível; heurísticas nos mesmos episódios.

### E-alt (prioridade 3, só se sobrar GPU)
- PPO com altitude em Canoas, 3 seeds. Motivo: a "reta a 150 m" supera a reta nos locais difíceis (Canoas 70,6% × 64,5%; ilhas 82,8% × 79,1%), enquanto em Taquari a PPO com altitude não sobe. A pergunta é se ela aprende a subir onde subir ajuda.

### Critérios (exploratórios)
Estatística: 1000 episódios pareados por método e local; bootstrap hierárquico seeds → janelas, B = 2000; política nos dois modos (argmax e amostrado), com o amostrado como principal; melhor heurística escolhida no treino de cada local ou nível.

| Id | Pergunta | Critério |
|---|---|---|
| **E1** | Onde há ambiguidade real, a PPO aprende mais que o movimento sem informação? | Canoas: PPO − aleatória ≥ 5 p.p. de sucesso, com limite inferior do IC > 0 |
| **E2** | A vantagem sobre a melhor heurística é maior no regime ambíguo? | Δ = (PPO − melhor heurística) em Canoas menos o mesmo Δ em Taquari S2 > 0, com IC que exclui 0 |
| **E3** | A vantagem cresce com a ambiguidade controlada? | Δ(L3) − Δ(L0) > 0, com IC que exclui 0; a curva Δ × nível é reportada inteira |
| E4 (descritivo) | Em que níveis a PPO iguala ou supera a melhor heurística, e a que custo? | sucesso, localização falsa e tempo de voo por nível, sem veredito |
| E5 (descritivo) | O gate passa mais vezes com mais ambiguidade? | fração de execuções aprovadas no gate (critério original) por nível e local |

---

## 4. Estrutura do artigo (8 páginas)

| Seção | Pág. | Conteúdo | Figuras e tabelas |
|---|---|---|---|
| **Abstract** | — | Pergunta, abordagem (PPO sobre a crença, à la ANL), dados reais de enchente, 3 achados com números, conclusão | — |
| **I. Introduction** | 0,7 | Drone sem GNSS e mapa anterior ao desastre; localização ativa como POMDP; pergunta "quando aprender a agir compensa?". Contribuições: (i) adaptação do ANL com PPO a imagens reais antes/depois de enchente em 4 locais; (ii) comparação pareada com heurísticas fortes, mostrando eficiência × risco; (iii) resultado negativo do MOPPO, com diagnóstico; (iv) evidência de que o valor da política depende da ambiguidade perceptual (v3 + experimento E) | — |
| **II. Background and Related Work** | 0,8 | POMDP e filtro de histograma; ANL e Deep Active Localization; PPO; MORL e AMOR; localização visual de UAVs (revisões, UAV-VisLoc, AnyLoc); AiRLoc; avaliação confiável em RL (rliable) | — |
| **III. Problem and Environments** | 0,8 | POMDP, ações, recompensa, sucesso, localização falsa, regra de parada pela crença. Dois ambientes: **v1** (grade de 50 m, descritor artesanal, 8 ações) e **v3** (25 m, descritor DINOv2 ajustado, footprint por altitude, deriva de odometria, tempo de voo). Dados: Taquari, Porto Alegre (ilhas, Centro), Canoas | Tab. I (locais, datas, papel de cada imagem, divisões) |
| **IV. Methods** | 0,7 | PPO sobre a crença (entradas como no ANL: crença, máscara, distintividade); heurísticas (reta, zigue-zague, aleatória, gulosa por ganho de informação com o mapa); MOPPO (crítico vetorial, pesos Dirichlet, memória de ações); gate de aprendizado; ambiente em lote em GPU com teste de paridade | Fig. 1 (visão geral: câmera → descritor → filtro → política) |
| **V. Experimental Protocol** | 0,4 | Episódios pareados, números aleatórios comuns; bootstrap hierárquico; dois modos de avaliação; critérios pré-especificados e o status exploratório do experimento E | — |
| **VI-A. RQ1: learning to act under ambiguity** | 0,6 | v1: PPO × heurísticas em S1 e S2; passos, distância nos sucessos, localização falsa; generalização norte→sul | Tab. II |
| **VI-B. RQ2: what makes it hard, and transfer** | 0,6 | v1: sucesso × índice de mudança; zero-shot em Porto Alegre; M1 não suportada. v3: baselines nos 6 alvos; a gulosa com o mapa é a pior nos locais difíceis | Fig. 2 |
| **VI-C. RQ3: controlling risk with preferences** | 0,4 | MOPPO: gate de condicionamento falhou; argmax idêntico; avaliação amostrada no mesmo modo para todos | linha curta na Tab. II ou Tab. III |
| **VI-D. RQ4: when does learning to act pay off?** | 0,9 | Taquari v3: teto das baselines e 0 de 30 no gate (H1 em uma frase); **experimento E**: Canoas × Taquari, curva da vantagem × nível de ruído, taxa de aprovação no gate | **Fig. 3** (figura principal: vantagem da PPO sobre a aleatória e sobre a melhor heurística × ambiguidade, com IC) + Tab. III |
| **VII. Discussion** | 0,7 | Valor da informação e ambiguidade; por que a reta é forte (cobre terreno novo e não confia no mapa); a política achou eficiência, não robustez; lições de avaliação em RL (gates, argmax × amostrado, seeds, ICs hierárquicos); limitações; ameaças à validade | — |
| **VIII. Conclusion** | 0,2 | Respostas às 4 RQs e trabalhos futuros | — |
| **Apêndice ou repositório** | — | Percepção da v3 (G1, ajuste fino, H1), prior de inundação e α = 0, vereditos de todas as hipóteses, tabela de alterações (V22–V46), hiperparâmetros | — |

**Orçamento visual:** 3 figuras + 3 tabelas no corpo (≈ 1,75 página).

---

## 5. Números que já podem ir para o texto (conferir na tabela canônica antes)

| Resultado | Valor | Fonte |
|---|---|---|
| v1, S2: sucesso PPO × reta × aleatória | 70,7% × 70,0% × 41,7% | `REPORT.md` |
| v1, S2: IC hierárquico de Δsucesso PPO − reta | [−7,1; +9,8] p.p. (B = 2000) | V18; **corrigir o valor antigo [−6,1; +9,8] na tag `v1-study-final`** |
| v1, S2: localização falsa PPO × reta | 20,2% × 13,0% | `REPORT.md` |
| v1, S2: distância nos sucessos PPO × reta | 1,25 × 1,89 km | `docs/conciliacao_numeros.md` nº 7 |
| v1, S2: passos penalizados (Wilcoxon) | p = 5,6e-6 | `REPORT.md` |
| v1, generalização norte→sul (S2) | 0,701 → 0,707 | `REPORT.md` |
| v3, Taquari S2 teste: aleatória × reta × zigue-zague × gulosa | 94,0% × 99,8% × 97,3% × 83,4% | `colab_out/eval_baselines_A/summary.csv` |
| v3, Canoas C1: aleatória × reta × reta a 150 m × gulosa | 40,8% × 64,5% × 70,6% × 36,9% | idem |
| v3, ilhas P1: aleatória × reta × reta a 150 m × gulosa | 44,4% × 79,1% × 82,8% × 43,3% | idem |
| v3, Centro P1: aleatória × reta × gulosa | 27,8% × 67,9% × 35,8% | idem |
| v3, gate de aprendizado | 0 de 30 execuções aprovadas (16 de 30 até 21/11 20:51; confirmar) | `colab_out/mx_A_*` |
| v3, H1 | +9,2 p.p. de top-1 em S1 e em S2 | `results_v3/verdicts/h1.json` |

---

## 6. Registro de afirmações

| Afirmação | Pode? | Forma correta |
|---|---|---|
| "O RL supera as heurísticas" | **não** (v1 e v3) | "A PPO não se distingue da reta em sucesso (IC [a; b]) e usa menos passos" |
| "O RL é mais eficiente" | sim, qualificada | sempre ao lado da localização falsa maior |
| "O RL aprende onde há ambiguidade" | só se E1 for suportada | número de Canoas com IC |
| "A vantagem do RL cresce com a ambiguidade" | só se E2 ou E3 forem suportadas | dizer qual, e que E é exploratório |
| "Planejar com o mapa desatualizado é pior que andar ao acaso" | sim, só nos locais em que os dados mostram | Canoas, Centro, ilhas (gulosa < aleatória); em Taquari, não |
| "Subir ajuda nos locais difíceis" | sim, para a reta | reta a 150 m × reta; a PPO com altitude só se E-alt rodar |
| "Change-aware" / "o prior de inundação melhora a localização" | **não** | "a calibração escolheu α = 0; o efeito do prior não foi testado na prática" |
| "Melhor percepção" na v3 | só para recuperação | H1 vale na grade de 50 m da v1 |
| "O MOPPO domina" / "controla o risco" | **não** | errata e diagnóstico |
| "Pré-registrado" | **não** | "pré-especificado"; E é exploratório |
| "Primeiro trabalho a…" | **não** | "até onde sabemos, …" |

---

## 7. Divisão e cronograma de escrita

| Dia | O quê | Quem |
|---|---|---|
| Sáb 21 (noite) | V46 commitado; jobs de E na fila | IA-1, Vitor aprova |
| Dom 22 | II, III e IV escritos; VI-A, VI-B e VI-C com os números do v1 e das baselines; figuras 1 e 2 | Vitor + IA-1 (texto), IA-2 (relacionados) |
| Seg 23, até 12:00 | Resultados de E; **congelamento**; Fig. 3 e Tab. III | IA-1 |
| Seg 23 (tarde) | VI-D, VII, VIII, Abstract | Vitor |
| Ter 24 | Conferência número a número contra a §5 e a §6; revisão cruzada | todos |
| Qua 25 | PDF final, release `v1.0` | Vitor |
| Até 27 | Slides e ensaio (a mesma estrutura das 4 RQs) | Vitor |

## 8. Se o tempo apertar

Cortar nesta ordem: E-alt → E-synth (fica só L0 e L3) → E-synth inteiro (fica só E-real). Sem nenhum E, a seção VI-D fica descritiva (Taquari no teto + baselines difíceis em Canoas e Porto Alegre), e a tese é dita como hipótese para trabalho futuro.

## 9. Checklist de entrega

- [ ] RL no centro: cada seção de resultados responde a uma RQ de RL
- [ ] Relação com o ANL (e com o AMOR) explícita
- [ ] V46 commitado antes de qualquer treino de E; E chamado de exploratório no texto
- [ ] Todo número confere com `paper/tables/` na tag `results-final`
- [ ] Localização falsa ao lado de todo ganho de eficiência
- [ ] "Change-aware" fora do título e da tese
- [ ] IC [−7,1; +9,8] corrigido em todo lugar
- [ ] Tabela de alterações e vereditos de todas as hipóteses no apêndice ou no repositório
- [ ] Dentro de 8 páginas; compila sem warnings

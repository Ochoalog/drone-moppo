# Plano de escrita do artigo (v2)

**Versão:** v2, de 18/11/2026. Substitui a v1 (histórico no git). Principais mudanças: o artigo passa a ser organizado em **3 perguntas de pesquisa** com uma tese central; entra um **registro de afirmações** (§4) que diz o que pode e o que não pode ser escrito; a MO deixa de ser manchete; os trabalhos relacionados ganham referências verificadas; as ameaças à validade foram corrigidas.

**Formato:** o escopo já foi validado com o professor (18/11); o template ainda precisa ser confirmado (ver `PLANO_DESENVOLVIMENTO.md` §2.11). O enunciado sugere o template arXiv/Prime AI; a proposta do grupo é IEEE Conference (`\documentclass[conference]{IEEEtran}`), duas colunas, **6 páginas de texto + referências**. A estrutura abaixo serve para os dois.
**Idioma:** confirmar com o professor. O padrão IEEE é inglês.
**Fonte da verdade:** só entram números da tag `results-final`, lidos de `paper/tables/*.csv` (gerados por `scripts/make_canonical_tables.py`). Cada número do texto tem a linha da tabela de onde saiu.

---

## 1. Tese e título

**Tese (uma frase, guia todas as seções):** quando o mapa de referência foi desatualizado por um desastre, **estratégias que confiam no mapa antigo para decidir para onde voar perdem robustez exatamente onde o mapa mais mudou**, e uma heurística que não confia no mapa (a linha reta) é uma baseline difícil de bater. A política PPO ganha em eficiência (menos passos e menos distância) onde o mapa ainda vale em boa parte, ao custo de mais localizações falsas.

Essa tese só fica no texto se a análise de mecanismo (`PLANO_DESENVOLVIMENTO.md` §2.7, M1) a sustentar. Se M1 não for suportada, a tese recua para a parte descritiva: "o desempenho de todos os métodos é governado pelo descasamento entre mapa e observação, e o RL não muda essa relação".

**Títulos propostos** (escolher depois dos resultados v2):
1. *Does Active Localization Help When the Map Is Outdated? A Post-Flood Benchmark for GNSS-Denied UAVs*
2. *Active Global Localization of GNSS-Denied UAVs over Outdated Post-Disaster Maps: When Does Reinforcement Learning Pay Off?*

Evitar "Multi-Objective RL" no título: é a parte com o resultado mais frágil.

---

## 2. Perguntas de pesquisa e hipóteses

O artigo usa os identificadores da coluna "Artigo". Os relatórios do repositório continuam com os identificadores originais; a tabela é o mapa entre os dois.

| Pergunta | Artigo | Original | Enunciado curto | Veredito atual | Fonte |
|---|---|---|---|---|---|
| **RQ1:** a localização ativa por RL supera heurísticas simples quando o mapa está desatualizado? | H1.1 | H1 | PPO bate cada baseline simples (S1, S2) | não suportada | `REPORT.md` |
| | H1.2 | H3 | Lacuna de generalização norte→sul pequena **e** PPO ainda vence no sul | não suportada (só pela 2ª parte; a lacuna é pequena) | `REPORT.md` |
| | H1.3 | HP1 | A relação PPO × reta se mantém zero-shot em Porto Alegre | inconclusiva (mantida no Centro, quebrada nas ilhas) | `REPORT_POA.md` |
| **RQ2:** o que governa a dificuldade? | H2.1 | HP2 | O sucesso cai com o índice de mudança; ilhas mais difíceis que o Centro | suportada | `REPORT_POA.md` |
| | H2.2 | H2 | A vantagem do RL cresce S0 < S1 < S2 | inconclusiva (eixo confundido, ver §5) | `REPORT.md` |
| | H2.3 | HP3 | A vantagem ativa é maior nas ilhas | não suportada (é menor) | `REPORT_POA.md` |
| | M1 | — (novo, exploratório) | Nas localizações falsas, a PPO visita células mais distintivas no mapa antigo e mais mudadas | a avaliar | `results_final/mechanism/` |
| **RQ3:** uma política condicionada em preferências controla o equilíbrio rapidez × risco? | H3.1 | H4 | MO(w_k) ≈ PPO fixo(w_k) | v1: não avaliável; v2: a avaliar | `REPORT_MO_ERRATA.md`, `results_mo_v2/` |
| | H3.2 | H5 | Algum w domina a reta em S2 | não suportada | idem |
| | H3.3 | H6 | Localização falsa cai com w_risco; distância cai com w_custo | v1: não suportada; v2: a avaliar | idem |
| | H3.4 | H7 | `w_seguro` reduz a localização falsa nas ilhas | não suportada | idem |

**Regras de texto para os vereditos:**
- Dizer "critérios **pré-especificados**", nunca "pré-registrados".
- Uma tabela de alterações de critério e de protocolo (HP1 depois do smoke; H4 de 3 para 2 presets na v1; avaliação da MO trocada de argmax para amostrada depois do diagnóstico de 18/11; H6 trocado para varredura 1-D na v2), com data e motivo. Quando houver alteração, reportar o veredito pelos dois critérios.
- H1.2: escrever as duas partes separadas. A lacuna de generalização é pequena (S2: 0,701 no norte → 0,707 no sul) e isso é um resultado positivo; o veredito "não suportada" vem de não vencer a reta no sul.
- A mensagem de que "a maioria das hipóteses não foi suportada" vale, mas sempre com o porquê de cada uma.

---

## 3. Estrutura das seções e orçamento de páginas

Orçamento: **4 figuras + 3 tabelas** no corpo (≈ 1,75 página). O resto vai para o repositório (link no rodapé) ou para um apêndice, se o template permitir.

### Abstract (150–250 palavras, sem citações), escrever por último
Problema (drone sem GNSS, mapa anterior à enchente) → abordagem (filtro de histograma + política PPO que escolhe para onde voar; extensão condicionada em preferências) → dados (Maxar 0,3 m/px, Taquari e Porto Alegre, 2023–2024) → **três** resultados com números e ICs: (1) em S2, sucesso estatisticamente indistinguível da reta, com menos passos e menos distância **nos episódios com sucesso**, mas mais localizações falsas; (2) o sucesso de todos os métodos cai com o descasamento mapa–observação, e a vantagem do RL some onde o mapa mais mudou (ilhas); (3) o resultado da MO conforme o desfecho (§6). Nunca "sucesso equivalente" sem margem de equivalência pré-especificada: usar "não distinguível (IC da diferença [a; b] p.p.)".

### Index Terms
Active localization, reinforcement learning, UAV, visual geo-localization, outdated maps, disaster response.

### I. Introduction (~0,7 página)
1. Motivação: enchentes do RS (Taquari em 2023–2024; Porto Alegre em maio/2024); drones de resposta sem GNSS; a única referência é a imagem **anterior** ao desastre.
2. Lacuna, com cuidado: a localização ativa aprendida (ANL; Deep Active Localization) foi avaliada com mapa correto; a localização aérea por RL (AiRLoc) busca um alvo, não a própria pose sobre um mapa desatualizado. "Até onde sabemos, nenhum trabalho avalia localização ativa sobre mapas desatualizados por um desastre real." **Nunca "ninguém testou".**
3. As 3 perguntas de pesquisa (uma frase cada).
4. Contribuições:
   (i) um benchmark de localização ativa com imagens reais antes e depois de enchentes, em dois locais (rural e urbano/ilhas), com um índice contínuo de descasamento mapa–observação;
   (ii) a adaptação do ANL (filtro de histograma + PPO) a esse domínio, com protocolo pareado, bootstrap hierárquico e critérios pré-especificados;
   (iii) evidência de **quando** o RL ajuda e quando não ajuda, com análise de mecanismo;
   (iv) uma extensão condicionada em preferências (à la AMOR) e o que ela mostra **ou não mostra** sobre controlar o risco de localização falsa.
5. Rodapé com o link do repositório.

### II. Related Work (~0,7 página)
- **A. Localização ativa:** Markov localization ativa (Burgard, Fox, Thrun); *Probabilistic Robotics*; ANL (Chaplot et al., 2018); Deep Active Localization (Gottipati et al., 2019).
- **B. Localização visual absoluta de UAVs:** as duas revisões de Couturier & Akhloufi (2021, 2024); trabalhos de casamento drone–satélite. A maioria é **passiva**. AiRLoc (Pirinen et al.) como o vizinho mais próximo: RL com vistas aéreas parciais, motivado por busca e salvamento, mas o problema é achar um alvo, não a própria pose, e o mapa não muda.
- **C. Mudança de aparência e mapas desatualizados:** localização sob mudança sazonal/temporal (pesquisar 1–2 trabalhos de 2020–2025; conferir na fonte).
- **D. RL multiobjetivo:** Hayes et al. (guia prático), políticas condicionadas em preferências, AMOR (Alegre et al., 2025), hipervolume e EUM (MORL-Baselines).
- Parágrafo de posicionamento: o que só este trabalho combina (mapa desatualizado por desastre real + decisão ativa + avaliação pareada contra heurísticas fortes).

### III. Problem Formulation (~0,5 página)
- POMDP: estado = pose na grade de 50 m; observação = recorte nadir de 150 m da imagem pós-evento; 8 ações; horizonte de 80 passos.
- Crença por filtro de histograma com verossimilhança temperada, β = κ·β₁ (κ = 0,25 em S1/S2), e **por que** temperar (observações sucessivas correlacionadas tornam o filtro superconfiante). Equações numeradas: atualização bayesiana e entropia.
- Definições formais de sucesso, localização falsa e timeout (glossário do `PLANO_DESENVOLVIMENTO.md` §2.6). Deixar explícito que a regra de parada usa só a crença (o drone não sabe se a localização é falsa).
- Recompensa escalar e a decomposição em r_t ∈ ℝ⁴ (informação, custo, sucesso, risco); w ~ Dirichlet(1).

### IV. Method and Data (~1,1 página)
- **A. Dados e cenários:** **Tabela I** (datas Maxar, papel de cada imagem, cobertura de nuvens, alinhamento, divisão norte/sul, regiões de Porto Alegre e sobreposição de 18,7 km²). Dizer explicitamente que **S0 é um teste de sanidade** (mesma imagem) e que S1 e S2 **não** são "níveis de desatualização" ordenados no tempo: S2 (out/23) é mais recente que S1 (mar/23), mas tem lama da enchente de set/23, ~17% de nuvens e 5 m de desalinhamento. Por isso o eixo principal do artigo é o índice de mudança.
- **B. Descritor e índice de mudança:** descritor "hand" (gradientes + cor); a ResNet18 foi descartada (top-1 de 10% contra 41% em S1). Índice de mudança = 1 − cos(descritor do mapa, observação). Declarar que ele depende do descritor e apresentar a covariável independente (fração de água; diferença de cor após igualar histogramas).
- **C. Agente PPO:** entrada (crença, máscara, distintividade e PCA-4 do mapa), CNN, hiperparâmetros principais numa linha; tabela completa no repositório.
- **D. Extensão condicionada em preferências:** MOPPO à la AMOR, crítico vetorial, memória de 8 ações (como no ANL), presets.
- **E. Baselines:** reta, zigue-zague, aleatória, gulosa de ganho de informação com o mapa (`greedy_map`, justa) e gulosa com informação privilegiada (referência).
- **Fig. 1:** pipeline (câmera → descritor → filtro → política → ação) + recortes antes/depois. Uma figura só para as duas coisas.

### V. Experimental Protocol (~0,4 página)
- Episódios pareados (semente 2024 + i fixa janela, início e ruído); 1000 no teste do Taquari, 500 em Porto Alegre, 300 na MO.
- **Estatística:** IC 95% por bootstrap **hierárquico** (seeds → janelas), diferenças pareadas, Wilcoxon nos passos penalizados, Holm onde há muitas comparações. Resultados por seed na tabela.
- Métricas: sucesso, localização falsa, mediana de passos nos sucessos, distância média nos sucessos (e em todos os episódios, no repositório), tempo de decisão; EUM (e hipervolume entre conjuntos de mesmo tamanho) para a MO.
- Hardware: CPU de notebook (i7-10510U); ~45 min por seed de PPO. Vale como argumento de baixo custo.

### VI. Results (~1,6 página)
- **A. RQ1:** **Tabela II** (Taquari S1/S2 e Porto Alegre P1 por região: sucesso, localização falsa, passos, distância nos sucessos, com IC e n). Frase de veredito para H1.1–H1.3.
- **B. RQ2:** **Fig. 2** (figura central): sucesso × quintil do índice de mudança, os dois locais, todos os métodos. **Fig. 3:** mecanismo (M1): distintividade e mudança das células visitadas, sucesso × localização falsa, PPO × reta. Vereditos H2.1–H2.3 e M1.
- **C. RQ3:** **Fig. 4** (projeções sucesso × localização falsa e sucesso × distância, com IC) e **Tabela III** (EUM, hipervolume de mesmo tamanho, H3.1–H3.4). Conteúdo conforme o desfecho (§6).
- Trajetórias qualitativas: só se sobrar espaço; senão, link para o vídeo no repositório.

### VII. Discussion (~0,6 página)
- **Por que a reta é forte:** cobre terreno novo a cada passo e não depende das previsões do mapa antigo; o filtro faz o trabalho de inferência. As estratégias que planejam com o mapa (`greedy_map`, PPO usando a distintividade) herdam os erros do mapa: a `greedy_map` é a pior em S2 (32,5% de localização falsa).
- **O equilíbrio central:** menos passos e menos distância × mais localizações falsas (20,2% × 13,0% em S2). É a motivação para controlar o risco.
- **O que a MO mostrou ou não mostrou** (§6).
- **Limitações:** grade 2D discreta, sem dinâmica de voo nem altitude; uma data pós-evento por local; descritor artesanal; poucas seeds; treino só em CPU; recompensa ajustada à mão; escalarização linear cobre só a parte convexa da fronteira.
- **Ameaças à validade:**
  - dependência espacial: um único corte norte/sul e janelas sobrepostas (mitigada pelo bootstrap por janela);
  - em Porto Alegre, padronização transdutiva dos descritores (usa a caixa inteira, incluindo as regiões avaliadas);
  - sobreposição das regiões de Porto Alegre e recalibração do β não totalmente disjunta;
  - desalinhamento **local** (prédios inclinados no Centro); o erro global de 2,5–5 m é só 5–10% de uma célula;
  - critérios alterados depois do smoke (tabela de alterações);
  - o índice de mudança depende do descritor (mitigado pela covariável independente).

### VIII. Conclusion (~0,2 página)
Responder às 3 perguntas em 3–4 frases. Trabalhos futuros: descritor aprendido entre datas, ações contínuas e altitude, planejamento que pondera a confiança no mapa (por exemplo, usar o índice de mudança esperado como entrada), mais eventos de desastre, teste com drone real.

### Acknowledgment
Professor da disciplina, UFRGS, Maxar Open Data e OpenAerialMap.

---

## 4. Registro de afirmações (o que pode e o que não pode ser escrito)

| Afirmação | Pode? | Forma correta e evidência |
|---|---|---|
| "O RL reduz a distância voada pela metade" | só qualificada | "Nos episódios com sucesso, a PPO voa X km contra Y km da reta" (coluna `distance_mean_success`). A média em todos os episódios (1,24 × 2,37 km) mistura episódios que terminam cedo por localização falsa |
| "Sucesso equivalente à reta em S2" | **não** | "Sucesso não distinguível da reta: diferença [a; b] p.p." (IC hierárquico) |
| "A PPO precisa de menos passos que a reta em S2" | sim | Wilcoxon nos passos penalizados, p = 5,6e-6 (recalcular com a versão final) |
| "A PPO generaliza do norte para o sul" | sim | S2: 0,701 → 0,707; S1: 0,917 → 0,840, com a reta também caindo (0,980 → 0,927) |
| "O RL tem mais localizações falsas" | sim, sempre que falar de ganho | 20,2% × 13,0% em S2 |
| "A dificuldade é governada pela mudança do mapa" | sim, como correlação | Spearman e quintis (H2.1); nunca causal sem a covariável independente |
| "O MOPPO domina os outros métodos" | **não** | Os dados v1 contradizem (MO 60% × reta 73% × PPO da PoC ~74%, nos mesmos episódios) |
| "A memória triplica o hipervolume" | **não** | Confundido por orçamento (348k × 246k), 1 seed e conjuntos de tamanho diferente |
| "A MO reduz a localização falsa nas ilhas" | **não** | IC da diferença inclui 0 (H3.4 não suportada) |
| "A avaliação com argmax apagou o condicionamento" | só qualificada | "O gate de condicionamento falhou no treino; a distribuição de ações muda pouco com w e o argmax não muda. Com avaliação amostrada (aplicada a todas as políticas aprendidas), …" |
| Comparar a MO amostrada com a PPO da PoC em argmax | **não** | Mesmo modo de avaliação para todas as políticas aprendidas; mostrar os dois modos |
| "Primeiro trabalho a…" | **não** | "Até onde sabemos, …" |
| "Hipóteses pré-registradas" | **não** | "Critérios pré-especificados", com a tabela de alterações |

Toda afirmação do resumo, da introdução e da conclusão entra nesta tabela antes da versão final.

---

## 5. Pontos de método que o texto precisa deixar claros

1. **S0–S2 não formam uma escala.** S0 é sanidade; S2 é mais difícil por lama, nuvens e alinhamento, não por ser mais antigo. Por isso H2.2 é inconclusiva por desenho, e o artigo usa o índice contínuo.
2. **O índice de mudança e o descritor.** Usar o termo "descasamento mapa–observação" ao interpretar o índice.
3. **Distância e localização falsa andam juntas.** Sempre reportar as duas lado a lado.
4. **Seeds.** A variação entre seeds é grande (S2: 81,0% / 68,3% / 72,3% na reavaliação de 300 episódios). Mostrar por seed.
5. **Por que κ.** Sem o temperamento, o filtro concentrava em 1–4 passos com 27–46% de localização falsa (`DECISIONS.md`).

---

## 6. Dois roteiros para a parte multiobjetivo (escolher depois do P3)

**Desfecho A: o retreino v2 passa nos gates e a MO responde a w.**
Seção VI-C mostra a fronteira e a resposta a w (H3.3 pela varredura 1-D), H3.1 pelo critério v2 com o critério original ao lado, e a errata da v1 numa frase ("uma primeira versão, treinada com ~¼ do orçamento, falhou no gate de condicionamento: a distribuição de ações mudava pouco com w e a ação mais provável não mudava").

**Desfecho B: plano B (`PLANO_DESENVOLVIMENTO.md` §2.9).**
Seção VI-C curta (~0,3 página): o condicionamento não foi demonstrado; o diagnóstico (gate de condicionamento que falhou, argmax idêntico entre presets, diferenças só na distribuição amostrada, orçamento); H3.1–H3.4 como não suportadas ou não avaliáveis; o que seria preciso (orçamento, normalização de w, amostragem de pesos). A discussão trata isso como barreira, como o enunciado pede.

Em nenhum dos dois a MO entra no título ou na primeira frase do resumo.

---

## 7. Figuras e tabelas → arquivos

| Item | Conteúdo | Origem |
|---|---|---|
| Fig. 1 | Pipeline + recortes antes/depois | **a criar** (TikZ ou draw.io) + `results/figs/data_overview.png` |
| Fig. 2 | Sucesso × quintil do índice de mudança (Taquari e Porto Alegre) | `results_final/change_index/` |
| Fig. 3 | Mecanismo M1 | `results_final/mechanism/` |
| Fig. 4 | Projeções da fronteira (MO v2 ou v1 com errata) | `results_mo_v2/figs/` ou `results_mo_v1fix/` |
| Tab. I | Dados e cenários | `REPORT.md` §1, `REPORT_POA.md` |
| Tab. II | RQ1: Taquari e Porto Alegre com IC hierárquico e por seed | `paper/tables/rq1.csv` |
| Tab. III | RQ3: EUM, hipervolume de mesmo tamanho, vereditos | `paper/tables/rq3.csv` |

Figuras em PDF vetorial, legíveis numa coluna (3,5 in), fonte ≥ 8 pt, paleta segura para daltônicos. `paper/make_figures.py` regenera tudo a partir dos CSVs.

---

## 8. Referências

Estilo IEEE numérico (`IEEEtran.bst`), via BibTeX. **Conferir cada uma na fonte original.** As marcadas com ✓ já foram conferidas em 18/11.

1. D. S. Chaplot, E. Parisotto, R. Salakhutdinov, "Active Neural Localization," ICLR 2018.
2. L. N. Alegre et al., "AMOR: Adaptive Character Control through Multi-Objective Reinforcement Learning," SIGGRAPH 2025 (autores e título exatos no PDF da raiz do projeto).
3. ✓ S. K. Gottipati, K. Seo, D. Bhatt, V. Mai, K. Murthy, L. Paull, "Deep Active Localization," *IEEE Robotics and Automation Letters*, 2019, doi:10.1109/LRA.2019.2932575.
4. ✓ A. Pirinen, A. Samuelsson, J. Backsund, K. Åström, "Aerial View Localization with Reinforcement Learning: Towards Emulating Search-and-Rescue," arXiv:2209.03694 (conferir o local de publicação final; há versão no SAIS 2023).
5. ✓ A. Couturier, M. A. Akhloufi, "A review on absolute visual localization for UAV," *Robotics and Autonomous Systems*, vol. 135, 103666, 2021.
6. ✓ A. Couturier, M. A. Akhloufi, "A Review on Deep Learning for UAV Absolute Visual Localization," *Drones*, vol. 8, no. 11, 622, 2024, doi:10.3390/drones8110622.
7. W. Burgard, D. Fox, S. Thrun, "Active Mobile Robot Localization," IJCAI 1997.
8. S. Thrun, W. Burgard, D. Fox, *Probabilistic Robotics*, MIT Press, 2005.
9. L. P. Kaelbling, M. L. Littman, A. R. Cassandra, "Planning and acting in partially observable stochastic domains," *Artificial Intelligence*, 1998.
10. J. Schulman et al., "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017.
11. A. Raffin et al., "Stable-Baselines3," JMLR, 2021.
12. C. F. Hayes et al., "A practical guide to multi-objective reinforcement learning and planning," JAAMAS, 2022.
13. F. Felten et al., MORL-Baselines e MO-Gymnasium, NeurIPS 2023 (Datasets and Benchmarks) — para hipervolume e EUM.
14. 1–2 trabalhos de localização sob mudança de aparência (a pesquisar).
15. Fonte dos dados: Maxar Open Data / OpenAerialMap (CC BY-NC 4.0).

---

## 9. Divisão de trabalho e prazos

| Seção | Autor | Revisor | Rascunho até |
|---|---|---|---|
| III Formulação + IV Método e dados | ____ | ____ | Dom 22/11 |
| V Protocolo + VI Resultados (depende do congelamento) | ____ | ____ | Dom 22/11 |
| II Trabalhos relacionados + referências | ____ | ____ | Sex 20/11 (não depende de resultados: começar já) |
| I Introdução + VII Discussão + VIII Conclusão | ____ | ____ | Seg 23/11 |
| Abstract, registro de afirmações, figuras finais, formato | ____ | todos | Ter 24/11 |

**Ordem de escrita:** Trabalhos relacionados (já) → Método → Resultados → Discussão → Introdução → Conclusão → Abstract.
Se a apresentação for antes de quarta 25/11, ver `PLANO_DESENVOLVIMENTO.md` §5.1.

## 10. Checklist de submissão

- [ ] Template e idioma confirmados com o professor (escopo já validado em 18/11)
- [ ] Todo número do texto confere com `paper/tables/` da tag `results-final`
- [ ] Toda afirmação do resumo, da introdução e da conclusão está no registro (§4) com a forma correta
- [ ] ICs de 95% (hierárquicos), n e número de seeds em todas as tabelas
- [ ] Tabela de alterações de critério presente
- [ ] Localização falsa reportada ao lado de todo ganho de distância ou passos
- [ ] Limitações e ameaças à validade explícitas
- [ ] Referências conferidas uma a uma
- [ ] Link do repositório no rodapé da introdução
- [ ] Dentro do limite de páginas; compila sem warnings

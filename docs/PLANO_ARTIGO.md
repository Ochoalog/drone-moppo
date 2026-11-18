# Plano de escrita do artigo (padrão IEEE)

**Formato:** IEEE Conference, classe `IEEEtran` (`\documentclass[conference]{IEEEtran}`), duas colunas, **6 a 8 páginas** sem contar as referências. O template está no Overleaf: "IEEE Conference Template".
**Idioma:** **confirmar com o professor**. O padrão IEEE é inglês. Se for em português, manter a estrutura e as convenções abaixo.
**Fonte da verdade:** só entram números da tag `results-final` (ver `PLANO_DESENVOLVIMENTO.md` §6). Cada número do texto aponta para o arquivo de onde saiu (tabela da seção 4).

> **Observação:** o enunciado sugere o template arXiv/Prime AI. Usar IEEE é uma escolha do grupo, e é bom avisar o professor. As seções exigidas (resumo, introdução, trabalhos relacionados, metodologia, resultados, discussão e conclusão) estão todas cobertas abaixo.

---

## 1. Título (proposta)

*Active Global Localization of GNSS-Denied UAVs over Outdated Post-Disaster Maps with Multi-Objective Reinforcement Learning*

(Em português: *Localização global ativa de drones sem GNSS sobre mapas desatualizados pós-desastre com aprendizado por reforço multiobjetivo*.)

---

## 2. Estrutura das seções

### Abstract (150–250 palavras, um parágrafo, sem citações)
Escrever **por último**. A ordem é: problema (drone sem GNSS e mapa anterior à enchente) → abordagem (filtro de histograma + política PPO que escolhe para onde voar, e uma extensão MOPPO condicionada em pesos) → dados (Maxar 0,3 m/px, Vale do Taquari e Porto Alegre, enchente de maio/2024) → resultado principal com números → conclusão honesta (a linha reta é uma baseline difícil; o RL reduz pela metade a distância voada com sucesso equivalente em S2).

### Index Terms (4–6)
Reinforcement learning, active localization, UAV, multi-objective RL, visual geo-localization, disaster response.

### I. Introduction (~0,75 página)
1. **Motivação:** enchentes do RS em 2024; drones de resposta perdem GNSS (interferência, falha); a única referência disponível é a imagem de satélite **anterior** ao desastre.
2. **Lacuna:** a localização ativa (ANL) foi avaliada em labirintos e em ambientes 3D sintéticos com mapa correto. Ninguém testou mapa desatualizado por um desastre real, nem o trade-off entre rapidez, distância e risco de localização falsa.
3. **Pergunta e hipóteses:** escrever uma frase por hipótese, H1–H7 (ver a seção 3 deste plano).
4. **Contribuições** em lista:
   (i) um benchmark de localização ativa com imagens reais pré e pós-enchente, em três níveis de desatualização (S0–S2), mais uma região de transferência (POA);
   (ii) a adaptação do ANL (filtro de histograma + PPO) a esse domínio, com protocolo estatístico pareado e hipóteses pré-registradas;
   (iii) uma extensão multiobjetivo inspirada no AMOR, com memória de ações;
   (iv) uma análise honesta, incluindo resultados negativos.
5. Parágrafo de organização do artigo ("O restante do artigo está organizado...").

### II. Related Work (~0,75 página, 3 subseções)
- **A. Localização ativa e Markov localization:** Burgard/Fox/Thrun (active Markov localization), Thrun et al. (*Probabilistic Robotics*), Chaplot et al. (ANL, ICLR 2018).
- **B. Geolocalização visual de UAVs sem GNSS:** casamento drone–satélite e cross-view geo-localization; surveys recentes. Mostrar que a maioria é **passiva** (não decide para onde voar).
- **C. RL multiobjetivo:** Hayes et al. (guia prático de MORL), políticas condicionadas em preferência, AMOR (Alegre et al., SIGGRAPH 2025), hipervolume e EUM.
- Fechar com um parágrafo de **posicionamento**: o que este trabalho combina que os outros não combinam.

### III. Problem Formulation (~0,5 página)
- POMDP: estado = pose real na grade de 50 m; observação = recorte nadir de 150 m da imagem pós-enchente; ação ∈ 8 direções; horizonte de 80 passos.
- Crença b_t por filtro de histograma com verossimilhança temperada (β, κ = 0,25). **Equações numeradas**: atualização bayesiana e entropia H(b_t).
- Definição formal de **sucesso** (massa ≥ 0,9 num raio de 100 m e erro ≤ 100 m) e de **localização falsa**.
- Versão multiobjetivo: r_t ∈ ℝ⁴ = (info, custo, sucesso, risco), escalarização w·r_t com w ~ Dirichlet.

### IV. Methodology (~1,25 página)
- **A. Dados e cenários:** tabela com as datas Maxar, o papel de cada imagem, o alinhamento (2,5 m e 5,0 m), a AOI e a divisão norte/sul entre treino e teste, e S0/S1/S2. **Fig. 1:** antes e depois, com a divisão norte/sul.
- **B. Descritor de observação:** o descritor "hand" (gradientes + cor) e por que a ResNet18 foi descartada (top-1 de 10% contra 41% em S1).
- **C. Ambiente e observação da política:** crença, máscara, distintividade e PCA-4 do mapa. **Fig. 2:** diagrama do pipeline (câmera → descritor → filtro → política → ação).
- **D. Agente PPO:** arquitetura da CNN e hiperparâmetros (a tabela completa vai para o apêndice ou fica resumida).
- **E. Extensão MOPPO:** condicionamento em w, amostragem por Dirichlet, memória curta de 8 ações (como no ANL), presets `fast`/`safe`/`unif`.
- **F. Baselines:** reta, zigue-zague, aleatória, greedy (oráculo) e greedy_map (justa).

### V. Experimental Setup (~0,5 página)
- Protocolo: 1000 episódios **pareados** no teste (sul), 3 seeds, 300 episódios no norte (H3). Porto Alegre zero-shot, sem retreino.
- Métricas: taxa de sucesso, taxa de localização falsa, mediana de passos, distância média, tempo de decisão, e hipervolume e EUM para a MO.
- Estatística: ICs de 95% por bootstrap pareado, Wilcoxon e critérios **pré-registrados** (citar `DECISIONS*.md`).
- Hardware e tempo: CPU i7-10510U, ~45 min por seed (dá peso ao argumento de baixo custo).

### VI. Results (~1,5 página)
- **A. Taquari (H1–H3):** **Tabela I** (sucesso, localização falsa, passos e distância por método e cenário). **Fig. 3:** sucesso × passos.
- **B. Transferência para Porto Alegre (HP1–HP3):** **Tabela II** por região (ilhas × centro) e efeito da intensidade de mudança (quintis).
- **C. Multiobjetivo (H4–H7):** **Fig. 4:** projeções da fronteira de Pareto. **Tabela III:** hipervolume e EUM com e sem memória. **Fig. 5:** adaptação de comportamento (H6).
- **D. Comportamento qualitativo:** **Fig. 6:** trajetórias (o PPO evita mata e busca lavouras e estruturas; a greedy persegue casamentos espúrios).
- Uma frase de veredito por hipótese, igual ao critério pré-registrado.

### VII. Discussion (~0,75 página)
- **Por que a linha reta é tão forte:** ela cobre área nova a cada passo, e o filtro já faz o trabalho pesado.
- **Trade-off central:** o RL reduz a distância em ~50%, mas tem mais localização falsa. É exatamente essa a motivação da MO.
- **Resultado do MOPPO:** escrever conforme o desfecho da pendência 5.1 do plano de desenvolvimento. Se a política ignora w, discutir o colapso do condicionamento como achado e explicar por que isso também afeta a leitura de H4.
- **Limitações:** grade 2D discreta, sem dinâmica de voo nem altitude; uma única data pós-enchente; descritor artesanal; poucas seeds; S0 degenerado; dados CC BY-NC; treino só em CPU.
- **Ameaças à validade:** vazamento norte/sul (citar `tests/test_leakage.py`) e erro de alinhamento de 2,5–5 m perto da resolução da grade.

### VIII. Conclusion (~0,25 página)
Responder à pergunta da introdução em 3–4 frases, retomar as hipóteses e listar trabalhos futuros: descritor aprendido entre datas, ações contínuas e altitude, mais eventos de desastre e teste com drone real.

### Acknowledgment
Professor da disciplina, UFRGS, OpenAerialMap/Maxar Open Data.

### References (estilo IEEE numérico, `IEEEtran.bst`)
Usar o BibTeX. **Conferir cada referência na fonte original** antes de submeter. Lista mínima:
1. D. S. Chaplot, E. Parisotto, R. Salakhutdinov, "Active Neural Localization," ICLR 2018.
2. L. N. Alegre et al., AMOR, SIGGRAPH 2025 (pegar o título e os autores exatos do PDF na raiz do projeto).
3. J. Schulman et al., "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017.
4. A. Raffin et al., "Stable-Baselines3," JMLR 2021.
5. W. Burgard, D. Fox, S. Thrun, "Active Mobile Robot Localization," IJCAI 1997.
6. S. Thrun, W. Burgard, D. Fox, *Probabilistic Robotics*, MIT Press, 2005.
7. C. F. Hayes et al., "A practical guide to multi-objective reinforcement learning and planning," AAMAS (JAAMAS) 2022.
8. L. P. Kaelbling, M. L. Littman, A. R. Cassandra, "Planning and acting in partially observable stochastic domains," AI 1998.
9. Pelo menos 2 trabalhos de geolocalização de UAV por imagem de satélite (pesquisar um survey de 2021–2024).
10. Fonte dos dados: Maxar Open Data / OpenAerialMap.

---

## 3. Hipóteses e onde está cada uma

| Hipótese | Enunciado curto | Veredito atual | Fonte |
|---|---|---|---|
| H1 | O PPO bate cada baseline simples nos cenários não degenerados | não suportada | `REPORT.md`, `results/hypotheses.json` |
| H2 | A vantagem do RL cresce com a desatualização (S0 < S1 < S2) | inconclusiva | idem |
| H3 | Generaliza do norte (treino) para o sul (teste) | não suportada | idem |
| HP1 | Transferência zero-shot para POA | inconclusiva | `REPORT_POA.md` |
| HP2 | A dificuldade cresce com a intensidade da mudança | suportada | idem |
| HP3 | Estruturas urbanas ajudam e a vantagem ativa é maior nas ilhas | não suportada | idem |
| H4 | Uma política MO substitui várias de peso fixo | suportada\* | `REPORT_MO.md` |
| H5 | A MO domina a reta em S2 | não suportada | idem |
| H6 | O comportamento se adapta a w | parcial | idem |
| H7 | `w_seguro` reduz a localização falsa nas ilhas | inconclusiva | idem |

\* **Revisar depois da pendência 5.1** (os presets dão números idênticos).

Mensagem do artigo: **mais da metade das hipóteses não foi suportada, e isso é um resultado.** O enunciado pede "validar ou refutar". O texto deve deixar claro que os critérios foram pré-registrados antes da execução.

---

## 4. Mapa de figuras e tabelas → arquivos

| Item | Conteúdo | Origem |
|---|---|---|
| Fig. 1 | Antes e depois da enchente, AOI e divisão norte/sul | `results/figs/data_overview.png`, `apresentacao_grupo/enchente_antes_depois.png` |
| Fig. 2 | Diagrama do pipeline | **a criar** (TikZ ou draw.io) |
| Fig. 3 | Sucesso × passos (Taquari) | `results/figs/` (curvas) |
| Fig. 4 | Fronteiras de Pareto | `results_mo/figs/pareto_success_*.png` |
| Fig. 5 | Adaptação a w (H6) | `results_mo/figs/behavior_adaptation_h6.png` |
| Fig. 6 | Trajetórias PPO × reta × greedy | `results/figs/trajectories_*.png` |
| Tab. I | Resultados Taquari | `results/summary_test.csv`, `paired_tests_test.csv` |
| Tab. II | Porto Alegre | `results_poa/summary.csv`, `region_contrast_P1.csv` |
| Tab. III | Hipervolume e EUM | `REPORT_MO.md` §3 (exportar CSV em `results_mo/`) |

Figuras para o artigo: PDF vetorial, legível em uma coluna (3,5 in), fonte ≥ 8 pt e cores seguras para daltônicos. Fazer um script `paper/make_figures.py` que regenera tudo a partir dos CSVs.

---

## 5. Divisão de trabalho e prazos

| Seção | Autor | Revisor | Rascunho até |
|---|---|---|---|
| III Formulação + IV Metodologia | ____ | ____ | Dom 22/11 |
| V Setup + VI Resultados | ____ | ____ | Dom 22/11 |
| II Trabalhos relacionados + referências | ____ | ____ | Dom 22/11 |
| I Introdução + VII Discussão + VIII Conclusão | ____ | ____ | Seg 23/11 |
| Abstract, figuras finais, revisão de formato | ____ | todos | Ter 24/11 |

**Ordem de escrita:** Metodologia → Resultados → Discussão → Trabalhos relacionados → Introdução → Conclusão → Abstract.

## 6. Checklist de submissão

- [ ] Todo número do texto confere com a tag `results-final`
- [ ] Toda figura e tabela é citada no texto antes de aparecer
- [ ] ICs de 95% e *n* informados em todas as tabelas
- [ ] Limitações explícitas
- [ ] Referências conferidas uma a uma, no estilo IEEE
- [ ] Link do repositório no artigo (nota de rodapé na introdução)
- [ ] Dentro do limite de páginas e compilando sem warnings no Overleaf

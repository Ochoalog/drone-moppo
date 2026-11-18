# Plano de escrita do artigo v3 (entrega 25/11/2026)

**Versão:** v3, de 18/11/2026. Substitui a v2 (preservada em `docs/PLANO_ARTIGO_v2.md`). O artigo deixa de ser "um estudo do PPO sobre mapas desatualizados" e passa a ser um **artigo de método**: o estudo atual vira o estudo preliminar que motiva o método, e o método novo (percepção aprendida, modelo realista, localização ciente da mudança, altitude) é avaliado com 10 seeds, validação espacial e transferência.

**Formato:** template a confirmar com o professor (o enunciado sugere arXiv/Prime AI; a proposta do grupo é IEEE Conference, duas colunas). Alvo: **8 páginas de texto + referências**. Se o limite for 6, cortar conforme a §8.
**Idioma:** confirmar. O padrão IEEE é inglês, e uma versão em inglês já serve de base para uma submissão futura (ICUAS 2027, prazo 05/02/2027, revisão duplo-cega).
**Fonte da verdade:** só números da tag `results-final` (congelamento de segunda, 23/11, 12:00), lidos de `paper/tables/*.csv`.
**Relação com o seminário:** o método é uma extensão direta do Active Neural Localization (filtro de histograma + política aprendida). Isso precisa estar explícito na introdução e no método, porque é exigência da disciplina.

---

## 1. Tese e título

**Tese:** quando o mapa de referência foi desatualizado por um desastre, a localização ativa falha exatamente onde o mapa mudou: tanto o filtro quanto a política confiam em lugares que hoje estão debaixo d'água. **Informar ao localizador onde o mapa provavelmente mudou** (com a delimitação de inundação que serviços de emergência publicam em dias) e **deixar a política escolher a altitude** recupera a robustez. Com isso, a política aprendida passa a pagar a pena contra heurísticas fortes.

A tese tem três partes, e cada uma depende de um resultado:
- "falha onde o mapa mudou": estudo v1 (M1 e índice de mudança, já especificados);
- "o filtro CA recupera robustez": H2;
- "a política aprendida passa a pagar a pena": H3 e H4.

Se H3 não for suportada, a tese recua para: "a percepção e o filtro cientes da mudança resolvem a maior parte do problema, e a escolha ativa de trajetória acrescenta pouco sobre uma heurística CA". Isso continua sendo um resultado útil e publicável.

**Títulos propostos** (escolher no domingo, 22/11, com os resultados):
1. *Change-Aware Active Localization for GNSS-Denied UAVs over Outdated Post-Flood Maps*
2. *Where the Map Lies: Change-Aware Active Localization of UAVs after Floods*
3. (se H3 não for suportada) *When Does Active Localization Pay Off over Outdated Post-Disaster Maps?*

---

## 2. Perguntas de pesquisa e hipóteses

Critérios completos em `DECISIONS_V3.md` (commitado antes da matriz) e em `PLANO_DESENVOLVIMENTO.md` §4.

| Pergunta | Hipótese | Enunciado curto |
|---|---|---|
| **RQ1:** por que a localização ativa falha sobre mapas desatualizados? | M1 + F1/F2 (estudo v1) | As falhas acontecem onde o mapa mudou, e a política v1 é atraída por células distintivas no mapa antigo |
| **RQ2:** a percepção e o filtro cientes da mudança resolvem a falha? | H1 | O descritor aprendido supera o hand |
| | H2 | O filtro CA reduz a localização falsa para todos os métodos |
| **RQ3:** com percepção forte e modelo realista, a escolha ativa de trajetória e de altitude paga a pena? | H3 (principal) | A PPO-CA supera (ou empata com menos tempo de voo) a melhor heurística CA |
| | H4 | A altitude como ação ajuda, e a política sobe mais onde o mapa mudou mais |
| | H5 | A relação do H3 se mantém zero-shot em Porto Alegre e na terceira área |

**Estudo v1 no artigo:** os vereditos antigos (H1–H7, HP1–HP3) aparecem numa tabela do apêndice, com a errata da MO e a tabela de alterações de critério e protocolo. No texto principal, o estudo v1 entra só como motivação (§3, seção III).

---

## 3. Estrutura das seções e orçamento de páginas (8 páginas)

Orçamento visual: **5 figuras + 3 tabelas** (≈ 2,25 páginas).

### Abstract (150–250 palavras), escrever na segunda, 23/11
Problema → achado do estudo preliminar (a falha se concentra onde o mapa mudou) → método (descritor aprendido, sensor e odometria realistas, filtro e política cientes da mudança com a delimitação do Copernicus EMS, altitude como ação) → avaliação (imagens Maxar de 2023–2024 do Taquari, de Porto Alegre e de uma terceira área; 10 seeds; validação cruzada espacial; transferência zero-shot) → dois ou três números principais com IC → conclusão.

### Index Terms
Active localization, reinforcement learning, UAV, visual geo-localization, outdated maps, change-aware localization, disaster response.

### I. Introduction (~0,8 página)
1. Motivação: enchentes do RS; drones de resposta sem GNSS; a única referência é a imagem anterior ao desastre; serviços como o Copernicus EMS publicam a delimitação da inundação em dias.
2. Lacuna: a localização ativa aprendida (ANL; Deep Active Localization) supõe mapa correto; a localização aérea por RL (AiRLoc) busca um alvo, não a própria pose; a localização visual absoluta de UAVs é quase toda passiva. "Até onde sabemos, nenhum trabalho trata a localização ativa sobre mapas desatualizados por um desastre real nem usa a delimitação do desastre para isso."
3. Achado preliminar em uma frase (a falha se concentra onde o mapa mudou) e a ideia do método.
4. Contribuições:
   (i) um estudo que mostra **onde e por que** a localização ativa falha sobre mapas pós-enchente (índice de mudança, mecanismo);
   (ii) um localizador ativo ciente da mudança, que estende o ANL com prior de mudança no filtro e na política e com escolha de altitude;
   (iii) um benchmark com imagens reais de antes e depois em três áreas, sensor e odometria realistas e validação cruzada espacial;
   (iv) avaliação com 10 seeds, critérios pré-especificados e resultados negativos reportados.
5. Rodapé com o link do repositório (anonimizar numa submissão duplo-cega).

### II. Related Work (~0,7 página)
- **A. Localização ativa:** Markov localization ativa; *Probabilistic Robotics*; ANL; Deep Active Localization.
- **B. Localização visual absoluta de UAVs e reconhecimento de lugar:** as revisões de Couturier & Akhloufi (2021, 2024); UAV-VisLoc; AnyLoc e características de modelos de fundação (DINOv2); AiRLoc.
- **C. Mapas desatualizados e mapeamento de desastres:** localização sob mudança de aparência; produtos de rapid mapping (Copernicus EMS) e máscaras de água de radar (Sentinel-1).
- **D. Avaliação confiável em RL:** rliable (IQM, bootstrap estratificado).
- Posicionamento: o que só este trabalho combina.

### III. Preliminary Study: Why Active Localization Fails on Outdated Maps (~0,8 página)
- O cenário do estudo v1 em um parágrafo (grade de 50 m, descritor hand, PPO à la ANL, Taquari e Porto Alegre).
- Três achados, com números canônicos:
  1. a PPO não se distingue da reta em sucesso (IC hierárquico da diferença em S2: [−6,1; +9,8] p.p.), com menos passos e ~2/3 da distância **nos sucessos** (1,25 × 1,89 km), mas mais localizações falsas (20,2% × 13,0%);
  2. o sucesso de todos os métodos cai com o índice de mudança (F1), e a covariável independente do descritor aponta na mesma direção (F2);
  3. mecanismo M1 (se suportado): as localizações falsas da PPO acontecem em células distintivas no mapa antigo e mudadas.
- **Fig. 2:** sucesso × quintil do índice de mudança (Taquari e Porto Alegre) + painel do M1.
- Frase de transição: "isso sugere dois remédios: percepção mais invariante e um localizador que saiba onde o mapa mudou."

### IV. Problem Formulation (~0,5 página)
- POMDP com posição contínua, altitude h ∈ {60, 100, 150} m, 10 ações, horizonte de 80 passos; custo em tempo de voo.
- Modelo de observação: footprint 2·h·tan(FOV/2); variantes com deslocamento, yaw, inclinação e fotometria.
- Modelo de movimento com deriva; predição do filtro.
- Equações numeradas: atualização bayesiana, verossimilhança temperada, **verossimilhança CA** (mistura com ρ_x = α·m_x), entropia.
- Definições de sucesso, localização falsa e timeout (a regra de parada usa só a crença).

### V. Method (~1,0 página)
- **A. Descritor:** DINOv2 + GeM + PCA; ajuste fino contrastivo com pares antes/depois só da faixa de treino.
- **B. Filtro ciente da mudança:** prior de mudança a partir do Copernicus EMS (EMSR720, EMSN194) ou de radar, rasterizado na grade; por que ele é independente da observação; calibração de α.
- **C. Política:** PPO (como no ANL, com crença, máscara, distintividade e prior de mudança na entrada); variante com altitude.
- **D. Ambiente vetorizado em GPU:** uma frase e o teste de paridade com o ambiente de referência.
- **Fig. 1:** visão geral (câmera em altitude → descritor → filtro CA com prior → política → ação), com um recorte antes/depois e o prior de inundação.

### VI. Experimental Setup (~0,6 página)
- **Tabela I:** áreas, datas, papel de cada imagem, fonte e data do prior de mudança, alinhamento, nuvens, dobras.
- Validação cruzada espacial em 3 faixas; zero-shot em Porto Alegre (ilhas, Centro), na terceira área e, se houver, no evento externo.
- Métodos: reta, zigue-zague, aleatória, greedy_map, cada um com filtro padrão e CA; PPO, PPO-CA, PPO-CA-ALT; ablações de descritor e deriva.
- Estatística: 10 seeds; IQM e probabilidade de melhora (rliable); bootstrap estratificado por dobra e região; bootstrap hierárquico nas diferenças pareadas; Holm entre locais. Critérios pré-especificados e tabela de alterações (apêndice).
- Hardware: notebook i7 (CPU) e uma T4 no Colab; tempo de GPU total.

### VII. Results (~1,9 página)
- **A. Percepção (H1):** top-1 de recuperação por descritor, em uma linha da Tabela III ou num painel pequeno.
- **B. Resultado principal (H3):** **Tabela II** (sucesso, localização falsa, tempo de voo, distância nos sucessos, por método e local, com IC e n) e **Fig. 3** (IQM com IC e probabilidade de melhora, por local).
- **C. Filtro ciente da mudança (H2):** **Fig. 4**: sucesso e localização falsa × quintil do índice de mudança, filtro padrão × CA, para a reta e a PPO. É a figura que fecha o argumento iniciado na Fig. 2.
- **D. Altitude (H4):** **Fig. 5**: altitude escolhida × índice de mudança, e o ganho em tempo de voo.
- **E. Transferência (H5)** e validação espacial: linhas da Tabela II.
- **F. Ablações:** **Tabela III** (descritor, filtro, entrada CA na política, altitude, deriva).
- Uma frase de veredito por hipótese, no formato do critério.

### VIII. Discussion (~0,6 página)
- O que o prior de mudança resolve e o que não resolve (mudanças fora da delimitação: lama, nuvens, ângulo de visada no Centro).
- Quando a escolha ativa paga a pena e quando uma heurística CA basta.
- A extensão multiobjetivo: uma frase com o resultado negativo e o link para o apêndice.
- **Limitações:** simulação com imagens reais, mas sem dinâmica de voo completa nem teste em voo; uma data pós-evento por área; prior de mudança com resolução e atraso próprios; 10 seeds numa única GPU; escolha manual de recompensa.
- **Ameaças à validade:** dependência espacial (mitigada pela validação em faixas e pelo bootstrap por janela); padronização transdutiva em Porto Alegre no estudo v1; sobreposição das regiões de Porto Alegre; desalinhamento local; o índice de mudança depende do descritor (mitigado pela covariável independente); critérios e protocolos alterados (tabela do apêndice).

### IX. Conclusion (~0,2 página)
Responder às 3 perguntas. Trabalhos futuros: teste em voo real, dinâmica e simulador (PX4 SITL + Gazebo), prior de mudança estimado online, outros tipos de desastre, multiobjetivo com orçamento adequado.

### Apêndice (se o template permitir; senão, no repositório)
- Vereditos do estudo v1 e tabela de alterações de critério e de protocolo.
- Extensão multiobjetivo: errata, diagnóstico (gate de condicionamento que falhou, argmax idêntico entre presets) e a reavaliação amostrada no mesmo modo para todas as políticas (`results_mo_v1fix/`).
- Hiperparâmetros, calibração de β, κ e α, detalhes do ambiente em GPU e da paridade.

---

## 4. Registro de afirmações

| Afirmação | Pode? | Forma correta e evidência |
|---|---|---|
| "O RL reduz a distância voada pela metade" (estudo v1) | **não** | "Nos episódios com sucesso, a PPO voa ~2/3 da distância da reta (1,25 × 1,89 km)" (`docs/conciliacao_numeros.md` nº 7) |
| "Sucesso equivalente à reta" (estudo v1) | **não** | "Não distinguível: IC hierárquico da diferença [−6,1; +9,8] p.p." |
| "A PPO generaliza do norte para o sul" (estudo v1) | sim | S2: 0,701 → 0,707; S1: 0,917 → 0,840, com a reta também caindo |
| "A dificuldade é governada pela mudança do mapa" | sim, como associação | F1 e F2; nunca causal |
| "O filtro CA reduz a localização falsa" | só se H2 for suportada | IC da diferença por método e local |
| "Nossa política supera as heurísticas" | só se H3 for suportada **por superioridade** | Probabilidade de melhora com IC; dizer em quais locais |
| "Nossa política iguala as heurísticas com menos tempo de voo" | só se H3 for suportada **por não inferioridade** | Margem de −3 p.p. pré-especificada + IC do tempo de voo |
| "A política aprende a subir onde o mapa mudou" | só se H4(b) for suportada | Spearman com IC |
| "Generaliza para outras áreas" | só pelo H5 | Dizer quais regiões; o evento externo é exploratório |
| "O MOPPO domina" / "a memória triplica o hipervolume" | **não** | Errata do estudo v1 |
| "A avaliação com argmax apagou o condicionamento" | só qualificada | Junto com "o gate de condicionamento falhou no treino" |
| "Primeiro trabalho a…" | **não** | "Até onde sabemos, …" |
| "Hipóteses pré-registradas" | **não** | "Critérios pré-especificados", com a tabela de alterações |
| "Funciona em tempo real a bordo" | só com medição | Tempo por decisão medido; dizer o hardware; não extrapolar para um computador de bordo sem medir |

Toda frase do resumo, da introdução e da conclusão passa por esta tabela antes da versão final (terça, 24/11).

---

## 5. Figuras e tabelas → arquivos

| Item | Conteúdo | Origem |
|---|---|---|
| Fig. 1 | Visão geral do método | **a criar** (TikZ ou draw.io) + recortes de `results_v3/figs/` |
| Fig. 2 | Estudo v1: sucesso × índice de mudança + M1 | `results_final/` (F1–F3) |
| Fig. 3 | Resultado principal: IQM e probabilidade de melhora por local | `results_v3/rliable/` |
| Fig. 4 | Filtro padrão × CA por quintil de mudança | `results_v3/change_quintiles/` |
| Fig. 5 | Altitude × índice de mudança; tempo de voo | `results_v3/altitude/` |
| Tab. I | Áreas, datas, prior de mudança, dobras | `results_v3/data_summary.csv` |
| Tab. II | Resultado principal e transferência | `paper/tables/main.csv` |
| Tab. III | Ablações | `paper/tables/ablations.csv` |

Figuras em PDF vetorial, legíveis numa coluna (3,5 in), fonte ≥ 8 pt, paleta segura para daltônicos. `paper/make_figures.py` regenera tudo.

---

## 6. Referências

Estilo IEEE (`IEEEtran.bst`). ✓ = conferida em 18/11; o resto a IA-2 confere na fonte antes de terça.

1. D. S. Chaplot, E. Parisotto, R. Salakhutdinov, "Active Neural Localization," ICLR 2018.
2. ✓ S. K. Gottipati et al., "Deep Active Localization," *IEEE RA-L*, 2019, doi:10.1109/LRA.2019.2932575.
3. ✓ A. Pirinen et al., "Aerial View Localization with Reinforcement Learning: Towards Emulating Search-and-Rescue," arXiv:2209.03694 (conferir o local de publicação final).
4. ✓ A. Couturier, M. A. Akhloufi, "A review on absolute visual localization for UAV," *Robotics and Autonomous Systems*, vol. 135, 103666, 2021.
5. ✓ A. Couturier, M. A. Akhloufi, "A Review on Deep Learning for UAV Absolute Visual Localization," *Drones*, vol. 8, no. 11, 622, 2024.
6. ✓ N. Keetha et al., "AnyLoc: Towards Universal Visual Place Recognition," *IEEE RA-L*, vol. 9, no. 2, pp. 1286–1293, 2023.
7. ✓ W. Xu et al., "UAV-VisLoc: A Large-scale Dataset for UAV Visual Localization," arXiv:2405.11936, 2024.
8. M. Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision" (conferir local e ano).
9. F. Radenović, G. Tolias, O. Chum, GeM pooling, *IEEE TPAMI*, 2019 (conferir).
10. A. van den Oord, Y. Li, O. Vinyals, "Representation Learning with Contrastive Predictive Coding" (InfoNCE), arXiv:1807.03748.
11. ✓ R. Agarwal et al., "Deep Reinforcement Learning at the Edge of the Statistical Precipice," NeurIPS 2021, arXiv:2108.13264.
12. ✓ Copernicus Emergency Management Service, ativação EMSR720 (Rapid Mapping, Rio Grande do Sul, maio de 2024) e EMSN194 (Porto Alegre).
13. W. Burgard, D. Fox, S. Thrun, "Active Mobile Robot Localization," IJCAI 1997.
14. S. Thrun, W. Burgard, D. Fox, *Probabilistic Robotics*, MIT Press, 2005.
15. J. Schulman et al., "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017.
16. 1–2 trabalhos de localização sob mudança de aparência e 1 de mapeamento de inundação por Sentinel-1 (a pesquisar).
17. Dados: Maxar Open Data / OpenAerialMap (CC BY-NC 4.0).
18. Apêndice: L. N. Alegre et al., AMOR, SIGGRAPH 2025; C. F. Hayes et al., guia prático de MORL, JAAMAS 2022.

---

## 7. Cronograma de escrita (alinhado ao plano de desenvolvimento)

| Dia | Seções | Quem |
|---|---|---|
| Qua 18 | Esqueleto LaTeX em `paper/`, macros de números lidos dos CSVs, Fig. 1 em rascunho | IA-1 |
| Qui 19 | II (Trabalhos relacionados) e referências conferidas | IA-2 → revisão do Vitor |
| Sex 20 | III (Estudo preliminar), com os números do estudo v1 já canônicos | Vitor + IA-1 |
| Sáb 21 | IV (Formulação) e V (Método) | Vitor |
| Dom 22 | VI (Setup) e VII (Resultados) com números provisórios; escolha do título | Vitor + IA-1 |
| Seg 23 | Números finais da tag `results-final`; I (Introdução), VIII (Discussão), IX (Conclusão), Abstract | Vitor |
| Ter 24 | Registro de afirmações, revisão cruzada (IA-2 + colegas), formato, slides | todos |
| Qua 25 | PDF final | Vitor |

**Ordem de escrita:** Trabalhos relacionados → Estudo preliminar → Formulação → Método → Setup → Resultados → Discussão → Introdução → Conclusão → Abstract.

---

## 8. Se o espaço ou o tempo apertar

Cortar nesta ordem:
1. Fig. 5 vira duas linhas da Tabela III;
2. apêndice vai para o repositório;
3. a seção III encolhe para meia página (Fig. 2 fica, o texto do M1 encolhe);
4. evento externo e terceira área viram uma linha cada na Tabela II.

Se a §8 do plano de desenvolvimento cortar componentes (altitude, ajuste fino, validação espacial), as seções correspondentes saem, e a hipótese vira trabalho futuro na conclusão. O artigo mínimo é: Estudo preliminar + filtro CA + PPO-CA × heurísticas CA + transferência para Porto Alegre.

---

## 9. Checklist de entrega (25/11)

- [ ] Template e idioma confirmados com o professor
- [ ] Relação com o ANL explícita na introdução e no método
- [ ] Todo número confere com `paper/tables/` da tag `results-final`
- [ ] Toda afirmação do resumo, da introdução e da conclusão está no registro (§4) com a forma correta
- [ ] IC de 95%, n, seeds e dobra em todas as tabelas
- [ ] Critérios pré-especificados citados e tabela de alterações no apêndice
- [ ] Fonte e data do prior de mudança declaradas (independente da imagem observada)
- [ ] Localização falsa ao lado de todo ganho de tempo, passos ou distância
- [ ] Limitações e ameaças à validade explícitas
- [ ] Referências conferidas uma a uma
- [ ] Link do repositório (anonimizável)
- [ ] Dentro do limite de páginas; compila sem warnings

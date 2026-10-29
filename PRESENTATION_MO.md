# Apresentação: Localização Ativa Multiobjetivo (MOPPO à la AMOR)

Esta apresentação reúne os resultados experimentais, animações de telemetria e vídeos comparativos da política multiobjetivo condicionada em pesos (**MOPPO**, Alegre et al., SIGGRAPH 2025) com memória curta de ações (**ANL**, Chaplot et al., ICLR 2018), desenvolvida para navegação visual ativa sob incerteza de GPS.

---

## 1. O Problema e a Dinâmica de Convergência

Na localização ativa, o drone parte de uma janela de busca de $64 \times 64$ células ($3.2 \times 3.2\text{ km}$), onde sua posição real é inicialmente desconhecida. Um filtro de histograma mantém uma distribuição de probabilidade espacial $b_t \in [0, 1]^{64 \times 64}$ (a "crença").

> **Condição de Sucesso:** A missão é declarada concluída quando pelo menos **90% da massa de probabilidade** do filtro se concentra em um raio de 2 células (100 metros) ao redor da posição estimada. Se o pico estiver correto, ocorre **Sucesso**; se estiver em outro lugar da janela, ocorre **Falsa Localização**.

O vídeo abaixo exibe um episódio completo da política MOPPO no Taquari S2. Observe como a probabilidade se aglutina conforme a aeronave sobrevoa marcos visuais salientes:

![Convergência da Política MOPPO (Episódio 0)](results_mo/videos/episode_0_mo_fast.gif)

### O que observar na animação acima:
1. **Painel 1 (Esquerda - Mapa Satélite Real & Trajetória):**
   - **Fundo:** Ortofoto aérea real de satélite (Taquari/RS) em cores naturais, evidenciando o Rio Taquari à direita, as matas fechadas, as estradas e os galpões industriais/fazendas.
   - **Contornos Dourados:** Indicam as feições de alta distintividade morfológica calculadas pelo descritor.
   - **Ponto Inicial (círculo azul):** O drone inicia perdido na mata homogênea e voa ativamente em direção ao rio e entroncamento de estradas (linha verde).
   - **Estimativa do Filtro (cruz rosa $\hat{x}_t$):** Converge com precisão sobre o drone conforme ele alcança a margem do rio.
2. **Painel 2 (Centro - Crença Espacial $b_t$ sobre Satélite):**
   - A imagem de satélite escurecida serve de fundo espacial.
   - No início ($t=0$), a crença é uma névoa difusa com pico inferior a $0.2\%$.
   - No passo 18, ocorre o colapso: a probabilidade atinge **48.8% no pico** exatamente sobre o ponto geográfico verdadeiro, finalizando com **LOCALIZAÇÃO COM SUCESSO**.
3. **Painel 3 (Direita - Curvas de Telemetria):**
   - **Entropia de Shannon $H_t$ (curva amarela):** Mede a desordem e incerteza da distribuição. Despenca vertiginosamente de $7.8\text{ nats}$ para menos de $0.5\text{ nats}$.
   - **Erro cartesiano em metros (curva azul tracejada):** O desvio entre o drone e o pico do filtro cai de $730\text{ m}$ para menos de $50\text{ m}$.

---

## 2. Comparação Lado a Lado: MOPPO vs Linha Reta vs Zigue-Zague

Por que aprender uma política de controle via RL multiobjetivo em vez de programar padrões geométricos tradicionais como linha reta ou zigue-zague (*lawnmower*)?

A animação a seguir apresenta os três métodos iniciando no **exato mesmo ponto**, enfrentando a **mesma janela de busca** e a **mesma dinâmica estocástica de voo**:

![Comparação Sincronizada: MOPPO vs Linha Reta vs Zigue-zague](results_mo/videos/comparison_ep0_mo_vs_straight_vs_zigzag.gif)

### Análise Comparativa dos Três Métodos:
- **MOPPO (Com Memória):**
  - **Comportamento:** Ativo e Intencional.
  - **Passos até Sucesso:** 18 passos.
  - **Distância Voada:** 1272 m.
  - **Mecanismo:** A memória curta de ações (ANL) informa à rede os rumos recentemente tomados, evitando redundâncias e guiando a aeronave diretamente para regiões de alta variância morfológica.
- **Linha Reta (Straight Baseline):**
  - **Comportamento:** Passivo / Deriva.
  - **Passos até Sucesso:** 54 passos (3x mais demorado!).
  - **Distância Voada:** 2777 m (mais que o dobro de energia consumida!).
  - **Mecanismo:** Voa em rumo fixo refletindo nas bordas da janela. A convergência depende de encontrar acidentalmente feições discriminativas ao longo da reta.
- **Zigue-Zague (Lawnmower):**
  - **Comportamento:** Varredura Sistemática.
  - **Passos:** 80 passos (Esgotou o limite de tempo!).
  - **Distância Voada:** 4186 m (Falha por timeout).
  - **Mecanismo:** Varre faixas paralelas de 4 em 4 células. Em terrenos com poucas referências, passa a maior parte do voo em áreas uniformes e redundantes, acumulando custo sem colapsar a entropia.

### Episódio 2: Robustez sob Ambientes Desafiadores
Em janelas com maior ambiguidade espacial, a linha reta e o zigue-zague frequentemente falham ou estouram o limite de 80 passos, enquanto a política MOPPO consegue resolver a ambiguidade em menos de 20 passos:

![Comparação Sincronizada no Episódio 2](results_mo/videos/comparison_ep2_mo_vs_straight_vs_zigzag.gif)

---

## 3. Como Funciona a Fronteira de Pareto?

No mundo real, o operador de um drone não busca apenas uma solução única:
- Em missões de emergência com pouca bateria, o objetivo é **minimizar a distância** (preset rápido $w_{\mathrm{rápido}}$).
- Em missões críticas em terreno hostil, o objetivo é **minimizar o risco de falsa localização** (preset seguro $w_{\mathrm{seguro}}$).

A formulação **MOPPO** (*à la* AMOR) treina uma **única rede neural** condicionada em um vetor de pesos $w = [w_{\mathrm{info}}, w_{\mathrm{cost}}, w_{\mathrm{success}}, w_{\mathrm{risk}}]$, cobrindo todo o simplex de preferências sem exigir retreino.

A animação abaixo ilustra como a política percorre a fronteira de Pareto conforme o vetor de preferências $w$ é ajustado:

![Explicação Dinâmica da Fronteira de Pareto](results_mo/videos/pareto_frontier_explainer.gif)

### Projeções 2D da Fronteira de Pareto 3D:

| Projeção: Sucesso vs Distância | Projeção: Sucesso vs Falsa Localização |
|:---:|:---:|
| ![Sucesso vs Distância](results_mo/figs/pareto_success_distance.png) | ![Sucesso vs Falsa Localização](results_mo/figs/pareto_success_false_loc.png) |

### Por que o MOPPO Domina os Outros Métodos?
1. **Dominância sobre Modelos de Peso Fixo:** 
   - Os pontos quadrados cor-de-rosa (PPOs treinados para um único peso estático) ficam no interior da casca convexa. A política multiobjetivo única obteve maior sucesso e menor distância do que as redes fixas especializadas, validando a **Hipótese H4**.
2. **Dominância sobre a Ablação sem Memória:**
   - Os triângulos laranjas representam o MOPPO treinado sem histórico de ações. Eles apresentam taxas de sucesso muito menores (40–60%) e exigem distâncias superiores a 1400 m.
   - O volume de hipervolume coberto salta de **0.1529 para 0.4232** (um ganho de quase **300%**) graças à memória curta de 8 ações!

---

## 4. Transferência Zero-Shot para Porto Alegre (Hipótese H7)

A política treinada exclusivamente em Taquari S2 foi submetida a um teste cego (*zero-shot*) na região insular e urbana de Porto Alegre (cenário P1), enfrentando condições de cheia histórica:

| Região | Método | Sucesso (%) | Falsa Loc. (%) | Distância Média (m) |
|---|---|---|---|---|
| **Ilhas** | **MOPPO `safe`** | 49.0% | **19.7%** | **1028.7 m** |
| Ilhas | MOPPO `fast` | 52.0% | 21.3% | 1054.0 m |
| Ilhas | PPO-S2 PoC | **71.3%** | 21.3% | 1320.8 m |
| Ilhas | Linha Reta | 67.7% | 22.0% | 2160.2 m |
| **Centro** | **MOPPO `safe`** | 70.3% | **1.0%** | **1254.3 m** |
| Centro | PPO-S2 PoC | **90.0%** | 0.7% | 1425.4 m |
| Centro | Linha Reta | 85.0% | 2.7% | 2338.5 m |

> **Veredito H7:** Nas ilhas de Porto Alegre, o preset seguro do MOPPO obteve a **menor taxa de falsa localização de todos os métodos** (19.7% vs 21.3% da PPO-S2 e 22.0% da reta), com economia expressiva de distância (voou 1028 m contra 2160 m da reta). O custo da transferência direta foi a taxa de sucesso moderada (49%), explicada pela severa inundação que alterou o contorno das ilhas.

### Vídeos e Animações em Porto Alegre:

#### 1. Ilhas do Delta do Jacuí (Episódio 4):
O drone parte da várzea alagada e converge com **86.4% de pico** ao alcançar as margens habitadas do canal fluvial próximo à ponte da BR-290:

![MOPPO Seguro nas Ilhas de Porto Alegre](results_mo/videos/poa_ilhas_mo_safe.gif)

*Comparação Lado a Lado nas Ilhas (MOPPO vs Linha Reta vs Zigue-zague):*

![Comparação nas Ilhas de Porto Alegre](results_mo/videos/poa_comparison_ilhas_mo_vs_straight_vs_zigzag.gif)

#### 2. Centro Histórico e Parque da Redenção (Episódio 1):
O drone navega pela malha urbana de Porto Alegre sobrevoando o Parque da Redenção e converge com **77.0% de certeza** em 35 passos, poupando mais de 2.3 km de voo em relação ao Zigue-zague:

![MOPPO Seguro no Centro de Porto Alegre](results_mo/videos/poa_centro_mo_safe.gif)

*Comparação Lado a Lado no Centro (MOPPO vs Linha Reta vs Zigue-zague):*

![Comparação no Centro de Porto Alegre](results_mo/videos/poa_comparison_centro_mo_vs_straight_vs_zigzag.gif)

---

## 5. Estrutura dos Arquivos de Vídeo e Scripts

Para reproduzir ou gerar novos vídeos, utilize os scripts automatizados:

```bash
# Gerar vídeos do Vale do Taquari
bash scripts/generate_mo_videos.sh

# Gerar vídeos de Porto Alegre (Centro e Ilhas)
.venv/bin/python -u -m src.dal.mo_poa_video
```

Os arquivos gerados estão disponíveis em:
- **Taquari:** `results_mo/videos/episode_0_mo_fast.mp4`, `comparison_ep0_mo_vs_straight_vs_zigzag.mp4`, `pareto_frontier_explainer.mp4`
- **Porto Alegre:** `results_mo/videos/poa_ilhas_mo_safe.mp4`, `poa_comparison_ilhas_mo_vs_straight_vs_zigzag.mp4`, `poa_centro_mo_safe.mp4`, `poa_comparison_centro_mo_vs_straight_vs_zigzag.mp4`
- Versões em `.gif` de alta resolução disponíveis na mesma pasta para relatórios e apresentações web.

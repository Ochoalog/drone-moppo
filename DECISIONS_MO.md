# DECISIONS_MO (Multi-Objective PPO à la AMOR)

Decisões de implementação para a extensão multiobjetivo de `drone_active_loc/` (MOPPO + memória ANL).

| # | Decisão | Motivo | Alternativa descartada |
|---|---|---|---|
| 1 | Vetor de recompensa $r_t \in \mathbb{R}^4$ com decomposição exata: info ($H_{t-1}-H_t$), custo ($-0.01 - 0.1\cdot\text{blocked}$), sucesso ($+1$ se terminal ok), risco ($-1$ se terminal falso). | Garante que para $w=(1,1,1,1)$, $r\cdot w$ é estritamente idêntico ao escalar original em todos os passos. | Reescalonar recompensas a priori (quebraria equivalência exata com PPO original). |
| 2 | Fatores de escala iniciais fixados em $s=[1.0, 1.0, 1.0, 1.0]$. | Permite que $w_{\text{uniforme}}=(0.25, 0.25, 0.25, 0.25)$ induza exatamente a recompensa da PoC escalada por $1/4$, mantendo a mesma política ótima e comparabilidade direta. | Normalizar por desvio padrão empírico da reta (alteraria a interpretação direta dos pesos fixados). |
| 3 | Vetor de observação composto por $w$ (4), histórico de ações (64, se com memória) e fração de passos restantes (1), totalizando 69 dimensões (ou 5 sem memória). | Fidelidade estrita à especificação da §2. | Manter `blocked_axes` no vetor (não solicitado na §2; a memória de ações já reflete movimentos bloqueados). |
| 4 | Normalização de $w$ na entrada da rede neural como $w_{\text{norm}} = (w - 0.25) \times 4.0 \in [-1, 3]$. | Coloca $w$ na mesma escala $[-1, 1]$ dos descritores PCA e da crença, evitando que os 1024 canais da CNN dominem os pesos no MLP inicial. | Deixar $w \in [0, 1]$ sem normalização (risco de ignorar o condicionamento de pesos). |
| 5 | GAE vetorial computado independentemente por objetivo e escalarizado por $\hat{A} = A \cdot w$ com normalização por minibatch. | Segue estritamente AMOR §4.2 e Eq. (5). | Escalarizar recompensas antes do GAE (perderia a estimativa de valor vetorial e vantagem por objetivo). |
| 6 | Orçamento de treino ajustado para medição real de throughput na CPU i7-10510U: ~43 FPS. | Conforme §5.1, o total deve caber em ~8 h de CPU. Com 43 FPS (~155k passos/h), 1.05M passos totais levam ~6.8 h de treino + ~1 h de avaliação. | Forçar 2M passos por seed (levaria mais de 24 horas, violando o teto de 8h). |
| 7 | Cortes de execuções ordenados conforme regra de desempate da §5.3. | §5.3 determina: "corte nesta ordem: o seed 1 da MO com memória, depois o w_uniforme fixo". Cortados: seed 1 da MO com memória e w_uniforme fixo (PPO da PoC serve de referência). Mantidos: MO com memória seed 0 (350k), MO sem memória seed 0 (250k), PPO fixo rápido (225k), PPO fixo seguro (225k). | Cortar ablação de memória ou PPOs de peso fixo essenciais para H4. |
| 8 | No máximo 3 checkpoints por execução (~25%, ~50%, ~75%) + modelo final. | Restrição estrita de disco (§0: runs_mo + results_mo < 1 GB). | Salvar checkpoints a cada 10 rollouts (esgotaria os ~1.7 GB livres em disco). |
| 9 | Trajetórias .npz salvas apenas para os primeiros 10 episódios dos 3 presets da MO com memória e da reta. | Atende a especificação da §6.5 e respeita a cota de disco. | Salvar todas as trajetórias de avaliação. |
| 10 | Verificação pareada com sementes fixadas `2024 + episode_idx`. | Garante que todas as políticas enfrentam exatamente a mesma sequência de janelas, posições iniciais e ruídos estocásticos de dinâmica. | Amostragem independente por método (aumentaria variância de comparação). |

---

## Critérios das Hipóteses (Pré-registrados antes da avaliação completa, §7)

Os critérios a seguir foram fixados antes de rodar os experimentos completos da §5.4 e §5.5:

- **H4 (Uma política no lugar de várias):**
  - *Critério:* Suportada se, para cada $w_k \in \{w_{\text{rápido}}, w_{\text{seguro}}\}$, o IC 95% da diferença do retorno escalarizado $r \cdot w_k$ entre $\text{MO}(w_k)$ e $\text{PPO-fixo}(w_k)$ contém 0 ou é favorável à MO.
  - *Não suportada:* se algum IC for inteiramente desfavorável à MO.
  - *Inconclusiva:* nos demais casos.

- **H5 (Dominar a linha reta):**
  - *Critério:* Suportada se existir algum peso $w$ na grade simplex com:
    1. Sucesso: limite inferior do IC 95% da diferença $\text{MO} - \text{reta} \ge -0.03$;
    2. Localização falsa: limite superior do IC 95% da diferença $\le 0$;
    3. Distância: limite superior do IC 95% da diferença $< 0$.
  - Como são 38 pesos avaliados, o resultado deve ser reportado tanto com IC nominal quanto com correção de Holm-Bonferroni para comparações múltiplas.

- **H6 (Controle do risco e adaptação de comportamento):**
  - *Critério:* Suportada se a correlação de Spearman entre $w_{\text{risco}}$ e taxa de localização falsa for negativa com IC 95% que exclui 0 ($\rho < 0$), E a correlação de Spearman entre $w_{\text{custo}}$ e distância for negativa com IC 95% que exclui 0 ($\rho < 0$).

- **H7 (Exploratória, Porto Alegre zero-shot):**
  - *Critério:* Suportada se a taxa de localização falsa do preset "seguro" ($w_{\text{seguro}}$) na região das `ilhas` (P1) tiver IC 95% da diferença pareada contra a PPO-S2 inteiramente abaixo de 0 ($\Delta \text{false\_loc} < 0$).
  - O custo em sucesso e distância será reportado como tradeoff.

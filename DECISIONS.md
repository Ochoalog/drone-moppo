# DECISIONS

Decisões tomadas durante a implementação (decisão, motivo, alternativa descartada).

## Ambiente
- **CPU apenas.** `nvidia-smi` falha (sem driver); CPU i7-10510U (4 núcleos / 8 threads), 15 GB RAM, ~4 GB livres em disco. Torch CPU (`download.pytorch.org/whl/cpu`). *Alternativa:* nenhuma.
- **`uv` + `.venv`, Python 3.12**; versões fixadas em `requirements.txt` (freeze) e `pyproject.toml`.

## Fase 1: dados
- **AOI em UTM:** a maior caixa alinhada à grade de 50 m **dentro** do retângulo lon/lat (x 407800–414850, y 6755150–6778300 → 141 × 463 células). *Motivo:* o retângulo lon/lat não é alinhado em UTM; a caixa interna garante que tudo está dentro da AOI pedida. *Alternativa:* caixa envolvente (incluiria pontos fora da AOI).
- **Leitura no overview fator 4 (1,2207 m/px) e miniatura no fator 16 (4,88 m/px)**, em blocos de 2048 px lidos em paralelo (8 threads, um handle por thread), recortados aos limites do dataset e completados com zeros. As três datas têm a mesma grade de pixels (origens diferem por múltiplos de 5000 m), então não há reamostragem. Download total ≈ 35 s.
- **Armazenamento como GeoTIFF JPEG q95** (~50–60 MB por data) em vez de `.npy` (355 MB por data). *Motivo:* sobram só ~4 GB de disco; a fonte já é JPEG, então a perda extra é desprezível.
- **Pixel válido = max(R,G,B) > 3** (e não ≠ 0). *Motivo:* o ringing do JPEG deixa valores 1–3 no nodata.
- **Frações por célula calculadas sobre a footprint de 150 m** (e não sobre o quadrado de 50 m). *Motivo:* o descritor vê a footprint inteira; nuvem nas vizinhanças corrompe o descritor.
- **Máscara de nuvem refinada:** HSV (V > 200, S < 40) → **abertura com disco de 15 px (~18 m)** → dilatação com disco de 41 px (~50 m). *Motivo:* o detector bruto marcava os telhados brancos da área urbana (manchas idênticas em `post` e `pre_mar23`), o que invalidaria justamente as células mais distintivas. A abertura remove objetos pequenos e mantém as nuvens (centenas de metros). A dilatação grande cobre bordas suaves. Resultado: nuvem bruta (como em §3) post 1,0%, mar23 0,85%, oct23 15,6%. Refinada: post 0,16%, mar23 0,12%, oct23 17,2%. *Alternativa descartada:* detector bruto com dilatação 7 px (descartava ~4% das células, quase todas urbanas).
- **Teste de §3:** a fração de nuvem é comparada (±30%) usando o detector **bruto**, que é o que reproduz a medição manual. A máscara refinada tem um teste próprio.

## Alinhamento
- Correlação de fase (OpenCV, janela de Hanning) em magnitude de Sobel, nos **60 tiles de 600 m mais estruturados** (maior gradiente médio em `post`) válidos nas duas datas. Usa a mediana dos tiles com resposta ≥ 0,05. Convenção verificada com deslocamento sintético (+3,3 px → +3,16 px estimado).
- **Resultados:** `pre_mar23` (2,22 E, 1,22 N) m, |d| = 2,5 m (MAD em x = 2,2 m); `pre_oct23` (4,92 E, 0,85 N) m, |d| = 5,0 m. Os dois passam de 2 m, então **a correção é aplicada** (as crops do mapa são amostradas em p + offset).
- `pre_mar23` deu um pouco acima dos 1,5–2 m de §3, com dispersão espacial grande em x (relevo e ortorretificação diferentes). Isso é ≪ 50 m (célula) e foi mantido com o offset global. *Alternativa:* campo de deslocamento local (desnecessário nessa escala).

## Fase 2: descritores
- **Recorte:** footprint de 150 m reamostrada para 128 × 128 px por uma única `warpAffine` (rotação + escala) a partir da imagem de 1,22 m/px. Recortes do mapa usam o centro da célula + offset de alinhamento.
- **Observações (K = 2):** jitter uniforme ±10 m em x e y, rotação uniforme ±5°, ganho de contraste ±10%, viés ±12,8 DN e ruído gaussiano σ = 3 DN. Mesma semente para "hand" e "cnn" (mesmas perturbações).
- **"hand" (224-d):** histograma de orientação do gradiente (8 bins, sem sinal, ponderado pela magnitude, com raiz quadrada estilo Hellinger) em grade 4 × 4, mais média e desvio de L, a, b em grade 4 × 4.
- **"cnn" (512-d):** ResNet18 IMAGENET1K_V1, saída do GAP, entrada 128 px normalizada pelo ImageNet.
- **Normalização:** z-score por dimensão **por data**, com estatísticas apenas das células de treino, seguido de L2. *Motivo:* features positivas (HOG/ReLU) têm cosseno ≈ 1 entre quaisquer pares, e a padronização por data também remove parte do deslocamento radiométrico entre datas sem usar rótulos. *Alternativa:* L2 cru (cossenos pouco discriminativos).
- **Throughput:** "hand" ~300 recortes/s (1 núcleo); "cnn" ~96 recortes/s com 7 threads (sozinha, ~175/s no micro-benchmark). "hand" e "cnn" rodaram **em paralelo**, totalizando 327k recortes (3 datas de mapa + 2 × post). O tempo de parede da CNN ficou em ~1 h 10 min, um pouco acima da meta de 1 h, por eu ter rodado calibração e testes ao mesmo tempo. Como o cálculo é feito uma única vez, mantive K = 2 e 128 px. *Alternativa:* 96 px / K = 1 (mais rápido, mas descritores piores e sem variação de observação).
- **Cache:** descritores padronizados em float16 e memory-mapped (`data/cache/<desc>/`), compartilhados pelos processos do `SubprocVecEnv` via page cache.

## Fase 3: filtro, β e baselines
- **Movimento no filtro:** deslocamento exato com bloqueio na borda (mesma regra da dinâmica real) + blur gaussiano separável σ = 0,5 com reflexão simétrica (conserva massa). Movimento real: 10% de chance de o destino ser trocado por um vizinho-8 aleatório (recortado à janela). Ações que saem da janela ficam bloqueadas (sem slip) e o geofence informa **qual** eixo bloqueou (`blocked_axes`). Essa informação também entra no vetor do PPO, para ele ter o mesmo que a baseline "reta" usa para refletir.
- **Célula de mapa inválida:** verossimilhança = média das verossimilhanças das células válidas, isto é, a marginal p(z) ("uniforme": não favorece nem penaliza). **Observação inválida** (post nublado/nodata na célula verdadeira): sem update. Prior uniforme só sobre células de mapa válidas.
- **Distintividade:** 1 − máx. cosseno com células válidas no **anel 3 ≤ d ≤ 5** células, em vez de no disco de raio 5. *Motivo:* footprints a < 3 células (150 m) se sobrepõem, então os vizinhos imediatos teriam sempre cosseno ≈ 1 e a distintividade seria ≈ 0 em todo lugar.
- **β em dois estágios:**
  1. β₁ por máxima log-verossimilhança da célula verdadeira em 20 janelas de treino (spec). Resultado "hand": S0 167, S1 21,6, S2 13,9.
  2. **Temperamento κ (β = κ·β₁).** No primeiro gate (β₁ puro), os episódios terminavam em 1 a 4 passos, com localização falsa de 27% em S1 e 46% em S2 (`results/gate` da 1ª tentativa, ver abaixo). O β₁ trata observações sucessivas como independentes, mas os erros são correlacionados (footprints sobrepostas, mesma data, mesmas mudanças no terreno), e por isso o filtro fica superconfiante. κ é escolhido em {1, 0,7, …, 0,04} maximizando **sucesso − localização falsa** (a utilidade terminal da recompensa) de uma política aleatória em 200 episódios de **treino**, com empate resolvido pelo maior κ.
  - Primeiro testei "maior κ com P(falsa | concentrou) ≤ 10%". Funcionou em S1 (κ = 0,18), mas em S2 degenerou para κ = 0,09, com 3% de sucesso, porque os erros de S2 são sistemáticos. Por isso foi descartado.
  - Resultado "hand": κ = 1 (S0), 0,25 (S1), 0,25 (S2).
- **Gulosa:** a especificação manda usar "as observações pré-computadas" das células amostradas. Em S1/S2 isso é informação **privilegiada** (o drone não conhece a aparência atual do terreno). Mantive esse modo como `greedy` (spec) e acrescentei `greedy_map`, que prevê a observação pelo descritor do **mapa** (o que um drone real tem). Os dois entram na avaliação.
- **Zigue-zague:** pernas de 16 células ao longo de um eixo cardinal aleatório, faixas de 4 células. Ao bater na borda durante a perna, faz imediatamente o deslocamento transversal; ao bater durante o transversal, inverte o lado da varredura.
- **Reta:** heading aleatório entre os 8, refletindo as componentes bloqueadas.

### Tentativas do gate de sanidade (S0, treino, 200 episódios)
| # | mudança | greedy S0 | zigue-zague S0 | S1 greedy / zz | S2 greedy / zz | resultado |
|---|---|---|---|---|---|---|
| 1 | "hand", β₁ puro | 1,00 | 1,00 | 0,73 / 0,74 (falsa 0,27 / 0,26) | 0,54 / 0,535 (falsa 0,46 / 0,465) | passa, mas episódios de 1–4 passos e filtro superconfiante |
| 2 | "hand", β = κ·β₁ | 1,00 | 1,00 | 0,90 / 0,96 (falsa 0,09 / 0,03) | 0,565 / 0,645 (falsa 0,275 / 0,075) | passa; episódios de 8–25 passos em S1/S2 |

## Fase 4: PPO
- **Extrator CNN:** a primeira versão (conv 5×5/s2 com 32 canais → 64 → 64, ~13 M MAC/amostra) rodava a ~28 fps, o que daria ~18 h por execução de 2M passos. A versão usada tem conv 4×4/s4 com 16 canais → 3×3/s2 com 32 → 3×3/s2 com 64 → MLP 1024+3 → 256 (~1,3 M MAC). A crença perde resolução de 1 célula na primeira camada, o que é aceitável para escolher entre 8 direções.
- **Smoke:** `batch_size = 512`, porque o rollout com 2 envs × 256 passos tem 512 amostras (< 2048).
- **`n_epochs = 5`** (default do SB3 = 10; a spec não fixa esse valor). *Motivo:* com 10 épocas, a atualização dominava (311 fps). Com 5, ~485–511 fps, e mais amostras novas valem mais que mais épocas sobre as mesmas. Os demais hiperparâmetros seguem a spec: n_steps 256, batch 2048, lr 3e-4, γ 0,99, ent 0,01, GAE λ 0,95, clip 0,2; cabeças π/V [128].
- **Descritores em float32 na RAM** (~175 MB/processo com "hand"). A conversão float16→float32 por episódio custava ~7 ms por `reset` (145 resets/s → 900/s após a mudança).
- **Orçamento noturno:** 2M passos × 9 execuções não cabiam (~9 × 70 min + avaliação). **S0 é degenerado**: no gate, todos os métodos localizam em 1 passo com 100% de sucesso, então não há o que aprender. Por isso recebe 300k passos (~12 min), e S1/S2 recebem 1,4M (~46–48 min cada). Estimativa: ~5,3 h de treino + ~1 h de avaliação. Ordem: seed 0 de S0, S1, S2, depois seed 1 e seed 2 (se o tempo acabar, todos os cenários já têm uma seed). *Alternativa:* 1,2M uniformes (desperdiça ~1,5 h em S0).

## Fase 5: avaliação e critérios das hipóteses
- **Sementes pareadas:** o episódio i usa a semente `eval.seed + i` (2024 + i), que fixa janela, início **e** o ruído de dinâmica e de variante de observação. Cada método vê exatamente os mesmos episódios. As janelas dependem do cenário (critério ≥ 70% válidas usa a máscara do mapa do cenário), então a comparação é pareada **dentro** de cada cenário.
- **PPO determinístico** (argmax) na avaliação. As 3 seeds são agregadas (3000 episódios), e para os testes pareados a métrica de cada episódio é a média das seeds. A tabela também traz o desvio-padrão do sucesso entre seeds.
- **Teste de Wilcoxon** sobre "passos penalizados" (falha ou localização falsa = 81), que combina sucesso e velocidade numa métrica pareada. A diferença de sucesso tem IC 95% por bootstrap pareado (2000 reamostras).
- **H1:** em cada cenário não degenerado, o PPO deve bater cada baseline simples (IC de Δsucesso > 0, Wilcoxon p < 0,05 e mediana de passos penalizados não pior). Contra uma baseline já com ≥ 98% de sucesso, basta Δsucesso ≥ −1 pp com passos significativamente menores. Além disso, Δsucesso vs `greedy` ≥ −5 pp.
- **Cenário degenerado** = todas as baselines simples com ≥ 98% de sucesso e mediana ≤ 2 passos (é o caso de S0). Fica excluído de H1/H3 e o relatório diz isso explicitamente.
- **H2:** vantagem = PPO − melhor baseline simples. Suportada se for monotônica S0 < S1 < S2 e os ICs de S0 e S2 não se sobrepuserem.
- **H3:** a política é treinada no norte e avaliada no sul (1000 episódios) e também no norte (300 episódios). Suportada se, no sul, o PPO ainda bate a melhor baseline simples (IC > 0) e sua diferença para a gulosa (referência sem treino, que mede a dificuldade intrínseca da região) cai no máximo 5 pp em relação ao norte.

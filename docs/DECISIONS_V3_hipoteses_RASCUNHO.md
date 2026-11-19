# RASCUNHO (substituído): o texto aprovado está na §2 do `DECISIONS_V3.md`, commitado em 19/11. Este arquivo fica só como histórico.

> **Status: rascunho, não pré-especificado ainda.** Depois da revisão, o texto vai para a §2 do `DECISIONS_V3.md` e é commitado **antes** de enfileirar a matriz completa (sexta, 20/11). A partir daí, qualquer mudança exige registro com data e motivo, e os vereditos saem pelos dois critérios.
> Base: §3.7 e §4 do `PLANO_DESENVOLVIMENTO.md` v3. Os pontos que o plano deixava abertos e que a IA-1 operacionalizou estão marcados com **[definido aqui]**.

## 1. Definições comuns

- **Episódio:** ambiente realista em lote (`dal.v3.envs.realistic_env`), janela de 3,2 km (128 × 128 células de 25 m), até 80 passos, deriva de 2% (exceto na ablação), seeds de avaliação 2024 + i (i = 0…999), ruído do episódio sorteado no reset (pareado entre métodos).
- **Sucesso:** massa da crença ≥ 0,9 num raio de 100 m em torno da célula MAP **e** erro ≤ 100 m. **Localização falsa:** concentrou a mais de 100 m. **Timeout:** não concentrou em 80 passos.
- **Tempo de voo:** 6 s por passo horizontal (5 s de voo + 1 s de captura); 16 s por mudança de altitude. **[definido aqui]** Métrica de custo principal: tempo de voo médio **nos episódios com sucesso**, sempre ao lado da localização falsa.
- **Melhor heurística CA** (H3) **[definido aqui]:** entre reta-CA, zigue-zague-CA, aleatória-CA, greedy_map-CA e reta-150-CA, a de **maior sucesso no conjunto de calibração** (janelas de treino da dobra). É escolhida antes de olhar o teste.
- **Descritor:** o escolhido pelo G1 completo (ajuste fino > zero-shot > hand, pelo top-1 no treino da dobra A).

## 2. Hipóteses

| Id | Hipótese | Critério (operacional) |
|---|---|---|
| **H1** | O descritor aprendido supera o hand | Top-1 de recuperação no **teste** da dobra A (S1 e S2), mesma função do `REPORT.md` §2 (`window_stats`, 20 janelas, semente 0), com IC por bootstrap de **células consulta** (B = 2000). Suportada se, em S1 **e** em S2, o limite inferior da diferença (aprendido − hand) > 0. **[definido aqui]** O hand é calculado sobre os mesmos recortes (COG), não o da v1 (JPEG); a referência da v1 (0,409 / 0,269) aparece ao lado |
| **H2** | O filtro CA reduz a localização falsa | Para a **reta** e para a **PPO-CA** (a política treinada com filtro CA, avaliada com filtro CA × padrão), no Taquari S2 (dobra A) **e** nas ilhas de Porto Alegre: IC da diferença de localização falsa (CA − padrão) inteiramente < 0 **e** limite inferior da diferença de sucesso ≥ −2 p.p. Suportada se valer nos 4 casos (2 métodos × 2 locais); não suportada se algum IC da localização falsa for inteiramente ≥ 0; inconclusiva nos demais |
| **H3** (principal) | A PPO-CA paga a pena contra a melhor heurística CA | Por local (Taquari S2 dobra A, ilhas, Centro, terceira área): **(a) superioridade:** probabilidade de melhora em sucesso (rliable, 10 seeds; P(PPO-CA > heurística) com a heurística como uma "seed" fixa) > 0,5 com IC 95% que exclui 0,5; **ou (b) não inferioridade com eficiência:** limite inferior da diferença de sucesso ≥ −3 p.p. **e** IC da diferença de tempo de voo nos sucessos inteiramente < 0. Holm entre os locais (os p-valores vêm de bootstrap). Suportada no local se (a) ou (b); reportar qual |
| **H4** | A altitude ajuda onde o mapa mais mudou | PPO-CA-ALT × PPO-CA, Taquari S2 dobra A: **(a)** sucesso não inferior (limite inferior ≥ −3 p.p.) **e** tempo de voo menor (IC < 0), **ou** sucesso superior (limite inferior > 0); **e (b)** Spearman entre a altitude média do episódio e o índice de mudança da janela > 0, com IC por bootstrap de janelas que exclui 0. Suportada se (a) **e** (b) |
| **H5** | A relação do H3 se mantém zero-shot | O H3 é suportado em ≥ 2 de 3 regiões (ilhas, Centro, terceira área). O evento externo é exploratório (sem veredito) |
| **M1** | Mecanismo (estudo v1) | Já avaliada pelo critério do `DECISIONS_FINAL.md` (F3): **não suportada** |

## 3. Estatística

- **ICs das diferenças pareadas:** bootstrap hierárquico (seeds → janelas, `dal.evaluation.statistics`), B = 2000, 95%. Para heurísticas (sem seed), só janelas.
- **rliable:** IQM do sucesso entre seeds e probabilidade de melhora, com bootstrap estratificado por dobra e região (biblioteca `rliable`).
- **Holm** entre locais no H3. Nos demais, IC nominal (e dito assim).
- **Vereditos gerados por código** a partir deste texto, nunca à mão; os ICs antigos nunca aparecem sozinhos.

## 4. Matriz e orçamento

| Bloco | Métodos | Seeds | Dobras | Execuções |
|---|---|---|---|---|
| Principal | PPO, PPO-CA, PPO-CA-ALT | 10 | A | 30 |
| Ablação de percepção | PPO-CA com hand | 3 | A | 3 |
| Validação espacial | PPO-CA, PPO-CA-ALT | 3 | B, C | 12 |
| Ablação de deriva | PPO-CA com deriva 0% e 5% | 3 | A | 6 |
| Baselines | reta, zigue-zague, aleatória, greedy_map (padrão e CA) e reta-150 | — | A, B, C | (só avaliação) |

- **Orçamento:** 1,4M passos por execução; 3M se o G4 der folga (≥ 2× o necessário para caber até sábado). **[definido aqui]** A decisão sai dos pilotos e fica registrada antes da matriz.
- **Avaliação:** 1000 episódios pareados por (método × local × dobra).
- **Ordem de corte:** a da §8 do plano.

## 5. Gates

| Gate | Critério |
|---|---|
| G1 | top-1 no **treino** da dobra A, S1 e S2: aprendido > hand (0,409 / 0,269 na v1; hand nos mesmos recortes em paralelo). Senão, o artigo usa o hand e reporta a percepção como resultado negativo |
| G3 | no controle (mapa = observação, S0), a reta com filtro padrão faz ≥ 90% de sucesso |
| G4 | ≥ 5.000 passos/s de treino na T4 (mediana de `progress.csv` do piloto) |
| Gate de aprendizado (aos 25% de cada treino) **[definido aqui]** | sucesso da política (amostrada) − sucesso da aleatória ≥ 5 p.p., em 200 episódios de treino da dobra. Falhou → `INVALID_GATE` (pasta + `DONE.json`), sem veredito |

# DECISIONS_FINAL: análises finais do artigo (P5–P7)

Critérios pré-especificados: escritos e commitados **antes** de rodar as análises correspondentes. Todas as saídas vão para `results_final/`, e os dados de `data/` e `data_poa/` são só lidos.

## F1. Índice de mudança no Taquari (P7.1), 18/11/2026

- **Definição:** a mesma função de Porto Alegre (`dal.poa.change_index`): 1 − cos(descritor do mapa, média dos K descritores de observação), por célula, com os descritores padronizados como no ambiente (`features.load_descriptors`, estatísticas das células de treino).
- **Por episódio:** média do índice nas células válidas da janela (`change_window`), para S1 e S2 no teste (sul), nos 1000 episódios já avaliados. A janela vem de `reconstruct_episode_window`.
- **Figura:** sucesso × quintil de `change_window`, com os quintis calculados **por local** (Taquari: S1 e S2 juntos; Porto Alegre: P0–P2 e regiões juntos, como no `REPORT_POA.md`). Todos os métodos.
- **Leitura:** descritiva (Spearman e Q1 − Q5 com IC), sem veredito novo. O índice depende do descritor, então ele mede o **descasamento entre mapa e observação**, não "mudança física".

## F2. Covariável independente do descritor (P7.2), 18/11/2026

- Porto Alegre: fração de água (já existe, `water_obs` − `water_map`).
- Taquari: diferença absoluta da cor média Lab por célula entre o mapa e a observação, depois de igualar o histograma do mapa ao da observação (canal a canal, sobre a AOI inteira).
- **Uso:** Spearman entre a covariável e o sucesso, por método, ao lado do índice de mudança. Se as duas covariáveis apontam na mesma direção, a queda de sucesso não é só uma tautologia do descritor. **Corte possível:** se não couber no prazo, fica como limitação declarada.

## F3. Mecanismo M1 (P7.3, exploratório), 18/11/2026

**Pergunta:** a PPO erra (localização falsa) porque busca células que **pareciam** distintivas no mapa antigo, mas que **mudaram**?

- **Dados:** PPO-S2 (seeds 0, 1 e 2) e reta, reexecutadas em argmax nos mesmos episódios (determinísticas: reproduzem os CSVs existentes, o que é conferido), registrando a célula de cada passo. Locais: Taquari S2 (1000 episódios, teste sul) e Porto Alegre P1 zero-shot, ilhas e centro (500 episódios cada).
- **Medidas por episódio:** média, sobre as células visitadas (incluindo a de partida), da **distintividade do mapa** (canal `distinct` do ambiente) e do **índice de mudança** (F1).
- **M1 é suportada em um local** se as duas condições valem para as **duas** medidas:
  - (a) nos episódios da PPO, média nas localizações falsas − média nos sucessos > 0, com IC 95% que exclui 0;
  - (b) nas localizações falsas, média da PPO − média da reta > 0, com IC 95% que exclui 0.

  O IC vem de bootstrap por janela (B = 2000). Para (b), as seeds da PPO entram como no bootstrap hierárquico, e o conjunto de episódios de cada método é o das suas localizações falsas.
- **Veredito:** **suportada** se valer no Taquari S2 **e** em pelo menos uma região de Porto Alegre; **não suportada** se, em algum local, uma das diferenças tiver IC inteiramente < 0; **inconclusiva** nos demais casos.
- **Leitura:** associação, não causalidade. Mesmo que M1 seja suportada, a tese do artigo só a usa como mecanismo plausível.

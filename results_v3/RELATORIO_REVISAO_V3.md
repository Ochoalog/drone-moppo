# Relatório para revisão independente: estudo v1 e v3 "change-aware" (IA-1, 18–23/11/2026)

> **Para o agente revisor.** Este arquivo resume o que foi feito, com números, decisões, desvios do plano e pontos fracos conhecidos, e diz onde conferir cada coisa. A fonte de verdade é o repositório. Tudo que está aqui tem um registro mais detalhado em `DECISIONS_V3.md` (V1–V53), `docs/STATUS_V3.md` e nos arquivos de resultado citados. Se algo aqui divergir desses arquivos, valem os arquivos, e a divergência é um achado.
>
> **Escrito por:** IA-1 (Claude Code), 21/11/2026, ≈ 15:00 (horário de Brasília). **Atualizado em 23/11, antes do congelamento:** §6 (H2 e E), §7 (V46–V53) e §11.
> **Branch:** `feat/v3-change-aware`, no worktree `/storage/projects/RL_project_final/dal_v3`; na hora deste relatório ela estava em `2c85667`.
> **Repositório:** https://github.com/Ochoalog/drone-moppo. Os resultados do Colab ficam na branch `colab/results`.

---

## 1. Contexto e regras

**Projeto:** localização ativa de um drone sem GNSS sobre mapas desatualizados de enchente. Os locais são Taquari, Porto Alegre e Canoas (RS, maio de 2024). O agente usa um filtro bayesiano de histograma, PPO e heurísticas. É o projeto final de RL (UFRGS): apresentação em 27/11 e artigo IEEE de conferência em duas colunas.

**Atores:**
- **IA-1 (Claude Code, local):** código, decisões, estatística e documentação.
- **IA-2 (Gemini, Colab T4):** jobs de GPU, por um contrato de arquivos: `colab/jobs_v3.csv`, `colab/CURRENT_COMMIT`, `DONE.json` e a branch `colab/results`.
- **Vitor (humano):** decide o que é dele e repassa mensagens entre as IAs.

**Regras do prompt v3 que devem ter sido seguidas (verificar):**
1. **Áreas protegidas, nunca alteradas:** `results/`, `results_poa/`, `runs/overnight/`, `data/`, `data_poa/`, `configs/`, `configs_poa/` e os manifestos em `../checkpoints/`, conferidos com `sha256sum -c`.
2. **Saídas novas só em:** `src/dal/v3/`, `configs_v3/`, `runs_v3/`, `results_v3/`, `colab/`, `paper/`, `tests/test_v3_*.py`, `scripts/*v3*`, `DECISIONS_V3.md` e `docs/STATUS_V3.md`.
3. **Vocabulário:** "pré-especificado", nunca "pré-registrado". Os critérios são fixados antes da avaliação.
4. **Prior de mudança:** vem de fonte independente com data ≤ à da imagem pós-evento, e nenhum código do prior abre a imagem pós-evento.
5. **Sem vazamento:** ajuste fino, PCA, β, κ e α usam só a faixa de treino da dobra.
6. **Gate que falha:** a execução vira `INVALID_GATE` e não gera veredito.
7. **Nada inventado:** todo CSV novo tem proveniência (`source_file`, `protocol`, `n_episodes`, `seeds`, `fold`, `region`, `git_commit`, `config_hash`).
8. **[PARAR E PERGUNTAR] antes de:** apagar resultado, reescrever histórico, mudar hipótese depois do commit da §2, cortar fora da ordem da §8 ou reexecutar no congelamento.
9. **Git:** Conventional Commits em inglês; identificadores em inglês; comentários e docstrings em português.

## 2. Linha do tempo resumida

| Data | Marco |
|---|---|
| antes de 18/11 | Padronização do código (inglês nos identificadores, português nos comentários), plano de desenvolvimento, plano do artigo IEEE, push para o GitHub. Revisão v2 (errata da MO, tabelas canônicas, avaliação amostrada) |
| 18/11 | **Estudo v1 fechado**, mergeado na `main` com a tag `v1-study-final`. Base da v3: `render.py`, spec `legacy_50m`, ambiente em lote (G2 passou), ambiente realista, filtro CA, calibração, PPO em lote, dobras, specs v3 com manifesto, prior com teste de proveniência. G0 aprovado para o prior. G1 preliminar: o DINOv2 zero-shot perde para o hand |
| 18–19/11 | G1 do grid4 zero-shot: **não passa** (V26). Ajuste fino da dobra A, com o critério do G1 dele pré-especificado nas janelas de teste: **passa** (V29) |
| 19/11 | Hipóteses aprovadas pelo Vitor e **commitadas na §2 do `DECISIONS_V3.md`** (`65c465b`). Valência cortada (V30); Canoas implementada (V31) |
| 20/11 | Extrações `ft` do Taquari e de Porto Alegre. Calibração da dobra A: κ = 0,04, α = 0. **G3 passa** (reta 100% em S0). Efeito teto em S2 (V34). **H1 suportada** (V36). Pilotos: `INVALID_GATE`; **G4 não passa** (V37). Emenda do gate decidida pelo Vitor (V38) |
| 21/11 | Patches da IA-2 revisados (V39). Matriz preparada (V40). Ablação hand cortada (V41). Código dos vereditos H2–H5 (V42). Pilotos `_v38` falham também no gate emendado (V43). Diagnóstico exploratório (V44). Decisão: só o bloco principal para medir o gate, e avaliação das baselines no teste (V45) |

## 3. Estudo v1 (fechado em 18/11; tag `v1-study-final`)

**O que foi feito** (detalhes em `REPORT_MO_ERRATA.md`, `DECISIONS_FINAL.md`, `DECISIONS_REVISAO.md`, `docs/conciliacao_numeros.md`, `results_final/` e `paper/tables/v1_*.csv`):
- **MOPPO:** o condicionamento colapsou. O gate de aprendizado já tinha falhado no treino; com a avaliação amostrada, todas as políticas aprendidas melhoram. A MO fica como apêndice negativo: H4 não avaliável; H5, H6 e H7 não suportadas.
- **Correção de autoreset do gymnasium 1.4:** a transição espúria NEXT_STEP foi trocada por SAME_STEP.
- **ICs hierárquicos** (seeds → janelas): o IC oficial de Δsucesso PPO − reta em S2 é **[−7,1; +9,8] p.p.** (B = 2000). O valor [−6,1; +9,8] que circulou veio de uma prévia com B = 200 e foi corrigido em V18. **Atenção:** a tag `v1-study-final` ainda tem o valor antigo em algum documento (V18).
- **F1–F3:**
  - F1: índice de mudança no Taquari.
  - F2: a covariável de cor (Lab) cai com o sucesso, ρ de −0,15 a −0,25.
  - **M1 não suportada.** As localizações falsas da PPO acontecem em células mais mudadas, mas não mais distintivas, nem mais mudadas que as da reta. **Isso mudou a tese do artigo:** "a falha acontece onde o mapa mudou" continua; "a política é atraída por células distintivas" saiu.

## 4. v3: o que foi construído (código em `src/dal/v3/`)

| Componente | Arquivo | Verificação |
|---|---|---|
| Renderizador único CPU/GPU (homografia com inclinação, FOV de 84°, ruído por hash, desfoque de raio fixo) | `sensor/render.py` | reproduz recortes da v1: pixels idênticos em ≥ 99,5% e hand com cosseno ≥ 0,9999999 (`colab/specs/legacy_50m_check.json`) |
| Specs determinísticas e manifesto por hash de conteúdo (Taquari, POA, Canoas; 25 m; 3 altitudes; K = 4 variantes) | `sensor/specs.py`, `colab/specs/v3_manifest.json` | a IA-2 regenera e confere (`--verify`) |
| Ambiente em lote, modo legado | `envs/batched_env.py` | **G2 passou**: 1000/1000 trajetórias idênticas; PPO da PoC idêntico nos 3 seeds (`results_v3/gates/G2.json`) |
| Ambiente realista (posição contínua, deriva de 2%, altitude 60/100/150 m, 10 ações, tempo de voo, sucesso em 100 m) | `envs/realistic_env.py` | testes `test_v3_realistic_env.py` |
| Filtro CA L ∝ (1−αm)·exp(βs) + αm·ū | `localization/ca_filter.py` | testes |
| Calibração (β₁ por altitude, κ, α; regra da PoC: maximizar sucesso − falsa da aleatória no treino) | `localization/calibration.py` | — |
| Prior de mudança e teste de proveniência | `localization/change_prior.py` | `test_v3_change_prior.py` |
| Descritores (DINOv2, grade 4×4, PCA por dobra, ajuste fino InfoNCE com vizinhas como negativos, saída `ft` da cabeça) | `descriptors/` | testes, incluindo `test_extractor_ft_output_matches_trained_head` |
| PPO em lote retomável com gate de aprendizado e `INVALID_GATE` | `agents/ppo_batched.py` | testes |
| Dobras espaciais A/B/C (3 faixas de latitude, margem de 200 m) | `evaluation/folds.py` | teste de vazamento |
| Vereditos (bootstrap estratificado, IQM/P(melhora) do rliable, Holm, H1–H5) | `evaluation/verdicts.py`, `scripts/verdict_h1_v3.py`, `scripts/verdicts_v3.py` | testes sintéticos |
| Jobs de calibração, treino e avaliação (sub-regiões ilhas/Centro, modos argmax/amostrado, gate original por episódio) | `jobs.py` | testes |

**Suíte:** `PYTHONPATH=$PWD/src .venv/bin/python -m pytest -q tests/test_v3_*.py`. Na última rodada completa, depois dos patches da IA-2, deu 65 passed; depois disso foram acrescentados testes. `ruff` limpo em `src/dal/v3`.

## 5. Gates

| Gate | Critério | Resultado | Fonte |
|---|---|---|---|
| G0 | dados e proveniência do prior | **aprovado** para Taquari (EMSR720, fontes ≤ 06/05/2024) e POA/Canoas (EMSN194, fontes de 08/05/2024, rótulo "otimista"). A linhagem do prior do Taquari é **autodeclarada pela IA-2** | `DECISIONS_V3.md` §5 |
| G1 zero-shot | top-1 > hand em S1 **e** S2, janelas de treino | **não passa.** GeM S1 0,139 / S2 0,063; grid4 S1 **0,377** × hand 0,411 (só S2 passa: 0,300 × 0,270) | `results_v3/g1/` |
| G1 ajuste fino | top-1 > hand nas janelas de **teste** da v1, S1 e S2 (pré-especificado em V26, antes do resultado) | **passa.** S1 0,460 × 0,368; S2 0,332 × 0,239. Em S0 o ft cai para 0,906 (hand 0,996) | `results_v3/g1/g1_legacy_ftA/` |
| G2 | paridade do ambiente em lote | **passa** | `results_v3/gates/G2.json` |
| G3 | controle S0: reta ≥ 90% | **passa.** 100% (1000 episódios de teste, κ = 0,04, α = 0) | `results_v3/gates/g3_A_ftA/` |
| G4 | ≥ 5.000 passos/s (mediana) | **não passa.** 4.248–4.785 (a IA-2 relatou "atingido"; corrigido em V37). Consequência: orçamento de 1,4 M passos | `results_v3/pilots/` |
| Gate de aprendizado | original: política − aleatória ≥ 5 p.p. aos 25%. Emendado (V38): ou sucesso ≥ aleatória − 2 p.p. **e** tempo nos sucessos ≤ 0,75 × o da aleatória | **todos os pilotos falham nos dois critérios** (6 execuções: 3 de `3da89d1` e 3 `_v38`) | `results_v3/pilots/` |

## 6. Hipóteses (§2 do `DECISIONS_V3.md`, pré-especificadas em 19/11) e situação

| Id | Situação | Números |
|---|---|---|
| **H1** (ft > hand) | **suportada** | S1 +9,2 p.p. [8,3; 10,1]; S2 +9,2 p.p. [8,4; 10,0]; bootstrap de células consulta, B = 2000. Cálculo refeito pela IA-1 a partir dos `.npz` publicados, idênticos byte a byte (`results_v3/verdicts/h1.json`) |
| **H2** (CA reduz a falsa) | **não avaliável.** Com α = 0 (V36), o filtro CA é idêntico ao padrão. O `eval_baselines_A` publicou só o `summary.csv` do filtro padrão, sem `episodes.csv`, então o `scripts/verdicts_v3.py` nunca rodou | `results_v3/eval/eval_baselines_A/summary.csv` |
| **H3** (PPO-CA × melhor heurística CA) | **não avaliável**: nenhuma execução passou no gate | — |
| **H4** (altitude) | **não avaliável** | diagnóstico: a PPO-CA-ALT quase não muda de altitude (média de 100–101 m) |
| **H5** (zero-shot) | **não avaliável** | — |
| M1 (v1) | não suportada | v1 |
| **E1, E2** (Canoas) | **cortadas** (V51) | Canoas tem só 31 janelas válidas |
| **E3** (Δ(L3) − Δ(L0) > 0, contra a melhor heurística) | **contrária** (amostrado e argmax) | −16,7 p.p. [−21,6; −12,5]; argmax −15,1 [−22,2; −8,2] (`results_v3/E/verdicts_E.json`) |
| E4, E5 | descritivas | Tab. III do artigo (`results_v3/E/table3.csv`, `curve.csv`) |

A tabela de vereditos de todas as hipóteses (v1, MO, v3 e E) está no apêndice do artigo (`paper/tables/tab_verdicts.tex`, gerada por `paper/make_tables_v4.py` a partir dos arquivos de veredito).

**Diagnóstico exploratório dos pilotos** (V44; só janelas de treino; sem veredito; `results_v3/diagnostics/diag_pilots_A/`):

| Ao fim do treino (1,4 M) | Sucesso | Tempo nos sucessos |
|---|---|---|
| aleatória | 93,0% | 142,2 s |
| PPO / PPO-CA / PPO-CA-ALT | 95,4–96,4% | 116,1–130,5 s |
| reta | 98,4% | 118,2 s |

As políticas aprendem algo em relação à aleatória, mas terminam no máximo empatadas com a reta. Argmax ≈ amostrado.

## 7. Desvios do plano, emendas e cortes (o revisor deve julgar cada um)

| # | O quê | Quando foi decidido em relação aos dados | Risco para a validade |
|---|---|---|---|
| V22–V24 | Descritor em grade 4×4 e ajuste fino com vizinhas como negativos | depois do G1 preliminar (GeM falhou), antes de qualquer resultado do grid4 e do ajuste fino | baixo: mudança de candidato, com o critério do G1 mantido |
| **V26** | **G1 do ajuste fino nas janelas de TESTE**, não nas de treino como no G1 original | antes do resultado do ajuste fino. Motivo: as janelas de treino da v1 estão dentro da faixa de treino da dobra A (memorização) | médio: o protocolo do gate mudou entre candidatos. Justificativa de vazamento registrada |
| V29 | Só a dobra A é usada no zero-shot | — | — |
| V30 / V41 / V45 | Cortes: Valência (§8 item 1), ablação hand (fora da §8; decisão do Vitor), deriva (§8 item 2), validação espacial (§8 item 4), vereditos de RL | depois de ver os pilotos (V45) | os cortes de V45 vieram depois de ver que o RL não passa no gate |
| V33 | Recalibração com grade de κ estendida e 1000 episódios; **não** adotar α = 0,25 no desempate | antes do resultado da recalibração | baixo. κ = 0,04 virou ótimo interior; α = 0 venceu sem empate |
| V34 | Efeito teto registrado (reta 99,8% em S2); **critérios mantidos** (decisão do Vitor) | antes dos pilotos | — |
| **V38** | **Emenda do gate de aprendizado** (tempo de voo) | depois dos pilotos falharem no gate original, **antes** de ver os tempos de voo dos pilotos | **alto, para o revisor olhar.** Foi uma mudança da §2 depois do commit, autorizada pelo Vitor; os vereditos sairiam pelos dois critérios. O limiar de 0,75 proposto pela IA-1 se mostrou **mais duro que a própria reta** (0,83; V44), um erro de calibração do limiar |
| **V39** | Aceito o patch da IA-2 em que, no Taquari, **todos os cenários usam a PCA gravada pela calibração (S2)**, em vez de uma PCA por cenário | depois do G3 e da calibração | baixo para vazamento (só faixa de treino), mas o **G3 rodou antes** dessa mudança |
| V42 | Operacionalizações dos vereditos: melhor heurística no treino de S2; p-valor do local para o Holm; índice de mudança da janela = prior independente (não o índice de imagem da v1) | antes de qualquer episódio da matriz | baixo |
| **V46** | **Experimento E** (exploratório); todas as execuções de E avaliadas no teste, passem ou não no gate | depois do teto das baselines em Taquari (V44); antes de qualquer job de E | **médio**: nasceu depois de ver dados; o artigo o chama de exploratório |
| V50 | Grade de ruído estendida (σ ∈ {4; 6; 8; 12}); regra dos níveis mantida | depois da primeira varredura de ruído (V49: nenhum σ deixava a aleatória < 75%); antes de treinar qualquer política do E | baixo: só a varredura (aleatória e top-1 no treino) foi vista |
| **V51** | E-real (treino e teste em Canoas), E1, E2 e E-alt cortados; zero-shot das 10 `mx_A_ppo` em Canoas como item descritivo | forçado pela falta de janelas válidas em Canoas (V48: toda divisão com ≥ 20 janelas de teste deixa 0 de treino); depois da primeira varredura de ruído; antes de treinar qualquer política do E | médio: o E perde o regime difícil real; Canoas fica só descritivo |
| **V53** | Ordem da regra dos níveis invertida: L3 → L2 → L1 (o mais difícil primeiro) | depois de a regra original dar níveis invertidos nas duas varreduras (V52: L3 = σ 6, o mais fácil); antes de treinar qualquer política do E | **médio**: mudança da regra depois de ver o resultado dela, mas só com dados das varreduras (aleatória no treino), sem nenhuma política. O L3 chegou a 66,2% de aleatória no treino (67,3% na varredura), não aos 35% visados |

## 8. Pontos fracos e erros conhecidos (autocrítica)

1. **A suspeita da V32 estava errada.** Eu disse que a calibração tinha degenerado porque a aleatória teria utilidade ≤ 0. Os dados mostraram a aleatória com 91–96% de sucesso. Corrigido em V33.
2. **O limiar de 0,75 do gate emendado (V38) foi proposto sem referência a uma heurística.** Nem a reta passaria. Um limiar relativo à melhor heurística teria sido mais defensável.
3. **O filtro CA ficou desligado pela calibração (α = 0).** A "contribuição CA" do método não foi exercida no Taquari. O H2 por construção não mostra efeito, e a PPO-CA = PPO + canal do prior.
4. **Os priors de POA e Canoas são "otimistas"** (fontes de 08/05, um dia antes da imagem pós-evento de 09/05). O de Valência teria sido desenhado sobre a mesma imagem pós-evento (V28), e por isso Valência foi cortada.
5. **A linhagem do prior do Taquari é autodeclarada pela IA-2** (G0). A IA-1 conferiu só o arquivo (hash) e a figura sobre a imagem pré-evento.
6. **Proveniência dos patches da IA-2:** os primeiros 3 pilotos rodaram com código **não versionado** (os `DONE.json` diziam `3da89d1`). Os patches entraram depois em `d398eb0`. Esses pilotos estão `INVALID_GATE` e não contam para nada, mas o fato é um desvio do contrato.
7. **Relatos da IA-2 com erros** corrigidos pela IA-1:
   - G4 "atingido com folga" (falso);
   - S1/S2 rotulados "seca/cheia" (são datas de mapa);
   - "o grid4 supera oficialmente o hand" (só em S2; o critério exigia S1 e S2).

   Toda conclusão da IA-2 foi conferida contra os `DONE.json` e CSVs publicados.
8. **Extração de 100 m do Taquari:** o primeiro `DONE.json` publicado era de uma extração antiga (saída `gem`, commit `b630f19`). A reextração com `ft` foi publicada depois (`498f068`, commit `65c465b`). Verificar que a calibração usou a versão `ft`.
9. **Canoas 100 m:** a extração foi retomada de outra sessão, com mapas a 0 recortes/s nesta. A validade das máscaras de Canoas não foi conferida pela IA-1 (os arquivos estão só no Drive).
10. **Extração local do hand** parada duas vezes por memória (V25; proteção às 04:44 de 19/11). Só o mapa `pre_mar23` de 100 m ficou completo. Não é usada no artigo.
11. **H1 usa as janelas de teste da v1** (grade de 50 m, faixa sul), não o ambiente v3 de 25 m. A ligação "melhor recuperação → melhor localização" no ambiente v3 não foi testada como hipótese.

## 9. Mapa de proveniência (commits e arquivos)

**`feat/v3-change-aware`:**

| Commit | Conteúdo |
|---|---|
| `65c465b` | §2 pré-especificada |
| `efe1aad` | V26 |
| `703de1d` | V29 |
| `6f4fcd6` | V33 |
| `3da89d1` | H1 |
| `ecd2176` | V36/V37 |
| `3700384` | V38 |
| `d398eb0` | patches da IA-2 |
| `1ca39a4` | V39 |
| `2158a2e` | V40 |
| `bcbab82` | V42 |
| `e93b354` | V43 |
| `5902fcc` | V44 |
| `7890281` | V45 |

**`colab/results`:**

| Commit | Conteúdo |
|---|---|
| `826d50b` | G1 grid4 |
| `146d0cc` | `finetune_A` |
| `f24d9b7` | G1 ft |
| `6a3c1ce`, `6e9460b` | calibrações |
| `8df9d3b` | G3 |
| `a8db119` | H1 |
| `d4ccb7a`, `d0f47e8`, `2e60d0b` | pilotos |
| `c629b1e`, `7184f2d`, `7f595a9` | pilotos `_v38` |
| `f076375` | diagnóstico |

**Cópias locais dos resultados publicados:** `results_v3/{g1,calibration,gates,pilots,diagnostics,h1,verdicts,colab_done,finetune}/`.

## 10. Como conferir (comandos)

```bash
cd /storage/projects/RL_project_final/dal_v3
unset PYTHONPATH; export PYTHONPATH=$PWD/src
# áreas protegidas
(cd ../drone_active_loc && for m in ../checkpoints/*.sha256; do sha256sum -c --quiet $m && echo "$m OK"; done)
# testes e lint
.venv/bin/python -m pytest -q tests/test_v3_*.py && .venv/bin/ruff check src/dal/v3 scripts/*v3*
# H1 a partir dos .npz publicados
.venv/bin/python scripts/verdict_h1_v3.py
# o hand das janelas de treino reproduz a v1 (0,409 / 0,269)
.venv/bin/python scripts/eval_retrieval_v3.py --descriptors hand-v1 --name hand_v1
# um DONE.json contra o contrato
.venv/bin/python -c "from dal.v3.reporting.colab_io import validate_done; import pandas as pd, json; from pathlib import Path; p=Path('results_v3/gates/g3_A_ftA/DONE.json'); print(validate_done(p, pd.read_csv('colab/jobs_v3.csv'), json.load(open(p))['git_commit'], Path('.')))"
# histórico das emendas
git log --oneline 65c465b..HEAD -- DECISIONS_V3.md
```

## 11. Estado no congelamento (23/11, antes das 12:00)

**Coleta fechada.** A fila de GPU terminou e nada mais será reexecutado, salvo um bug confirmado.

- **Matriz v3 (V45):** **0 de 30** execuções `mx_A_*` passaram no gate, nem pelo critério original nem pelo emendado. A vantagem sobre a aleatória aos 25% ficou entre −1,5 e +4,0 p.p. Por isso H3–H5 não são avaliáveis.
- **Baselines v3** (`eval_baselines_A`, filtro padrão, 1000 episódios): em Taquari S2 (teste), a aleatória acerta 94,0% e a reta 99,8%. Em Canoas, nas ilhas e no Centro o regime é difícil (tabela de baselines no apêndice do artigo).
- **Experimento E** (exploratório; V46, V50–V53):
  - **Níveis.** L0 σ = 0; L1 σ = 6; L2 σ = 8; L3 σ = 12. A aleatória no treino fica em 93,0 / 87,8 / 83,5 / 66,2%.
  - **Melhor heurística no treino:** a reta a 150 m, em todos os níveis.
  - **Curva** (PPO amostrada, IC hierárquico):

    | Nível | PPO − aleatória | PPO − reta a 150 m |
    |---|---|---|
    | L0 | +3,8 | −1,8 |
    | L1 | +8,1 | −4,3 |
    | L2 | +9,6 | −6,5 |
    | L3 | +10,3 [+5,1; +15,5] | −18,5 [−23,3; −14,3] |

  - **E3 contrária:** −16,7 p.p. [−21,6; −12,5].
  - **E5 (gate):** L0 0/10, L1 0/5, L2 2/5, L3 4/5; o critério original e o emendado dão o mesmo resultado.
  - **Altitude não explica a diferença:** em L3, a reta a 100 m (a altitude da PPO) faz 89,9% e o zigue-zague 79,2%, contra 72,7% da PPO; a gulosa faz 71,3%.
  - **Ressalva de orçamento:** os 1,4 M passos são fixos. Em L3 os episódios são mais longos (tempo nos sucessos da PPO 132 s → 277 s), então a PPO vê menos episódios e pode estar subtreinada lá. Está nas limitações do artigo.
- **Canoas zero-shot** (descritivo; V51): as 10 `mx_A_ppo` acertam 54,2% [47,5; 60,3].
  - Acima da aleatória (+13,4) e da gulosa (+17,3).
  - Empatada com o zigue-zague (+1,5 [−3,5; +6,3]).
  - Abaixo da reta (−10,3) e da reta a 150 m (−16,4).
- **Artigo** (`paper/`, todos os números por macro de `paper/numbers.tex`, gerados por `paper/make_figures.py`):
  - abstract, introdução, VI-D (RQ4, com a Fig. 3 e a Tab. III), discussão, conclusão e apêndice estão escritos;
  - o apêndice tem a tabela de vereditos de todas as hipóteses, a de alterações V22–V53 e as baselines v3.
- **Pendências** (Vitor):
  - compilar no Overleaf (tamanho, overfull; o abstract passou de 250 palavras);
  - tag `results-final` às 12:00, depois do `pytest` e dos dois manifestos;
  - release v1.0 em 25/11 e apresentação em 27/11.

## 12. Perguntas sugeridas ao revisor

1. O G1 do ajuste fino nas janelas de teste (V26) é um critério justo frente ao G1 zero-shot nas de treino, ou o "passa" depende da troca de protocolo?
2. A emenda do gate (V38), decidida depois da falha no critério original, compromete a pré-especificação, mesmo com os dois critérios reportados e sem nenhum resultado de RL que dependa dela?
3. Os cortes de V45 depois de ver o diagnóstico configuram escolha guiada por resultado? O artigo está sendo honesto ao apresentar o RL como resultado negativo?
4. Com α = 0, faz sentido manter "change-aware" no título e na tese? O que o artigo pode afirmar sobre o filtro CA?
5. O efeito teto (reta 98–99,8%) indica um ambiente fácil demais (descritor ajustado + células de 25 m + sucesso em 100 m)? Isso deve ser discutido como limitação principal?
6. A H1 (recuperação na grade v1 de 50 m) sustenta a afirmação de "melhor percepção" para o ambiente v3?
7. Há algum número nos documentos sem fonte rastreável (`source_file`/commit)?

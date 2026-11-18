# Plano de desenvolvimento: finalização do projeto

**Projeto:** localização global ativa de drone sem GNSS com RL sobre mapas desatualizados (enchentes do RS, 2024)
**Repositório:** https://github.com/Ochoalog/drone-moppo
**Situação em 18/11/2026:** PoC (PPO), bateria zero-shot em Porto Alegre e extensão multiobjetivo (MOPPO à la AMOR) prontas e com resultados. Faltam a padronização do código, uma pendência científica no MOPPO, o congelamento dos resultados, o artigo e a apresentação.
**Prazo:** apresentação na semana de 23/11 (**confirmar o dia com o professor**).

---

## 1. Diagnóstico do código atual

| Tema | Situação | O que muda |
|---|---|---|
| Idioma dos identificadores | Quase tudo em inglês, com mistura de português em nomes de módulos, funções e variáveis (`mo_poa_nuvem_video.py`, `generate_taquari_nuvem_comparison_video`, `success_ilhas`, `centro_minus_ilhas`, `adv_ilhas`) | Tudo em inglês: `nuvem` → `cloud`, `ilhas` → `islands`, `centro` → `downtown`. Os nomes das regiões continuam em português só como **valores** (rótulos nos CSVs), para não quebrar os resultados |
| Idioma dos comentários | ~776 comentários, quase todos em inglês, e docstrings em inglês | Comentários e docstrings em **português** |
| Organização | 26 módulos soltos em `src/dal/`. O código de vídeo e de PPTX (~3.000 linhas) está misturado com o núcleo científico | Separar em subpacotes (seção 3) |
| Duplicação | Cinco geradores de vídeo (`mo_video`, `mo_poa_video`, `mo_poa_nuvem_video`, `mo_taquari_nuvem_video`, `mo_*_presentation_video`) repetem renderização de quadro, legenda e mapa | Extrair um módulo `presentation/rendering.py` comum |
| Funções longas | 15 funções com mais de 100 linhas. As maiores: `train_moppo` (254), `generate_taquari_nuvem_comparison_video` (231), `generate_poa_ilhas_nuvem_comparison_video` (219), `build_report_markdown` (208), `run_pipeline` (190), `poa_report.build` (187) | Quebrar em funções de no máximo ~40 linhas com nomes descritivos. Prioridade para as do núcleo (`train_moppo`, `run_pipeline`) |
| Nomes curtos | Scripts auxiliares (`dados_enchente/analyze_v2.py`, `apresentacao_grupo/*.py`) usam `m`, `pa`, `pb`, `cb`, `inl`, `d` | Renomear ou mover para `legacy/` e documentar que estão fora do pipeline |
| Mensagens de commit | Misturam português e inglês e formatos diferentes | Conventional Commits em inglês: `feat:`, `fix:`, `refactor:`, `docs:`, `test:` |
| Lint | Não há | `ruff` com regras `E,F,I,N,B,UP`. A `N` (pep8-naming) pega nomes fora do padrão |

---

## 2. Convenções (valem para todo o grupo)

1. **Identificadores** (módulos, classes, funções, variáveis, constantes, chaves de config, colunas novas de CSV): **inglês**, em `snake_case`, `PascalCase` para classes e `UPPER_CASE` para constantes.
2. **Comentários e docstrings:** **português**. A docstring diz *o que* a função faz e *por quê*, e não repete o código. As referências a artigos ficam assim: `# Ver AMOR (Alegre et al., 2025), §5`.
3. **Clean code:**
   - uma função faz uma coisa, com no máximo ~40 linhas e ~5 parâmetros (se precisar de mais, usar uma `dataclass` de configuração);
   - nada de números mágicos no código: tudo vai para o YAML ou para uma constante nomeada;
   - nomes que revelam intenção (`belief_entropy`, não `h`; `false_localization_rate`, não `fl`);
   - type hints em todas as funções públicas;
   - sem código morto nem comentado.
4. **Termos técnicos padronizados** (glossário do código):

| Conceito | Nome no código |
|---|---|
| crença / histograma | `belief` |
| localização falsa | `false_localization` |
| sucesso | `success` |
| passo | `step` |
| vetor de pesos (preferência) | `preference_weights` (`w` só dentro de fórmulas) |
| nuvem | `cloud` |
| ilhas / centro (regiões de POA) | `islands` / `downtown` (rótulo de dado: `"ilhas"` / `"centro"`) |
| linha reta / zigue-zague | `straight` / `lawnmower` |

5. **Exemplo:**

```python
def compute_false_localization_rate(episodes: pd.DataFrame) -> float:
    """Calcula a fração de episódios em que a crença convergiu para o lugar errado.

    Um episódio conta como localização falsa quando a massa da crença passa de 0,9
    num raio de 100 m em torno de um ponto a mais de 100 m da posição real.
    """
    # Episódios truncados por limite de passos não contam como localização falsa.
    finished = episodes[episodes["terminated"]]
    return float(finished["false_localization"].mean())
```

---

## 3. Estrutura alvo do repositório

```
drone-moppo/
├── README.md                    # visão geral, instalação, reprodução (português)
├── CONTRIBUTING.md              # convenções da seção 2
├── pyproject.toml               # + configuração do ruff
├── configs/                     # taquari/, poa/, moppo/ (hoje: configs, configs_poa, configs_mo)
├── src/dal/
│   ├── geo/                     # data.py, align.py, grid.py
│   ├── localization/            # features.py, calibrate.py, filter.py
│   ├── envs/                    # env.py, mo_env.py
│   ├── agents/                  # train.py (PPO), mo_ppo.py (MOPPO), baselines.py
│   ├── evaluation/              # evaluate.py, mo_eval.py, poa.py, mo_poa_eval.py, statistics.py
│   ├── reporting/               # report.py, poa_report.py, mo_report.py, figures.py
│   └── pipelines/               # mo_pipeline.py e pontos de entrada CLI
├── presentation/                # vídeos e PPTX (fora do pacote científico)
│   ├── rendering.py             # código comum de quadro, mapa e legenda
│   └── ...
├── scripts/                     # .sh e utilitários
├── tests/
├── docs/                        # este plano, plano do artigo, DECISIONS*, REPORT*
├── legacy/                      # dados_enchente/ e apresentacao_grupo/ (fora do pipeline)
└── paper/                       # LaTeX IEEEtran + figuras finais
```

---

## 4. Regra de segurança da refatoração: não mudar nenhum número

A refatoração é **só de forma**. Para garantir isso:

1. **Antes de começar:** tag `v0-pre-refactor` (já criada no push inicial) e manifestos sha256 em `checkpoints/`.
2. Gerar uma **linha de base determinística**: `scripts/smoke.sh`, `scripts/smoke_poa.sh` e `scripts/run_mo_smoke.sh` com seed fixa, guardando os CSVs de episódios em `tests/golden/`.
3. Criar o teste `tests/test_regression_golden.py`, que roda os três smokes e compara os CSVs com a linha de base (`pd.testing.assert_frame_equal`, com tolerância só para colunas de tempo).
4. **Cada PR de refatoração** precisa passar em `pytest -q`, no golden e no `ruff check`.
5. Os modelos treinados (`runs*/`) **não** são retreinados por causa da refatoração. Se uma renomeação mudar uma chave de checkpoint, escrever um adaptador de carregamento em vez de retreinar.

---

## 5. Pendências científicas (prioridade **antes** da refatoração grande)

### 5.1 Crítico: a política MOPPO parece ignorar o vetor de pesos

No `REPORT_MO.md`, os presets `unif`, `fast` e `safe` dão **exatamente** os mesmos números (60,0% de sucesso, 25,0% de localização falsa, 1019,7 m de distância e mediana de 20 passos). A correlação de risco em H6 sai com "variância nula". Isso indica **colapso do condicionamento**: a rede não usa `w`, ou `w` não chega à rede na avaliação.

Consequências:
- o veredito "H4 SUPORTADA" fica sob suspeita, porque compara três políticas idênticas com PPOs fixos diferentes;
- o hipervolume e o EUM podem estar inflados ou mal interpretados;
- a fronteira de Pareto da MO pode, na prática, ser um único ponto.

Investigação (responsável: _____, até 20/11):
1. Conferir em `mo_eval.py` se o `w` do preset está mesmo no vetor de observação na inferência, e não o `default_w`.
2. Conferir se a avaliação determinística (argmax) não está apagando diferenças pequenas nos logits: comparar as distribuições de ação para `w` diferentes no mesmo estado.
3. Teste de sensibilidade: o gradiente da saída do ator em relação a `w` deve ser diferente de zero.
4. Conferir a escala das recompensas por objetivo (`scale_factors`). Se um objetivo domina, o `w` não muda a política.

Desfechos possíveis:
- **bug na avaliação:** corrigir, reavaliar (não precisa retreinar) e regenerar `REPORT_MO.md`;
- **colapso no treino:** um retreino curto com a correção (normalização por objetivo, `w` também na entrada da CNN, ou mais amostras de `w` por lote), se couber numa noite. Se não couber, **relatar como resultado negativo**, explicando o mecanismo. Esse caminho é válido pelo enunciado ("comentar limitações, desafios e barreiras").

### 5.2 Conciliar os números do PPO da PoC

O `REPORT.md` dá 70,7% de sucesso para o PPO em S2 (1000 episódios × 3 seeds). O `REPORT_MO.md` dá 81,0%, 68,3% e 72,3% por seed (300 episódios, com outro protocolo). O artigo precisa de **uma tabela canônica** que diga claramente o *n* e o protocolo de cada número.

### 5.3 Validar com o professor

Levar ao professor o escopo final e as hipóteses H1–H7. O enunciado exige essa validação.

---

## 6. Cronograma

> D = dia da apresentação (semana de 23/11). Ajustar quando a data estiver confirmada.

| Quando | Entrega | Responsáveis |
|---|---|---|
| **Qua 18/11** | Código no GitHub (`main` + branches + tags); planos publicados; papéis definidos | Vitor |
| **Qui 19/11** | `CONTRIBUTING.md`, ruff configurado, linha de base golden (§4); início da investigação 5.1 | A, B |
| **Qui 19 – Sex 20/11** | Investigação 5.1 concluída e decisão tomada: corrigir e reavaliar, retreinar à noite, ou relatar como negativo. Validação com o professor | A + professor |
| **Sex 20 – Sáb 21/11** | Refatoração em PRs pequenos, na ordem: (1) renomeações para inglês; (2) comentários e docstrings em português; (3) mover `presentation/` e `legacy/`; (4) subpacotes; (5) quebrar funções longas do núcleo | B, C (revisão cruzada) |
| **Sáb 21/11** | **Congelamento dos resultados**: tag `results-final`, tabelas e figuras exportadas para `paper/figures/` | A |
| **Sáb 21 – Seg 23/11** | Rascunho completo do artigo (ver `PLANO_ARTIGO.md`) | todos, por seção |
| **Seg 23 – Ter 24/11** | Revisão cruzada do artigo, slides e ensaio da apresentação | todos |
| **D − 1** | Versão final do artigo (PDF) e release no GitHub `v1.0` | Vitor |

**Regra de corte:** se a refatoração atrasar, os passos (4) e (5) ficam para depois da entrega. Os passos (1) e (2) são obrigatórios. O artigo e os números corretos vêm antes da estética do código.

---

## 7. Fluxo de trabalho no GitHub

- Branch `main` protegida. Cada tarefa ganha uma branch `refactor/...`, `fix/...` ou `docs/...` e um PR com um revisor do grupo.
- Abrir uma issue por item das seções 1, 5 e 6. Usar os rótulos `refactor`, `science`, `paper` e `presentation`.
- Os dados (`data*/`, ~1,4 GB) e os modelos (`runs*/`) **não** vão para o git. Eles são reproduzidos por `scripts/prepare_data.sh`, e os modelos finais vão como asset da release `v1.0`.
- Dados Maxar/OpenAerialMap sob **CC BY-NC 4.0**: citar no README e no artigo.

## 8. Definição de pronto

- [ ] `ruff check` sem erros e `pytest -q` verde, incluindo o golden
- [ ] nenhum identificador em português (verificar com `ruff` regra N e com grep pelo glossário)
- [ ] comentários e docstrings em português
- [ ] pendência 5.1 resolvida ou documentada como limitação
- [ ] README em português, com reprodução passo a passo
- [ ] tag `results-final` e release `v1.0`
- [ ] artigo em PDF no padrão IEEE em `paper/`

# Revisão do trabalho feito sob a v1, pelos planos v2

**Data:** 18/11/2026. **Branch:** `science/plan-v2-review`. Decisões desta revisão: `DECISIONS_REVISAO.md`.

## 0. Pausa segura (§0 do prompt)

- **Processos:** só a reavaliação amostrada (`scripts/evaluate_mo_stochastic.py`, 8 workers) estava rodando. Nenhum treino. Ela foi mantida até o fim (R1).
- **Mudanças não commitadas:** os planos v2 e `docs/PLANO_DESENVOLVIMENTO_v1.md`, commitados como primeiro commit da branch (`810f568`).
- **Manifestos:** `sha256sum -c ../checkpoints/taquari_2026-10-28.sha256 --quiet` e `... poa_2026-10-28.sha256 --quiet` → **passam** (início da revisão; conferidos de novo no fim, §5).
- **Disco:** `df -h .` → 705 GB livres de 916 GB (19% usado).

## 1. Branches, tags e PRs

- **Branches locais e remotas:** `main` (= tag `v0-pre-refactor`), `master`, `poa-tests`, `amor-mo` (históricas), `fix/mo-stochastic-eval`, `chore/conventions`, `science/plan-v2-review` (esta) e `refactor/integration` (criada nesta revisão, aponta para `fe4d12c`).
- **Tags:** `checkpoint-taquari-2026-10-28`, `checkpoint-poa-2026-10-28`, `v0-pre-refactor`.
- **PRs antes da revisão:** nenhum. **Issues:** nenhuma da v1; criadas #1–#10 (P1–P10).
- **A `main` não recebeu nenhuma refatoração:** a ponta dela é a própria tag `v0-pre-refactor`.

## 2. Inventário dos commits desde `v0-pre-refactor`

### 2.1 Commits feitos sob a v1 (antes da revisão)

| Commit | O que fez | Item da v1 | Classificação v2 | Ação |
|---|---|---|---|---|
| `4a9a031` fix(mo) | `deterministic=False` opcional em `mo_eval`; scripts de diagnóstico e de reavaliação amostrada | 5.1 | **(a) mantém**, com a ressalva (c) abaixo | A reavaliação amostrada iniciada por ele cobria só a MO e os PPOs fixos; complementada por `results_mo_v1fix/`, que inclui a PPO da PoC no mesmo modo (A2) |
| `0cbe146` chore | `CONTRIBUTING.md`, ruff, template de PR, golden | §2, §4 | **(c) corrigir:** golden sem `use_deterministic_algorithms(True)` | Corrigido em `474405d` (fixture com 1 thread e algoritmos determinísticos); 2 execuções seguidas iguais |
| `ecb00c1`, `fe4d12c` refactor | Comentários e docstrings em português em 20 módulos | §6 (1)–(2) | **(b) reordenar:** refatoração antes de `results-final`. Não está na `main`, mas é ancestral desta branch | Mantidos (AST idêntica, conferida arquivo por arquivo; golden e teste de carga passam). `refactor/integration` criada. O PR avisa; o grupo decide no merge (R8) |
| `409df30` fix(mo) | Avaliação amostrada opcional em Porto Alegre, **só para a MO** | 5.1 | **(c) corrigir:** a PPO-S2 continuava em argmax (comparação injusta) | Corrigido em `474405d`: o modo vale para a MO e a PPO-S2; episódios gravados um a um |
| `35b2a65` fix(mo) | Gate de condicionamento amostrado; marcador `CONDITIONING_GATE_FAILED` | 5.1 | **(c) corrigir:** a v2 pede `INVALID_GATE`, os dois modos e recusa na avaliação | Corrigido em `4a7ba9a` |
| `03fd9b2` (na `main`, = tag) | Planos v1, geradores de PPTX | — | (a) mantém (histórico) | Planos substituídos pela v2 |

### 2.2 Commits desta revisão

| Commit | O que fez | Item v2 |
|---|---|---|
| `810f568` | Planos v2 commitados | — |
| `26a18f0` | `DECISIONS_MO_V2.md`: protocolo A1–A6, critérios v2, teste D, condição do retreino. **Antes de qualquer avaliação v2** | §2.1, §2.3 |
| `4a7ba9a` | `INVALID_GATE`: gates em todo treino, nos dois modos; `load_mo_model` recusa execuções marcadas; testes | P1 (§2.1, item 3) |
| `474405d` | Teste de carga dos 13 modelos (referência gerada com o código da tag v0); golden determinístico; `evaluate_v1fix.py`; H7 com a PPO-S2 no mesmo modo | §4.4, P1 |
| `f73ad4b` | perf: BLAS de 1 thread nos ambientes; todos os núcleos no update; sem forward extra | P2 |
| `291f5b5` | fix: autoreset `SAME_STEP` opcional (A7), `configs_mo_v2/`, teste | P2/P3 |
| `560af6a` | `REPORT_MO_ERRATA.md`; `PRESENTATION_MO.md`, decisão nº 12 e geradores corrigidos | P4, P8 |
| `971c263` | Bootstrap hierárquico e reconstrução de janelas (72/72 trajetórias conferidas) | P5 |
| `9f674c4`, `980599c` | Tabelas canônicas RQ1 | P6 |
| `44d6dd2` | `docs/conciliacao_numeros.md` | P6 |
| `690ae98` | `docs/diagnostico_mo.md` | P1 |
| `4e5cdf6` | `DECISIONS_FINAL.md`: critérios F1–F3 **antes** das análises | P7 |
| `4c7cd21` | Índice de mudança no Taquari; script do M1 | P7 |
| `bbe1fa8` | Benchmark de fps v0 × atual | P2 |
| `301536d` | `DECISIONS_REVISAO.md` | — |

### 2.3 Saídas inválidas pela v2 (d): mantidas como histórico

| Saída | Por que é (d) | Substituída por |
|---|---|---|
| `REPORT_MO.md` (vereditos H4–H7) | Vereditos fora do critério; gate reprovado; orçamentos diferentes | `REPORT_MO_ERRATA.md`; depois, `results_mo_v1fix/` (e `results_mo_v2/`, se houver retreino) |
| `results_mo/summary_test.csv`, `results_mo/hypervolume_eum.json` | Só argmax; hipervolume entre conjuntos de tamanhos diferentes, com referência a posteriori | `results_mo_v1fix/` (dois modos; EUM com referência a priori) |
| `results_mo/poa_summary.csv` | MO e PPO-S2 em argmax, sem episódios um a um | `results_mo_v1fix/poa_*_{argmax,sampled}.csv` |
| Vídeos-deck em `results_mo/videos/` e `results_mo/apresentacao_drone_active_loc.pptx` | Texto afirma superioridade e "~300%" | Geradores corrigidos; regenerar (P8, pendente). Até lá: **não usar** |
| `runs_mo/*` (modelos v1) | Gate reprovado (MO com memória), 225–348k passos, coleta com `NEXT_STEP` | Retreino v2 (P3), se a condição for atendida; senão, plano B |
| `results_mo/stochastic/` | Iniciado antes das regras v2: não inclui a PoC; o BLAS não estava limitado (só afeta a velocidade) | `results_mo_v1fix/` para comparações; mantido para conferência |

## 3. Pendência 5.1 da v1 confrontada com a §2.1 da v2

Diagnóstico completo, comandos e saídas: `docs/diagnostico_mo.md`. Confronto:

- **Por que o script não parou quando o gate falhou?** `train_moppo` gravava `gate_check_25pct.json` e seguia: nenhum código lia o resultado. **Corrigido:** gate em toda execução, nos dois modos; reprovação grava `INVALID_GATE` no log e na pasta, emite aviso, e `load_mo_model` recusa a execução (`4a7ba9a`, testes em `tests/test_mo_gates.py`).
- **"A avaliação com argmax apaga o condicionamento":** a frase aparece na mensagem do commit `4a9a031` e no `docs/PLANO_DESENVOLVIMENTO_v1.md`, mas nos dois casos junto da falha do gate. O que faltava era a leitura de que o condicionamento é **fraco no treino**: as diferenças entre presets são de 3–5% de massa de probabilidade, e os "33%" são o máximo entre pesos extremos. Essa leitura está agora em `docs/diagnostico_mo.md`, no `DECISIONS_MO_V2.md` (A1) e no `REPORT_MO_ERRATA.md`.
- **Troca post hoc do modo de avaliação:** registrada em `DECISIONS_MO_V2.md` (A1–A3), com data, e aplicada a todas as políticas aprendidas (MO, PPOs fixos e PPO da PoC s0–s2) nos mesmos 300 episódios, com `torch.manual_seed(2024 + i)` (`scripts/evaluate_v1fix.py`). Os dois modos são reportados (§5.2).
- **Achados novos durante a revisão:**
  - o OpenBLAS abria 8 threads em cada processo de ambiente (fork depois do import do numpy), o que derruba o fps;
  - o autoreset `NEXT_STEP` do gymnasium 1.4 colocava ≈ 1 transição espúria por episódio no buffer do MOPPO (A7);
  - o fps real do MOPPO v1 foi 88, não 43 (`docs/conciliacao_numeros.md`, item 5).

## 4. Textos com afirmações a corrigir

Saída do grep do §1.4 do prompt, excluindo os planos e os falsos positivos (código: "dominam o ponto de referência", "IC superior", etc.):

| Arquivo:linha | Texto | Forma correta (registro de afirmações) | Feito |
|---|---|---|---|
| `REPORT_MO.md`:9, 11 | "pré-registrados" | "pré-especificados" | Histórico; errata publicada |
| `REPORT_MO.md`:19 | "A política multiobjetivo única superou..." (H4) | "H4 não avaliável" | Errata §3 |
| `REPORT_MO.md`:48 | "ganho de quase 3x no hipervolume [...] papel crucial" | "inconclusivo: orçamento, 1 seed, conjuntos de tamanhos diferentes" | Errata §3 |
| `REPORT_MO.md`:24 | "disparidade radiométrica insular" (causal) | "a queda é da política MO: a PPO-S2 faz 71,3% nos mesmos episódios" | Errata §3 |
| `PRESENTATION_MO.md`:78–79 | "3x mais demorado!", "mais que o dobro de energia" | números do episódio, marcado como individual | ✅ |
| `PRESENTATION_MO.md`:88 | "frequentemente falham [...] enquanto a MOPPO resolve" | episódio individual; taxas agregadas | ✅ |
| `PRESENTATION_MO.md`:112–117 | "Por que o MOPPO domina", "validando a H4", "~300%" | "O que a fronteira v1 mostra e o que não mostra" | ✅ |
| `PRESENTATION_MO.md`:135 | "menor taxa de falsa localização de todos" + causa na inundação | "H7 não suportada; IC inclui 0; a queda é da MO" | ✅ |
| `DECISIONS_MO.md`:18 (nº 12) | "superioridade do MOPPO" | linha `12-errata` | ✅ |
| `DECISIONS_MO.md`:22 | "Pré-registrados" | nota "leia pré-especificados" | ✅ |
| `src/dal/mo_report.py`:494, 496, 504, 533 | "pré-registrados"; veredito H4 e "quase 3x" **em texto fixo com números da v1** | "pré-especificados"; texto neutro que aponta para a errata | ✅ |
| `src/dal/mo_pitch_pptx.py`:282, 294, 540, 589 | "Supera [...] e domina", "quase 300%", "triplicando", "Comprovadas" | vereditos da errata | ✅ (o `.pptx` já gerado não foi regenerado) |
| `src/dal/mo_poa_presentation_video.py`:145, 229 | "~300%", "MULTIOBJETIVO SUPERA" | idem | ✅ (vídeo não regenerado) |
| `src/dal/mo_taquari_presentation_video.py`:139 | "~300%" | idem | ✅ (vídeo não regenerado) |
| `src/dal/mo_report.py`:415 | `h4_inconclusive` nunca usado: H4 só podia sair suportada/não suportada | o critério prevê "inconclusiva" | Registrado; o gerador v1 é histórico, e os vereditos v2 vêm de código novo |

## 5. Entrega

*Preenchido no fim da execução das filas: testes, tabela da MO nos dois modos, fps, estado de P1–P10 e como acompanhar o que estiver rodando.*

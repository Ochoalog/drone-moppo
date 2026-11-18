# Como contribuir

Guia rápido para o grupo. O plano completo está em [`docs/PLANO_DESENVOLVIMENTO.md`](docs/PLANO_DESENVOLVIMENTO.md).

## Idioma

| O quê | Idioma |
|---|---|
| Módulos, classes, funções, variáveis, constantes, chaves de config, colunas novas de CSV | **inglês** |
| Comentários e docstrings | **português** |
| Mensagens de commit | inglês, no formato [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`) |
| README, relatórios, artigo | português (o artigo segue o que for combinado com o professor) |

Rótulos de dados já gravados nos resultados (`"ilhas"`, `"centro"`) continuam como estão: são valores, não identificadores.

## Glossário de nomes

| Conceito | Nome no código |
|---|---|
| crença / histograma | `belief` |
| localização falsa | `false_localization` |
| sucesso | `success` |
| passo | `step` |
| vetor de pesos (preferência) | `preference_weights` (`w` só dentro de fórmulas) |
| nuvem | `cloud` |
| ilhas / centro | `islands` / `downtown` |
| linha reta / zigue-zague | `straight` / `lawnmower` |

## Clean code

- Uma função faz uma coisa, com no máximo ~40 linhas e ~5 parâmetros. Se precisar de mais, use uma `dataclass`.
- Sem números mágicos: tudo vai para o YAML ou para uma constante nomeada.
- Nomes que revelam intenção (`belief_entropy`, não `h`).
- Type hints nas funções públicas.
- Sem código morto ou comentado.
- O comentário explica o **porquê**, não repete o código.

```python
def compute_false_localization_rate(episodes: pd.DataFrame) -> float:
    """Calcula a fração de episódios em que a crença convergiu para o lugar errado."""
    # Episódios truncados por limite de passos não contam como localização falsa.
    finished = episodes[episodes["terminated"]]
    return float(finished["false_localization"].mean())
```

## Antes de abrir um PR

```bash
unset PYTHONPATH                 # o shell de algumas máquinas exporta caminhos do ROS
.venv/bin/ruff check src scripts tests
.venv/bin/pytest -q              # inclui o teste de regressão "golden"
```

- **Refatoração não muda números.** Se o teste golden falhar, o PR mudou comportamento: corrija ou justifique no PR.
- Um PR por assunto, pequeno, com pelo menos um revisor do grupo.
- Branches: `refactor/...`, `fix/...`, `feat/...`, `docs/...`, `chore/...`.
- Nunca commitar `data*/`, `runs*/`, vídeos ou credenciais.

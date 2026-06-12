# Guia de Contribuição

## Fluxo de Branches

```
main        → branch protegida, só merge via PR aprovado
develop     → integração contínua do time
feat/<nome> → novas funcionalidades (branch a partir de develop)
fix/<nome>  → correções (branch a partir de develop)
```

Sempre crie sua branch a partir de `develop`:

```bash
git checkout develop
git pull origin develop
git checkout -b feat/minha-feature
```

## Convenção de Commits

Seguimos o padrão [Conventional Commits](https://www.conventionalcommits.org/):

| Prefixo    | Quando usar                                  |
|------------|----------------------------------------------|
| `feat:`     | nova funcionalidade                          |
| `fix:`      | correção de bug                              |
| `chore:`    | tarefas de manutenção (deps, configs)        |
| `docs:`     | documentação                                 |
| `test:`     | testes                                       |
| `refactor:` | refatoração sem mudança de comportamento     |
| `ci:`       | mudanças no pipeline CI/CD                   |

Exemplos:

```
feat: add MLP model with early stopping
fix: correct precision@k calculation for empty predictions
chore: update torch to 2.3.0
```

## Processo de Pull Request

1. Abrir PR de `feat/<nome>` → `develop`
2. CI deve estar verde (ruff + pytest)
3. Ao menos **1 aprovação** de outro membro do time
4. Squash merge com mensagem semântica

## Setup Local

Para novos devs, um único comando configura tudo:

```bash
git clone <repo-url>
cd tech-challenge-02
make setup
```

Isso executa `uv sync`, instala os pre-commit hooks e cria o `.env` a partir do `.env.example`.

## Padrões de Código

- Funções com **no máximo 20 linhas**
- **Type hints** em todas as funções públicas
- **Sem `print()`** — use `logging` estruturado
- **Docstrings Google style** em todas as funções públicas
- Zero erros no `ruff check .` antes de abrir PR

Exemplo de função no padrão:

```python
def compute_precision_at_k(recommended: list[int], relevant: list[int], k: int) -> float:
    """Calcula Precision@K para uma lista de recomendações.

    Args:
        recommended: IDs dos itens recomendados, ordenados por score.
        relevant: IDs dos itens relevantes (ground truth).
        k: Número de itens a considerar no topo da lista.

    Returns:
        Proporção de itens relevantes entre os top-K recomendados.
    """
    top_k = set(recommended[:k])
    hits = len(top_k & set(relevant))
    return hits / k if k > 0 else 0.0
```

# Como Contribuir

Bem-vindo ao projeto **Tech Challenge  02** — um sistema de recomendação de produtos baseado em redes neurais, desenvolvido na Fase 02 da pós-graduação em Machine Learning (FIAP).

Este documento descreve as convenções e o fluxo de trabalho adotados pelo time para manter o repositório organizado e o histórico legível.

---

## Fluxo de Branches

```
main        → branch protegida; aceita apenas merges via PR aprovado
develop     → branch de integração; representa o estado mais recente do time
feat/<nome> → nova funcionalidade (criada a partir de develop)
fix/<nome>  → correção de bug (criada a partir de develop)
```

Sempre crie sua branch a partir de `develop`:

```bash
git checkout develop
git pull origin develop
git checkout -b feat/minha-feature
```

---

## Convenção de Commits

Seguimos o padrão [Conventional Commits](https://www.conventionalcommits.org/). Mensagens claras facilitam revisão, geração de changelog e rastreabilidade.

| Prefixo      | Quando usar                                   |
|--------------|-----------------------------------------------|
| `feat:`      | nova funcionalidade                           |
| `fix:`       | correção de bug                               |
| `chore:`     | manutenção (dependências, configs, arquivos)  |
| `docs:`      | documentação                                  |
| `test:`      | adição ou correção de testes                  |
| `refactor:`  | refatoração sem mudança de comportamento      |
| `ci:`        | mudanças no pipeline de CI/CD                 |

Exemplos:

```
feat: add MLP model with configurable hidden layers
fix: correct Recall@K when ground truth is empty
chore: pin torch to 2.3.0 in uv.lock
test: add smoke test for forward pass shape
```

---

## Processo de Pull Request

1. Abrir PR de `feat/<nome>` ou `fix/<nome>` → `develop`
2. CI deve estar **verde** (ruff + pytest)
3. Ao menos **1 aprovação** de outro membro do time
4. **Squash merge** com mensagem semântica no título do PR

---

## Setup Local

Para novos membros, um único comando configura o ambiente completo:

```bash
git clone <repo-url>
cd tech-challenge-02
make setup
```

O target `setup` executa `uv sync`, instala os pre-commit hooks e cria o `.env` a partir do `.env.example`. Edite o `.env` com os valores do seu ambiente antes de rodar qualquer pipeline.

---

## Padrões de Código

| Regra | Detalhe |
|-------|---------|
| Tamanho de função | No máximo **20 linhas** |
| Tipagem | **Type hints** em todas as funções públicas |
| Logging | **Sem `print()`** — use o módulo `logging` |
| Documentação | **Docstrings Google style** em todas as funções públicas |
| Linting | `ruff check .` deve retornar **zero erros** antes de abrir PR |

Exemplo de função no padrão exigido:

```python
def compute_precision_at_k(recommended: list[int], relevant: list[int], k: int) -> float:
    """Calcula Precision@K para uma lista de recomendações.

    Args:
        recommended: IDs dos itens recomendados, ordenados por score decrescente.
        relevant: IDs dos itens relevantes (ground truth do usuário).
        k: Número de posições do topo da lista a considerar.

    Returns:
        Proporção de itens relevantes entre os top-K recomendados.
    """
    top_k = set(recommended[:k])
    hits = len(top_k & set(relevant))
    return hits / k if k > 0 else 0.0
```

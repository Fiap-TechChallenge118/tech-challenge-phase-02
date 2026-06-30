# Tech Challenge — Fase 02

![CI](https://github.com/<seu-usuario>/tech-challenge-phase-02/actions/workflows/ci.yml/badge.svg)

> Sistema de Recomendação de Produtos com Rede Neural (MLP/Embedding-based) + Pipeline MLOps End-to-End.

Pipeline completo de Machine Learning aplicado a um problema de e-commerce: recomendar produtos com base no histórico de compras dos usuários, utilizando o dataset **Instacart Market Basket**. O projeto cobre desde a ingestão e versionamento de dados até o registro e promoção do modelo em produção.

Consulte o [CONTRIBUTING.md](./CONTRIBUTING.md) para o fluxo de branches, convenção de commits e processo de PR.

---

## Stack Tecnológica

| Camada | Tecnologia |
|---|---|
| Modelagem | PyTorch (MLP / Embedding-based) |
| Baselines & Pré-processamento | Scikit-Learn |
| Tracking de Experimentos | MLflow |
| Versionamento de Dados & Pipeline | DVC |
| Containerização | Docker (multi-stage) + Docker Compose |
| Gerenciamento de Dependências | uv (`pyproject.toml` + lock file) |
| Configurações | Pydantic Settings + `.env` |
| Qualidade de Código | Ruff + pre-commit hooks |
| Testes | pytest |

---

## Estrutura de Pastas

```
tech-challenge-02/
│
├── configs/                    # Arquivos de configuração do projeto
│   └── config.yaml             # Hiperparâmetros, paths e settings do pipeline
│
├── data/                       # Dados (arquivos grandes ignorados pelo git, versionados via DVC)
│   ├── raw/                    # Dados brutos originais, nunca modificados
│   └── processed/              # Dados transformados e prontos para treino
│
├── docs/                       # Documentação complementar
│   ├── model_card.md           # Model Card: performance, limitações e vieses do modelo
│   └── ml_canvas.md            # ML Canvas: problema, métricas de negócio e SLOs
│
├── infra/                      # Infraestrutura como código para deploy em nuvem
│   ├── aws/                    # Recursos AWS (ECS, ECR, S3, etc.) via Terraform
│   └── azure/                  # Recursos Azure (ACI, ACR, Blob Storage, etc.) via Terraform
│
├── metrics/                    # Métricas geradas pelo pipeline (rastreadas pelo DVC)
│   └── *.json                  # Precision@K, Recall@K, NDCG, MAP e outras métricas
│
├── models/                     # Artefatos de modelos treinados (versionados via DVC/MLflow)
│   └── *.pt / *.pkl            # Checkpoints PyTorch e modelos Scikit-Learn serializados
│
├── notebooks/                  # Notebooks exploratórios (não fazem parte do pipeline)
│   ├── data/                   # Imagens e outputs gerados pelos notebooks
│   └── 01_eda.ipynb            # Análise exploratória do dataset Instacart
│
├── scripts/                    # Scripts utilitários e de operação
│   ├── validate_env.py         # Valida se o ambiente está configurado corretamente
│   └── promote_model.py        # Promove modelo no MLflow Registry (Staging → Production)
│
├── src/                        # Código-fonte principal do projeto (pacote Python)
│   ├── settings.py             # Configurações centralizadas via Pydantic Settings
│   ├── data/                   # Ingestão, leitura e validação de dados brutos
│   ├── features/               # Engenharia de features — padrão Strategy para preprocessors
│   ├── models/                 # Arquiteturas PyTorch — padrão Factory para instanciar modelos
│   └── training/               # Loop de treino, early stopping, avaliação e logging MLflow
│
├── tests/                      # Testes automatizados
│   ├── smoke/                  # Smoke tests: importações, instâncias, forward pass
│   ├── unit/                   # Testes unitários de funções e classes isoladas
│   └── integration/            # Testes de integração do pipeline completo
│
├── .github/
│   └── workflows/
│       └── ci.yml              # CI/CD: lint (ruff) + testes (pytest) em cada push/PR
│
├── .env.example                # Template de variáveis de ambiente (nunca commitar o .env real)
├── .gitignore
├── .pre-commit-config.yaml     # Hooks: ruff --fix, ruff-format, conventional-pre-commit
├── .python-version             # Python 3.12 fixado para o projeto
├── docker-compose.yml          # Orquestração: serviço de treino + MLflow server
├── Dockerfile                  # Imagem multi-stage (builder → runtime)
├── dvc.yaml                    # Definição do pipeline DVC (stages reprodutíveis)
├── Makefile                    # Atalhos para comandos comuns do projeto
├── pyproject.toml              # Configuração do projeto, dependências e ferramentas
├── uv.lock                     # Lock file de dependências (uv)
└── README.md                   # Este arquivo
```

---

## Responsabilidades dos Módulos Principais

### `src/` — Código de Produção

- **`src/data/`** — leitura dos CSVs brutos, validação de schema (Pandera) e split treino/validação/teste.
- **`src/features/`** — construção de embeddings de usuários e itens, normalização e codificação categórica. Implementa o padrão **Strategy** (`PreprocessorStrategy` ABC) para intercambiar preprocessors sem alterar o pipeline.
- **`src/models/`** — arquiteturas PyTorch (MLP e Embedding-based) + baselines (popularidade, Sklearn). Implementa o padrão **Factory** (`ModelFactory`) para instanciar modelos a partir de `configs/config.yaml`.
- **`src/training/`** — loop de treino com early stopping, logging por epoch via MLflow, cálculo de métricas@K e serialização de checkpoints.

### `tests/` — Testes Automatizados

- **`smoke/`** — instancia cada componente e verifica forward pass com tensores aleatórios.
- **`unit/`** — testa funções de pré-processamento, cálculo de métricas e componentes isolados.
- **`integration/`** — executa o pipeline de ponta a ponta em um subset reduzido dos dados.

---

## Decisões Técnicas

### Gerenciamento de Dependências — `uv` (não Poetry)

**uv** foi escolhido por três razões objetivas:

- Resolução de dependências ~100x mais rápida (escrito em Rust)
- `uv sync` instala o ambiente em segundos — relevante para builds Docker e CI
- `uv.lock` é determinístico e commitado no repositório, garantindo reprodutibilidade total

O `pyproject.toml` segue PEP 517/518 com produção em `[project].dependencies` e dev em `[dependency-groups].dev`. O DVC já é instalado por `uv sync` — não é necessário instalá-lo separadamente.

### Versão do Python — 3.12 (fixada)

Fixado em `.python-version`. Razão: suporte completo a type hints modernos (PEP 695), melhoria de ~5% de performance sobre 3.11, e compatibilidade garantida com PyTorch 2.x, MLflow 2.x e DVC 3.x.

### Linter e Formatter — `ruff`

`ruff` substitui `flake8`, `isort` e `black` simultaneamente. Configuração em `ruff.toml`:

- `line-length = 88` (padrão black)
- `target-version = "py312"`
- Rules ativas: `E` (pycodestyle errors), `F` (pyflakes), `W` (warnings), `I` (isort), `N` (pep8-naming)

Zero tolerância a erros: `ruff check .` deve retornar limpo antes de qualquer PR.

### Pre-commit Hooks

| Hook | Versão | Ação |
|---|---|---|
| `ruff` | v0.15.17 | Lint com `--fix` automático |
| `ruff-format` | v0.15.17 | Formatação do código |
| `conventional-pre-commit` | v4.4.0 | Valida mensagem de commit (stage `commit-msg`) |

### Fluxo de Branches e Commits

```
main        → protegida; aceita apenas merges via PR aprovado
develop     → integração contínua do time
feat/<nome> → nova funcionalidade (criada a partir de develop)
fix/<nome>  → correção de bug (criada a partir de develop)
```

Merge strategy: **squash merge** com mensagem semântica no título do PR.

Prefixos de commit obrigatórios: `feat:` · `fix:` · `chore:` · `docs:` · `test:` · `refactor:` · `ci:`

---

## Decisões de ML

### Dataset — Instacart Market Basket Analysis

**Instacart Market Basket Analysis** — dataset público com mais de 3 milhões de pedidos de ~200.000 usuários.

Arquivos principais:
- `orders.csv` — metadados dos pedidos (usuário, dia da semana, hora, dias desde o último pedido)
- `order_products__prior.csv` — itens de pedidos anteriores (~32M interações)
- `order_products__train.csv` — itens do pedido mais recente de cada usuário (ground truth)
- `products.csv`, `departments.csv`, `aisles.csv` — catálogo de produtos

Escolhido por:
- Volume adequado: ~32M interações user-item (muito acima do mínimo exigido)
- Estrutura real de e-commerce com histórico temporal de pedidos
- Benchmark público amplamente utilizado em papers de recomendação

Os arquivos brutos (`data/raw/`) estão no `.gitignore` e são gerenciados exclusivamente pelo DVC.

### Métricas de Avaliação

O modelo é avaliado com métricas orientadas a ranking:

- **Precision@K** — fração de itens recomendados que são relevantes
- **Recall@K** — fração de itens relevantes que foram recomendados
- **NDCG@K** — Normalized Discounted Cumulative Gain (considera a posição do ranking)
- **MAP@K** — Mean Average Precision

### Resultados (após `dvc repro`)

> Os valores abaixo são populados automaticamente ao final do pipeline de avaliação.

| Modelo              | Precision@10 | Recall@10 | NDCG@10 | MAP@10 |
|---------------------|-------------|-----------|---------|--------|
| MLP (proposto)      | —           | —         | —       | —      |
| Popularity Baseline | —           | —         | —       | —      |
| Sklearn Baseline    | —           | —         | —       | —      |

### Padrões de Código (não negociáveis)

| Regra | Detalhe |
|---|---|
| Tamanho de função | ≤ 20 linhas |
| Tipagem | Type hints em todas as funções públicas |
| Logging | Proibido `print()` — usar módulo `logging` |
| Documentação | Docstrings Google style em todas as funções públicas |
| Linting | `ruff check .` com zero erros antes de abrir PR |

---

## Pré-requisitos

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) instalado:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- Docker e Docker Compose

> DVC, pytest, ruff e demais dependências são instalados automaticamente via `make setup`.

---

## Início Rápido

### 1. Clonar e configurar o ambiente

```bash
git clone <url-do-repositorio>
cd tech-challenge-phase-02

# Instala dependências, configura pre-commit e cria o .env a partir do .env.example
make setup
```

Edite o `.env` gerado com seus valores locais antes de continuar.

### 2. Baixar os dados versionados pelo DVC

```bash
dvc pull
```

> Se o remote DVC não estiver configurado, copie os arquivos brutos do Instacart para `data/raw/` manualmente.

### 3. Validar o ambiente

```bash
make validate
```

### 4. Reproduzir o pipeline completo

```bash
dvc repro
```

Executa os stages em ordem:
1. `preprocess` — limpeza e divisão dos dados brutos
2. `feature_eng` — construção das features de usuário e item
3. `train` — treino do modelo PyTorch com early stopping
4. `evaluate` — cálculo de métricas e comparação com baselines

### 5. Executar os testes

```bash
make test
```

### 6. Visualizar experimentos no MLflow

```bash
docker compose up mlflow
# Acesse http://localhost:5000
```

### 7. Executar treino completo com Docker Compose

```bash
docker compose up --build
```

---

## Pipeline DVC

```
preprocess → feature_eng → train → evaluate
```

| Stage | Entrada | Saída |
|---|---|---|
| `preprocess` | `data/raw/*.csv` | `data/processed/interactions.parquet` |
| `feature_eng` | `data/processed/interactions.parquet` | `data/processed/features.parquet` |
| `train` | `data/processed/features.parquet` | `models/model.pt`, `models/encoders.pkl` |
| `evaluate` | `models/model.pt`, dados de teste | `metrics/evaluation.json` |

---

## Qualidade de Código

```bash
# Verificar linting
make lint

# Corrigir automaticamente
make lint-fix

# Formatar código
make format

# Executar testes
make test

# Rodar todos os hooks manualmente
pre-commit run --all-files
```

---

## Licença

MIT

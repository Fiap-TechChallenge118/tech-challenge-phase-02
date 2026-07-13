# Tech Challenge — Fase 02

[![CI](https://github.com/Fiap-TechChallenge118/tech-challenge-phase-02/actions/workflows/ci.yml/badge.svg)](https://github.com/Fiap-TechChallenge118/tech-challenge-phase-02/actions/workflows/ci.yml)
[![API na AWS](https://img.shields.io/badge/API-18.233.199.59%3A8000-FF9900?logo=amazonaws&logoColor=white)](http://18.233.199.59:8000/docs)
[![Swagger](https://img.shields.io/badge/Swagger-%2Fdocs-85EA2D?logo=swagger&logoColor=black)](http://18.233.199.59:8000/docs)
[![Deploy](https://img.shields.io/badge/deploy-EC2%20t3.micro%20%2B%20ECR-232F3E?logo=terraform&logoColor=white)](infra/aws)

> Sistema de Recomendação de Produtos com Rede Neural (MLP/Embedding-based) + Pipeline MLOps End-to-End.

**API de recomendação publicada:** http://18.233.199.59:8000/docs (Swagger UI)

```bash
curl "http://18.233.199.59:8000/recommend/1?k=5"
```

> O deploy roda em EC2 provisionada por Terraform. Ao destruir a infraestrutura
> (`terraform -chdir=infra/aws destroy`), o endereço deixa de responder — recriar leva
> ~3 minutos (ver [Deploy em Nuvem](#deploy-em-nuvem-aws)).

Pipeline completo de Machine Learning aplicado a um problema de e-commerce: recomendar produtos com base no histórico de compras dos usuários, utilizando o dataset **Instacart Market Basket**. O projeto cobre desde a ingestão e versionamento de dados até o registro e promoção do modelo em produção.

Consulte o [CONTRIBUTING.md](./CONTRIBUTING.md) para o fluxo de branches, convenção de commits e processo de PR.

---

## Stack Tecnológica

| Camada | Tecnologia |
|---|---|
| Modelagem | PyTorch (MLP / Embedding-based) |
| Baselines & Pré-processamento | Scikit-Learn |
| Tracking de Experimentos | MLflow (tracking + Model Registry) |
| Versionamento de Dados & Pipeline | DVC (remote S3) |
| Containerização | Docker (multi-stage) + Docker Compose |
| API de Inferência | FastAPI + Uvicorn (Swagger/OpenAPI) |
| Deploy em Nuvem | AWS ECR + EC2, provisionados via Terraform |
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
│   └── model_card.md           # Model Card: performance, limitações e vieses do modelo
│
├── infra/                      # Infraestrutura como código para deploy em nuvem
│   └── aws/                    # Terraform: ECR + EC2 + Elastic IP + IAM que servem a API
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
│   ├── download_dataset.py     # Baixa o dataset Instacart do Kaggle para data/raw/
│   └── push_image.sh           # Build da imagem da API e push para o ECR
│
├── src/                        # Código-fonte principal do projeto (pacote Python)
│   ├── settings.py             # Configurações centralizadas via Pydantic Settings
│   ├── api/                    # API de inferência (FastAPI): /health e /recommend
│   ├── data/                   # Ingestão, leitura e validação de dados brutos
│   ├── features/               # Engenharia de features — padrão Strategy para preprocessors
│   ├── models/                 # Arquiteturas PyTorch — padrão Factory para instanciar modelos
│   └── training/               # Loop de treino, early stopping, avaliação e logging MLflow
│
├── tests/
│   └── unit/                   # Testes unitários de funções e classes isoladas
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
├── Dockerfile                  # Multi-stage: builder → runtime (treino) → api (serving)
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

Avaliação em 3.938 usuários de teste, top-K = 10 (valores gerados por
`metrics/evaluation.json`, stage `evaluate`):

| Modelo               | Precision@10 | Recall@10  | NDCG@10    | MAP@10     |
|----------------------|--------------|------------|------------|------------|
| MLP (proposto)       | 0,00767      | 0,03862    | 0,02284    | 0,01333    |
| Popularity Baseline  | **0,00841**  | **0,04248**| **0,02440**| **0,01389**|
| Sklearn Baseline     | 0,00061      | 0,00229    | 0,00108    | 0,00039    |

A MLP supera o baseline Scikit-Learn (LogReg) por mais de 12× em todas as métricas,
mas **fica ~5–9% abaixo do baseline de popularidade**. Três experimentos rastreados no
MLflow (embedding 64 vs 128, `pos_weight` 1 vs 3) confirmam o resultado: o gargalo não
é a capacidade do modelo, e sim o descompasso entre o objetivo de treino (classificação
binária com 1 negativo por positivo) e a tarefa de avaliação (ranquear ~50 mil itens).
Ver [Model Card](docs/model_card.md) para a análise completa e os caminhos de melhoria.

### Padrões de Código (não negociáveis)

| Regra | Detalhe |
|---|---|
| Tamanho de função | ≤ 20 linhas |
| Tipagem | Type hints em todas as funções públicas |
| Logging | Proibido `print()` — usar módulo `logging` |
| Documentação | Docstrings Google style **em português** em todas as funções públicas |
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
make dvc-pull
```

> Se o remote DVC não estiver configurado, copie os arquivos brutos do Instacart para `data/raw/` manualmente.

### 3. Validar o ambiente

```bash
make validate
```

O script `scripts/validate_env.py` executa 5 verificações em sequência e encerra com código de saída específico para cada tipo de falha:

| Etapa | O que verifica | Código de saída em caso de falha |
|-------|---------------|----------------------------------|
| 1 — Versão do Python | Python ≥ 3.12 | `1` |
| 2 — Variáveis de ambiente | Todas as vars obrigatórias presentes | `2` |
| 3 — Imports críticos | torch, sklearn, mlflow, pandas, numpy, pydantic, pandera, yaml, joblib | `3` |
| 4 — Settings Pydantic | Todos os campos do `.env` são válidos | `4` |
| 5 — Diretórios | `data/`, `configs/`, `models/`, `metrics/` existem | aviso (não erro) |

**Saída esperada em ambiente configurado:**

```
INFO  ============================================================
INFO  Validação do Ambiente — Tech Challenge 02
INFO  ============================================================

[1/5] Versão do Python
INFO    ✓ Python 3.12 (≥ 3.12)

[2/5] Variáveis de ambiente obrigatórias
INFO    ✓ Arquivo .env encontrado
INFO    ✓ PROJECT_NAME definida
INFO    ✓ MLFLOW_TRACKING_URI definida
...

[3/5] Imports críticos
INFO    ✓ PyTorch importado com sucesso (v2.x)
INFO    ✓ MLflow importado com sucesso (v2.x)
...

[4/5] Validação do Settings (Pydantic)
INFO    ✓ Settings válido — projeto 'tech-challenge-02' | env 'development' | seed 42

[5/5] Diretórios do projeto
INFO    ✓ configs/ existe
WARNING ⚠ data/raw não encontrado (rode dvc pull)
...

INFO  ✓ Ambiente validado com sucesso! (N avisos, 0 erros)
```

Se o script encerrar com erro, a mensagem indica exatamente o que falta corrigir antes de continuar.

> Avisos em `[5/5]` sobre `data/` são esperados antes do `dvc pull` e não bloqueiam o ambiente.

### 4. Reproduzir o pipeline completo

```bash
make repro
```

Executa os stages em ordem:
1. `preprocess` — limpeza e divisão dos dados brutos
2. `train` — treino do modelo PyTorch com early stopping
3. `evaluate` — cálculo de métricas e comparação com baselines
4. `register` — registro e promoção do modelo no MLflow Registry

> Para executar um stage isolado: `make preprocess`, `make train`, `make evaluate` ou `make register`.

### 5. Executar os testes

```bash
make test
```

### 6. Visualizar experimentos no MLflow

```bash
make mlflow-ui
# Acesse http://localhost:5000
```

### 7. Executar treino completo com Docker Compose

```bash
docker compose up --build
```

---

## Pipeline DVC

```
preprocess → feature_eng → train → evaluate → register
```

| Stage | Entrada | Saída |
|---|---|---|
| `preprocess` | `data/raw/*.csv` | `data/processed/interactions.parquet`, `mappings.json` |
| `feature_eng` | `interactions.parquet`, `mappings.json` | `train_pairs.parquet`, `val_pairs.parquet`, `test_pairs.parquet` |
| `train` | `train_pairs.parquet`, `val_pairs.parquet` | `models/model.pt` |
| `evaluate` | `models/model.pt`, `test_pairs.parquet` | `metrics/evaluation.json` |
| `register` | `models/model.pt`, `metrics/evaluation.json` | modelo promovido a Production no MLflow Registry |

Os dados e artefatos são versionados no remote S3 configurado em `.dvc/config`.
Após clonar, `dvc pull` recupera tudo do cache e `dvc repro` reproduz o pipeline sem
retreinar (os hashes do `dvc.lock` batem com os artefatos publicados).

---

## Deploy em Nuvem (AWS)

A API de inferência (`src/api/`) roda em container na AWS, servindo o modelo promovido
a Production. A infraestrutura é declarada em `infra/aws/` (Terraform).

**URL pública:** http://18.233.199.59:8000 &nbsp;·&nbsp; **Swagger:** http://18.233.199.59:8000/docs

### Endpoints

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/health` | Status do serviço e do modelo carregado |
| `GET` | `/recommend/{user_id}?k=10` | Top-K produtos para o usuário (1 ≤ k ≤ 50) |
| `GET` | `/docs` | Documentação interativa (Swagger UI) |
| `GET` | `/openapi.json` | Schema OpenAPI 3.1 |

```bash
curl http://18.233.199.59:8000/health
# {"status":"ok","model_loaded":true,"n_users":206209,"n_items":49685}

curl "http://18.233.199.59:8000/recommend/1?k=5"
# {"user_id":1,"count":5,"recommendations":[
#   {"product_id":24852,"score":0.9986},   # Banana
#   {"product_id":47626,"score":0.9978},   # Large Lemon
#   ...
# ]}
```

> O endereço é **HTTP** (sem TLS): HTTPS exigiria um load balancer ou domínio próprio,
> fora do escopo do Free Tier. O IP é um Elastic IP, portanto estável entre reboots.

### Rodar a API localmente

```bash
# Opção 1 — via uv (hot reload para desenvolvimento)
uv run --group api uvicorn src.api.main:app --reload

# Opção 2 — mesma imagem que roda na AWS
docker build --target api -t tc02-api .
docker run --rm -p 8000:8000 tc02-api
```

Ambas expõem o Swagger em http://localhost:8000/docs. É necessário ter
`models/model.pt` no diretório (recuperável com `dvc pull`).

Usuários fora do vocabulário do modelo retornam `404` — o modelo é colaborativo e não
trata cold start (ver [Model Card](docs/model_card.md), seção 4).

### Arquitetura

`ECR` (imagem) → `EC2 t3.micro` (Docker) → `Elastic IP` (endereço público estável).

**Por que EC2 e não App Runner/Fargate:** a conta usada está no plano gratuito da AWS,
que não habilita esses serviços (`SubscriptionRequiredException`). A `t3.micro` é
elegível ao Free Tier e roda o mesmo container. Como ela tem apenas 1 GB de RAM e o
import do PyTorch é pesado, o `user_data` provisiona 2 GB de swap.

### Subir a infraestrutura

```bash
# 1. Criar o ECR
terraform -chdir=infra/aws apply

# 2. Build da imagem (stage `api`) e push
./scripts/push_image.sh

# 3. Subir a EC2 que serve a API
terraform -chdir=infra/aws apply -var deploy_service=true
terraform -chdir=infra/aws output api_url

# 4. Destruir tudo (evita consumir horas do Free Tier)
terraform -chdir=infra/aws destroy
```

O modelo é **embutido na imagem** (`COPY models/model.pt`), tornando o container
autossuficiente: ele não depende do MLflow nem do remote DVC em runtime.

---

## Qualidade de Código

```bash
make lint          # verifica erros sem corrigir
make lint-fix      # corrige automaticamente
make format        # formata o código
make test          # executa os testes unitários
make check         # lint + format check combinados (usado no CI)
```

Para rodar todos os hooks manualmente:

```bash
pre-commit run --all-files
```

Consulte a [Referência de Comandos](#referência-de-comandos-make) para a lista completa.

---

## Referência de Comandos (`make`)

Execute `make` ou `make help` para listar todos os comandos disponíveis com suas descrições.

### Ambiente

| Comando | Descrição |
|---|---|
| `make setup` | Configura o ambiente do zero — instala dependências, cria `.env`. Rodar uma vez ao clonar. |
| `make install` | Instala/atualiza dependências via `uv sync`. |
| `make validate` | Valida o ambiente: Python, variáveis de ambiente, imports e diretórios. |

### Qualidade de Código

| Comando | Descrição |
|---|---|
| `make lint` | Verifica erros de linting sem corrigir. |
| `make lint-fix` | Corrige erros de linting automaticamente. |
| `make format` | Formata o código. |
| `make check` | Lint + format check combinados (usado no CI). |

### Testes

| Comando | Descrição |
|---|---|
| `make test` | Executa todos os testes unitários. |
| `make test-cov` | Executa testes com relatório de cobertura. |
| `make test-integration` | Executa testes de integração (requer artefatos do DVC). |

### CI

| Comando | Descrição |
|---|---|
| `make ci` | Sequência completa do CI: `check` + `test` + `dvc-dag`. |

### Pipeline DVC

| Comando | Descrição |
|---|---|
| `make repro` | Reproduz o pipeline DVC completo (`preprocess → train → evaluate → register`). |
| `make preprocess` | Executa apenas o stage de pré-processamento. |
| `make train` | Executa apenas o stage de treino. |
| `make evaluate` | Executa apenas o stage de avaliação. |
| `make register` | Executa apenas o stage de registro no MLflow Model Registry. |

### DVC — Dados e Versionamento

Os comandos `dvc-pull` e `dvc-push` aceitam o argumento `ARGS` para especificar quais artefatos operar.
Arquivos rastreados via `dvc add` (ex: `data/raw`) usam o arquivo `.dvc` como referência; outputs de stages usam o caminho direto.

```bash
# Baixar todos os artefatos
make dvc-pull

# Baixar apenas os dados brutos
make dvc-pull ARGS="data/raw.dvc"

# Baixar dados processados e modelos
make dvc-pull ARGS="data/processed models"

# Enviar artefatos para o remote
make dvc-push ARGS="data/raw.dvc"
```

| Comando | Descrição |
|---|---|
| `make dvc-pull [ARGS=...]` | Baixa artefatos do remote DVC. |
| `make dvc-push [ARGS=...]` | Envia artefatos para o remote DVC. |
| `make dvc-status` | Exibe o status do pipeline e dos dados rastreados. |
| `make dvc-dag` | Exibe o grafo de dependências do pipeline no terminal. |

### MLflow

| Comando | Descrição |
|---|---|
| `make mlflow-ui` | Sobe a UI do MLflow em `http://localhost:5000`. |

---

## Licença

MIT

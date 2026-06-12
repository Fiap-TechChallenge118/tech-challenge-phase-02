# TODO List — Tech Challenge Fase 02 (Sistema de Recomendação)
> Agente: execute cada item sequencialmente. Marque `[x]` ao concluir.

---

## ETAPA 1 — Clean Code e Estrutura

### 1.1 Estrutura do Projeto
- [x] Criar estrutura de pastas:
  ```
  tech-challenge-02/
  ├── data/
  │   ├── raw/
  │   └── processed/
  ├── src/
  │   ├── __init__.py
  │   ├── settings.py
  │   ├── data/
  │   │   ├── __init__.py
  │   │   └── preprocessing.py
  │   ├── models/
  │   │   ├── __init__.py
  │   │   ├── factory.py          # Design Pattern: Factory
  │   │   ├── mlp.py
  │   │   └── baselines.py
  │   ├── features/
  │   │   ├── __init__.py
  │   │   └── engineering.py
  │   ├── training/
  │   │   ├── __y
  │   │   ├── trainer.py
  │   │   └── evaluate.py
  │   └── pipeline.py
  ├── tests/
  │   ├── conftest.py
  │   ├── test_smoke.py
  │   ├── test_schema.py
  │   └── test_pipeline.py
  ├── configs/
  │   └── config.yaml
  ├── scripts/
  │   ├── __init__.py
  │   ├── validate_env.py
  │   ├── docker_build.py
  │   ├── docker_up.py
  │   ├── docker_down.py
  │   └── docker_logs.py
  ├── models/                     # artefatos MLflow/DVC
  ├── metrics/                    # métricas DVC
  ├── notebooks/
  │   └── 01_eda.ipynb
  ├── docs/
  │   ├── model_card.md
  │   └── ml_canvas.md
  ├── infra/
  │   ├── aws/
  │   │   ├── main.tf
  │   │   ├── variables.tf
  │   │   ├── outputs.tf
  │   │   └── terraform.tfvars.example
  │   └── azure/
  │       ├── main.tf
  │       ├── variables.tf
  │       ├── outputs.tf
  │       └── terraform.tfvars.example
  ├── .github/
  │   └── workflows/
  │       └── ci.yml
  ├── .env.example
  ├── .gitignore
  ├── .dockerignore
  ├── .pre-commit-config.yaml
  ├── CONTRIBUTING.md
  ├── dvc.yaml
  ├── dvc.lock
  ├── Dockerfile
  ├── docker-compose.yml
  ├── Makefile
  ├── pyproject.toml
  ├── uv.lock
  └── README.md
  ```
- [ ] Criar todos os `__init__.py` (incluindo `scripts/__init__.py`)
- [x] Criar `.gitignore`:
  - `data/raw/`, `mlruns/`, `__pycache__/`, `.env`, `*.pth`, `*.pkl`, `.dvc/cache/`
  - `.terraform/`, `*.tfstate`, `*.tfstate.backup`, `terraform.tfvars`
  - `.venv/`, `dist/`, `*.egg-info/`
- [ ] Criar `.dockerignore` (`.git`, `mlruns/`, `data/raw/`, `__pycache__/`, `.env`, `infra/`, `.github/`)
- [ ] Criar `.env.example` com todas as variáveis necessárias (sem valores reais)
- [ ] Inicializar repositório git: `git init`
- [ ] Commit: `chore: initial project structure`

### 1.2 CONTRIBUTING.md
- [ ] Criar `CONTRIBUTING.md` com seções:
  - **Fluxo de branches:**
    ```
    main        → branch protegida, só merge via PR aprovado
    develop     → integração contínua do time
    feat/<nome> → novas funcionalidades (branch a partir de develop)
    fix/<nome>  → correções (branch a partir de develop)
    ```
  - **Convenção de commits** (Conventional Commits):
    ```
    feat:     nova funcionalidade
    fix:      correção de bug
    chore:    tarefas de manutenção (deps, configs)
    docs:     documentação
    test:     testes
    refactor: refatoração sem mudança de comportamento
    ci:       mudanças no pipeline CI/CD
    ```
  - **Processo de PR:**
    1. Abrir PR de `feat/X` → `develop`
    2. CI deve estar verde (ruff + pytest)
    3. Ao menos 1 aprovação de outro membro
    4. Squash merge com mensagem semântica
  - **Setup local:** `make setup` (único comando para novos devs)
  - **Padrões de código:** funções ≤ 20 linhas, type hints, sem `print()`, docstrings Google style
- [ ] Commit: `docs: add CONTRIBUTING.md`

### 1.3 Dataset
- [x] ✅ **Dataset definido:** [Instacart Online Grocery Basket Analysis](https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset)
  - Arquivos: `orders.csv`, `order_products__prior.csv`, `order_products__train.csv`, `products.csv`, `aisles.csv`, `departments.csv`
  - **Objetivo:** retornar top-K produtos recomendados para um usuário (ranking, não classificação binária)
  - Sinal de relevância: histórico de compras do usuário (`order_products__prior`)
  - Filtro mínimo: usuários com ≥ 5 pedidos e produtos com ≥ 5 compras
- [ ] Copiar arquivos brutos para `data/raw/`
- [ ] Verificar: `python -c "import pandas as pd; df = pd.read_csv('data/raw/...'); assert len(df) >= 10000; print('OK', df.shape)"`
- [ ] Documentar dataset escolhido no `docs/ml_canvas.md`

### 1.4 ML Canvas
- [ ] Criar `docs/ml_canvas.md` com seções:
  - Stakeholders e problema de negócio (e-commerce, aumento de conversão)
  - Definição de interação positiva (compra, clique ou rating >= threshold)
  - Métrica técnica: Precision@K, Recall@K, NDCG@K, MAP
  - Métrica de negócio: aumento de CTR, receita incremental estimada
  - SLOs: latência de inferência < 100ms para top-K recomendações

### 1.5 Design Patterns
- [ ] Implementar `src/models/factory.py`:
  - Classe `ModelFactory` com método `create(model_type: str, config: dict) -> nn.Module`
  - Suportar tipos: `"mlp"`, `"embedding"`, `"baseline"`
- [ ] Implementar `src/data/preprocessing.py` com Strategy pattern:
  - Interface `PreprocessorStrategy(ABC)` com método `fit_transform(df)`
  - Implementações: `RatingPreprocessor`, `ImplicitFeedbackPreprocessor`
- [ ] Type hints em todas as funções públicas
- [ ] Docstrings Google style em todas as funções públicas
- [ ] Garantir funções ≤ 20 linhas

### 1.6 Qualidade de Código
- [ ] Configurar `ruff` no `pyproject.toml` (target Python 3.11, regras E/F/W/I/N)
- [ ] Criar `.pre-commit-config.yaml`:
  ```yaml
  repos:
    - repo: https://github.com/astral-sh/ruff-pre-commit
      rev: v0.4.4
      hooks:
        - id: ruff
          args: [--fix]
        - id: ruff-format
  ```
- [ ] Rodar `ruff check .` → corrigir até zero erros
- [ ] Commit: `chore: configure ruff and pre-commit hooks`

---

## ETAPA 2 — Ambiente e Dependências

### 2.1 uv e pyproject.toml
- [ ] Instalar uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- [ ] Inicializar projeto: `uv init --no-workspace` (ou criar `pyproject.toml` manualmente)
- [ ] Adicionar dependências de produção:
  ```bash
  uv add torch scikit-learn mlflow dvc pandas numpy pydantic pydantic-settings python-dotenv
  ```
- [ ] Adicionar dependências de desenvolvimento:
  ```bash
  uv add --group dev pytest ruff pre-commit ipykernel
  ```
- [ ] Registrar entrypoints no `pyproject.toml`:
  ```toml
  [project.scripts]
  docker-build  = "scripts.docker_build:main"
  docker-up     = "scripts.docker_up:main"
  docker-down   = "scripts.docker_down:main"
  docker-logs   = "scripts.docker_logs:main"
  validate-env  = "scripts.validate_env:main"
  ```
- [ ] Verificar que `uv.lock` foi gerado
- [ ] Commitar lock file: `git add uv.lock && git commit -m "chore: add uv lock file"`

### 2.2 Configurações com Pydantic Settings
- [ ] Criar `src/settings.py`:
  - Classe `Settings(BaseSettings)` lendo do `.env`
  - Campos: `DATA_PATH`, `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME`, `DVC_REMOTE`, `RANDOM_SEED`, `TOP_K`
- [ ] Criar `.env.example` com todos os campos (sem valores secretos)
- [ ] Criar `.env` local (não commitado) com valores reais
- [ ] Criar `configs/config.yaml` com schema documentado:
  ```yaml
  model:
    type: mlp          # mlp | embedding
    hidden_dims: [256, 128, 64]
    dropout: 0.3
  training:
    epochs: 50
    batch_size: 512
    lr: 0.001
    weight_decay: 1e-5
    patience: 5        # early stopping
  data:
    top_k: 10
    test_size: 0.2
    seed: 42
  ```
- [ ] Referenciar `Settings()` em todos os módulos (sem hardcoding)

### 2.3 Scripts Docker via uv
- [ ] Criar `scripts/__init__.py` (vazio)
- [ ] Criar `scripts/docker_build.py` — executa `docker build -t tc02-trainer .`
- [ ] Criar `scripts/docker_up.py` — executa `docker compose up -d`
- [ ] Criar `scripts/docker_down.py` — executa `docker compose down`
- [ ] Criar `scripts/docker_logs.py` — executa `docker compose logs -f`
- [ ] Testar: `uv run docker-build` → imagem construída sem erros
- [ ] Commit: `feat: docker helper scripts via uv`

### 2.4 Script de Validação
- [ ] Criar `scripts/validate_env.py`:
  - Verificar importação de todas as libs obrigatórias (torch, sklearn, mlflow, dvc)
  - Verificar presença das variáveis de ambiente obrigatórias
  - Verificar versão do Python >= 3.11
  - Logar resultado via `logging` (sem `print()`)
- [ ] Testar: `uv run validate-env` → saída sem erros
- [ ] Commit: `feat: pydantic settings and env validation script`

### 2.5 Makefile com Setup Único
- [ ] Criar `Makefile` com target `setup` como ponto de entrada para novos devs:
  ```makefile
  setup:           ## Configura o ambiente do zero (rodar uma vez ao clonar)
      uv sync
      pre-commit install
      cp -n .env.example .env || true
      @echo "✅ Setup completo. Edite o .env com seus valores."

  install:
      uv sync

  lint:
      ruff check .

  test:
      pytest tests/ -v

  train:
      dvc repro

  mlflow-ui:
      mlflow ui --backend-store-uri mlruns/

  docker-build:
      uv run docker-build

  docker-up:
      uv run docker-up

  docker-down:
      uv run docker-down

  docker-logs:
      uv run docker-logs

  validate:
      uv run validate-env
  ```
- [ ] Documentar `make setup` no README como **primeiro comando após clonar**
- [ ] Commit: `chore: makefile with setup target`

---

## ETAPA 3 — Containerização e Versionamento

### 3.1 GitHub Actions (CI)
- [ ] Criar `.github/workflows/ci.yml`:
  ```yaml
  name: CI
  on:
    push:
      branches: [main, develop]
    pull_request:
      branches: [main, develop]
  jobs:
    lint-and-test:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - uses: astral-sh/setup-uv@v3
        - run: uv sync --frozen
        - run: uv run ruff check .
        - run: uv run pytest tests/ -v
  ```
- [ ] Verificar: abrir PR → CI executa automaticamente
- [ ] Configurar branch protection em `main` e `develop`: exigir CI verde + 1 aprovação
- [ ] Commit: `ci: add GitHub Actions workflow for lint and test`

### 3.2 DVC
- [ ] Inicializar DVC: `dvc init`
- [ ] Versionar dataset: `dvc add data/raw/`
- [ ] Configurar remote local: `dvc remote add -d local_remote /tmp/dvc-storage`
- [ ] (Opcional) Configurar remote S3/Azure Blob para o time compartilhar dados:
  ```bash
  # AWS S3
  dvc remote add -d s3remote s3://<bucket>/dvc
  # Azure Blob
  dvc remote add -d azremote azure://<container>/dvc
  ```
- [ ] Fazer push dos dados: `dvc push`
- [ ] Commitar arquivos DVC: `git add data/raw/.gitignore data/raw/*.dvc .dvc/config && git commit -m "feat: add dataset versioning with DVC"`

### 3.3 Pipeline DVC
- [ ] Criar `dvc.yaml` com os stages:
  ```yaml
  stages:
    preprocess:
      cmd: python -m src.data.preprocessing
      deps: [data/raw/, src/data/preprocessing.py]
      outs: [data/processed/interactions.parquet]

    feature_eng:
      cmd: python -m src.features.engineering
      deps: [data/processed/interactions.parquet, src/features/engineering.py]
      outs: [data/processed/features.parquet, data/processed/user_encoder.pkl, data/processed/item_encoder.pkl]

    train:
      cmd: python -m src.training.trainer
      deps: [data/processed/features.parquet, src/training/trainer.py, src/models/, configs/config.yaml]
      outs: [models/mlp_model.pth]
      metrics: [metrics/train_metrics.json]

    evaluate:
      cmd: python -m src.training.evaluate
      deps: [models/mlp_model.pth, data/processed/features.parquet, src/training/evaluate.py]
      metrics: [metrics/eval_metrics.json]
  ```
- [ ] Implementar cada stage como módulo executável (`if __name__ == "__main__"`)
- [ ] Testar: `dvc repro` → pipeline executa do zero sem erros
- [ ] Commit: `feat: DVC pipeline with 4 stages`

### 3.4 MLflow Tracking
- [ ] Configurar `MLFLOW_TRACKING_URI` no `.env` (ex: `http://localhost:5000`)
- [ ] No stage `train`: logar com MLflow:
  - `mlflow.log_params(config)` — hiperparâmetros
  - `mlflow.log_metric("loss", ..., step=epoch)` — loss por epoch
  - `mlflow.log_metric("precision_at_k", ...)` — métricas finais
  - `mlflow.pytorch.log_model(model, "model")` — artefato do modelo
- [ ] Garantir ≥ 3 runs distintos com configurações diferentes
- [ ] Commit: `feat: MLflow tracking integrated into DVC pipeline`

### 3.5 Docker
- [ ] Criar `Dockerfile` multi-stage:
  ```dockerfile
  # Stage 1: builder
  FROM python:3.11-slim AS builder
  WORKDIR /app
  COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
  COPY pyproject.toml uv.lock ./
  RUN uv export --frozen --no-dev -o requirements.txt
  RUN pip wheel --no-cache-dir -r requirements.txt -w /wheels

  # Stage 2: runtime
  FROM python:3.11-slim AS runtime
  WORKDIR /app
  COPY --from=builder /wheels /wheels
  RUN pip install --no-cache-dir /wheels/*.whl
  COPY src/ src/
  COPY configs/ configs/
  ENV PYTHONPATH=/app
  CMD ["python", "-m", "src.pipeline"]
  ```
- [ ] Criar `docker-compose.yml` com serviços:
  - `trainer`: executa `dvc repro`
  - `mlflow`: `ghcr.io/mlflow/mlflow mlflow server --host 0.0.0.0`
  - Volumes para `data/`, `models/`, `mlruns/`
- [ ] Testar: `uv run docker-build` → sem erros
- [ ] Testar: `docker compose up mlflow` → MLflow UI acessível em `localhost:5000`
- [ ] Commit: `feat: multi-stage Dockerfile and docker-compose`

---

## ETAPA 4 — Rede Neural, Registry e Entrega

### 4.1 Modelo MLP PyTorch
- [ ] Criar `src/models/mlp.py`:
  - Classe `RecommendationMLP(nn.Module)`:
    - Input: features de usuário + item concatenadas
    - Camadas: `Linear → BatchNorm → ReLU → Dropout` (configurável via `configs/config.yaml`)
    - Output: score de relevância (sigmoid)
  - Método `forward(user_features, item_features) -> Tensor`
- [ ] Criar `src/models/baselines.py`:
  - `PopularityBaseline`: recomenda itens mais populares
  - `SklearnBaseline`: wrapper para `LogisticRegression` ou `RandomForestClassifier`
- [ ] Fixar seeds: `torch.manual_seed(SEED)`, `np.random.seed(SEED)`, `random.seed(SEED)`

### 4.2 Loop de Treino
- [ ] Criar `src/training/trainer.py`:
  - Função `train(model, train_loader, val_loader, config)`:
    - `BCEWithLogitsLoss` com `pos_weight` para interações esparsas
    - Optimizer: `Adam` com `lr` e `weight_decay` configuráveis
    - Early stopping: parar se `val_loss` não melhora por `patience` epochs
    - Logar loss por epoch no MLflow
  - Retornar modelo com melhor val_loss

### 4.3 Avaliação com ≥ 4 Métricas
- [ ] Criar `src/training/evaluate.py`:
  - Função `evaluate_at_k(model, test_data, k=10) -> dict`:
    - `Precision@K`
    - `Recall@K`
    - `NDCG@K`
    - `MAP` (Mean Average Precision)
  - Comparar MLP vs todos os baselines em tabela
  - Salvar resultados em `metrics/eval_metrics.json`

### 4.4 MLflow Model Registry
- [ ] Registrar modelo no Registry:
  ```python
  mlflow.pytorch.log_model(model, "model", registered_model_name="recommendation-mlp")
  ```
- [ ] Promover para Staging:
  ```python
  client.transition_model_version_stage(name="recommendation-mlp", version=1, stage="Staging")
  ```
- [ ] Após validação, promover para Production:
  ```python
  client.transition_model_version_stage(name="recommendation-mlp", version=1, stage="Production")
  ```
- [ ] Verificar ≥ 3 runs rastreados no experimento

### 4.5 Testes
- [ ] Criar `tests/conftest.py` com fixtures compartilhadas (ex: modelo instanciado, dataset mock)
- [ ] Criar `tests/test_smoke.py`:
  - Instanciar `RecommendationMLP` com dimensões fixas
  - Forward pass com tensores aleatórios
  - Assertar shape do output `== (batch_size, 1)`
- [ ] Criar `tests/test_schema.py`:
  - Carregar dataset processado
  - Verificar schema: tipos, ranges, ausência de nulos pós-processamento
  - Verificar que user_ids e item_ids estão dentro do vocabulário dos encoders
- [ ] Criar `tests/test_pipeline.py`:
  - Rodar pipeline completo em subset pequeno do dataset (smoke test)
  - Verificar que `metrics/eval_metrics.json` é gerado corretamente
- [ ] Rodar `pytest tests/ -v` → todos os testes passando
- [ ] Commit: `test: smoke, schema and pipeline tests passing`

### 4.6 Model Card
- [ ] Criar `docs/model_card.md` com seções:
  - Descrição do modelo (arquitetura MLP, dataset, data de treino)
  - Métricas finais (Precision@K, Recall@K, NDCG@K, MAP no test set)
  - Comparação com baselines (tabela)
  - Limitações (cold start, popularidade bias, dados históricos)
  - Uso pretendido e uso não recomendado
  - Decisões de design e trade-offs

### 4.7 README
- [ ] Criar `README.md` com seções:
  - Descrição do problema e solução
  - Arquitetura do projeto (diagrama ou texto)
  - **Setup (novos devs):** `git clone ... && make setup`
  - Como rodar: `make train`, `make test`, `make mlflow-ui`
  - Como reproduzir com Docker: `make docker-up`
  - Tabela de resultados finais (métricas comparativas)
  - Links para Model Card, ML Canvas e CONTRIBUTING

### 4.8 Deploy em Nuvem com Terraform (Bônus)

#### 4.8.1 AWS (ECR + ECS Fargate)
- [ ] Criar `infra/aws/`:
  - `main.tf`: provider `aws`, ECR repository, ECS cluster, task definition, service, ALB, Security Groups
  - `variables.tf`: `aws_region`, `image_tag`, `container_port`, `cpu`, `memory`
  - `outputs.tf`: `ecr_repository_url`, `alb_dns_name`
  - `terraform.tfvars.example`
- [ ] Build e push para ECR:
  ```bash
  uv run docker-build
  aws ecr get-login-password | docker login --username AWS --password-stdin <ecr-url>
  docker tag tc02-trainer:latest <ecr-url>/tc02-trainer:latest
  docker push <ecr-url>/tc02-trainer:latest
  ```
- [ ] Provisionar: `cd infra/aws && terraform init && terraform apply`
- [ ] Verificar URL pública: `terraform output alb_dns_name`
- [ ] Commit: `feat: terraform AWS deploy (ECR + ECS Fargate)`

#### 4.8.2 Azure (ACR + ACI)
- [ ] Criar `infra/azure/`:
  - `main.tf`: provider `azurerm`, resource group, ACR, ACI (container group)
  - `variables.tf`: `location`, `resource_group_name`, `image_tag`, `container_port`
  - `outputs.tf`: `acr_login_server`, `container_fqdn`
  - `terraform.tfvars.example`
- [ ] Build e push para ACR:
  ```bash
  uv run docker-build
  az acr login --name <acr-name>
  docker tag tc02-trainer:latest <acr-login-server>/tc02-trainer:latest
  docker push <acr-login-server>/tc02-trainer:latest
  ```
- [ ] Provisionar: `cd infra/azure && terraform init && terraform apply`
- [ ] Verificar URL pública: `terraform output container_fqdn`
- [ ] Commit: `feat: terraform Azure deploy (ACR + ACI)`

#### 4.8.3 Documentação
- [ ] Documentar ambas as URLs públicas no README
- [ ] Adicionar seção "Deploy" ao README com comandos para cada cloud

### 4.9 Checklist Final e Entrega
- [ ] Rodar checklist de qualidade:
  - `ruff check .` → 0 erros
  - `pytest tests/ -v` → todos passando
  - `uv run docker-build` → sem erros
  - `dvc repro` → pipeline roda do zero
  - `grep -r "print(" src/` → saída vazia (zero prints)
  - Nenhuma chave/secret hardcoded no código
- [ ] Verificar CI verde na branch `main`
- [ ] Verificar histórico git: `git log --oneline` → commits semânticos limpos
- [ ] Verificar que `uv.lock` está commitado
- [ ] Verificar que `.env` não está no repositório: `git status`
- [ ] Gravar vídeo 5 min (método STAR):
  - **S**ituation: problema de recomendação no e-commerce
  - **T**ask: pipeline ML end-to-end com MLP embedding-based
  - **A**ction: EDA → DVC pipeline → MLP PyTorch → MLflow Registry → Docker
  - **R**esult: métricas finais vs baselines, `dvc repro` ao vivo, MLflow UI
- [ ] Publicar repositório público no GitHub
- [ ] Commit final: `docs: finalize README and model card`

---

## Resumo de Critérios de Avaliação

| Critério                  | Peso | Itens Relacionados                                    |
|---------------------------|------|-------------------------------------------------------|
| Clean code e estrutura    | 15%  | 1.5, 1.6 (SOLID, naming, type hints, design patterns) |
| Reprodutibilidade         | 15%  | 2.1, 2.2, 2.4 (uv, lock file, .env)                  |
| Docker                    | 15%  | 3.5 (multi-stage, imagem otimizada, compose)          |
| DVC + Pipeline            | 15%  | 3.2, 3.3 (≥ 3 stages, dvc repro funcional)            |
| Rede neural PyTorch       | 15%  | 4.1, 4.2, 4.3 (MLP, early stopping, baselines)        |
| MLflow + Registry         | 10%  | 3.4, 4.4 (≥ 3 runs, modelo em Production)             |
| Vídeo STAR                | 10%  | 4.9 (clareza, 4 elementos, ≤ 5 min)                   |
| Bônus: deploy em nuvem    | 5%   | 4.8 (Terraform + container acessível via URL pública) |

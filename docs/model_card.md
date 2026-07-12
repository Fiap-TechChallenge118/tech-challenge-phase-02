# Model Card — RecommendationMLP

**Versão:** 1 (Production)
**Registrado em:** MLflow Model Registry (`recommender-mlp`)
**Responsável:** Matheus Ferreira (Dev C)
**Data de criação:** 2026-07-04

---

## 1. Descrição do Modelo

RecommendationMLP é uma rede neural de recomendação baseada em **embeddings
latentes + MLP**. O modelo aprende representações densas de usuários e itens
(cada um com dimensão 64) e as combina via concatenação antes de passar por
uma rede fully-connected com 3 camadas ocultas.

### Arquitetura

```
user_idx ──→ Embedding(206k, 64) ──┐
                                    ├──→ concat(128) ──→ MLP ──→ logit
item_idx ──→ Embedding(50k, 64) ──┘
```

| Componente | Especificação |
|---|---|
| Embeddings | 2 camadas independentes (`nn.Embedding`, dim=64) |
| Camadas ocultas | `[256, 128, 64]` |
| Ativação | ReLU |
| Normalização | BatchNorm1d após cada Linear |
| Regularização | Dropout (p=0.3) |
| Saída | 1 logit (sem sigmoid — `BCEWithLogitsLoss` aplica internamente) |
| Parâmetros totais | **16.452.353** |

### Treinamento

| Hiperparâmetro | Valor |
|---|---|
| Loss | `BCEWithLogitsLoss` (sem `pos_weight`) |
| Otimizador | Adam (`lr=0.001`, `weight_decay=1e-5`) |
| Batch size | 1024 |
| Épocas máximas | 50 |
| Early stopping | patience=5, min_delta=1e-4 |
| Melhor época | 10 (parou na 15) |

### Dataset

**Instacart Online Grocery Basket Analysis** (fonte: [Kaggle](https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset)):

| Estatística | Valor |
|---|---|
| Pedidos | 3.421.083 |
| Interações user-item | 33.819.106 |
| Usuários | 206.209 |
| Itens (produtos) | 49.685 |
| Sinal de relevância | Compra do produto (interação positiva) |

**Treino:** 20% do dataset (5.545.498 pares) com split 80/10/10
(treino/validação/teste). Estratégia de pares: `InteractionPairsStrategy`
(positivos = compras reais, negativos = amostragem aleatória com ratio 1:1).

---

## 2. Métricas de Avaliação

Avaliado em **3.938 usuários do test set** (amostra de 5.000 com interações
positivas), recomendando os top-10 itens para cada usuário.

### Resultados (@10)

| Modelo | Precision | Recall | NDCG | MAP |
|---|---|---|---|---|
| **Popularity** | **0,0084** | **0,0425** | **0,0244** | **0,0139** |
| MLP (PyTorch) | 0,0009 | 0,0042 | 0,0022 | 0,0010 |
| Sklearn (LogReg) | 0,0006 | 0,0023 | 0,0011 | 0,0004 |

### Interpretação

- **Popularity** lidera: recomendar os itens mais comprados globalmente é
  surpreendentemente eficaz para datasets de mercado com forte concentração
  de compras em poucos produtos.
- **MLP supera Sklearn em ~2-3×**: as embeddings latentes capturam sinal de
  preferência que features agregadas por usuário (LogReg) não conseguem
  modelar.
- **Valores absolutos baixos**: o catálogo tem ~50 mil itens e recomendar
  apenas 10 é inerentemente difícil. Além disso, o modelo foi treinado em só
  20% dos dados para viabilizar iterações rápidas durante o desenvolvimento.

---

## 3. Comparação com Baselines

| Baseline | Tipo | Descrição |
|---|---|---|
| **PopularityBaseline** | Não-personalizado | Recomenda os itens mais comprados globalmente. Mesma lista para todos os usuários. |
| **SklearnBaseline** | Regressão logística | `LogisticRegression(max_iter=1000, class_weight="balanced")` + `StandardScaler` treinada sobre features agregadas por usuário (nº de itens distintos, nº de pedidos, etc.) via `AggregatedFeaturesStrategy`. |
| **RecommendationMLP** | Rede neural | Embeddings latentes + MLP de 3 camadas (este modelo). |

---

## 4. Limitações e Vieses

### Cold start
O modelo não consegue recomendar para **novos usuários** (sem histórico de
compras) nem para **novos itens** (sem interações registradas). Ambos
dependem dos embeddings aprendidos durante o treino — um usuário/item
ausente do vocabulário causaria erro de índice.

### Viés de popularidade
Assim como a maioria dos sistemas de recomendação colaborativos, o modelo
tende a recomendar itens populares. A PopularityBaseline lidera as métricas
justamente porque concentrar recomendações nos itens mais comprados é uma
heurística forte neste dataset.

### Dados históricos e sazonalidade
O dataset Instacart é um snapshot de compras de supermercado. Mudanças de
estação, lançamentos de produtos ou alterações no comportamento do
consumidor não são capturadas. O modelo deve ser retreinado periodicamente.

### Treino parcial (20%)
Os resultados atuais refletem treino com apenas 20% dos dados. Um treino
completo (100%) deve melhorar as métricas da MLP e possivelmente ultrapassar
a Popularity, já que a MLP tem capacidade de aprender preferências
personalizadas que a baseline de popularidade ignora.

### Ausência de features de conteúdo
O modelo usa apenas interações passadas (filtragem colaborativa pura).
Features de produto (departamento, corredor, nome) e de usuário
(frequência, horário) não são usadas, o que limita a capacidade de
generalização.

---

## 5. Uso Pretendido e Uso Não Recomendado

### ✅ Uso pretendido
- Sistemas de recomendação em e-commerce/supermercado com catálogo estável
- Geração de top-K recomendações personalizadas para usuários com histórico
- Baseline para experimentos com arquiteturas mais complexas (Transformers,
  Graph Neural Networks)

### ❌ Uso não recomendado
- **Decisões de alto risco** (saúde, finanças, justiça) — o modelo não foi
  validado para esses domínios
- **Recomendações em tempo real** com catálogo dinâmico — cold start de
  itens não é tratado
- **Usuários sem histórico** — não há fallback para novos usuários
- **Substituir curadoria humana** sem supervisão — viés de popularidade
  pode criar bolhas de recomendação

---

## 6. Decisões de Design e Trade-offs

| Decisão | Justificativa | Trade-off |
|---|---|---|
| **Embedding dim=64** | Bom equilíbrio entre expressividade e tamanho do modelo (16M params) | dim=128 poderia capturar mais nuance, mas aumentaria VRAM em ~2× |
| **BCEWithLogitsLoss (sem sigmoid no forward)** | Estabilidade numérica — evita dupla-sigmoid | Requer aplicar `torch.sigmoid` manualmente para obter probabilidades na inferência |
| **Pickle (não pt2) para serialização** | Compatível com qualquer arquitetura PyTorch sem exigir `TensorSpec` | Menos seguro que `torch.export` (avisos do MLflow) |
| **SQLite como backend MLflow** | Registry funciona sem servidor dedicado | Não escala para times grandes ou múltiplos acessos simultâneos |
| **Train/Val/Test 80/10/10 com permutação determinística** | Reprodutibilidade garantida entre execuções | Pode não capturar sazonalidade se a ordem temporal for relevante |
| **3 camadas ocultas [256,128,64]** | Arquitetura clássica de funil, testada em benchmarks de recomendação | Alternativas (ex: tower separada para user/item) podem performar melhor |

---

## 7. Reprodução

```bash
# 1. Setup do ambiente
git clone https://github.com/Fiap-TechChallenge118/tech-challenge-phase-02.git
cd tech-challenge-phase-02
uv sync

# 2. Baixar dados versionados pelo DVC
make dvc-pull

# 3. Reproduzir o pipeline completo (preprocess → feature_eng → train → evaluate → register)
make repro

# 4. Visualizar experimentos no MLflow
make mlflow-ui
# Acesse http://localhost:5000
```

---

## 8. Referências

- Dataset: [Instacart Online Grocery Basket Analysis (Kaggle)](https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset)
- MLflow: [mlflow.org](https://mlflow.org)
- PyTorch: [pytorch.org](https://pytorch.org)
- Model Cards: [Mitchell et al. (2019)](https://arxiv.org/abs/1810.03993)

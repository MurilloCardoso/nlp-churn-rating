# Churn Prediction — Telco (Tech Challenge Fase 1)

Previsão de churn de clientes de telecomunicações via rede neural (MLP em PyTorch) comparada com baselines (Regressão Logística, XGBoost) e combinada em um ensemble. Pipeline completo: EDA, treino, tracking de experimentos com MLflow, API de inferência em FastAPI e suíte de testes automatizados.

---

## Contexto

Uma operadora de telecom está perdendo clientes e precisa de um modelo preditivo que sinalize quem tem risco de cancelamento, para agir com ofertas de retenção antes da perda. Este repositório entrega o pipeline end-to-end exigido pelo Tech Challenge Fase 1 da Pos-Tech: do dado cru até um endpoint HTTP servindo predições.

**Dataset:** [Telco Customer Churn (IBM)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — ~7.000 registros, 20 features, desbalanceado (~27% positivos).

---

## Arquitetura

```
nlp-churn-rating/
├── data/raw/              # dataset original (telco_churn.csv)
├── docs/                  # Model Card e documentação técnica
├── models/                # artefatos de produção (scaler, threshold, pesos)
├── notebooks/             # EDA, baselines exploratórios, métricas
├── scripts/               # utilitários (batch_predict.py)
├── src/
│   ├── main.py            # pipeline de treino: CV + baselines + MLP + ensemble
│   └── api.py             # FastAPI: /health, /predict
├── tests/                 # smoke, schema (pandera), unit, API
├── Makefile               # atalhos (install, lint, test, train, run)
└── pyproject.toml         # single source of truth (deps, ruff, pytest)
```

### Pipeline de ML

```
CSV raw
  ↓  limpeza + feature engineering (ChargesPerMonth, HighSpender, NewCustomer, LongTermCustomer)
  ↓  one-hot encoding + StandardScaler
split 80/20 estratificado
  ├── train+val (80%)
  │     ├── StratifiedKFold(k=5) → Dummy / LR / XGBoost  [MLflow CV metrics]
  │     └── inner split 80/20 → MLP com early stopping (val_loss)
  └── test (20%) → avaliação única de todos os modelos + ensemble
         ↓  threshold ótimo via análise de custo (FN=10× FP)
         ↓  métricas: AUC-ROC, PR-AUC, F1, Recall, Accuracy
         ↓  MLflow + artefatos salvos em models/
```

### Modelos

| Modelo | Lib | Hiperparâmetros-chave | Papel |
|---|---|---|---|
| DummyClassifier | scikit-learn | `strategy="stratified"` | baseline aleatório |
| LogisticRegression | scikit-learn | `C=0.1`, `class_weight="balanced"` | baseline linear |
| XGBoost | xgboost | `n_estimators=300`, `max_depth=4`, `lr=0.05`, `scale_pos_weight=n_neg/n_pos` | baseline árvore |
| ChurnMLP | PyTorch | `Linear(N,64)-BN-ReLU-Dropout(0.4)-Linear(64,32)-BN-ReLU-Dropout(0.3)-Linear(32,1)`, Adam `lr=1e-3`, `wd=1e-4`, early stopping | **modelo central** |
| Ensemble | — | `mean(P_LR, P_XGB, P_MLP)` | combinação |

### Decisões técnicas

- **Balanceamento:** nos baselines usamos `class_weight='balanced'` (LR) e `scale_pos_weight` (XGB). No MLP **não** usamos `pos_weight` — o desbalanceamento é tratado downstream via threshold.
- **Threshold:** escolhido por minimização de custo esperado `FN*cost_fn + FP*cost_fp`, com `cost_fn=10` e `cost_fp=1` (perder um churner dói 10× mais que oferecer desconto errado). Isso materializa o trade-off FP vs FN exigido pela rubrica.
- **Reprodutibilidade:** seeds fixas para `random`, `numpy`, `torch` + `cudnn.deterministic=True`.
- **Validação:** `StratifiedKFold(k=5)` para baselines; split interno 80/20 para early stopping do MLP. **Test set nunca é tocado durante o treino.**
- **Logging:** `logging` estruturado, `print` proibido.

---

## Setup

### Pré-requisitos

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

### Instalação

```bash
git clone https://github.com/seu-usuario/nlp-churn-rating.git
cd nlp-churn-rating

# Instala deps de produção + desenvolvimento (pytest, ruff, pandera, httpx)
make install
# equivalente a: uv sync --extra dev
```

### Dataset

Baixar `telco_churn.csv` e salvar em `data/raw/`. O teste `test_dataset_exists` valida o caminho.

---

## Uso

Todos os comandos têm atalho no `Makefile`.

### Treinar o pipeline completo

```bash
make train
# equivalente a: uv run python -m src.main
```

Isso executa:
1. Carrega e limpa o CSV.
2. Aplica feature engineering.
3. Roda `StratifiedKFold(k=5)` para Dummy/LR/XGBoost.
4. Treina o MLP com early stopping.
5. Treina baselines finais em `train+val`.
6. Avalia todos no test set com threshold ótimo por custo.
7. Salva artefatos em `models/` e registra tudo no MLflow.

**Artefatos gerados em `models/`:**
- `best_model.pt` — pesos do MLP
- `scaler.pkl` — StandardScaler ajustado
- `feature_columns.json` — ordem das colunas após one-hot
- `threshold.json` — threshold ótimo do MLP
- `lr_model.pkl` / `xgb_model.pkl` — baselines serializados

### Subir a API

```bash
make run
# equivalente a: uv run uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health` → status + `model_loaded` + `input_dim`
- `POST /predict` → recebe `CustomerFeatures`, retorna `{churn_probability, churn_prediction, threshold}`

Exemplo:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85, "TotalCharges": 358.2
  }'
```

Doc interativa (Swagger UI): http://localhost:8000/docs

### Batch de predições de teste

Com a API rodando em outro terminal:

```bash
uv run python scripts/batch_predict.py --n 200 --seed 7
```

### Testes automatizados

```bash
make test
# 26 testes: smoke (MLP), preprocessing (threshold), schema (pandera), API (TestClient)
```

### Linting

```bash
make lint     # ruff check
make format   # ruff format
```

---

## MLflow tracking

Subir a UI (padrão em `./mlruns`):

```bash
make mlflow
# equivalente a: uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --port 5000
# http://127.0.0.1:5000
```

Cada run do `make train` registra:
- **Params:** seed, splits, custos FN/FP, hiperparâmetros de todos os 3 modelos, arquitetura do MLP.
- **Métricas CV:** `cv_{dummy,lr,xgb}_{auc_mean,auc_std,pr_auc_mean,pr_auc_std,f1_mean,recall_mean}`.
- **Métricas test:** `test_{dummy,lr,xgboost,mlp,ensemble}_{auc_roc,pr_auc,f1,recall,accuracy,threshold}`.
- **Matriz de confusão** do ensemble: `test_confusion_{tn,fp,fn,tp}`.
- **Custo total** do ensemble no test: `test_ensemble_cost`.
- **Artefato:** modelo PyTorch serializado.

---

## Resultados (execução de referência)

### CV 5-fold no conjunto train+val

| Modelo | AUC-ROC | PR-AUC | F1 (t=0.5) | Recall (t=0.5) |
|---|---|---|---|---|
| Dummy | 0.499 ± 0.014 | 0.266 ± 0.005 | 0.266 | 0.268 |
| Logistic Regression | **0.850 ± 0.005** | **0.669 ± 0.017** | 0.639 | 0.799 |
| XGBoost | 0.844 ± 0.003 | 0.662 ± 0.008 | 0.632 | 0.776 |

### Test set (threshold ótimo por custo, FN=10× FP)

| Modelo | AUC-ROC | PR-AUC | F1 | Recall | Accuracy | Threshold |
|---|---|---|---|---|---|---|
| Dummy | 0.507 | 0.269 | 0.277 | 0.278 | 0.615 | 0.010 |
| LR | 0.839 | 0.651 | 0.554 | 0.968 | 0.585 | 0.200 |
| XGBoost | 0.833 | 0.642 | 0.526 | 0.979 | 0.530 | 0.100 |
| MLP | 0.836 | 0.639 | 0.503 | 0.995 | 0.478 | 0.030 |
| **Ensemble** | **0.840** | **0.653** | 0.524 | 0.984 | 0.525 | 0.090 |

**Matriz de confusão do ensemble:** TN=371 · FP=662 · **FN=6** · TP=368
→ 98.4% de recall em churn, com accuracy baixa como consequência direta do peso 10× em FN.

---

## Testes

| Arquivo | O que cobre |
|---|---|
| `tests/test_smoke.py` | Forward/backward do MLP, overfit de batch pequeno, formato de saída |
| `tests/test_preprocessing.py` | `best_threshold_recall` — faixas, min_precision respeitado, parametrizado |
| `tests/test_schema.py` | Validação do CSV com `pandera` — tipos, domínios, contagem mínima |
| `tests/test_api.py` | `/health`, `/predict`, validação Pydantic (422), header de latência |

---

## Documentação adicional

- [docs/MODEL_CARD.md](docs/MODEL_CARD.md) — Model Card completo: uso pretendido, métricas, vieses, cenários de falha, plano de monitoramento.

---


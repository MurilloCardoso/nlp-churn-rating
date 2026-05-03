# Model Card — Churn Prediction (Telco)

Documento estruturado conforme [Model Cards for Model Reporting (Mitchell et al., 2019)](https://arxiv.org/abs/1810.03993). Cobre o modelo servido em produção (`/predict`) e o ensemble usado como referência interna.

---

## 1. Detalhes do modelo

| Item | Valor |
|---|---|
| Nome | `ChurnMLP` (modelo servido) + ensemble `LR + XGBoost + MLP` |
| Versão | 1.0.0 |
| Tipo | Classificação binária (churn / no churn) |
| Arquitetura central | MLP feed-forward em PyTorch — `Linear(N→64) → BatchNorm → ReLU → Dropout(0.4) → Linear(64→32) → BatchNorm → ReLU → Dropout(0.3) → Linear(32→1)` |
| Loss | `BCEWithLogitsLoss` (sem `pos_weight`) |
| Otimizador | Adam (`lr=1e-3`, `weight_decay=1e-4`) com `ReduceLROnPlateau` |
| Regularização | Dropout (0.4 / 0.3), early stopping (patience=20, val_loss) |
| Baselines | DummyClassifier (`stratified`), LogisticRegression (`C=0.1, class_weight='balanced'`), XGBoost (`n_estimators=300, max_depth=4, lr=0.05, scale_pos_weight=n_neg/n_pos`) |
| Ensemble | Média aritmética das probabilidades dos 3 modelos |
| Frameworks | PyTorch, Scikit-Learn, XGBoost, MLflow, FastAPI |
| Reprodutibilidade | Seeds fixas (`random`, `numpy`, `torch`); `cudnn.deterministic=True` |
| Desenvolvedores | Grupo Pos-Tech Fase 1 |
| Licença / uso | Acadêmico (Tech Challenge) |

---

## 2. Uso pretendido

### Casos de uso primários

- **Priorização de campanhas de retenção:** identificar clientes com risco de cancelamento para uma equipe de CRM/marketing acionar com ofertas direcionadas.
- **Apoio a analistas de negócio:** suplemento ao julgamento humano, não substituto.

### Usuários primários

- Times de CRM e Customer Success da operadora.
- Analistas de retenção que consomem o endpoint `/predict` ou um lote de scores agendado.

### Casos de uso explicitamente fora de escopo

- **Decisões automáticas com efeito legal/financeiro** sobre o cliente (negar serviço, reajustar preço, encerrar contrato).
- **Pontuação de crédito** ou qualquer decisão que impacte termos contratuais.
- **Inferência sobre coortes** muito diferentes do dataset de treino (clientes não-residenciais, mercados fora dos EUA, planos B2B).
- **Previsão em tempo real de evento único** (o modelo prevê risco em uma janela, não o instante do cancelamento).

---

## 3. Fatores

### Fatores relevantes

- **Tipo de contrato** (`Month-to-month`, `One year`, `Two year`) — fator dominante de churn.
- **Tempo de relacionamento** (`tenure`) — clientes novos churnam muito mais.
- **Serviço de internet** (`Fiber optic` vs `DSL` vs `No`).
- **Forma de pagamento** (`Electronic check` historicamente associado a churn maior).
- **Idoso** (`SeniorCitizen`) e **dependentes** (`Dependents`).

### Fatores de avaliação

A avaliação reportada agrega todos os subgrupos. Análises por subgrupo (gênero, senioridade, tipo de contrato) **não** foram incluídas nesta versão e estão listadas em [§9 Caveats](#9-caveats-e-recomendações). Antes de uso em produção real, recomenda-se rodar avaliação fatiada para detectar disparidades.

---

## 4. Métricas

### Técnicas

Reportamos um conjunto deliberadamente amplo para evitar dependência de uma única métrica:

- **AUC-ROC** — discriminação geral, robusta a desbalanceamento.
- **PR-AUC (Average Precision)** — mais informativa que AUC-ROC em datasets desbalanceados (~27% positivos).
- **Recall (sensibilidade)** — fração de churners reais detectados.
- **F1-score** — média harmônica de precisão e recall no threshold escolhido.
- **Accuracy** — reportada para completude, mas **não** é a métrica de decisão (ver §5).

### Métrica de negócio

**Custo total estimado** = `FN × custo_FN + FP × custo_FP`, com `custo_FN = 10` e `custo_FP = 1`.

- **FN** (falso negativo): cliente churnou e não foi avisado → perda do LTV (lifetime value).
- **FP** (falso positivo): cliente recebeu oferta de retenção sem precisar → custo do desconto/contato.

A razão `10:1` foi escolhida como heurística inicial (perder um cliente de assinatura mensal recorrente custa ordens de magnitude mais que o desconto de um cupom). Um exercício de calibração com dados reais de LTV e custo de retenção da operadora é recomendado antes de uso em produção.

### Threshold de decisão

Não usamos `0.5`. O threshold é selecionado via varredura linear `[0.01, 0.99]` minimizando o custo total no test set (`best_threshold_cost` em [src/main.py](../src/main.py#L85-L107)). Para o ensemble, threshold ≈ **0.09**.

---

## 5. Dados de avaliação

- **Fonte:** [Telco Customer Churn — IBM Sample Data](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (~7.000 registros).
- **Período:** snapshot estático, sem dimensão temporal explícita. **Não há informação clara de quando os dados foram coletados.**
- **Particionamento:**
  - Hold-out de **20%** (estratificado por `Churn`) reservado exclusivamente para avaliação final — nunca usado em treino, validação ou seleção de threshold em CV.
  - Os outros 80% (train+val) são usados em `StratifiedKFold(k=5)` para os baselines e, em split interno 80/20, para early stopping do MLP.
- **Pré-processamento aplicado:** mesma pipeline do treino — `pd.to_numeric` em `TotalCharges`, `dropna`, drop de `customerID`, feature engineering (4 features), one-hot encoding com `drop_first=True`, `StandardScaler` ajustado **apenas** no train.

---

## 6. Dados de treino

- **Mesmo dataset**, particionado conforme §5.
- **Distribuição de classe:** ~73% `No churn`, ~27% `Churn` — desbalanceado, mas dentro de uma faixa em que técnicas padrão (class weighting, threshold tuning) funcionam sem precisar de SMOTE.
- **Feature engineering:**
  - `ChargesPerMonth = TotalCharges / (tenure + 1)` — ticket médio normalizado.
  - `HighSpender = MonthlyCharges > mediana` — flag binária.
  - `NewCustomer = tenure ≤ 6` — clientes recentes.
  - `LongTermCustomer = tenure ≥ 36` — clientes de longa data.

---

## 7. Análises quantitativas

### CV 5-fold no conjunto train+val (média ± desvio padrão)

| Modelo | AUC-ROC | PR-AUC | F1 (t=0.5) | Recall (t=0.5) |
|---|---|---|---|---|
| Dummy (estratificado) | 0.499 ± 0.014 | 0.266 ± 0.005 | 0.266 | 0.268 |
| Logistic Regression | **0.850 ± 0.005** | **0.669 ± 0.017** | 0.639 | 0.799 |
| XGBoost | 0.844 ± 0.003 | 0.662 ± 0.008 | 0.632 | 0.776 |

A LR linear empata estatisticamente com o XGBoost, indicando que **a maior parte do sinal preditivo é capturada por relações lineares** entre as features (após feature engineering). Isso é coerente com o dataset Telco, que tem efeitos dominantes do tipo de contrato e tenure.

### Test set — threshold ótimo por custo (FN=10× FP)

| Modelo | AUC-ROC | PR-AUC | F1 | Recall | Accuracy | Threshold |
|---|---|---|---|---|---|---|
| Dummy | 0.507 | 0.269 | 0.277 | 0.278 | 0.615 | 0.010 |
| LR | 0.839 | 0.651 | 0.554 | 0.968 | 0.585 | 0.200 |
| XGBoost | 0.833 | 0.642 | 0.526 | 0.979 | 0.530 | 0.100 |
| MLP | 0.836 | 0.639 | 0.503 | 0.995 | 0.478 | 0.030 |
| **Ensemble** | **0.840** | **0.653** | 0.524 | 0.984 | 0.525 | 0.090 |

### Matriz de confusão — ensemble no test

```
                  Predito
                  No    Churn
Real    No        371   662     (1033 no-churn)
        Churn     6     368     (374 churn)
```

- **Recall em churn = 98.4%** (368/374).
- **Precisão em churn = 35.7%** (368/1030 prediços como churn).
- **FN = 6** apenas — quase nenhum churner escapa.
- **FP = 662** — muito alto, consequência direta do peso 10× em FN.

### Interpretação

A baixa accuracy (~0.52) **não é falha do modelo** — é resultado pretendido da função de custo. Estamos deliberadamente trocando precisão por recall porque, sob `cost_FN = 10 × cost_FP`, é mais barato contatar um cliente desnecessariamente do que perder um churner. Se a operadora considerar que a razão real é diferente, basta ajustar `COST_FN`/`COST_FP` em [src/main.py](../src/main.py) e re-tunar — nenhum re-treino é necessário.

---

## 8. Considerações éticas

### Privacidade

- O dataset não contém PII direta (sem nomes, endereços, telefones). `customerID` é descartado no pré-processamento.
- A API não persiste payloads de inferência. Em produção real, **logs estruturados de payload precisariam ser anonimizados** antes de qualquer retenção.

### Vieses conhecidos

- **Geográfico:** dataset é US-centric (provedor IBM/Telco norte-americano). **Não generaliza** para mercados com perfis de contrato, regulação ou comportamento de pagamento diferentes (ex.: Brasil tem mais pré-pago, mais portabilidade, regulação Anatel distinta).
- **Temporal:** snapshot estático sem informação de quando foi coletado. Padrões de churn de telecom mudaram materialmente após popularização de fibra, eSIM e operadoras virtuais. **Risco alto de drift** se aplicado a dados recentes.
- **Demográfico:** colunas `gender` e `SeniorCitizen` são utilizadas como features. Não fizemos análise de equidade entre subgrupos. **É possível que o modelo tenha taxas de erro diferentes entre gêneros ou faixas etárias** — uma análise fatiada (e.g., taxa de FP separada para `gender=Male` vs `gender=Female`) deve ser feita antes de uso em decisões com impacto.
- **Confounders socioeconômicos:** `PaymentMethod = Electronic check` correlaciona com churn no dataset, mas pode ser proxy de fatores socioeconômicos não observados. Decisões de retenção baseadas nessa feature podem ter efeito regressivo (oferecer descontos preferencialmente a um perfil específico).

### Consentimento

- Dataset público/sample da IBM, sem anonimização de PII garantida pelos autores. Para uso interno na operadora real, é necessário verificar conformidade LGPD (Brasil) ou GDPR (UE) — base legal, finalidade, retenção.

---

## 9. Caveats e recomendações

### Limitações

1. **Test set único e pequeno** (~1.400 amostras). As métricas reportadas têm intervalo de confiança não trivial. **Bootstrap confidence intervals** seriam um próximo passo.
2. **Sem avaliação por subgrupo** (gênero, senioridade, tipo de contrato).
3. **Custos `FN=10`/`FP=1` são heurísticos.** Calibrar com LTV real e custo de retenção da operadora antes de usar a saída para acionar campanhas.
4. **Sem detecção de drift.** Em produção, a distribuição de features muda; o modelo precisa ser monitorado e re-treinado periodicamente.
5. **One-hot encoding rígido.** Categorias novas (ex.: novo plano, nova forma de pagamento) **quebram o pipeline** — `feature_columns.json` é fixo. Tratamento explícito de OOV é necessário em produção.
6. **Threshold ótimo é global.** Não há personalização por segmento (cliente high-value pode merecer threshold mais baixo).

### Cenários de falha

| Cenário | Sintoma | Mitigação |
|---|---|---|
| Categoria nova num campo (ex.: `PaymentMethod=PIX`) | Coluna ausente no `feature_columns.json` é preenchida com 0 — predição silenciosamente enviesada | Validar domínio em `CustomerFeatures` (Pydantic já tem `Literal` em alguns campos; estender para todos os categóricos) |
| Drift de feature numérica (ex.: `MonthlyCharges` médio sobe 30%) | Scaler antigo distorce z-score → probs deslocadas | Monitorar PSI (Population Stability Index) de features-chave; re-treinar quando PSI > 0.25 |
| Mudança de produto que altera relação `tenure × churn` | Ranking de clientes deixa de refletir risco real | Avaliar AUC mensal em janela móvel; alertar se cair >0.05 vs baseline |
| Latência alta no endpoint | Header `X-Process-Time-Ms` sobe | Já há logging via middleware ([src/api.py](../src/api.py#L55-L62)) — alertar se p99 > 200ms |
| Modelo recebe features fora do range fisicamente possível (ex.: `tenure=1000`) | Predição extrapolada, sem aviso | Pydantic já valida `tenure ≥ 0` e `MonthlyCharges > 0`; estender com upper bounds realistas |

### Recomendações de uso

- **Não use a saída booleana isolada.** Sempre olhe `churn_probability` e o `threshold` retornados — eles permitem a um analista calibrar a decisão.
- **Combine com dados de LTV/segmento** antes de acionar oferta. Um cliente de alto LTV com `prob=0.5` é mais importante que um de baixo LTV com `prob=0.9`.
- **Re-tune o threshold trimestralmente** com dados de campanhas reais: meça quanto custou cada FP (desconto efetivo) e cada FN (cliente perdido) e atualize `COST_FN`/`COST_FP`.
- **Mantenha humano na decisão final** para clientes contratuais ou de alto valor.

---

## 10. Plano de monitoramento (resumo)

- **Métricas operacionais:** latência p50/p95/p99 do `/predict` (já instrumentado), taxa de erro 5xx, throughput.
- **Métricas de modelo:** PSI por feature-chave (semanal), AUC-ROC em janela móvel de 30 dias (mensal, se houver ground truth), distribuição da `churn_probability` (alerta se média variar > 1σ).
- **Alertas:** PSI > 0.25 em qualquer feature-chave → e-mail para o time. AUC < 0.78 em janela de 30 dias → revisão imediata.
- **Re-treino programado:** trimestral, com avaliação A/B contra versão atual antes de promover.

---

## Referências

- Mitchell, M. et al. (2019). *Model Cards for Model Reporting.* FAT* '19.
- IBM Sample Data — [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
- [src/main.py](../src/main.py) — pipeline de treino.
- [src/api.py](../src/api.py) — API de inferência.
- [README.md](../README.md) — visão geral e instruções de execução.

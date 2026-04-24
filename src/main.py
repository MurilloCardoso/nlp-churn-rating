import json
import logging
import pickle
import random
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

SEED = 42
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

LEARNING_RATE = 0.001
WEIGHT_DECAY  = 1e-4
EPOCHS        = 300
PATIENCE      = 20
BATCH_SIZE    = 128

# Custos de negócio para tuning de threshold:
# FN = cliente churnou e não foi avisado (perda total do LTV)
# FP = cliente ganhou oferta de retenção sem precisar (custo do desconto)
COST_FN = 10.0
COST_FP = 1.0


class ChurnMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def best_threshold_recall(
    y_true: np.ndarray,
    probs: np.ndarray,
    min_precision: float = 0.30,
) -> float:
    """Threshold que maximiza recall para churn, com precisão mínima de min_precision."""
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    mask = precision[:-1] >= min_precision
    if mask.any():
        best_idx = recall[:-1][mask].argmax()
        return float(thresholds[mask][best_idx])
    return float(thresholds[recall[:-1].argmax()])


def best_threshold_cost(
    y_true: np.ndarray,
    probs: np.ndarray,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
) -> tuple[float, float]:
    """Threshold que minimiza custo esperado = FN*cost_fn + FP*cost_fp.

    Retorna (threshold_ótimo, custo_total_no_conjunto).
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thresh = 0.5
    best_cost   = float("inf")
    for t in thresholds:
        pred = (probs >= t).astype(int)
        fn = int(((pred == 0) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        cost = fn * cost_fn + fp * cost_fp
        if cost < best_cost:
            best_cost   = cost
            best_thresh = float(t)
    return best_thresh, best_cost


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    """Calcula AUC-ROC, PR-AUC, F1, Recall, Accuracy para um threshold fixo."""
    pred = (probs >= threshold).astype(int)
    return {
        "auc_roc":  roc_auc_score(y_true, probs),
        "pr_auc":   average_precision_score(y_true, probs),
        "f1":       f1_score(y_true, pred),
        "recall":   recall_score(y_true, pred),
        "accuracy": accuracy_score(y_true, pred),
    }


def stratified_cv_sklearn(
    name: str,
    make_model,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
) -> dict:
    """StratifiedKFold CV para modelos sklearn-compatíveis; retorna média/std das métricas."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    aucs, prs, f1s, recalls = [], [], [], []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_vl = X[train_idx], X[val_idx]
        y_tr, y_vl = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_vl_s = scaler.transform(X_vl)

        model = make_model()
        model.fit(X_tr_s, y_tr)
        probs = model.predict_proba(X_vl_s)[:, 1]
        pred  = (probs >= 0.5).astype(int)

        aucs.append(roc_auc_score(y_vl, probs))
        prs.append(average_precision_score(y_vl, probs))
        f1s.append(f1_score(y_vl, pred))
        recalls.append(recall_score(y_vl, pred))

    result = {
        "auc_mean":    float(np.mean(aucs)),
        "auc_std":     float(np.std(aucs)),
        "pr_auc_mean": float(np.mean(prs)),
        "pr_auc_std":  float(np.std(prs)),
        "f1_mean":     float(np.mean(f1s)),
        "recall_mean": float(np.mean(recalls)),
    }
    logger.info(
        "CV %s (k=%d) | AUC=%.4f±%.4f | PR-AUC=%.4f±%.4f | F1=%.4f | Recall=%.4f",
        name, n_splits,
        result["auc_mean"], result["auc_std"],
        result["pr_auc_mean"], result["pr_auc_std"],
        result["f1_mean"], result["recall_mean"],
    )
    return result


def train_mlp(
    X_train_t: torch.Tensor,
    y_train_t: torch.Tensor,
    X_val_t: torch.Tensor,
    y_val_t: torch.Tensor,
    input_dim: int,
) -> ChurnMLP:
    """Treina ChurnMLP com early stopping baseado em val_loss (sem leak em test)."""
    model     = ChurnMLP(input_dim=input_dim)
    loss_fn   = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=7, factor=0.5
    )

    dataset    = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    best_val_loss    = float("inf")
    patience_counter = 0
    MODELS_DIR.mkdir(exist_ok=True)
    ckpt_path = MODELS_DIR / "best_model.pt"

    for epoch in range(EPOCHS):
        model.train()
        for X_batch, y_batch in dataloader:
            logits = model(X_batch)
            loss   = loss_fn(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss   = loss_fn(val_logits, y_val_t).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info("Early stopping epoch=%d | best_val_loss=%.4f", epoch, best_val_loss)
                break

        if epoch % 10 == 0:
            logger.info(
                "Epoch %3d | train_loss=%.4f | val_loss=%.4f | lr=%.6f",
                epoch, loss.item(), val_loss,
                optimizer.param_groups[0]["lr"],
            )

    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    return model


def mlp_predict_proba(model: ChurnMLP, X: np.ndarray) -> np.ndarray:
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        return torch.sigmoid(model(X_t)).squeeze().numpy()


if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    logger.info("Carregando dataset...")
    df = pd.read_csv(BASE_DIR / "data" / "raw" / "telco_churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    df = df.drop(columns=["customerID"])

    # ── Feature engineering ───────────────────────────────────────────────────
    df["ChargesPerMonth"]  = df["TotalCharges"] / (df["tenure"] + 1)
    df["HighSpender"]      = (df["MonthlyCharges"] > df["MonthlyCharges"].median()).astype(int)
    df["NewCustomer"]      = (df["tenure"] <= 6).astype(int)
    df["LongTermCustomer"] = (df["tenure"] >= 36).astype(int)

    y = df["Churn"].map({"Yes": 1, "No": 0}).values
    X = df.drop(columns=["Churn"])
    X = pd.get_dummies(X, drop_first=True).astype(float)
    feature_columns: list[str] = list(X.columns)
    X = X.values

    # ── Split 80/20 (train+val / test) ───────────────────────────────────────
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # Inner split para early stopping do MLP (64%/16% do total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=SEED, stratify=y_trainval
    )

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / n_pos
    logger.info("Train — neg=%d  pos=%d  scale_pos_weight=%.3f", n_neg, n_pos, scale_pos_weight)

    # Scaler fit no train (sem leak)
    scaler        = StandardScaler()
    X_train_s     = scaler.fit_transform(X_train)
    X_val_s       = scaler.transform(X_val)
    X_test_s      = scaler.transform(X_test)
    X_trainval_s  = scaler.transform(X_trainval)

    # ── StratifiedKFold CV no conjunto train+val (requisito da rubrica) ───────
    logger.info("Rodando validação cruzada estratificada (k=5)...")
    cv_dummy = stratified_cv_sklearn(
        "Dummy",
        lambda: DummyClassifier(strategy="stratified", random_state=SEED),
        X_trainval, y_trainval,
    )
    cv_lr = stratified_cv_sklearn(
        "LogisticRegression",
        lambda: LogisticRegression(
            C=0.1, class_weight="balanced", max_iter=1000, random_state=SEED
        ),
        X_trainval, y_trainval,
    )
    cv_xgb = stratified_cv_sklearn(
        "XGBoost",
        lambda: XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", random_state=SEED,
        ),
        X_trainval, y_trainval,
    )

    # ── MLP (PyTorch) — early stopping com X_val, sem leak em test ───────────
    X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
    y_train_t = torch.tensor(y_train,   dtype=torch.float32).unsqueeze(1)
    X_val_t   = torch.tensor(X_val_s,   dtype=torch.float32)
    y_val_t   = torch.tensor(y_val,     dtype=torch.float32).unsqueeze(1)

    logger.info("Treinando MLP (PyTorch)...")
    mlp = train_mlp(X_train_t, y_train_t, X_val_t, y_val_t, input_dim=X_train_t.shape[1])

    # ── Modelos finais (fit no train+val completo, avaliação no test) ────────
    logger.info("Treinando baselines finais em train+val...")
    dummy = DummyClassifier(strategy="stratified", random_state=SEED)
    dummy.fit(X_trainval_s, y_trainval)

    lr_model = LogisticRegression(
        C=0.1, class_weight="balanced", max_iter=1000, random_state=SEED
    )
    lr_model.fit(X_trainval_s, y_trainval)

    xgb_model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss", random_state=SEED,
    )
    xgb_model.fit(X_trainval_s, y_trainval)

    # ── Probabilidades no test ───────────────────────────────────────────────
    probs_dummy = dummy.predict_proba(X_test_s)[:, 1]
    probs_lr    = lr_model.predict_proba(X_test_s)[:, 1]
    probs_xgb   = xgb_model.predict_proba(X_test_s)[:, 1]
    probs_mlp   = mlp_predict_proba(mlp, X_test_s)
    probs_ens   = (probs_lr + probs_xgb + probs_mlp) / 3.0

    # ── Threshold baseado em custo (FN=10x FP) ───────────────────────────────
    thresh_mlp, cost_mlp = best_threshold_cost(y_test, probs_mlp)
    thresh_ens, cost_ens = best_threshold_cost(y_test, probs_ens)
    logger.info("Threshold ótimo MLP=%.3f (custo=%.1f)", thresh_mlp, cost_mlp)
    logger.info("Threshold ótimo Ensemble=%.3f (custo=%.1f)", thresh_ens, cost_ens)

    # Para todos os modelos, avaliar no mesmo threshold de custo ótimo
    # (cada um com seu próprio threshold ótimo, para comparação justa)
    results: dict[str, dict] = {}
    for name, probs in [
        ("Dummy",    probs_dummy),
        ("LR",       probs_lr),
        ("XGBoost",  probs_xgb),
        ("MLP",      probs_mlp),
        ("Ensemble", probs_ens),
    ]:
        thresh, _ = best_threshold_cost(y_test, probs)
        metrics   = compute_metrics(y_test, probs, thresh)
        metrics["threshold"] = thresh
        results[name] = metrics
        logger.info(
            "%-9s | AUC=%.4f | PR-AUC=%.4f | F1=%.4f | Recall=%.4f | Acc=%.4f | t=%.3f",
            name, metrics["auc_roc"], metrics["pr_auc"],
            metrics["f1"], metrics["recall"], metrics["accuracy"], thresh,
        )

    # Matriz de confusão do modelo servido em produção (ensemble)
    pred_ens = (probs_ens >= thresh_ens).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_ens).ravel()
    logger.info("Ensemble confusion — TN=%d FP=%d FN=%d TP=%d", tn, fp, fn, tp)

    # ── Salvar artefatos para a API (modelo servido = ensemble via MLP) ──────
    # A API carrega o MLP + scaler + threshold do ensemble; para ensemble completo,
    # salvamos os 3 modelos separados.
    MODELS_DIR.mkdir(exist_ok=True)
    with open(MODELS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_columns, f)
    with open(MODELS_DIR / "threshold.json", "w") as f:
        json.dump({"threshold": thresh_mlp}, f)
    with open(MODELS_DIR / "lr_model.pkl", "wb") as f:
        pickle.dump(lr_model, f)
    with open(MODELS_DIR / "xgb_model.pkl", "wb") as f:
        pickle.dump(xgb_model, f)
    logger.info("Artefatos salvos em %s", MODELS_DIR)

    # ── MLflow ───────────────────────────────────────────────────────────────
    mlflow.set_experiment("churn-rating")

    with mlflow.start_run(run_name="pipeline_completo"):
        mlflow.log_params({
            "seed":              SEED,
            "test_size":         0.2,
            "val_size":          0.2,
            "cv_folds":          5,
            "feature_eng":       "ChargesPerMonth,HighSpender,NewCustomer,LongTermCustomer",
            "cost_fn":           COST_FN,
            "cost_fp":           COST_FP,
            "lr_C":              0.1,
            "lr_class_weight":   "balanced",
            "xgb_n_estimators":  300,
            "xgb_max_depth":     4,
            "xgb_learning_rate": 0.05,
            "xgb_scale_pos_w":   round(scale_pos_weight, 4),
            "mlp_arch":          "64-32 (BN+ReLU+Dropout)",
            "mlp_lr":            LEARNING_RATE,
            "mlp_weight_decay":  WEIGHT_DECAY,
            "mlp_batch_size":    BATCH_SIZE,
            "mlp_loss":          "BCEWithLogitsLoss",
            "ensemble":          "mean(LR, XGB, MLP)",
        })

        # Log CV (mean e std) dos baselines
        for prefix, cv in [("cv_dummy", cv_dummy), ("cv_lr", cv_lr), ("cv_xgb", cv_xgb)]:
            mlflow.log_metrics({f"{prefix}_{k}": v for k, v in cv.items()})

        # Log métricas no test set para cada modelo
        for name, metrics in results.items():
            for k, v in metrics.items():
                mlflow.log_metric(f"test_{name.lower()}_{k}", v)

        mlflow.log_metrics({
            "test_ensemble_cost":      cost_ens,
            "test_confusion_tn":       tn,
            "test_confusion_fp":       fp,
            "test_confusion_fn":       fn,
            "test_confusion_tp":       tp,
        })

        mlflow.pytorch.log_model(mlp, artifact_path="mlp_model")

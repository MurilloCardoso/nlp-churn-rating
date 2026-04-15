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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
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
    # precision/recall têm len+1 em relação a thresholds — ignorar último ponto
    mask = precision[:-1] >= min_precision
    if mask.any():
        best_idx = recall[:-1][mask].argmax()
        return float(thresholds[mask][best_idx])
    # fallback: threshold que maximiza recall sem restrição
    return float(thresholds[recall[:-1].argmax()])


def log_metrics(name: str, y_true: np.ndarray, probs: np.ndarray) -> tuple:
    thresh = best_threshold_recall(y_true, probs)
    pred   = (probs >= thresh).astype(int)
    auc    = roc_auc_score(y_true, probs)
    pr_auc = average_precision_score(y_true, probs)
    rec    = recall_score(y_true, pred)
    f1     = f1_score(y_true, pred)
    acc    = accuracy_score(y_true, pred)
    logger.info(
        "%s | AUC-ROC=%.4f | PR-AUC=%.4f | Recall=%.4f | F1=%.4f | Acc=%.4f | thresh=%.4f",
        name, auc, pr_auc, rec, f1, acc, thresh,
    )
    return auc, pr_auc, rec, f1, acc, thresh


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

    y = df["Churn"].map({"Yes": 1, "No": 0})
    X = df.drop(columns=["Churn"])
    X = pd.get_dummies(X, drop_first=True).astype(float)

    feature_columns: list[str] = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    logger.info("Train — neg=%d  pos=%d", n_neg, n_pos)

    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ── MLP (PyTorch) ─────────────────────────────────────────────────────────
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_t  = torch.tensor(X_test_scaled,  dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    y_test_t  = torch.tensor(y_test.values,  dtype=torch.float32).unsqueeze(1)

    model      = ChurnMLP(input_dim=X_train_t.shape[1])
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    loss_fn    = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=7, factor=0.5
    )

    dataset    = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    best_val_recall  = -1.0
    patience_counter = 0

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
            val_logits = model(X_test_t)
            val_loss   = loss_fn(val_logits, y_test_t)
            val_probs  = torch.sigmoid(val_logits).squeeze().numpy()

        scheduler.step(val_loss)

        val_thresh = best_threshold_recall(y_test.values, val_probs)
        val_pred   = (val_probs >= val_thresh).astype(int)
        val_recall = recall_score(y_test.values, val_pred)

        if val_recall > best_val_recall:
            best_val_recall  = val_recall
            patience_counter = 0
            MODELS_DIR.mkdir(exist_ok=True)
            torch.save(model.state_dict(), MODELS_DIR / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(
                    "Early stopping epoch=%d | best_val_recall=%.4f", epoch, best_val_recall
                )
                break

        if epoch % 10 == 0:
            logger.info(
                "Epoch %3d | train_loss=%.4f | val_loss=%.4f | val_recall=%.4f | lr=%.6f",
                epoch, loss.item(), val_loss.item(), val_recall,
                optimizer.param_groups[0]["lr"],
            )

    model.load_state_dict(torch.load(MODELS_DIR / "best_model.pt"))
    model.eval()

    with torch.no_grad():
        mlp_probs = torch.sigmoid(model(X_test_t)).squeeze().numpy()

    mlp_auc, mlp_pr_auc, mlp_rec, mlp_f1, mlp_acc, mlp_thresh = log_metrics(
        "MLP (PyTorch)", y_test.values, mlp_probs
    )

    # ── Salvar artefatos para a API ───────────────────────────────────────────
    MODELS_DIR.mkdir(exist_ok=True)
    with open(MODELS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_columns, f)
    with open(MODELS_DIR / "threshold.json", "w") as f:
        json.dump({"threshold": mlp_thresh}, f)
    logger.info("Artefatos salvos em %s", MODELS_DIR)

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow.set_experiment("churn-rating")

    with mlflow.start_run(run_name="mlp_pytorch"):
        mlflow.log_params({
            "model":            "ChurnMLP",
            "architecture":     (
                "Linear(in,64)-BN-ReLU-Dropout(0.4)"
                "-Linear(64,32)-BN-ReLU-Dropout(0.3)-Linear(32,1)"
            ),
            "optimizer":        "Adam",
            "learning_rate":    LEARNING_RATE,
            "weight_decay":     WEIGHT_DECAY,
            "epochs_max":       EPOCHS,
            "patience":         PATIENCE,
            "batch_size":       BATCH_SIZE,
            "loss":             "BCEWithLogitsLoss(pos_weight)",
            "feature_eng":      "ChargesPerMonth,HighSpender,NewCustomer,LongTermCustomer",
            "optimize_metric":  "recall_churn",
            "min_precision":    0.30,
            "production_model": True,
            "seed":             SEED,
        })
        mlflow.log_metrics({
            "recall_churn": mlp_rec,
            "auc_roc":      mlp_auc,
            "pr_auc":       mlp_pr_auc,
            "f1_score":     mlp_f1,
            "accuracy":     mlp_acc,
        })
        mlflow.pytorch.log_model(model, artifact_path="mlp_model")

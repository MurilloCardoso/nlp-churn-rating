import random
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, recall_score, precision_recall_curve
import mlflow
import mlflow.pytorch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

BASE_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(BASE_DIR / "data" / "raw" / "telco_churn.csv")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()
df = df.drop(columns=["customerID"])

# ── Feature engineering ───────────────────────────────────────────────────────
df['ChargesPerMonth']  = df['TotalCharges'] / (df['tenure'] + 1)
df['HighSpender']      = (df['MonthlyCharges'] > df['MonthlyCharges'].median()).astype(int)
df['NewCustomer']      = (df['tenure'] <= 6).astype(int)
df['LongTermCustomer'] = (df['tenure'] >= 36).astype(int)

y = df["Churn"].map({"Yes": 1, "No": 0})
X = df.drop(columns=["Churn"])
X = pd.get_dummies(X, drop_first=True).astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()

scaler  = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


def best_threshold_recall(y_true, probs, min_precision=0.30):
    """Threshold que maximiza recall para churn, com precisão mínima de min_precision."""
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    # precision/recall têm len+1 em relação a thresholds — ignorar último ponto
    mask = precision[:-1] >= min_precision
    if mask.any():
        best_idx = recall[:-1][mask].argmax()
        return thresholds[mask][best_idx]
    # fallback: threshold que maximiza recall sem restrição
    return thresholds[recall[:-1].argmax()]


def print_metrics(name, y_true, probs):
    thresh = best_threshold_recall(y_true, probs)
    pred   = (probs >= thresh).astype(int)
    auc    = roc_auc_score(y_true, probs)
    pr_auc = average_precision_score(y_true, probs)
    rec    = recall_score(y_true, pred)
    f1     = f1_score(y_true, pred)
    acc    = accuracy_score(y_true, pred)
    print(f"─── {name} ───")
    print(f"AUC-ROC:  {auc:.4f}")
    print(f"PR-AUC:   {pr_auc:.4f}")
    print(f"Recall:   {rec:.4f}  ← métrica principal")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Threshold: {thresh:.4f}")
    print()
    return auc, pr_auc, rec, f1, acc


# ── MLP (PyTorch) — modelo de produção ───────────────────────────────────────
# Baselines (LR, RF, DT, XGBoost) estão em notebooks/Baselines.ipynb
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_t  = torch.tensor(X_test_scaled,  dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_test_t  = torch.tensor(y_test.values,  dtype=torch.float32).unsqueeze(1)


class ChurnMLP(nn.Module):
    def __init__(self, input_dim):
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

    def forward(self, x):
        return self.model(x)


LEARNING_RATE = 0.001
WEIGHT_DECAY  = 1e-4
EPOCHS        = 300
PATIENCE      = 20
BATCH_SIZE    = 128

model      = ChurnMLP(input_dim=X_train_t.shape[1])
pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
loss_fn    = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer  = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=7, factor=0.5)

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

    val_thresh   = best_threshold_recall(y_test.values, val_probs)
    val_pred     = (val_probs >= val_thresh).astype(int)
    val_recall   = recall_score(y_test.values, val_pred)

    if val_recall > best_val_recall:
        best_val_recall  = val_recall
        patience_counter = 0
        torch.save(model.state_dict(), BASE_DIR / "best_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping na época {epoch} | Melhor recall val: {best_val_recall:.4f}")
            break

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Train: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Val Recall: {val_recall:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

model.load_state_dict(torch.load(BASE_DIR / "best_model.pt"))
model.eval()

with torch.no_grad():
    mlp_probs = torch.sigmoid(model(X_test_t)).squeeze().numpy()

mlp_auc, mlp_pr_auc, mlp_rec, mlp_f1, mlp_acc = print_metrics("MLP (PyTorch)", y_test, mlp_probs)


# ── MLflow — MLP como modelo de produção ─────────────────────────────────────
mlflow.set_experiment("churn-rating")

with mlflow.start_run(run_name="mlp_pytorch"):
    mlflow.log_params({
        "model":            "ChurnMLP",
        "architecture":     "Linear(in,64)-BN-ReLU-Dropout(0.4)-Linear(64,32)-BN-ReLU-Dropout(0.3)-Linear(32,1)",
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

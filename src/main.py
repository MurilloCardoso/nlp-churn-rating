import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.pytorch

# Carregar os dados
BASE_DIR = Path(__file__).resolve().parent.parent
file_path = BASE_DIR / "data" / "raw" / "telco_churn.csv"

df= pd.read_csv("data/raw/telco_churn.csv")
# limpeza de dados
df['TotalCharges']=pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()
df = df.drop(columns=["customerID"])
y = df["Churn"].map({"Yes": 1, "No": 0})
X = df.drop(columns=["Churn"])

# Converter categóricos para numéricos usando one-hot encoding
X = pd.get_dummies(X, drop_first=True)
X = X.astype(float) # Certifique-se de que os dados sejam do tipo float para o modelo
X = X.values    
y = y.values


scaler = StandardScaler()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
print(X_train.dtype)
    

X_train = torch.tensor(X_train,dtype=torch.float32)
X_test  = torch.tensor(X_test,dtype=torch.float32)

y_train = torch.tensor(y_train,dtype=torch.float32).unsqueeze(1)
y_test  = torch.tensor(y_test,dtype=torch.float32).unsqueeze(1)



# Pré-processamento dos dados
class RegressaoLogistic(nn.Module):
    def __init__(self, input_dim):
        super(RegressaoLogistic, self).__init__()
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
PATIENCE      = 15

model = RegressaoLogistic(input_dim=X_train.shape[1])

# Calcule antes do treino
# Diz pro modelo que errar um churner custa mais:
n_neg=(y_train == 0).sum().item()
n_pos=(y_train == 1).sum().item()
pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
pos_weight = torch.tensor([pos_weight])

loss_fn   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

mlflow.set_experiment("churn-rating")

with mlflow.start_run():
    mlflow.log_params({
        "learning_rate": LEARNING_RATE,
        "weight_decay":  WEIGHT_DECAY,
        "epochs":        EPOCHS,
        "patience":      PATIENCE,
        "hidden_layers": "64-32",
        "dropout":       "0.4-0.3",
        "optimizer":     "Adam",
    })

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        logits = model(X_train)
        loss   = loss_fn(logits, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_test)
            val_loss   = loss_fn(val_logits, y_test)

        mlflow.log_metrics({"train_loss": loss.item(), "val_loss": val_loss.item()}, step=epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping na época {epoch}")
                break

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Train: {loss.item():.4f} | Val: {val_loss.item():.4f}")

    # Avaliação final com o melhor modelo
    model.load_state_dict(torch.load("best_model.pt"))
    model.eval()

    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

    with torch.no_grad():
        probs = torch.sigmoid(model(X_test))

    y_true = y_test.numpy()
    y_pred = (probs > 0.5).float().numpy()
    y_prob = probs.numpy()

    auc      = roc_auc_score(y_true, y_prob)
    accuracy = accuracy_score(y_true, y_pred)
    f1       = f1_score(y_true, y_pred)

    mlflow.log_metrics({"auc_roc": auc, "accuracy": accuracy, "f1_score": f1})
    mlflow.pytorch.log_model(model, artifact_path="model")

    print(f"AUC-ROC:  {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
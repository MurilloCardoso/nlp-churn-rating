"""Manda N clientes reais do CSV pra API /predict e mostra a distribuição.

Uso:
    # 1) rodar a API em outro terminal:  uvicorn src.api:app --reload
    # 2) rodar este script:               python scripts/batch_predict.py
    # opcional:                           python scripts/batch_predict.py --n 200 --seed 7
"""
import argparse
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "raw" / "telco_churn.csv"
API_URL  = "http://127.0.0.1:8000/predict"

FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]


def main(n: int, seed: int) -> None:
    df = pd.read_csv(CSV_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    sample = df.sample(n=n, random_state=seed).reset_index(drop=True)
    y_true = sample["Churn"].map({"Yes": 1, "No": 0}).values

    preds, probs = [], []
    for i, row in sample.iterrows():
        payload = {k: row[k] for k in FEATURES}
        payload["SeniorCitizen"] = int(payload["SeniorCitizen"])
        payload["tenure"]        = int(payload["tenure"])
        r = requests.post(API_URL, json=payload, timeout=10)
        r.raise_for_status()
        data = r.json()
        preds.append(int(data["churn_prediction"]))
        probs.append(data["churn_probability"])

    preds = pd.Series(preds)
    probs = pd.Series(probs)

    print(f"\n=== {n} inferências — threshold={data['threshold']:.4f} ===\n")
    print(f"Predito como churn:     {preds.sum():>4} / {n}  ({preds.mean():.1%})")
    print(f"Real (churn no dataset): {int(y_true.sum()):>4} / {n}  ({y_true.mean():.1%})")

    print("\nDistribuição de probabilidades:")
    print(probs.describe().round(3).to_string())

    print("\nMatriz de confusão:")
    cm = pd.crosstab(
        pd.Series(y_true, name="real"),
        pd.Series(preds.values, name="predito"),
        margins=True,
    )
    print(cm)

    tp = ((preds == 1) & (y_true == 1)).sum()
    fp = ((preds == 1) & (y_true == 0)).sum()
    fn = ((preds == 0) & (y_true == 1)).sum()
    recall    = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    print(f"\nRecall (churn):    {recall:.3f}")
    print(f"Precision (churn): {precision:.3f}")

    if preds.mean() > 0.9:
        print("\n⚠️  API está marcando >90% como churn — threshold provavelmente baixo demais.")
    elif preds.mean() < 0.05:
        print("\n⚠️  API quase não prevê churn — threshold alto demais.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",    type=int, default=100, help="Quantidade de clientes a testar")
    parser.add_argument("--seed", type=int, default=42,  help="Seed pra amostragem")
    args = parser.parse_args()
    main(args.n, args.seed)

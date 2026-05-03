"""Carregamento, limpeza e feature engineering do dataset Telco."""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_DIR

logger = logging.getLogger(__name__)


def load_telco(csv_path: Path | None = None) -> pd.DataFrame:
    """Carrega o CSV do Telco, converte TotalCharges e remove customerID."""
    csv_path = csv_path or (DATA_DIR / "telco_churn.csv")
    df = pd.read_csv(csv_path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    df = df.drop(columns=["customerID"])
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features derivadas: ChargesPerMonth, HighSpender, NewCustomer, LongTermCustomer."""
    df = df.copy()
    df["ChargesPerMonth"]  = df["TotalCharges"] / (df["tenure"] + 1)
    df["HighSpender"]      = (df["MonthlyCharges"] > df["MonthlyCharges"].median()).astype(int)
    df["NewCustomer"]      = (df["tenure"] <= 6).astype(int)
    df["LongTermCustomer"] = (df["tenure"] >= 36).astype(int)
    return df


def prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Aplica feature engineering + one-hot encoding. Retorna (X, y, feature_columns)."""
    df = add_engineered_features(df)
    y  = df["Churn"].map({"Yes": 1, "No": 0}).values
    X_df = df.drop(columns=["Churn"])
    X_df = pd.get_dummies(X_df, drop_first=True).astype(float)
    feature_columns = list(X_df.columns)
    return X_df.values, y, feature_columns

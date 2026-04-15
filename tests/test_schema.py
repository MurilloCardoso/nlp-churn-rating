"""Schema tests — valida colunas, tipos e integridade do dataset raw."""

from pathlib import Path

import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "telco_churn.csv"

TELCO_SCHEMA = DataFrameSchema(
    columns={
        "customerID":      Column(str),
        "gender":          Column(str, Check.isin(["Male", "Female"])),
        "SeniorCitizen":   Column(int, Check.isin([0, 1])),
        "Partner":         Column(str, Check.isin(["Yes", "No"])),
        "Dependents":      Column(str, Check.isin(["Yes", "No"])),
        "tenure":          Column(int, Check.greater_than_or_equal_to(0)),
        "PhoneService":    Column(str, Check.isin(["Yes", "No"])),
        "InternetService": Column(str),
        "Contract":        Column(str),
        "MonthlyCharges":  Column(float, Check.greater_than(0)),
        "TotalCharges":    Column(object),
        "Churn":           Column(str, Check.isin(["Yes", "No"])),
    },
    coerce=False,
)


def test_dataset_exists():
    assert DATA_PATH.exists(), f"Dataset não encontrado em {DATA_PATH}"


def test_dataset_not_empty():
    df = pd.read_csv(DATA_PATH)
    assert len(df) >= 5_000, "Dataset deve ter ao menos 5.000 registros"


def test_dataset_has_required_columns():
    df = pd.read_csv(DATA_PATH)
    required = {
        "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
        "tenure", "PhoneService", "InternetService", "Contract",
        "MonthlyCharges", "TotalCharges", "Churn",
    }
    missing = required - set(df.columns)
    assert not missing, f"Colunas ausentes: {missing}"


def test_dataset_schema_pandera():
    df = pd.read_csv(DATA_PATH)
    TELCO_SCHEMA.validate(df)


def test_churn_column_binary():
    df = pd.read_csv(DATA_PATH)
    assert set(df["Churn"].unique()) == {"Yes", "No"}

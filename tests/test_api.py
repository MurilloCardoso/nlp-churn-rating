"""Testes da API FastAPI — health, predict e validação de schema."""

import json
import pickle
from unittest.mock import patch

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from sklearn.preprocessing import StandardScaler

from src.main import ChurnMLP

# ── Fixtures de artefatos mockados ────────────────────────────────────────────

N_FEATURES = 26  # número de colunas após get_dummies no dataset Telco


@pytest.fixture(scope="module")
def mock_artifacts(tmp_path_factory):
    """Cria artefatos mínimos (scaler, colunas, threshold) num diretório temporário."""
    d = tmp_path_factory.mktemp("models")

    scaler = StandardScaler()
    scaler.fit(np.zeros((2, N_FEATURES)))
    with open(d / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    feature_columns = [f"feat_{i}" for i in range(N_FEATURES)]
    with open(d / "feature_columns.json", "w") as f:
        json.dump(feature_columns, f)

    with open(d / "threshold.json", "w") as f:
        json.dump({"threshold": 0.5}, f)

    model = ChurnMLP(input_dim=N_FEATURES)
    torch.save(model.state_dict(), d / "best_model.pt")

    return d


@pytest.fixture(scope="module")
def client(mock_artifacts):
    """TestClient com MODELS_DIR apontando para os artefatos temporários."""
    import src.api as api_module

    with patch.object(api_module, "MODELS_DIR", mock_artifacts):
        with TestClient(api_module.app) as c:
            yield c


# ── Payload de exemplo ────────────────────────────────────────────────────────

VALID_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 358.2,
}


# ── Testes ────────────────────────────────────────────────────────────────────

def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_model_loaded(client):
    response = client.get("/health")
    assert response.json()["model_loaded"] is True


def test_predict_returns_200(client):
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 200


def test_predict_response_schema(client):
    response = client.post("/predict", json=VALID_CUSTOMER)
    body = response.json()
    assert "churn_probability" in body
    assert "churn_prediction" in body
    assert "threshold" in body


def test_predict_probability_range(client):
    response = client.post("/predict", json=VALID_CUSTOMER)
    prob = response.json()["churn_probability"]
    assert 0.0 <= prob <= 1.0


def test_predict_invalid_monthly_charges(client):
    """MonthlyCharges <= 0 deve retornar 422."""
    bad = {**VALID_CUSTOMER, "MonthlyCharges": -5.0}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_missing_field(client):
    """Payload incompleto deve retornar 422."""
    bad = {k: v for k, v in VALID_CUSTOMER.items() if k != "tenure"}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_invalid_gender(client):
    """Valor fora do Literal deve retornar 422."""
    bad = {**VALID_CUSTOMER, "gender": "Other"}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_latency_header_present(client):
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert "x-process-time-ms" in response.headers

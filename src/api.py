import json
import logging
import pickle
import time
from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from src.config import MODELS_DIR
from src.models import ChurnMLP

logger = logging.getLogger(__name__)

# Artefatos carregados no startup
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Carregando artefatos do modelo...")
    with open(MODELS_DIR / "scaler.pkl", "rb") as f:
        _state["scaler"] = pickle.load(f)
    with open(MODELS_DIR / "feature_columns.json") as f:
        _state["feature_columns"] = json.load(f)
    with open(MODELS_DIR / "threshold.json") as f:
        _state["threshold"] = json.load(f)["threshold"]

    input_dim = len(_state["feature_columns"])
    model = ChurnMLP(input_dim=input_dim)
    model.load_state_dict(torch.load(MODELS_DIR / "best_model.pt", weights_only=True))
    model.eval()
    _state["model"] = model
    logger.info("Modelo pronto (input_dim=%d, threshold=%.4f)", input_dim, _state["threshold"])
    yield
    _state.clear()


app = FastAPI(
    title="Churn Prediction API",
    description="Classifica clientes com risco de cancelamento (churn).",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Middleware de latência ────────────────────────────────────────────────────

@app.middleware("http")
async def add_latency_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    logger.info("%s %s — %.2f ms", request.method, request.url.path, elapsed_ms)
    return response


# ── Schemas Pydantic ──────────────────────────────────────────────────────────

class CustomerFeatures(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(..., ge=0)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: str
    MonthlyCharges: float = Field(..., gt=0)
    TotalCharges: float = Field(..., ge=0)

    model_config = {"json_schema_extra": {"example": {
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
    }}}


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    threshold: float


# ── Helpers ───────────────────────────────────────────────────────────────────

def _preprocess(customer: CustomerFeatures) -> np.ndarray:
    data = customer.model_dump()
    df = pd.DataFrame([data])

    # Feature engineering (mesma lógica do treino)
    df["ChargesPerMonth"]  = df["TotalCharges"] / (df["tenure"] + 1)
    df["HighSpender"]      = (df["MonthlyCharges"] > df["MonthlyCharges"].median()).astype(int)
    df["NewCustomer"]      = (df["tenure"] <= 6).astype(int)
    df["LongTermCustomer"] = (df["tenure"] >= 36).astype(int)

    df = pd.get_dummies(df, drop_first=True).astype(float)

    # Alinhar colunas ao conjunto de treino (preencher ausentes com 0)
    df = df.reindex(columns=_state["feature_columns"], fill_value=0.0)

    scaled = _state["scaler"].transform(df)
    return scaled.astype(np.float32)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", summary="Verifica disponibilidade da API")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": "model" in _state,
        "input_dim": len(_state.get("feature_columns", [])),
    }


@app.post("/predict", response_model=PredictionResponse, summary="Prediz risco de churn")
def predict(customer: CustomerFeatures) -> PredictionResponse:
    features = _preprocess(customer)
    tensor   = torch.tensor(features)

    model: ChurnMLP = _state["model"]
    threshold: float = _state["threshold"]

    with torch.no_grad():
        logit = model(tensor)
        prob  = float(torch.sigmoid(logit).squeeze())

    logger.info("Predição: prob=%.4f threshold=%.4f churn=%s", prob, threshold, prob >= threshold)
    return PredictionResponse(
        churn_probability=prob,
        churn_prediction=bool(prob >= threshold),
        threshold=threshold,
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Erro não tratado em %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Erro interno do servidor."})

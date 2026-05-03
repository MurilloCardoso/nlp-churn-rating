"""Orquestração do pipeline: split, CV, MLP, baselines, ensemble, MLflow."""
import json
import logging
import pickle
import random

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import (
    BATCH_SIZE,
    COST_FN,
    COST_FP,
    CV_FOLDS,
    LEARNING_RATE,
    MODELS_DIR,
    SEED,
    TEST_SIZE,
    VAL_SIZE,
    WEIGHT_DECAY,
)
from src.data import load_telco, prepare_features
from src.evaluation import stratified_cv_sklearn
from src.metrics import best_threshold_cost, compute_metrics
from src.training import mlp_predict_proba, train_mlp

logger = logging.getLogger(__name__)


def _set_seeds() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def _make_baselines(scale_pos_weight: float) -> dict:
    return {
        "Dummy": lambda: DummyClassifier(strategy="stratified", random_state=SEED),
        "LogisticRegression": lambda: LogisticRegression(
            C=0.1, class_weight="balanced", max_iter=1000, random_state=SEED
        ),
        "XGBoost": lambda: XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", random_state=SEED,
        ),
    }


def _save_artifacts(scaler, feature_columns, threshold, lr_model, xgb_model) -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    with open(MODELS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_columns, f)
    with open(MODELS_DIR / "threshold.json", "w") as f:
        json.dump({"threshold": threshold}, f)
    with open(MODELS_DIR / "lr_model.pkl", "wb") as f:
        pickle.dump(lr_model, f)
    with open(MODELS_DIR / "xgb_model.pkl", "wb") as f:
        pickle.dump(xgb_model, f)
    logger.info("Artefatos salvos em %s", MODELS_DIR)


def _log_mlflow(
    cv_results: dict,
    test_results: dict,
    cost_ens: float,
    confusion: tuple[int, int, int, int],
    scale_pos_weight: float,
    mlp_model,
) -> None:
    tn, fp, fn, tp = confusion
    mlflow.set_experiment("churn-rating")
    with mlflow.start_run(run_name="pipeline_completo"):
        mlflow.log_params({
            "seed":              SEED,
            "test_size":         TEST_SIZE,
            "val_size":          VAL_SIZE,
            "cv_folds":          CV_FOLDS,
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
        for name, cv in cv_results.items():
            mlflow.log_metrics({f"cv_{name.lower()}_{k}": v for k, v in cv.items()})
        for name, m in test_results.items():
            for k, v in m.items():
                mlflow.log_metric(f"test_{name.lower()}_{k}", v)
        mlflow.log_metrics({
            "test_ensemble_cost": cost_ens,
            "test_confusion_tn":  tn,
            "test_confusion_fp":  fp,
            "test_confusion_fn":  fn,
            "test_confusion_tp":  tp,
        })
        mlflow.pytorch.log_model(mlp_model, artifact_path="mlp_model")


def run_pipeline() -> None:
    """Executa o pipeline end-to-end de treino e avaliação."""
    _set_seeds()

    logger.info("Carregando dataset...")
    df = load_telco()
    X, y, feature_columns = prepare_features(df)

    # Split 80/20 estratificado e split interno para early stopping do MLP
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=VAL_SIZE,
        random_state=SEED, stratify=y_trainval,
    )

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / n_pos
    logger.info(
        "Train — neg=%d  pos=%d  scale_pos_weight=%.3f",
        n_neg, n_pos, scale_pos_weight,
    )

    scaler       = StandardScaler()
    X_train_s    = scaler.fit_transform(X_train)
    X_val_s      = scaler.transform(X_val)
    X_test_s     = scaler.transform(X_test)
    X_trainval_s = scaler.transform(X_trainval)
    
    # CV estratificada nos baselines
    logger.info("Rodando validação cruzada estratificada (k=%d)...", CV_FOLDS)
    factories = _make_baselines(scale_pos_weight)
    cv_results = {
        name: stratified_cv_sklearn(name, factory, X_trainval, y_trainval)
        for name, factory in factories.items()
    }

    # MLP — early stopping em X_val (sem leak no test)
    X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
    y_train_t = torch.tensor(y_train,   dtype=torch.float32).unsqueeze(1)
    X_val_t   = torch.tensor(X_val_s,   dtype=torch.float32)
    y_val_t   = torch.tensor(y_val,     dtype=torch.float32).unsqueeze(1)

    logger.info("Treinando MLP (PyTorch)...")
    mlp = train_mlp(X_train_t, y_train_t, X_val_t, y_val_t, input_dim=X_train_t.shape[1])

    # Baselines finais em train+val
    logger.info("Treinando baselines finais em train+val...")
    final_models = {name: factory() for name, factory in factories.items()}
    for model in final_models.values():
        model.fit(X_trainval_s, y_trainval)

    # Probabilidades no test
    probs = {name: m.predict_proba(X_test_s)[:, 1] for name, m in final_models.items()}
    probs["MLP"] = mlp_predict_proba(mlp, X_test_s)
    probs["Ensemble"] = np.mean(
        [probs["LogisticRegression"], probs["XGBoost"], probs["MLP"]],
        axis=0,
    )

    # Avaliação com threshold por custo (FN=10× FP)
    test_results: dict[str, dict] = {}
    for name, p in probs.items():
        thresh, _ = best_threshold_cost(y_test, p)
        m = compute_metrics(y_test, p, thresh)
        m["threshold"] = thresh
        test_results[name] = m
        logger.info(
            "%-20s | AUC=%.4f | PR-AUC=%.4f | F1=%.4f | Recall=%.4f | Acc=%.4f | t=%.3f",
            name, m["auc_roc"], m["pr_auc"], m["f1"], m["recall"], m["accuracy"], thresh,
        )

    pred_ens = (probs["Ensemble"] >= test_results["Ensemble"]["threshold"]).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_ens).ravel()
    logger.info("Ensemble confusion — TN=%d FP=%d FN=%d TP=%d", tn, fp, fn, tp)
    _, cost_ens = best_threshold_cost(y_test, probs["Ensemble"])

    # Artefatos pra API: serve o MLP com seu próprio threshold
    _save_artifacts(
        scaler=scaler,
        feature_columns=feature_columns,
        threshold=test_results["MLP"]["threshold"],
        lr_model=final_models["LogisticRegression"],
        xgb_model=final_models["XGBoost"],
    )

    _log_mlflow(
        cv_results=cv_results,
        test_results=test_results,
        cost_ens=cost_ens,
        confusion=(tn, fp, fn, tp),
        scale_pos_weight=scale_pos_weight,
        mlp_model=mlp,
    )

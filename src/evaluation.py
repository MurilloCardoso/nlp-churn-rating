"""Cross-validation estratificada para baselines sklearn-compatíveis."""
import logging
from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.config import CV_FOLDS, SEED

logger = logging.getLogger(__name__)


def stratified_cv_sklearn(
    name: str,
    make_model: Callable[[], Any],
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = CV_FOLDS,
) -> dict:
    """StratifiedKFold em modelo sklearn; retorna média/std de AUC, PR-AUC, F1, Recall."""
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

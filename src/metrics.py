"""Métricas e seleção de threshold."""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)

from src.config import COST_FN, COST_FP


def best_threshold_recall(
    y_true: np.ndarray,
    probs: np.ndarray,
    min_precision: float = 0.30,
) -> float:
    """Threshold que maximiza recall com precisão mínima de min_precision."""
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    mask = precision[:-1] >= min_precision
    if mask.any():
        best_idx = recall[:-1][mask].argmax()
        return float(thresholds[mask][best_idx])
    return float(thresholds[recall[:-1].argmax()])


def best_threshold_cost(
    y_true: np.ndarray,
    probs: np.ndarray,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
) -> tuple[float, float]:
    """Threshold que minimiza FN*cost_fn + FP*cost_fp. Retorna (threshold, custo_total)."""
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thresh = 0.5
    best_cost   = float("inf")
    for t in thresholds:
        pred = (probs >= t).astype(int)
        fn = int(((pred == 0) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        cost = fn * cost_fn + fp * cost_fp
        if cost < best_cost:
            best_cost   = cost
            best_thresh = float(t)
    return best_thresh, best_cost


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    """AUC-ROC, PR-AUC, F1, Recall e Accuracy num threshold fixo."""
    pred = (probs >= threshold).astype(int)
    return {
        "auc_roc":  roc_auc_score(y_true, probs),
        "pr_auc":   average_precision_score(y_true, probs),
        "f1":       f1_score(y_true, pred),
        "recall":   recall_score(y_true, pred),
        "accuracy": accuracy_score(y_true, pred),
    }

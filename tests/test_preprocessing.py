"""Testes unitários — funções de pré-processamento e threshold."""

import numpy as np
import pytest

from src.metrics import best_threshold_recall


def _make_perfect_probs(n=100):
    """Retorna y_true e probs onde a classe positiva tem score 1.0."""
    y_true = np.array([0] * 70 + [1] * 30)
    probs  = np.where(y_true == 1, 1.0, 0.0).astype(float)
    return y_true, probs


def test_best_threshold_recall_returns_float():
    y_true, probs = _make_perfect_probs()
    thresh = best_threshold_recall(y_true, probs)
    assert isinstance(thresh, float)

# Testa se o threshold retornado está entre 0 e 1, o que é esperado para probabilidades.    
def test_best_threshold_recall_range():
    y_true, probs = _make_perfect_probs()
    thresh = best_threshold_recall(y_true, probs)
    assert 0.0 <= thresh <= 1.0

def test_best_threshold_recall_with_noisy_probs():
    rng    = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=200)
    probs  = rng.uniform(0, 1, size=200)
    thresh = best_threshold_recall(y_true, probs)
    assert 0.0 <= thresh <= 1.0


def test_best_threshold_recall_min_precision_respected():
    """Com probs perfeitas e min_precision=0.9, threshold deve ser ≥ 0.5."""
    y_true, probs = _make_perfect_probs()
    thresh = best_threshold_recall(y_true, probs, min_precision=0.90)
    pred   = (probs >= thresh).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    if tp + fp > 0:
        precision = tp / (tp + fp)
        assert precision >= 0.90 - 1e-6


@pytest.mark.parametrize("min_prec", [0.0, 0.30, 0.50, 0.80])
def test_best_threshold_recall_various_min_precision(min_prec):
    rng    = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=500)
    probs  = rng.uniform(0, 1, size=500)
    thresh = best_threshold_recall(y_true, probs, min_precision=min_prec)
    assert isinstance(thresh, float)

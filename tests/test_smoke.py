"""Smoke tests — verifica que o modelo MLP pode ser instanciado e executado."""

import torch

from src.main import ChurnMLP


def test_churn_mlp_instantiation():
    model = ChurnMLP(input_dim=10)
    assert model is not None


def test_churn_mlp_output_shape():
    model = ChurnMLP(input_dim=10)
    model.eval()
    x = torch.randn(8, 10)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (8, 1)


def test_churn_mlp_output_is_finite():
    model = ChurnMLP(input_dim=20)
    model.eval()
    x = torch.randn(16, 20)
    with torch.no_grad():
        out = model(x)
    assert torch.isfinite(out).all()

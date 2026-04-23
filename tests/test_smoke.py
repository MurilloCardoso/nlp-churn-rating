"""Smoke tests — verifica que o modelo MLP pode ser instanciado e executado."""

import torch

from src.main import ChurnMLP

def test_backward_pass():
    model = ChurnMLP(input_dim=10)
    x = torch.randn(8, 10)
    y = torch.randint(0, 2, (8, 1)).float()

    out = model(x)
    loss = ((out - y) ** 2).mean()
    loss.backward()

    for p in model.parameters():
        assert p.grad is not None

def test_model_can_overfit_small_batch():
    model = ChurnMLP(input_dim=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    x = torch.randn(20, 5)
    y = torch.randint(0, 2, (20, 1)).float()

    for _ in range(200):
        out = model(x)
        loss = ((out - y) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert loss.item() < 0.1
    
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

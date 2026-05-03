"""Treino e inferência do MLP."""
import logging

import numpy as np
import torch
import torch.nn as nn

from src.config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MODELS_DIR,
    PATIENCE,
    WEIGHT_DECAY,
)
from src.models import ChurnMLP

logger = logging.getLogger(__name__)


def train_mlp(
    X_train_t: torch.Tensor,
    y_train_t: torch.Tensor,
    X_val_t: torch.Tensor,
    y_val_t: torch.Tensor,
    input_dim: int,
) -> ChurnMLP:
    """Treina ChurnMLP com early stopping em val_loss (sem leak no test)."""
    model     = ChurnMLP(input_dim=input_dim)
    loss_fn   = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=7, factor=0.5
    )

    dataset    = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    best_val_loss    = float("inf")
    patience_counter = 0
    MODELS_DIR.mkdir(exist_ok=True)
    ckpt_path = MODELS_DIR / "best_model.pt"

    for epoch in range(EPOCHS):
        model.train()
        for X_batch, y_batch in dataloader:
            logits = model(X_batch)
            loss   = loss_fn(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss   = loss_fn(val_logits, y_val_t).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info("Early stopping epoch=%d | best_val_loss=%.4f", epoch, best_val_loss)
                break

        if epoch % 10 == 0:
            logger.info(
                "Epoch %3d | train_loss=%.4f | val_loss=%.4f | lr=%.6f",
                epoch, loss.item(), val_loss,
                optimizer.param_groups[0]["lr"],
            )

    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    return model


def mlp_predict_proba(model: ChurnMLP, X: np.ndarray) -> np.ndarray:
    """Retorna probabilidades de churn para um batch de features escaladas."""
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        return torch.sigmoid(model(X_t)).squeeze().numpy()

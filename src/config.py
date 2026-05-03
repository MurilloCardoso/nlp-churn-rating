"""Constantes e configuração centralizada do projeto."""
from pathlib import Path

SEED = 42

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data" / "raw"
MODELS_DIR = BASE_DIR / "models"

# Hiperparâmetros do MLP
LEARNING_RATE = 0.001
WEIGHT_DECAY  = 1e-4
EPOCHS        = 300
PATIENCE      = 20
BATCH_SIZE    = 128

# Custos de negócio para tuning de threshold
# FN = cliente churnou e não foi avisado (perda de LTV)
# FP = cliente recebeu oferta de retenção sem precisar (custo do desconto)
COST_FN = 10.0
COST_FP = 1.0

# Splits
TEST_SIZE = 0.2
VAL_SIZE  = 0.2  # do conjunto train+val

# Cross-validation
CV_FOLDS = 5

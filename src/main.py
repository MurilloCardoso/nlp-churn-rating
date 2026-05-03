"""Entry point CLI — executa o pipeline completo via `python -m src.main`."""
import logging

from src.pipeline import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    force=True,
)


if __name__ == "__main__":
    run_pipeline()

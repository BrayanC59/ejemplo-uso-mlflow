"""Configuración del pronóstico de PIB per cápita y del tracking de MLflow."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parents[1]
load_dotenv(RAIZ / ".env")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME", "pib-per-capita"
)
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "pib-per-capita")
MLFLOW_MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "challenger")
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

_ruta_datos = Path(
    os.getenv("DATA_PATH", "data/pib_per_capita_countries_dataset.csv")
)
DATA_PATH = _ruta_datos if _ruta_datos.is_absolute() else RAIZ / _ruta_datos

TARGET = "gdp_per_capita"
METRIC = "validation_mae"
TRAIN_END_YEAR = int(os.getenv("TRAIN_END_YEAR", "2017"))
VALIDATION_START_YEAR = int(os.getenv("VALIDATION_START_YEAR", "2018"))
VALIDATION_END_YEAR = int(os.getenv("VALIDATION_END_YEAR", "2020"))
TEST_START_YEAR = int(os.getenv("TEST_START_YEAR", "2021"))
TEST_END_YEAR = int(os.getenv("TEST_END_YEAR", "2023"))
FORECAST_YEAR = int(os.getenv("FORECAST_YEAR", "2024"))


def activar_seguimiento() -> None:
    """Apunta MLflow al almacén y al experimento configurados."""
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

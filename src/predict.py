"""Pronostica PIB per cápita para el año configurado usando el alias de MLflow."""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

import config
from data import preparar_proyeccion


def _uri() -> str:
    return f"models:/{config.MLFLOW_MODEL_NAME}@{config.MLFLOW_MODEL_ALIAS}"


def cargar():
    """Carga el pipeline registrado bajo el alias configurado."""
    config.activar_seguimiento()
    return mlflow.sklearn.load_model(_uri())


def predecir(features: pd.DataFrame):
    """Predice PIB per cápita a partir de las features temporales."""
    return cargar().predict(features)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pronostica el PIB per cápita para cada país."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=config.DATA_PATH,
        help="Dataset CSV (por defecto, el definido en DATA_PATH).",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=config.FORECAST_YEAR,
        help="Año que se desea pronosticar.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV opcional donde guardar los pronósticos.",
    )
    args = parser.parse_args()

    features, resultado = preparar_proyeccion(args.input, args.year)
    predicciones = cargar().predict(features)
    resultado["gdp_per_capita_predicho"] = pd.Series(predicciones).clip(lower=0)
    resultado = resultado.sort_values(
        "gdp_per_capita_predicho", ascending=False
    ).reset_index(drop=True)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        resultado.to_csv(args.output, index=False)
        print(f"Pronósticos guardados en {args.output}")
    print(resultado.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

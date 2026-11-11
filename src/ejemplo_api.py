"""Genera un ejemplo JSON de consumo del modelo de PIB-per-cápita en MLflow ya subido :D

El JSON incluye la petición compatible con `/invocations` y la respuesta
calculada por la versión asignada al alias configurado.

    python src/ejemplo_api.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow.models import Model
from mlflow.tracking import MlflowClient

import config

SALIDA = config.RAIZ / "ejemplos-api"
PUERTO = 5001


def _uri(modelo: str, alias: str) -> str:
    return f"models:/{modelo}@{alias}"


def _jsonable(valor):
    """Convierte valores de numpy/pandas y faltantes a tipos JSON estándar."""
    if isinstance(valor, dict):
        return {str(clave): _jsonable(item) for clave, item in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_jsonable(item) for item in valor]
    if pd.isna(valor):
        return None
    if hasattr(valor, "item"):
        return _jsonable(valor.item())
    return valor


def _esquema(firma, lado: str):
    campo = getattr(firma, lado, None) if firma is not None else None
    if campo is None:
        return None
    return _jsonable(campo.to_dict())


def generar(
    modelo: str, alias: str, puerto: int = PUERTO
) -> tuple[Path, Path, Path]:
    """Escribe los JSON de petición, respuesta e información del modelo."""
    config.activar_seguimiento()
    uri = _uri(modelo, alias)
    cliente = MlflowClient(tracking_uri=config.MLFLOW_TRACKING_URI)
    version = cliente.get_model_version_by_alias(modelo, alias)
    info = mlflow.models.get_model_info(uri)
    ejemplo = Model.load(uri).load_input_example(uri)
    if not isinstance(ejemplo, pd.DataFrame) or ejemplo.empty:
        raise ValueError(
            f"El modelo {uri} no tiene un input_example de tipo DataFrame."
        )

    esquema_entrada = _esquema(info.signature, "inputs")
    if esquema_entrada:
        columnas_esperadas = [campo["name"] for campo in esquema_entrada]
        if list(ejemplo.columns) != columnas_esperadas:
            raise ValueError(
                "Las columnas del input_example no coinciden con la firma del modelo: "
                f"esperadas {columnas_esperadas}, recibidas {list(ejemplo.columns)}."
            )

    ejemplo = ejemplo.head(1)
    peticion = {
        "dataframe_split": {
            "columns": list(ejemplo.columns),
            "data": _jsonable(ejemplo.to_numpy().tolist()),
        }
    }

    predictor = mlflow.pyfunc.load_model(uri)
    predicciones = predictor.predict(ejemplo)
    if hasattr(predicciones, "tolist"):
        predicciones = predicciones.tolist()
    respuesta = {"predictions": _jsonable(predicciones)}

    SALIDA.mkdir(parents=True, exist_ok=True)
    nombre = f"{modelo}__{alias}"
    destino_peticion = SALIDA / f"{nombre}.json"
    destino_respuesta = SALIDA / f"{nombre}__respuesta.json"
    for destino, contenido in (
        (destino_peticion, peticion),
        (destino_respuesta, respuesta),
    ):
        destino.write_text(
            json.dumps(contenido, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    metadatos = {
        "modelo": uri,
        "version": version.version,
        "endpoint": f"POST http://127.0.0.1:{puerto}/invocations",
        "esquema_entrada": esquema_entrada,
        "esquema_salida": _esquema(info.signature, "outputs"),
    }
    destino_info = SALIDA / f"{nombre}__info.json"
    destino_info.write_text(
        json.dumps(metadatos, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destino_peticion, destino_respuesta, destino_info


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera un JSON para consumir el modelo registrado en MLflow."
    )
    parser.add_argument(
        "--modelo",
        default=config.MLFLOW_MODEL_NAME,
        help="Nombre en el registro. Por defecto, MLFLOW_MODEL_NAME.",
    )
    parser.add_argument(
        "--alias",
        default=config.MLFLOW_MODEL_ALIAS,
        help="Alias registrado. Por defecto, MLFLOW_MODEL_ALIAS.",
    )
    parser.add_argument(
        "--puerto",
        type=int,
        default=PUERTO,
        help="Puerto del endpoint de MLflow Models.",
    )
    args = parser.parse_args()
    peticion, respuesta, informacion = generar(
        args.modelo, args.alias, args.puerto
    )
    print(f"Petición JSON: {peticion}")
    print(f"Respuesta JSON: {respuesta}")
    print(f"Información del modelo: {informacion}")


if __name__ == "__main__":
    main()

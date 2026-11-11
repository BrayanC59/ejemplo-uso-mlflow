"""Registra como challenger el regresor con menor MAE de validación."""

from __future__ import annotations

import mlflow
from mlflow.tracking import MlflowClient

import config


def _uri_modelo(cliente: MlflowClient, run_id: str) -> str:
    run = cliente.get_run(run_id)
    salidas = []
    if run.outputs is not None and run.outputs.model_outputs:
        salidas = list(run.outputs.model_outputs)
    if not salidas:
        raise SystemExit(f"El run {run_id} no tiene un modelo registrado en MLflow.")
    elegido = max(salidas, key=lambda salida: salida.step)
    return f"models:/{elegido.model_id}"


def registrar() -> None:
    """Asigna el alias configurado al mejor modelo de regresión evaluado."""
    config.activar_seguimiento()
    cliente = MlflowClient(tracking_uri=config.MLFLOW_TRACKING_URI)
    experimento = cliente.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME)
    if experimento is None:
        raise SystemExit(
            f"No existe el experimento {config.MLFLOW_EXPERIMENT_NAME}. "
            "Ejecuta antes python src/train.py."
        )

    runs = cliente.search_runs(
        experiment_ids=[experimento.experiment_id],
        filter_string=(
            "attributes.status = 'FINISHED' and "
            "tags.modelo_registrable = 'true' and "
            f"metrics.{config.METRIC} >= 0"
        ),
        order_by=[f"metrics.{config.METRIC} ASC"],
        max_results=1,
    )
    if not runs:
        raise SystemExit(
            f"No hay modelos candidatos con la métrica {config.METRIC}. "
            "Ejecuta antes python src/train.py."
        )

    mejor = runs[0]
    mae = mejor.data.metrics[config.METRIC]
    nombre = mejor.data.tags.get("mlflow.runName", mejor.info.run_id)
    version = mlflow.register_model(
        _uri_modelo(cliente, mejor.info.run_id),
        config.MLFLOW_MODEL_NAME,
    )
    cliente.set_registered_model_alias(
        config.MLFLOW_MODEL_NAME,
        config.MLFLOW_MODEL_ALIAS,
        version.version,
    )
    print(
        f"Mejor modelo: {nombre} ({config.METRIC}=${mae:,.2f}). "
        f"models:/{config.MLFLOW_MODEL_NAME}@{config.MLFLOW_MODEL_ALIAS} "
        f"-> versión {version.version}"
    )


if __name__ == "__main__":
    registrar()

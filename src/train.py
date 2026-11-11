"""Entrena y compara regresores temporales para PIB per cápita con MLflow."""

from __future__ import annotations

import os

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config
from data import (
    COLUMNAS_CATEGORICAS,
    COLUMNAS_NUMERICAS,
    cargar_particiones,
)


def _modelos():
    return [
        (
            "random_forest",
            RandomForestRegressor(
                n_estimators=100,
                random_state=config.RANDOM_SEED,
                n_jobs=-1,
            ),
        ),
        ("regresion_lineal", LinearRegression()),
    ]


def _pipeline(modelo) -> Pipeline:
    categoricas = Pipeline(
        [
            ("imputacion", SimpleImputer(strategy="most_frequent")),
            ("codificacion", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numericas = Pipeline(
        [
            ("imputacion", SimpleImputer(strategy="median")),
            ("escalado", StandardScaler()),
        ]
    )
    preprocesamiento = ColumnTransformer(
        [
            ("categoricas", categoricas, list(COLUMNAS_CATEGORICAS)),
            ("numericas", numericas, list(COLUMNAS_NUMERICAS)),
        ]
    )
    return Pipeline(
        [("preprocesamiento", preprocesamiento), ("modelo", modelo)]
    )


def _registrar_metricas(prefijo, reales, predicciones) -> None:
    mae = mean_absolute_error(reales, predicciones)
    rmse = mean_squared_error(reales, predicciones) ** 0.5
    r2 = r2_score(reales, predicciones)
    mlflow.log_metrics(
        {
            f"{prefijo}_mae": float(mae),
            f"{prefijo}_rmse": float(rmse),
            f"{prefijo}_r2": float(r2),
        }
    )
    print(f"{prefijo}: MAE=${mae:,.2f}, RMSE=${rmse:,.2f}, R2={r2:.3f}")


def _dataset(features: pd.DataFrame, objetivo: pd.Series, nombre: str):
    tabla = features.copy()
    tabla[config.TARGET] = objetivo
    return mlflow.data.from_pandas(
        tabla,
        source=config.DATA_PATH.resolve().as_uri(),
        targets=config.TARGET,
        name=nombre,
    )


def _parametros(modelo) -> dict:
    return {
        clave: valor if valor is not None else "None"
        for clave, valor in modelo.get_params().items()
        if isinstance(valor, (int, float, str, bool)) or valor is None
    }


def entrenar() -> None:
    """Evalúa con cortes por año y registra un modelo final por estrategia."""
    os.environ.pop("MLFLOW_RUN_ID", None)
    config.activar_seguimiento()
    mlflow.sklearn.autolog(disable=True)

    (
        x_train,
        y_train,
        x_validacion,
        y_validacion,
        x_prueba,
        y_prueba,
        resumen,
    ) = cargar_particiones()

    print(
        f"Filas: entrenamiento={len(x_train)}, "
        f"validación={len(x_validacion)}, prueba={len(x_prueba)}"
    )

    with mlflow.start_run(run_name="persistencia"):
        mlflow.set_tag("estrategia", "persistencia")
        mlflow.set_tag("modelo_registrable", "false")
        mlflow.log_dict(resumen, "datos/resumen.json")
        for nombre, reales, features in (
            ("validation", y_validacion, x_validacion),
            ("test", y_prueba, x_prueba),
        ):
            _registrar_metricas(
                nombre, reales, features["gdp_per_capita_lag1"]
            )

    x_final = pd.concat([x_train, x_validacion, x_prueba]).sort_index()
    y_final = pd.concat([y_train, y_validacion, y_prueba]).sort_index()
    limite_final = x_final["year"] <= config.TEST_END_YEAR
    x_final = x_final.loc[limite_final]
    y_final = y_final.loc[limite_final]

    datos_train = _dataset(x_train, y_train, "pib_per_capita_train")
    datos_validacion = _dataset(
        x_validacion, y_validacion, "pib_per_capita_validation"
    )
    datos_prueba = _dataset(x_prueba, y_prueba, "pib_per_capita_test")

    for nombre, estimador in _modelos():
        modelo = _pipeline(estimador)
        modelo.fit(x_train, y_train)
        prediccion_validacion = modelo.predict(x_validacion)
        prediccion_prueba = modelo.predict(x_prueba)

        with mlflow.start_run(run_name=nombre):
            mlflow.set_tag("estrategia", nombre)
            mlflow.set_tag("modelo_registrable", "true")
            mlflow.log_params(_parametros(estimador))
            mlflow.log_params(
                {
                    "train_end_year": config.TRAIN_END_YEAR,
                    "validation_start_year": config.VALIDATION_START_YEAR,
                    "validation_end_year": config.VALIDATION_END_YEAR,
                    "test_start_year": config.TEST_START_YEAR,
                    "test_end_year": config.TEST_END_YEAR,
                    "random_seed": config.RANDOM_SEED,
                }
            )
            mlflow.log_input(datos_train, context="training")
            mlflow.log_input(datos_validacion, context="validation")
            mlflow.log_input(datos_prueba, context="test")
            _registrar_metricas(
                "validation", y_validacion, prediccion_validacion
            )
            _registrar_metricas("test", y_prueba, prediccion_prueba)

            modelo.fit(x_final, y_final)
            mlflow.sklearn.log_model(
                modelo,
                name="model",
                input_example=x_final.head(1),
                serialization_format="pickle",
            )
            print(f"Modelo final registrado: {nombre}")


if __name__ == "__main__":
    entrenar()

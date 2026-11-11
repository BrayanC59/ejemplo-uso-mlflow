"""Carga el panel de países y crea features temporales sin usar datos futuros."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import config

COLUMNAS_REQUERIDAS = (
    "country_code",
    "country_name",
    "sub_region",
    "year",
    "gdp_per_capita",
    "gdp_variation",
)
COLUMNAS_CATEGORICAS = ("country_code", "sub_region")
COLUMNAS_NUMERICAS = (
    "year",
    "gdp_per_capita_lag1",
    "gdp_variation_lag1",
)
COLUMNAS_FEATURES = (*COLUMNAS_CATEGORICAS, *COLUMNAS_NUMERICAS)


def _leer_csv(ruta: Path | None = None) -> pd.DataFrame:
    """Lee y valida las columnas que necesita el modelo."""
    archivo = ruta or config.DATA_PATH
    if not archivo.is_file():
        raise FileNotFoundError(
            f"No se encontró el dataset en {archivo}. Revisa DATA_PATH."
        )

    datos = pd.read_csv(archivo)
    faltan = [columna for columna in COLUMNAS_REQUERIDAS if columna not in datos]
    if faltan:
        raise ValueError(f"Faltan columnas en {archivo.name}: {faltan}")

    datos = datos.loc[:, COLUMNAS_REQUERIDAS].copy()
    datos["year"] = pd.to_numeric(datos["year"], errors="raise").astype(int)
    for columna in ("gdp_per_capita", "gdp_variation"):
        datos[columna] = pd.to_numeric(datos[columna], errors="raise")

    if datos["country_code"].isna().any():
        raise ValueError("El dataset tiene filas sin country_code.")
    if datos.duplicated(["country_code", "year"]).any():
        raise ValueError("Hay más de una observación por país y año.")

    datos["country_code"] = datos["country_code"].astype(str)
    datos["country_name"] = datos["country_name"].fillna("Sin dato").astype(str)
    datos["sub_region"] = datos["sub_region"].fillna("Sin dato").astype(str)
    datos = datos.sort_values(["country_code", "year"]).reset_index(drop=True)
    return datos


def _crear_features(datos: pd.DataFrame) -> pd.DataFrame:
    """Asocia cada año con los valores observados del mismo país el año previo."""
    ordenados = datos.sort_values(["country_code", "year"]).copy()
    grupos = ordenados.groupby("country_code", sort=False)
    ordenados["year_lag1"] = grupos["year"].shift(1)
    ordenados["gdp_per_capita_lag1"] = grupos["gdp_per_capita"].shift(1)
    ordenados["gdp_variation_lag1"] = grupos["gdp_variation"].shift(1)

    consecutivos = ordenados["year"].eq(ordenados["year_lag1"] + 1)
    validos = (
        consecutivos
        & ordenados["gdp_per_capita"].gt(0)
        & ordenados["gdp_per_capita_lag1"].gt(0)
    )
    resultado = ordenados.loc[validos].copy()
    resultado = resultado.rename(columns={"gdp_per_capita": config.TARGET})
    return resultado.reset_index(drop=True)


def cargar_particiones():
    """Devuelve los conjuntos temporales y un resumen para registrarlos en MLflow."""
    datos = _leer_csv()
    supervisados = _crear_features(datos)
    anios = supervisados["year"]

    mascara_train = anios <= config.TRAIN_END_YEAR
    mascara_validacion = anios.between(
        config.VALIDATION_START_YEAR, config.VALIDATION_END_YEAR
    )
    mascara_test = anios.between(config.TEST_START_YEAR, config.TEST_END_YEAR)

    particiones = (mascara_train, mascara_validacion, mascara_test)
    nombres = ("entrenamiento", "validacion", "prueba")
    for nombre, mascara in zip(nombres, particiones):
        if not mascara.any():
            raise ValueError(
                f"No hay observaciones para {nombre}. Revisa los años configurados "
                "y la cobertura del dataset."
            )

    x = supervisados.loc[:, COLUMNAS_FEATURES]
    y = supervisados[config.TARGET]
    resumen = {
        "archivo": config.DATA_PATH.name,
        "filas_originales": int(len(datos)),
        "filas_con_pib_positivo": int(datos[config.TARGET].gt(0).sum()),
        "filas_supervisadas": int(len(supervisados)),
        "filas_entrenamiento": int(mascara_train.sum()),
        "filas_validacion": int(mascara_validacion.sum()),
        "filas_prueba": int(mascara_test.sum()),
        "anio_entrenamiento_hasta": config.TRAIN_END_YEAR,
        "anios_validacion": [
            config.VALIDATION_START_YEAR,
            config.VALIDATION_END_YEAR,
        ],
        "anios_prueba": [config.TEST_START_YEAR, config.TEST_END_YEAR],
        "columnas_features": list(COLUMNAS_FEATURES),
        "objetivo": config.TARGET,
    }

    return (
        x.loc[mascara_train],
        y.loc[mascara_train],
        x.loc[mascara_validacion],
        y.loc[mascara_validacion],
        x.loc[mascara_test],
        y.loc[mascara_test],
        resumen,
    )


def preparar_proyeccion(
    ruta: Path | None = None, anio_objetivo: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepara una fila por país para pronosticar el año siguiente al último dato."""
    anio = anio_objetivo or config.FORECAST_YEAR
    datos = _leer_csv(ruta)
    disponibles = datos.loc[
        (datos["year"] < anio) & datos["gdp_per_capita"].gt(0)
    ].copy()
    if disponibles.empty:
        raise ValueError(f"No hay observaciones válidas anteriores a {anio}.")

    ultimos = (
        disponibles.sort_values(["country_code", "year"])
        .groupby("country_code", sort=False)
        .tail(1)
        .copy()
    )
    ultimos["year"] = anio
    ultimos["gdp_per_capita_lag1"] = ultimos["gdp_per_capita"]
    ultimos["gdp_variation_lag1"] = ultimos["gdp_variation"]

    features = ultimos.loc[:, COLUMNAS_FEATURES].reset_index(drop=True)
    metadata = ultimos.loc[
        :, ["country_code", "country_name", "sub_region"]
    ].reset_index(drop=True)
    metadata["year"] = anio
    return features, metadata

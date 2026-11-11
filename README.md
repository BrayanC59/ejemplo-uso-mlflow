# Pronóstico de PIB per cápita con MLflow

Proyecto de regresión para pronosticar el PIB per cápita (`gdp_per_capita`) por país. Usa `data/pib_per_capita_countries_dataset.csv`, registra métricas y modelos en MLflow, compara un bosque aleatorio con regresión lineal y conserva la persistencia del año anterior como referencia.

> El notebook `notebooks/modelo_de_prediccion_PIB_ML.ipynb` llama a `/content/proyecto_regresion_pib.py`, un script de Google Colab que no está incluido en el repositorio. La implementación de `src/` sigue los cortes temporales y resultados que sí se guardaron en el notebook: entrenamiento hasta 2017, validación 2018–2020 y prueba 2021–2023. No presupone poder reproducir exactamente el script externo ausente.

## Flujo y features

- Usa `country_code` y `sub_region` como categorías.
- Predice el PIB per cápita del año actual usando el PIB per cápita y la variación del año anterior, además del año objetivo.
- Solo usa observaciones consecutivas con PIB positivo tanto en el año objetivo como en el anterior. Los valores ausentes de la variación se imputan dentro del pipeline.
- Ajusta los candidatos con datos hasta 2017; elige entre bosque aleatorio y regresión lineal por menor `validation_mae`; la persistencia se registra como referencia, pero no se registra como modelo candidato. Reporta también MAE, RMSE y R² en la prueba de 2021–2023.
- Una vez evaluado, guarda en cada run el pipeline final reajustado con los datos hasta 2023, listo para proyectar 2024.

## Preparación en Windows

Desde PowerShell, en la carpeta raíz del proyecto:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Si `.venv` ya existe, actívalo e instala las dependencias solo si hace falta. `.env` es local; no lo subas a Git.

Las variables principales están en `.env`:

```dotenv
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
MLFLOW_EXPERIMENT_NAME=pib-per-capita
MLFLOW_MODEL_NAME=pib-per-capita
MLFLOW_MODEL_ALIAS=challenger
DATA_PATH=data/pib_per_capita_countries_dataset.csv
RANDOM_SEED=42
```

Las rutas relativas de `DATA_PATH` se resuelven desde la raíz del proyecto. Se pueden cambiar los límites de años y el año de pronóstico con las variables opcionales documentadas en `.env.example`.

## Entrenar y ver resultados

En una terminal, activa el entorno y ejecuta:

```powershell
.\.venv\Scripts\Activate.ps1
python src\train.py
```

El entrenamiento crea el almacén local `mlflow.db` y registra el experimento `pib-per-capita`. Los runs incluyen el baseline `persistencia`, `random_forest` y `regresion_lineal`; los candidatos guardan las métricas de validación y prueba, parámetros, conjuntos de datos y el pipeline final.

En otra terminal, desde la raíz y con el mismo entorno:

```powershell
.\.venv\Scripts\Activate.ps1
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5050
```

Abre <http://127.0.0.1:5050>, selecciona el experimento `pib-per-capita` y compara `validation_mae` (menor es mejor). Si el puerto está ocupado, cambia `--port` y abre esa dirección. La terminal de la interfaz debe seguir abierta mientras la usas.

## Registrar y pronosticar

Después de entrenar, registra el modelo con menor MAE de validación y asígnale el alias `challenger`:

```powershell
python src\register.py
```

Pronostica el año configurado (por defecto 2024) y guarda los resultados:

```powershell
python src\predict.py --output resultados\pronostico_2024.csv
```

También se puede cambiar el año o el CSV de entrada:

```powershell
python src\predict.py --year 2024 --input data\pib_per_capita_countries_dataset.csv --output resultados\pronostico_2024.csv
```

`src/predict.py` imprime los diez mayores pronósticos y guarda todas las filas si se especifica `--output`. El archivo de salida no se crea hasta ejecutar el comando.

## Consumir el modelo por API

Genera un cuerpo de petición JSON listo para enviar, la respuesta de ejemplo del alias y los metadatos de la firma:

```powershell
python src\ejemplo_api.py
```

Los archivos quedan en `ejemplos-api/`:

- `pib-per-capita__challenger.json`: cuerpo de la petición para el endpoint MLflow.
- `pib-per-capita__challenger__respuesta.json`: forma de la respuesta generada localmente.
- `pib-per-capita__challenger__info.json`: versión, endpoint y firma de entrada/salida.

En una terminal, inicia el servidor del modelo:

```powershell
mlflow models serve -m "models:/pib-per-capita@challenger" --port 5001 --env-manager local
```

En otra terminal, envía el JSON de petición y compara la respuesta con el archivo de ejemplo:

```powershell
curl.exe http://127.0.0.1:5001/invocations `
  -H "Content-Type: application/json" `
  --data-binary "@ejemplos-api\pib-per-capita__challenger.json"
```

## MLflow local o remoto

Con la configuración predeterminada, los resultados se guardan **localmente** en `mlflow.db` y en el almacén de artefactos de MLflow. No se suben automáticamente a un servicio remoto ni a GitHub.

Para registrar en un servidor remoto de MLflow, configura `MLFLOW_TRACKING_URI` en `.env` con el URI entregado por ese servidor y asegúrate de que tenga almacenamiento de artefactos accesible. Si el servidor requiere autenticación, configura sus credenciales en variables de entorno seguras; no las escribas en el código ni las subas al repositorio.

## Archivos principales

```text
src/config.py    configuración, dataset, experimento y años
src/data.py      validación, features temporales y particiones
src/train.py     baseline, evaluación, ajuste y logging de candidatos
src/register.py  registro del candidato con menor MAE de validación
src/predict.py   pronósticos desde el alias de MLflow
data/            dataset de PIB per cápita
mlflow.db        tracking local generado por MLflow
```

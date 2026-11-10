# ejemplo-mlops

Flujo mínimo con MLflow: explorar los datos, entrenar, registrar el mejor run con el alias `challenger` y predecir con ese alias. Este repositorio es la estructura del proyecto. Los archivos de `src/` y el notebook describen qué hace cada pieza; no contienen implementación.

## Estructura

```
├── README.md                 # este documento
├── requirements.txt          # librerías con versión exacta
├── .env.example              # variables de ejemplo, sin secretos
├── .gitignore                # mlflow.db, mlruns/, .venv/ y .env
├── MLproject                 # entrada estándar: mlflow run .
├── notebooks/
│   └── 01_exploracion.ipynb  # explorar y prototipar
├── src/
│   ├── config.py             # experimento, modelo y semilla
│   ├── data.py               # cargar y partir los datos
│   ├── train.py              # entrenar y registrar en MLflow
│   ├── register.py           # mejor run → alias challenger
│   └── predict.py            # cargar por alias y predecir
├── data/                     # datos pequeños de ejemplo
├── mlflow.db                 # generado: parámetros, métricas, versiones
└── mlruns/                   # generado: modelos, gráficos, CSV
```

`mlflow.db` y `mlruns/` no están en el repositorio. MLflow los crea al entrenar o al abrir la interfaz de seguimiento.

## Instalar

Requisitos: Python 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

En esta copia ya existe el entorno `.envmlops`, con las mismas versiones que `requirements.txt`. Para usarlo:

```bash
source .envmlops/bin/activate
```

`.env` es local. No lo subas a Git. Si más adelante hace falta una credencial, va solo en `.env`, nunca en `.env.example`.

## Entrenar

1. Deja el conjunto pequeño en `data/`. La ruta por defecto es `data/ejemplo.csv` (`DATA_PATH`).
2. Revisa nombres y semilla en `.env`. `src/config.py` es quien los concentra.
3. `src/data.py` carga ese archivo y lo parte en entrenamiento y prueba con `RANDOM_SEED`.
4. `src/train.py` entrena y deja en MLflow los parámetros, las métricas y el modelo.
5. El seguimiento local usa `MLFLOW_TRACKING_URI` (`sqlite:///mlflow.db`).

La entrada del proyecto es `MLproject`:

```bash
mlflow run .
```

Eso corresponde al entry point `main`, que llama a `src/train.py`. Para ver los runs:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

La interfaz queda en `http://127.0.0.1:5000`. Ahí aparecen el experimento, los parámetros, las métricas y los archivos de cada run (también guardados en `mlruns/`).

## Registrar el challenger

`src/register.py` elige el mejor run del experimento `MLFLOW_EXPERIMENT_NAME` y publica esa versión del modelo `MLFLOW_MODEL_NAME` con el alias `challenger`.

```bash
mlflow run . -e register
```

El alias es el contrato: predicción y servicio apuntan a `challenger`, no a un número de versión fijo.

## Servir y predecir

`src/predict.py` carga el modelo con el alias y devuelve la predicción:

```text
models:/ejemplo-modelo@challenger
```

El nombre y el alias salen de `MLFLOW_MODEL_NAME` y `MLFLOW_MODEL_ALIAS`.

Para exponer el mismo modelo en un servicio local, cuando ya exista una versión con ese alias:

```bash
mlflow models serve \
  -m "models:/ejemplo-modelo@challenger" \
  -p 5001 \
  --no-conda
```

El servicio escucha en `http://127.0.0.1:5001`. Una petición de ejemplo:

```bash
curl http://127.0.0.1:5001/invocations \
  -H 'Content-Type: application/json' \
  -d '{"inputs": [[0.0, 1.0, 2.0]]}'
```

Las columnas reales dependen del conjunto que quede en `data/`.

## Datos

`data/` es para un ejemplo pequeño que sí puede vivir en Git. Conjuntos grandes, descargas y copias intermedias se quedan fuera del repositorio. La política está escrita en `data/README.md`.

## Qué no se versiona

| Ruta | Motivo |
| --- | --- |
| `.env` | valores locales; puede llegar a tener secretos |
| `.venv/`, `.envmlops/` | entornos de Python |
| `mlflow.db` | metadatos generados: parámetros, métricas, versiones |
| `mlruns/` | archivos generados: modelos, gráficos, CSV |

## Cargar imagen a docker hub 

- docker login

- mlflow models build-docker \
  -m "models:/nombre_de_tu_modelo@challenger" \
  -n "tu_usuario_dockerhub/mi-modelo-challenger:latest"

- docker push tu_usuario_dockerhub/mi-modelo-challenger:latest

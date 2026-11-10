# ejemplo-mlops

Flujo con MLflow: limpiar el conjunto de crédito, entrenar cuatro clasificadores, registrar el de mayor accuracy con el alias `challenger` y servir ese clasificador. El artefacto `model` es solo el estimador. La API recibe las ocho columnas ya codificadas y escaladas. Los filtros, el mapa del `LabelEncoder` y la media y la escala quedan en el run `preprocesamiento`.

Los nombres del experimento, del modelo y del alias salen de `.env` (`MLFLOW_EXPERIMENT_NAME`, `MLFLOW_MODEL_NAME`, `MLFLOW_MODEL_ALIAS`). En esta copia son `credit-risk` y `challenger`.

## Estructura

```
├── README.md                 # este documento
├── requirements.txt          # librerías con versión exacta
├── .env.example              # variables de ejemplo, sin secretos
├── .gitignore                # mlflow.db, mlruns/, .venv/ y .env
├── MLproject                 # mlflow run . --env-manager local
├── notebooks/
│   └── 01_exploracion.ipynb  # explorar y prototipar
├── src/
│   ├── config.py             # experimento, modelo y semilla
│   ├── data.py               # limpiar, codificar, escalar y partir
│   ├── train.py              # entrenar y dejar el clasificador en MLflow
│   ├── register.py           # mejor run → alias challenger
│   ├── predict.py            # predecir con columnas ya transformadas
│   └── ejemplo_api.py        # documento de petición y respuesta
├── ejemplos-api/             # generado: un Markdown por modelo y alias
├── data/                     # credit_risk_dataset.csv, fuera de Git
├── mlflow.db                 # generado: parámetros, métricas, versiones
└── mlruns/                   # generado: modelos y artefactos
```

`mlflow.db` y `mlruns/` no están en el repositorio. MLflow los crea al entrenar o al abrir la interfaz.

## Instalar

Requisitos: Python 3.12 y Docker, si se va a construir la imagen.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

En esta copia ya existe el entorno `.envmlops`, con las mismas versiones que `requirements.txt`:

```bash
source .envmlops/bin/activate
```

`.env` es local. No lo subas a Git.

## Interfaz y almacén

Desde la raíz del proyecto, con el entorno activado. Esta terminal se queda ocupada:

```bash
mlflow ui --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db
```

La interfaz queda en http://127.0.0.1:5000. Lee `sqlite:///mlflow.db` y los archivos de `mlruns/`. El entrenamiento y el servicio van en otra terminal.

## Entrenar

`DATA_PATH` apunta a `data/credit_risk_dataset.csv`. `src/train.py` abre un run `preprocesamiento` y, después, un run por clasificador: regresión logística, SVM, árbol de decisión y random forest. Cada uno guarda accuracy, parámetros y el estimador.

```bash
set -a && source .env && set +a
mlflow run . --env-manager local
```

`--env-manager local` reutiliza el entorno ya instalado. Sin esa opción MLflow intenta crear otro entorno con pyenv.

## Registrar el challenger

`src/register.py` elige el run terminado con mayor `accuracy` y publica ese modelo con el alias `challenger`.

```bash
mlflow run . -e register --env-manager local
```

Predicción y servicio usan `models:/credit-risk@challenger`, no un número de versión fijo.

## Servir el modelo

`mlflow models serve` expone el clasificador. La petición lleva una fila ya codificada y escalada. Una fila produce una sola clase, `[0]` o `[1]`.

```bash
mlflow --env-file .env models serve \
  -m "models:/credit-risk@challenger" \
  --port 5001 \
  --env-manager local
```

El servicio escucha en http://127.0.0.1:5001. Para regenerar el ejemplo de esa versión:

```bash
python src/ejemplo_api.py --modelo credit-risk --alias challenger --puerto 5001
```

Eso escribe `ejemplos-api/credit-risk__challenger.md`. La petición de una fila es:

```bash
curl http://127.0.0.1:5001/invocations \
  -H 'Content-Type: application/json' \
  -d '{"dataframe_split": {"columns": ["person_age", "person_income", "person_home_ownership", "person_emp_length", "loan_intent", "loan_amnt", "loan_int_rate", "loan_percent_income"], "data": [[-0.44220311260130146, -0.12484565079562059, 3.0, -0.4415669058833856, 3.0, -1.0521712618466563, 0.7503988094474996, -1.1235941837207022]]}}'
```

La respuesta tiene esta forma:

```json
{"predictions": [1]}
```

`src/predict.py` hace lo mismo en local, sin levantar el servicio:

```bash
mlflow run . -e predict --env-manager local
```

## Imagen Docker

`mlflow models build-docker` empaqueta el alias `challenger`. Dentro del contenedor el proceso escucha en el puerto 8080.

```bash
mlflow --env-file .env models build-docker \
  -m "models:/credit-risk@challenger" \
  -n "tu_usuario_dockerhub/mi-modelo-challenger:latest"
```

Para probarla en el puerto 5002 del equipo:

```bash
docker run --rm -p 5002:8080 tu_usuario_dockerhub/mi-modelo-challenger:latest
```

La misma fila, contra el contenedor:

```bash
curl http://127.0.0.1:5002/invocations \
  -H 'Content-Type: application/json' \
  -d '{"dataframe_split": {"columns": ["person_age", "person_income", "person_home_ownership", "person_emp_length", "loan_intent", "loan_amnt", "loan_int_rate", "loan_percent_income"], "data": [[-0.44220311260130146, -0.12484565079562059, 3.0, -0.4415669058833856, 3.0, -1.0521712618466563, 0.7503988094474996, -1.1235941837207022]]}}'
```

`--rm` borra el contenedor al detenerlo con Ctrl+C.

## Cargar imagen a Docker Hub

```bash
docker login

mlflow --env-file .env models build-docker \
  -m "models:/credit-risk@challenger" \
  -n "tu_usuario_dockerhub/mi-modelo-challenger:latest"

docker push tu_usuario_dockerhub/mi-modelo-challenger:latest
```

Sustituye `tu_usuario_dockerhub` por el usuario de Docker Hub. Si la imagen ya se construyó en el paso anterior, basta con `docker login` y `docker push`.

## Datos

`data/` guarda el conjunto de trabajo. `DATA_PATH` es `data/credit_risk_dataset.csv` y ese archivo no se sube a Git. La política está en `data/README.md`.

## Qué no se versiona

| Ruta | Motivo |
| --- | --- |
| `.env` | valores locales; puede llegar a tener secretos |
| `.venv/`, `.envmlops/` | entornos de Python |
| `mlflow.db` | metadatos generados: parámetros, métricas, versiones |
| `mlruns/` | archivos generados: modelos y artefactos |

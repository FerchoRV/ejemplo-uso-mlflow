"""Nombres del experimento y del modelo, y semilla.

`src/` lee estos valores desde aquí. Salen de `.env`; si falta el
archivo, se usan los de `.env.example`.

El objetivo, la partición y la métrica salen del prototipo de
`notebooks/01_exploracion.ipynb`: clasificar `loan_status` y elegir
el run con mayor accuracy.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parents[1]
load_dotenv(RAIZ / ".env")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "ejemplo-mlops")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "ejemplo-modelo")
MLFLOW_MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "challenger")
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

_ruta_datos = os.getenv("DATA_PATH", "data/credit_risk_dataset.csv")
DATA_PATH = (
    Path(_ruta_datos)
    if Path(_ruta_datos).is_absolute()
    else RAIZ / _ruta_datos
)

# Decisiones cerradas en el notebook. No se repiten en el resto de src/.
TARGET = "loan_status"
TEST_SIZE = 0.3
METRIC = "accuracy"
EDAD_MIN = 20
EDAD_MAX = 70
CATEGORICAS = (
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
)


def activar_seguimiento():
    """Apunta MLflow al almacén y al experimento de este proyecto."""
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

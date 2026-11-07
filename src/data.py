"""Cargar, limpiar y partir los datos del prototipo de crédito.

Sigue `notebooks/01_exploracion.ipynb`:

- quita duplicados y filas incompletas;
- deja edades entre 20 y 70 años, y no menores que la antigüedad laboral;
- codifica las columnas de texto con `LabelEncoder`;
- escala el resto de features con `StandardScaler`, sin tocar el objetivo;
- parte entrenamiento y prueba con `TEST_SIZE` y `RANDOM_SEED`.

El codificador y el escalador se ajustan con todo el conjunto limpio,
igual que en el cuaderno, y después se hace la partición.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import config


def cargar_limpio() -> pd.DataFrame:
    """Lee `DATA_PATH` y aplica los filtros del notebook."""
    if not config.DATA_PATH.is_file():
        raise FileNotFoundError(
            f"No está el conjunto en {config.DATA_PATH}. "
            "Revisa DATA_PATH en .env."
        )

    datos = pd.read_csv(config.DATA_PATH)
    faltan = [c for c in (config.TARGET, *config.CATEGORICAS) if c not in datos.columns]
    if faltan:
        raise ValueError(f"Faltan columnas en {config.DATA_PATH.name}: {faltan}")

    datos = datos.drop_duplicates()
    datos = datos.dropna()
    datos = datos[
        (datos["person_age"] >= config.EDAD_MIN)
        & (datos["person_age"] <= config.EDAD_MAX)
    ]
    datos = datos[datos["person_age"] >= datos["person_emp_length"]]
    return datos.reset_index(drop=True)


def ajustar_preprocesador(datos: pd.DataFrame) -> dict:
    """Ajusta el codificador y el escalador. No incluye `loan_status`."""
    trabajo = datos.copy()
    encoders = {}
    for columna in config.CATEGORICAS:
        encoder = LabelEncoder()
        trabajo[columna] = encoder.fit_transform(trabajo[columna].astype(str))
        encoders[columna] = encoder

    numericas = [
        columna
        for columna in trabajo.columns
        if columna not in config.CATEGORICAS and columna != config.TARGET
    ]
    scaler = StandardScaler()
    scaler.fit(trabajo[numericas])
    features = [columna for columna in datos.columns if columna != config.TARGET]
    return {
        "encoders": encoders,
        "scaler": scaler,
        "numericas": numericas,
        "features": features,
    }


def aplicar_preprocesador(datos: pd.DataFrame, preprocesador: dict) -> pd.DataFrame:
    """Devuelve las features codificadas y escaladas, en el orden de entrenamiento."""
    trabajo = datos.copy()
    if config.TARGET in trabajo.columns:
        trabajo = trabajo.drop(columns=[config.TARGET])

    faltan = [c for c in preprocesador["features"] if c not in trabajo.columns]
    if faltan:
        raise ValueError(f"Faltan features para predecir: {faltan}")

    for columna, encoder in preprocesador["encoders"].items():
        trabajo[columna] = encoder.transform(trabajo[columna].astype(str))

    numericas = preprocesador["numericas"]
    escaladas = pd.DataFrame(
        preprocesador["scaler"].transform(trabajo[numericas]),
        columns=numericas,
        index=trabajo.index,
    )
    trabajo[numericas] = escaladas
    return trabajo[preprocesador["features"]]


def cargar_particion():
    """Devuelve train, test y el preprocesador ajustado en el prototipo."""
    datos = cargar_limpio()
    preprocesador = ajustar_preprocesador(datos)
    features = aplicar_preprocesador(datos, preprocesador)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        datos[config.TARGET],
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
    )
    return x_train, x_test, y_train, y_test, preprocesador


def fila_cruda_ejemplo() -> pd.DataFrame:
    """Una observación sin el objetivo, para probar `predict` sin un archivo extra."""
    datos = cargar_limpio()
    return datos.drop(columns=[config.TARGET]).head(1)

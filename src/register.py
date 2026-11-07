"""Registra el mejor run con el alias `challenger`.

Documento de responsabilidad. No hay implementación.

Cuando se escriba el código, este módulo:

- busca los runs del experimento definido en `src/config.py`;
- elige el mejor según la métrica que fije el entrenamiento;
- registra esa versión del modelo;
- le asigna el alias `challenger` (`MLFLOW_MODEL_ALIAS`).

Es el entry point `register` de `MLproject`
(`mlflow run . -e register`).

El alias es el contrato con predicción y con el servicio. No se
publica un número de versión fijo.
"""

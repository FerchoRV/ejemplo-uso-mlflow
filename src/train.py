"""Entrena y registra el run en MLflow.

Documento de responsabilidad. No hay implementación.

Cuando se escriba el código, este módulo:

- toma los datos ya partidos por `src/data.py`;
- entrena con la semilla y los nombres de `src/config.py`;
- abre el seguimiento en `MLFLOW_TRACKING_URI`;
- registra parámetros, métricas y el modelo dentro del experimento.

Es el entry point `main` de `MLproject` (`mlflow run .`).
Al ejecutarse, MLflow genera `mlflow.db` y `mlruns/`. Esos caminos
están en `.gitignore`.
"""

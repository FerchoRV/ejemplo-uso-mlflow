"""Nombres del experimento y del modelo, y semilla.

Documento de responsabilidad. No hay implementación.

Cuando se escriba el código, este módulo será el único lugar donde se
definen:

- el nombre del experimento de MLflow (`MLFLOW_EXPERIMENT_NAME`);
- el nombre del modelo registrado (`MLFLOW_MODEL_NAME`);
- el alias con el que se publica y se consume (`MLFLOW_MODEL_ALIAS`);
- la semilla (`RANDOM_SEED`).

Los valores de ejemplo viven en `.env.example`. El resto de `src/`
debe leerlos desde aquí, no repetirlos.
"""

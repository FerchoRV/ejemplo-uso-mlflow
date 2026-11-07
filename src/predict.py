"""Carga el modelo por alias y predice.

Documento de responsabilidad. No hay implementación.

Cuando se escriba el código, este módulo:

- resuelve el modelo como `models:/<MLFLOW_MODEL_NAME>@<alias>`;
- el alias por defecto es `challenger`;
- recibe las features de una observación o de un lote;
- devuelve la predicción del modelo cargado.

Es el entry point `predict` de `MLproject`
(`mlflow run . -e predict`).

No elige una versión numérica. Si `register.py` mueve el alias, esta
pieza sigue al candidato actual.
"""

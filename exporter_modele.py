# ===== 1. Importer les bibliothèques =====
import os
import shutil

import mlflow
import mlflow.sklearn

# ===== 2. Configuration (Docker Compose ou poste local) =====
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_NAME = os.getenv("MODEL_NAME", "churn-model")
ALIAS = os.getenv("MODEL_ALIAS", "champion")
DESTINATION = os.getenv("MODEL_DIR", "model")

# ===== 3. Charger le champion depuis le registre =====
mlflow.set_tracking_uri(TRACKING_URI)
model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{ALIAS}")

# ===== 4. Remplacer l'ancien dossier d'export =====
if os.path.exists(DESTINATION):
    shutil.rmtree(DESTINATION)

# ===== 5. Exporter le modèle pour un usage hors registre =====
mlflow.sklearn.save_model(
    model,
    DESTINATION,
    serialization_format="cloudpickle",
)

print(f"Champion {MODEL_NAME}@{ALIAS} exporté dans {DESTINATION}/")

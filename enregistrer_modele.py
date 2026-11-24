# ===== 1. Importer les bibliothèques =====
import os
import sys

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.ensemble import RandomForestClassifier

# Windows : éviter le crash d'affichage de l'emoji MLflow sur une console cp1252
for flux in (sys.stdout, sys.stderr):
    if hasattr(flux, "reconfigure"):
        flux.reconfigure(errors="replace")

# ===== 2. Configuration (Docker Compose ou poste local) =====
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "churn-prediction")
MODEL_NAME = os.getenv("MODEL_NAME", "churn-model")
ALIAS = os.getenv("MODEL_ALIAS", "champion")
FORCE_RETRAIN = os.getenv("FORCE_RETRAIN", "0") == "1"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

# ===== 3. Ne rien refaire si le champion existe déjà =====
if not FORCE_RETRAIN:
    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
        print(
            f"'{MODEL_NAME}@{ALIAS}' existe déjà "
            f"(version {champion.version}) : rien à faire."
        )
        print("Relancer avec FORCE_RETRAIN=1 pour réentraîner.")
        raise SystemExit(0)
    except MlflowException:
        pass  # pas encore de champion : on entraîne

# ===== 4. Lire et préparer les données =====
df = pd.read_csv("churn_data.csv")
X = pd.get_dummies(df.drop(columns=["churn"]), columns=["abonnement"])
y = df["churn"]

# ===== 5. Entraîner le modèle =====
modele = RandomForestClassifier(n_estimators=100, random_state=42)
modele.fit(X, y)

# ===== 6. Enregistrer le modèle dans le Registry =====
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="champion-compose"):
    mlflow.sklearn.log_model(
        modele,
        name="model",
        registered_model_name=MODEL_NAME,
        serialization_format="cloudpickle",
    )

# ===== 7. Poser l'alias sur la version la plus récente =====
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
version = max(versions, key=lambda v: int(v.version)).version

client.set_registered_model_alias(MODEL_NAME, ALIAS, version)
print(f"Version {version} de {MODEL_NAME} enregistrée avec l'alias '{ALIAS}'")

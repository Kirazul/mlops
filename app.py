# ===== 1. Importer les bibliothèques =====
import os
from typing import Literal

import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ===== 2. Configuration (surchargée par Docker / Docker Compose) =====
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_URI = os.getenv("MODEL_URI", "model")

# ===== 3. Créer l'application =====
app = FastAPI(title="API Churn", version="1.0.0")

# ===== 4. Charger le modèle une seule fois au démarrage =====
mlflow.set_tracking_uri(TRACKING_URI)

try:
    model = mlflow.sklearn.load_model(MODEL_URI)
except Exception as exc:
    raise RuntimeError(
        f"Impossible de charger le modèle depuis '{MODEL_URI}'. "
        "Vérifiez MLFLOW_TRACKING_URI et MODEL_URI."
    ) from exc

# Récupérer les colonnes attendues par le modèle
if not hasattr(model, "feature_names_in_"):
    raise RuntimeError(
        "Le modèle chargé n'expose pas feature_names_in_ : "
        "il doit être entraîné avec un DataFrame nommé."
    )

features = list(model.feature_names_in_)
print(f"Modèle chargé depuis {MODEL_URI} ({len(features)} variables)")

# ===== 5. Décrire les données reçues =====
class Client(BaseModel):
    age: int = Field(ge=0)
    salaire: float = Field(ge=0)
    abonnement: Literal["Basique", "Standard", "Premium"]
    anciennete: float = Field(ge=0)
    consommation: float = Field(ge=0)


# ===== 6. Routes de vérification =====
@app.get("/")
def root():
    return {"service": "churn-api", "model": MODEL_URI, "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


# ===== 7. Route de prédiction =====
@app.post("/predict")
def predict(client: Client):
    # a) Convertir les données en DataFrame
    ligne = pd.DataFrame([client.model_dump()])

    # b) Transformer abonnement en colonnes indicatrices
    ligne = pd.get_dummies(
        ligne,
        columns=["abonnement"],
        dtype=int
    )

    # c) Aligner les colonnes sur celles du modèle
    ligne = ligne.reindex(
        columns=features,
        fill_value=0
    )

    # d) Effectuer la prédiction
    try:
        resultat = int(model.predict(ligne)[0])
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur pendant la prédiction : {str(e)}"
        )

    # e) Retourner le résultat au format JSON
    return {
        "churn": resultat,
        "message": (
            "Le client va partir"
            if resultat == 1
            else "Le client reste"
        )
    }

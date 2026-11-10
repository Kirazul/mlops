# ===== 1. Importer les bibliothèques =====
import mlflow
import mlflow.sklearn

# ===== 2. Se connecter à la base MLflow =====
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# ===== 3. Charger le champion depuis le registre =====
model = mlflow.sklearn.load_model("models:/churn-model@champion")

# ===== 4. Exporter le modèle dans le dossier model/ =====
mlflow.sklearn.save_model(
    model,
    "model",
    skops_trusted_types=["sklearn.tree._tree.Tree"],
)

print("Modèle champion exporté dans model/")

import sys

import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Windows : éviter le crash d'affichage de l'emoji MLflow sur une console cp1252
for flux in (sys.stdout, sys.stderr):
    if hasattr(flux, "reconfigure"):
        flux.reconfigure(errors="replace")

df = pd.read_csv("churn_data.csv")
X = pd.get_dummies(df[["age", "salaire", "abonnement", "anciennete", "consommation"]],
                   drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, df["churn"], test_size=0.2, random_state=42)

mlflow.set_experiment("churn-prediction")

mlflow.sklearn.autolog(serialization_format="cloudpickle")

with mlflow.start_run(run_name="essai-auto"):
    model = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    print("score test :", model.score(X_test, y_test))

# MLOps - Churn Prediction

**Author:** Mohamed Aziz Mansour

A small MLOps project for customer churn prediction. It generates a synthetic
customer dataset, trains and evaluates a first classification model, and saves
the trained model for later use.

## Contents

- `generate_data.py` - generates the synthetic churn dataset.
- `churn_data.csv` - generated customer data.
- `churn_model.ipynb` - notebook used to train and evaluate the first model.
- `model.pkl` - saved trained model.
- `train_mlflow.py` - trains three churn models and logs them as MLflow runs.
- `lire_runs.py` - reads the MLflow runs and ranks them by F1 score.
- `charger_modele.py` - reloads the best model from a run and makes a prediction.
- `train_autolog.py` - bonus: logs a run automatically with MLflow autolog.

## MLflow experiment tracking

This branch adds experiment tracking with MLflow 3.17.1. All runs are grouped in
the `churn-prediction` experiment.

- Runs: `essai-1`, `essai-2`, `essai-3`, plus `essai-auto` from autolog.
- Logged per run: parameters (`n_estimators`, `max_depth`, `features`), metrics
  (accuracy, precision, recall, f1), and the trained model.
- Results on the test set: `essai-1` and `essai-3` reach an F1 of 0.889, while
  `essai-2` reaches 0.333.

Run the experiments and open the MLflow UI (activate the virtual environment
first):

```powershell
python train_mlflow.py
mlflow ui
```

Then open http://127.0.0.1:5000 and select the `churn-prediction` experiment.
`mlruns/` and `mlflow.db` stay local and are ignored by Git; the scripts can be
re-run at any time to reproduce the runs.

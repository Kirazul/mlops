# Reflection

## TP3 - Experiment tracking with MLflow

1. **What is the best model? Can you prove it?**
   Yes. Each attempt is saved as an MLflow run with its parameters and metrics.
   Sorting the runs by F1 (in the MLflow UI or with `search_runs`) shows that
   `essai-1` and `essai-3` are the best models (F1 = 0.889), while `essai-2` is
   the weakest one (F1 = 0.333).

2. **Can you give back exactly the model from Essai 1?**
   Yes. Every run keeps its own model artifact.
   `mlflow.sklearn.load_model("runs:/<run_id>/model")` reloads exactly the model
   of the chosen run, without retraining.

3. **Can a classmate reproduce your work on another machine?**
   Partly. The code is versioned with Git, and each run stores its parameters
   and the versions of the libraries used. A fully identical environment would
   require Docker (TP5-6).

# MLOps – Prédiction du churn client

Projet MLOps du cours : modèle de classification qui prédit si un client va résilier son abonnement (churn), suivi des expériences avec MLflow, API FastAPI et déploiement conteneurisé avec Docker / Docker Compose.

## Sommaire

1. [Structure du dépôt](#structure-du-dépôt)
2. [TP1 – Données et modèle](#tp1--données-et-modèle)
3. [TP3 – Suivi des expériences MLflow](#tp3--suivi-des-expériences-mlflow)
4. [TP4 – Registre de modèles et API FastAPI](#tp4--registre-de-modèles-et-api-fastapi)
5. [TP5 – Docker et Docker Compose](#tp5--docker-et-docker-compose)
6. [Branches](#branches)

## Structure du dépôt

| Fichier | Rôle |
|---|---|
| `generate_data.py` | Génère 500 clients fictifs dans `churn_data.csv` |
| `churn_data.csv` | Jeu de données généré |
| `churn_model.ipynb` | Exploration, entraînement et évaluation |
| `model.pkl` | Modèle entraîné avec `joblib` |
| `train_mlflow.py` | Entraîne 3 essais et les journalise dans MLflow |
| `register_model.py` | Enregistre le meilleur run dans le registre et pose l'alias `champion` |
| `load_champion.py` | Charge le champion depuis le registre et teste une prédiction |
| `deplacer_alias.py` | Déplace / restaure l'alias `champion` |
| `app.py` | API FastAPI (`/health`, `/predict`) |
| `test_api.py` | Tests de l'API (Health, prédictions, validation) |
| `export_champion.py` | Exporte le champion dans `model/` pour l'image Docker |
| `Dockerfile` | Image de l'API churn |
| `.dockerignore` | Fichiers exclus de l'image |
| `requirements.txt` | Dépendances épinglées de l'image |
| `docker-compose.yml` | Stack MLflow + API |

## TP1 – Données et modèle

Chaque client est décrit par `age`, `salaire`, `abonnement`, `anciennete`, `consommation` et la cible `churn` (1 si `consommation < 30` et `anciennete < 2`).

`RandomForestClassifier` (scikit-learn), découpage 80/20 (`random_state=42`).

| Essai | Paramètres | Features | Accuracy |
|---|---|---|---|
| 1 | 100 arbres, `max_depth=5` | 5 | 0.99 |
| 2 | 50 arbres, `max_depth=3` | 5 | 0.99 |
| 3 | 100 arbres, `max_depth=5` | 4 (sans `salaire`) | 1.00 |

## TP3 – Suivi des expériences MLflow

```powershell
python train_mlflow.py     # crée l'expérience churn-prediction et les 3 runs
mlflow ui                  # interface sur http://localhost:5000
```

Chaque run journalise les paramètres, les métriques (`accuracy`, `precision`, `recall`, `f1`) et le modèle.

## TP4 – Registre de modèles et API FastAPI

```powershell
python register_model.py   # meilleur run par f1 -> churn-model, alias champion
python load_champion.py    # vérifie le chargement du champion
uvicorn app:app --host 0.0.0.0 --port 8000
python test_api.py         # GET /health + POST /predict
```

L'API lit `MLFLOW_TRACKING_URI` (défaut `sqlite:///mlflow.db`) et `MODEL_URI` (défaut `model`).

## TP5 – Docker et Docker Compose

### Prérequis

- Docker Desktop démarré (`docker --version`, `docker info`)

### Image seule

```powershell
python export_champion.py                  # exporte le champion dans model/
docker build -t churn-api:1.0 .
docker run -d -p 8000:8000 --name api churn-api:1.0
curl.exe http://localhost:8000/health
python test_api.py
docker logs api
docker stop api
docker rm api
```

### Docker Compose (MLflow + API)

Au premier lancement, le registre MLflow est vide : on l'alimente avant de démarrer l'API.

```powershell
docker compose up -d mlflow
curl.exe http://localhost:5000/health

$env:MLFLOW_TRACKING_URI="http://localhost:5000"
$env:PYTHONUTF8="1"
python train_mlflow.py
python register_model.py
Remove-Item Env:MLFLOW_TRACKING_URI
Remove-Item Env:PYTHONUTF8

docker compose up -d --build
docker compose ps
curl.exe http://localhost:8000/health
python test_api.py
docker compose logs -f api
```

Arrêt :

```powershell
docker compose stop        # arrête sans supprimer
docker compose down        # supprime conteneurs et réseau (volume conservé)
docker compose down -v     # supprime aussi le volume mlflow-data
```

Architecture :

```
Navigateur / test_api.py
        |  :8000
        v
      api  ──(http://mlflow:5000)──>  mlflow  ──> volume mlflow-data
 (modèle champion)                 (registre + artefacts)
```

## Branches

- `master` : version stable
- `develop` : intégration
- `feature/*` : développement des nouvelles fonctionnalités

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Couche de dépendances mise en cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code de l'API + scripts utilisés par le service d'initialisation Compose
COPY app.py enregistrer_modele.py exporter_modele.py churn_data.csv ./

# Par défaut : champion du registre MLflow (surchargé par Compose)
ENV MODEL_URI=models:/churn-model@champion

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

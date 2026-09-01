# Diabetes Classification MLOps

A lightweight end-to-end MLOps project for binary diabetes classification.

## Stack
Python, Pandas, Scikit-learn, MLflow, FastAPI, Docker, Kubernetes/Minikube, GitHub Actions.

## Pipeline
Raw CSV -> ingestion -> validation -> preprocessing -> model comparison -> MLflow -> best model -> FastAPI -> Docker -> Kubernetes.

## Models
- Logistic Regression
- Random Forest

The best model is selected by F1 score.

## Dataset
Pima Indians Diabetes Database. Put the CSV at `data/raw/diabetes.csv`.

## Local run
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/ingest.py
python src/validate.py
python src/train.py
mlflow ui
uvicorn api.app:app --reload
```

API docs: http://127.0.0.1:8000/docs

## Docker
```bash
docker build -t diabetes-api .
docker run --rm -p 8000:8000 diabetes-api
```

## Kubernetes
```bash
minikube start --cpus=2 --memory=2048
eval $(minikube docker-env)
docker build -t diabetes-api:latest .
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
minikube service diabetes-service
```

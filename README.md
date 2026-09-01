# Diabetes Classification MLOps

An end-to-end, lightweight MLOps pipeline for binary diabetes classification — built to demonstrate a full production-style ML workflow: data validation, experiment tracking, model serving, containerization, and orchestration, without any cloud dependency.

## Project story

Built a reproducible classification pipeline with automated data ingestion and quality validation, multi-model evaluation with MLflow experiment tracking, and F1-based model selection. The selected model is served through a FastAPI REST API, containerized with Docker, and deployed on Kubernetes (Minikube) with liveness/readiness health checks. GitHub Actions runs automated tests and a Docker build on every push.

## Architecture

```
Raw CSV
   │
   ▼
Ingestion (src/ingest.py)
   │
   ▼
Validation (src/validate.py)  ──► FAIL ──► stop
   │ PASS
   ▼
Preprocessing + Training (src/train.py)
   │
   ├── Logistic Regression
   └── Random Forest
   │
   ▼
MLflow tracking ──► best model selected by F1
   │
   ▼
FastAPI (api/app.py)
   /health  /predict  /model-info
   │
   ▼
Docker image (diabetes-api:1.0)
   │
   ▼
Kubernetes / Minikube
   Deployment ──► Pod ──► Service ──► client
```

CI (GitHub Actions) runs on every push: install deps → run tests → build Docker image.

## Stack

Python, Pandas, Scikit-learn, MLflow, FastAPI, Docker, Kubernetes (Minikube), GitHub Actions, pytest.

## Models

- Logistic Regression
- Random Forest

Best model selected by **F1 score**, with accuracy, precision, and recall also logged for comparison.

## Dataset

Pima Indians Diabetes Database. Place the CSV at `data/raw/diabetes.csv`.

## Project structure

```
diabetes-mlops/
├── data/
│   ├── raw/diabetes.csv
│   └── processed/          (generated, git-ignored)
├── src/
│   ├── ingest.py
│   ├── validate.py
│   └── train.py
├── api/
│   └── app.py
├── model/
│   ├── model.pkl
│   └── metadata.json
├── tests/
│   ├── test_validation.py
│   └── test_api.py
├── kubernetes/
│   ├── deployment.yaml
│   └── service.yaml
├── .github/workflows/ci.yml
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md
```

## Local run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/ingest.py
python src/validate.py
python src/train.py

mlflow ui          # view experiment tracking at http://127.0.0.1:5000
uvicorn api.app:app --reload
```

API docs (Swagger UI): http://127.0.0.1:8000/docs

### Example request

```json
POST /predict
{
  "Pregnancies": 2,
  "Glucose": 120,
  "BloodPressure": 70,
  "SkinThickness": 25,
  "Insulin": 80,
  "BMI": 30.5,
  "DiabetesPedigreeFunction": 0.45,
  "Age": 35
}
```

```json
{
  "prediction": 0,
  "probability": 0.31
}
```

## Docker

```bash
docker build -t diabetes-api:1.0 .
docker run -d --name diabetes-api -p 8000:8000 diabetes-api:1.0
```

Test at http://127.0.0.1:8000/docs

## Kubernetes (Minikube)

```bash
# Start a local cluster
minikube start --driver=docker

# Build the image, then load it into Minikube's image store
docker build -t diabetes-api:1.0 .
minikube image load diabetes-api:1.0

# Deploy
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml

# Check status
kubectl get pods
kubectl get svc

# Access the API
minikube service diabetes-api-service --url
```

The Deployment includes `/health` liveness and readiness probes, so Kubernetes automatically restarts the pod if the API becomes unresponsive.

## Testing

```bash
pytest
```

## CI/CD

Every push to `main` triggers a GitHub Actions workflow (`.github/workflows/ci.yml`) that:
1. Installs dependencies
2. Runs the `pytest` suite
3. Builds the Docker image

from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = Path("model/model.pkl")
METADATA_PATH = Path("model/metadata.json")


# --------------------------------------------------
# Load Model
# --------------------------------------------------

if not MODEL_PATH.exists():
    raise RuntimeError(
        "Model not found. Run: python src/train.py"
    )

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# Load Metadata
# --------------------------------------------------

metadata = {}

if METADATA_PATH.exists():
    import json

    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)


# --------------------------------------------------
# FastAPI Application
# --------------------------------------------------

app = FastAPI(
    title="Diabetes Classification API",
    description="Lightweight MLOps prediction API",
    version="1.0.0",
)


# --------------------------------------------------
# Request Schema
# --------------------------------------------------

class PatientData(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float


# --------------------------------------------------
# Root
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Diabetes Classification API",
        "docs": "/docs",
    }


# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# --------------------------------------------------
# Model Information
# --------------------------------------------------

@app.get("/model-info")
def model_info():
    return metadata


# --------------------------------------------------
# Prediction
# --------------------------------------------------

@app.post("/predict")
def predict(patient: PatientData):

    try:

        features = [[
            patient.Pregnancies,
            patient.Glucose,
            patient.BloodPressure,
            patient.SkinThickness,
            patient.Insulin,
            patient.BMI,
            patient.DiabetesPedigreeFunction,
            patient.Age,
        ]]

        prediction = int(
            model.predict(features)[0]
        )

        probability = float(
            model.predict_proba(features)[0][1]
        )

        return {
            "prediction": prediction,
            "probability": round(
                probability,
                4
            ),
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


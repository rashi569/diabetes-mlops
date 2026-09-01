from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from validate import validate_data


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA = "data/processed/diabetes.csv"
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

MIN_F1 = 0.70


# --------------------------------------------------
# Preprocessing + Model Pipeline
# --------------------------------------------------

def make_pipeline(model):
    return Pipeline([
        (
            "imputer",
            SimpleImputer(
                missing_values=0,
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            model
        ),
    ])


# --------------------------------------------------
# Train + Evaluate One Model
# --------------------------------------------------

def evaluate(
    name,
    model,
    X_train,
    X_test,
    y_train,
    y_test
):
    mlflow.set_experiment("Diabetes-Classification")

    with mlflow.start_run(run_name=name) as run:

        # Train
        model.fit(X_train, y_train)

        # Predict
        predictions = model.predict(X_test)

        # Metrics
        metrics = {
            "accuracy": accuracy_score(
                y_test,
                predictions
            ),
            "precision": precision_score(
                y_test,
                predictions,
                zero_division=0
            ),
            "recall": recall_score(
                y_test,
                predictions,
                zero_division=0
            ),
            "f1": f1_score(
                y_test,
                predictions,
                zero_division=0
            ),
        }

        # MLflow parameters
        mlflow.log_param("model", name)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        # MLflow metrics
        for metric_name, value in metrics.items():
            mlflow.log_metric(
                metric_name,
                float(value)
            )

        # Save model to MLflow
        mlflow.sklearn.log_model(
            model,
            name="model",
            serialization_format="cloudpickle"
        )

        # Display results
        print(
            f"{name}: "
            f"F1={metrics['f1']:.4f}, "
            f"Accuracy={metrics['accuracy']:.4f}"
        )

        return (
            model,
            metrics,
            run.info.run_id
        )


# --------------------------------------------------
# Main Training Pipeline
# --------------------------------------------------

def train():

    # 1. Validate data
    df = validate_data(DATA)

    # 2. Separate features and target
    X = df[FEATURES]
    y = df["Outcome"]

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 4. Candidate models
    candidates = [

        (
            "LogisticRegression",
            make_pipeline(
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            )
        ),

        (
            "RandomForest",
            make_pipeline(
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=42,
                    n_jobs=1
                )
            )
        ),
    ]

    # 5. Store results
    results = []

    # 6. Train each candidate
    for name, model in candidates:

        fitted_model, metrics, run_id = evaluate(
            name,
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        results.append(
            (
                metrics["f1"],
                name,
                fitted_model,
                metrics,
                run_id
            )
        )

    # 7. Select model with highest F1
    (
        best_f1,
        best_name,
        best_model,
        best_metrics,
        best_run
    ) = max(
        results,
        key=lambda x: x[0]
    )

    # 8. Model quality gate
    if best_f1 < MIN_F1:

        raise RuntimeError(
            f"No model met the minimum F1 threshold "
            f"of {MIN_F1}. "
            f"Best F1: {best_f1:.4f}"
        )

    # 9. Save winning model
    model_path = MODEL_DIR / "model.pkl"

    joblib.dump(
        best_model,
        model_path
    )

    # 10. Save model metadata
    metadata = {
        "model": best_name,
        "version": "1",
        "f1": round(
            float(best_metrics["f1"]),
            4
        ),
        "accuracy": round(
            float(best_metrics["accuracy"]),
            4
        ),
        "precision": round(
            float(best_metrics["precision"]),
            4
        ),
        "recall": round(
            float(best_metrics["recall"]),
            4
        ),
        "min_f1_threshold": MIN_F1,
        "mlflow_run_id": best_run,
    }

    metadata_path = MODEL_DIR / "metadata.json"

    pd.Series(metadata).to_json(
        metadata_path
    )

    # 11. Final output
    print()
    print(f"BEST MODEL: {best_name}")
    print(f"BEST F1: {best_f1:.4f}")
    print(f"Saved model to {model_path}")
    print(f"Saved metadata to {metadata_path}")


# --------------------------------------------------
# Entry Point
# --------------------------------------------------

if __name__ == "__main__":
    train()

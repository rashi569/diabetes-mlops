import pandas as pd

REQUIRED = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

def validate_data(path="data/processed/diabetes.csv"):
    df = pd.read_csv(path)

    if list(df.columns) != REQUIRED:
        raise ValueError(f"Invalid schema. Expected: {REQUIRED}")
    if df.empty:
        raise ValueError("Dataset is empty")
    if df.duplicated().any():
        raise ValueError("Duplicate rows found")
    if not set(df["Outcome"].unique()).issubset({0, 1}):
        raise ValueError("Outcome must contain only 0 and 1")

    for col in REQUIRED[:-1]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"{col} must be numeric")
        if (df[col] < 0).any():
            raise ValueError(f"Negative value found in {col}")

    print("DATA VALIDATION PASSED")
    return df

if __name__ == "__main__":
    validate_data()

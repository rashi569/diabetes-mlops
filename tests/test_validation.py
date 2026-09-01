import pandas as pd
import pytest
from src.validate import validate_data

def test_validation_accepts_valid_data(tmp_path):
    cols = [
        "Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin",
        "BMI","DiabetesPedigreeFunction","Age","Outcome"
    ]
    df = pd.DataFrame([[1,120,70,20,80,30,0.5,30,0]], columns=cols)
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    assert len(validate_data(path)) == 1

def test_validation_rejects_bad_target(tmp_path):
    cols = [
        "Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin",
        "BMI","DiabetesPedigreeFunction","Age","Outcome"
    ]
    df = pd.DataFrame([[1,120,70,20,80,30,0.5,30,2]], columns=cols)
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError):
        validate_data(path)

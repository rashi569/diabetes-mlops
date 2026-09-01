from pathlib import Path
import pandas as pd

RAW = Path("data/raw/diabetes.csv")
PROCESSED = Path("data/processed/diabetes.csv")

def ingest_data():
    if not RAW.exists():
        raise FileNotFoundError(
            "Put the Pima diabetes CSV at data/raw/diabetes.csv"
        )
    df = pd.read_csv(RAW)
    if df.empty:
        raise ValueError("Dataset is empty")
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED, index=False)
    print(f"Loaded {len(df)} rows and saved {PROCESSED}")

if __name__ == "__main__":
    ingest_data()

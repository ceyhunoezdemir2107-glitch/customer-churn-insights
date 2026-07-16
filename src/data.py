from pathlib import Path

import pandas as pd


RAW_DATA_PATH = Path("data/raw/telco_customer_churn.csv")
TARGET_COLUMN = "Churn"


def load_raw_data(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw Telco churn dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Add the Telco churn CSV before running the pipeline."
        )
    return pd.read_csv(path)


def clean_telco_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean common Telco churn data issues."""
    cleaned = df.copy()

    if "customerID" in cleaned.columns:
        cleaned = cleaned.drop(columns=["customerID"])

    if "TotalCharges" in cleaned.columns:
        cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
        if "tenure" in cleaned.columns:
            cleaned["TotalCharges"] = cleaned["TotalCharges"].fillna(
                cleaned["tenure"] * cleaned.get("MonthlyCharges", 0)
            )
        cleaned["TotalCharges"] = cleaned["TotalCharges"].fillna(cleaned["TotalCharges"].median())

    if TARGET_COLUMN in cleaned.columns:
        cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].map({"No": 0, "Yes": 1})

    return cleaned


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned dataframe into features and target."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in dataframe.")
    return df.drop(columns=[TARGET_COLUMN]), df[TARGET_COLUMN]

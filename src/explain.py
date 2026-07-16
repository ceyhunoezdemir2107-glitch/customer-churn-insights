from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("models/churn_model.joblib")


def load_model(path: str | Path = MODEL_PATH):
    """Load a persisted churn model."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Run training first.")
    return joblib.load(path)


def permutation_importance_frame(feature_names: list[str], importances: list[float]) -> pd.DataFrame:
    """Create a sorted feature-importance dataframe."""
    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

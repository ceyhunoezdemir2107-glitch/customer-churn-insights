from dataclasses import asdict
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data import clean_telco_data, load_raw_data, split_features_target
from src.evaluate import evaluate_classifier
from src.features import build_preprocessor


MODEL_DIR = Path("models")
REPORTS_DIR = Path("reports")
RANDOM_STATE = 42


def build_models(x_train):
    """Create candidate churn models."""
    return {
        "dummy_baseline": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(x_train)),
                (
                    "classifier",
                    DummyClassifier(strategy="most_frequent"),
                ),
            ]
        ),
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(x_train)),
                (
                    "classifier",
                    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(x_train)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(x_train)),
                (
                    "classifier",
                    GradientBoostingClassifier(random_state=RANDOM_STATE),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(x_train)),
                (
                    "classifier",
                    HistGradientBoostingClassifier(random_state=RANDOM_STATE),
                ),
            ]
        ),
    }


def metrics_to_frame(results: dict) -> pd.DataFrame:
    """Convert model metrics into a sorted comparison table."""
    rows = [
        {"model": model_name, **asdict(metrics)}
        for model_name, metrics in results.items()
    ]
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)


def train_and_select_model():
    """Train candidate models and persist the best model by ROC-AUC."""
    raw = load_raw_data()
    cleaned = clean_telco_data(raw)
    x, y = split_features_target(cleaned)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    results = {}
    fitted_models = {}
    for name, model in build_models(x_train).items():
        model.fit(x_train, y_train)
        metrics = evaluate_classifier(model, x_test, y_test)
        results[name] = metrics
        fitted_models[name] = model

    comparison = metrics_to_frame(results)
    best_name = comparison.iloc[0]["model"]

    MODEL_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    joblib.dump(fitted_models[best_name], MODEL_DIR / "churn_model.joblib")
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    return best_name, comparison


def main() -> None:
    best_name, comparison = train_and_select_model()
    print(f"Best model: {best_name}")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float


def evaluate_classifier(model, x_test: pd.DataFrame, y_test: pd.Series) -> ClassificationMetrics:
    """Evaluate a fitted binary classifier."""
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]

    return ClassificationMetrics(
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred, zero_division=0),
        recall=recall_score(y_test, y_pred, zero_division=0),
        f1=f1_score(y_test, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_test, y_proba),
    )


def confusion_matrix_frame(model, x_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Return confusion matrix as a labeled dataframe."""
    matrix = confusion_matrix(y_test, model.predict(x_test))
    return pd.DataFrame(
        matrix,
        index=["Actual no churn", "Actual churn"],
        columns=["Predicted no churn", "Predicted churn"],
    )

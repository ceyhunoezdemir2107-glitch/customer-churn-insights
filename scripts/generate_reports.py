from pathlib import Path
import sys

import joblib
import matplotlib
import pandas as pd
import seaborn as sns
import shap
from matplotlib import pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import clean_telco_data, load_raw_data, split_features_target
from src.train import RANDOM_STATE


MODEL_PATH = Path("models/churn_model.joblib")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"


matplotlib.use("Agg")
sns.set_theme(style="whitegrid")


def save_current_figure(filename: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close()


def plot_churn_distribution(raw: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(data=raw, x="Churn", hue="Churn", palette="Set2", legend=False)
    total = len(raw)
    for patch in ax.patches:
        count = patch.get_height()
        ax.annotate(f"{count / total:.1%}", (patch.get_x() + patch.get_width() / 2, count), ha="center", va="bottom", fontsize=10)
    ax.set_title("Churn Distribution")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Customers")
    save_current_figure("churn_distribution.png")


def plot_churn_by_contract(raw: pd.DataFrame) -> None:
    contract_rates = (
        raw.assign(churn_flag=raw["Churn"].map({"No": 0, "Yes": 1}))
        .groupby("Contract", as_index=False)["churn_flag"]
        .mean()
        .sort_values("churn_flag", ascending=False)
    )

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=contract_rates, x="Contract", y="churn_flag", hue="Contract", palette="Set2", legend=False)
    ax.set_title("Churn Rate by Contract Type")
    ax.set_xlabel("Contract")
    ax.set_ylabel("Churn rate")
    ax.set_ylim(0, max(contract_rates["churn_flag"]) * 1.2)
    for patch in ax.patches:
        value = patch.get_height()
        ax.annotate(f"{value:.1%}", (patch.get_x() + patch.get_width() / 2, value), ha="center", va="bottom", fontsize=10)
    save_current_figure("churn_by_contract.png")


def plot_tenure_distribution(raw: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 5))
    ax = sns.histplot(data=raw, x="tenure", hue="Churn", bins=30, multiple="layer", palette="Set2")
    ax.set_title("Tenure Distribution by Churn Status")
    ax.set_xlabel("Tenure in months")
    ax.set_ylabel("Customers")
    save_current_figure("tenure_distribution_by_churn.png")


def plot_monthly_charges(raw: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5))
    ax = sns.boxplot(data=raw, x="Churn", y="MonthlyCharges", hue="Churn", palette="Set2", legend=False)
    ax.set_title("Monthly Charges by Churn Status")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Monthly charges")
    save_current_figure("monthly_charges_by_churn.png")


def plot_model_comparison() -> None:
    comparison_path = REPORTS_DIR / "model_comparison.csv"
    if not comparison_path.exists():
        raise FileNotFoundError("reports/model_comparison.csv not found. Run `python -m src.train` first.")

    comparison = pd.read_csv(comparison_path).sort_values("roc_auc", ascending=True)
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=comparison, x="roc_auc", y="model", hue="model", palette="Set2", legend=False)
    ax.set_title("Model Comparison by ROC-AUC")
    ax.set_xlabel("ROC-AUC")
    ax.set_ylabel("Model")
    ax.set_xlim(0.45, 0.9)
    for patch in ax.patches:
        value = patch.get_width()
        ax.annotate(f"{value:.3f}", (value + 0.005, patch.get_y() + patch.get_height() / 2), va="center", fontsize=10)
    save_current_figure("model_roc_auc_comparison.png")


def clean_feature_name(feature_name: str) -> str:
    return feature_name.replace("numeric__", "").replace("categorical__", "")


def prepare_test_data() -> tuple[pd.DataFrame, pd.Series]:
    cleaned = clean_telco_data(load_raw_data())
    x, y = split_features_target(cleaned)
    _, x_test, _, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    return x_test, y_test


def plot_confusion_matrix(y_test: pd.Series, y_proba: pd.Series, threshold: float = 0.5) -> None:
    y_pred = (y_proba >= threshold).astype(int)
    matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No churn", "Churn"],
        yticklabels=["No churn", "Churn"],
    )
    ax.set_title(f"Confusion Matrix at Threshold {threshold:.2f}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    save_current_figure("confusion_matrix.png")


def plot_roc_curve(y_test: pd.Series, y_proba: pd.Series) -> None:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}", color="#2f6f9f")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random baseline")
    plt.title("ROC Curve")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(loc="lower right")
    save_current_figure("roc_curve.png")


def plot_precision_recall_curve(y_test: pd.Series, y_proba: pd.Series) -> None:
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"Average precision = {avg_precision:.3f}", color="#3d8b40")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="upper right")
    save_current_figure("precision_recall_curve.png")


def build_threshold_analysis(y_test: pd.Series, y_proba: pd.Series) -> pd.DataFrame:
    rows = []
    for threshold in [round(value / 100, 2) for value in range(5, 96, 5)]:
        y_pred = (y_proba >= threshold).astype(int)
        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "predicted_churn_rate": float(y_pred.mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_threshold_analysis(thresholds: pd.DataFrame) -> None:
    thresholds.to_csv(REPORTS_DIR / "threshold_analysis.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds["threshold"], thresholds["precision"], marker="o", label="Precision")
    plt.plot(thresholds["threshold"], thresholds["recall"], marker="o", label="Recall")
    plt.plot(thresholds["threshold"], thresholds["f1"], marker="o", label="F1")
    plt.title("Threshold Trade-off")
    plt.xlabel("Decision threshold")
    plt.ylabel("Metric value")
    plt.ylim(0, 1)
    plt.legend()
    save_current_figure("threshold_tradeoff.png")


def plot_permutation_importance(model, x_test: pd.DataFrame, y_test: pd.Series) -> None:
    result = permutation_importance(
        model,
        x_test,
        y_test,
        n_repeats=8,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
        n_jobs=-1,
    )
    importance = (
        pd.DataFrame(
            {
                "feature": x_test.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(REPORTS_DIR / "permutation_importance.csv", index=False)

    top_features = importance.head(12).sort_values("importance_mean", ascending=True)
    plt.figure(figsize=(9, 6))
    ax = sns.barplot(data=top_features, x="importance_mean", y="feature", hue="feature", palette="Set2", legend=False)
    ax.set_title("Top Features by Permutation Importance")
    ax.set_xlabel("Mean ROC-AUC decrease")
    ax.set_ylabel("Feature")
    save_current_figure("permutation_importance.png")


def plot_shap_summary(model, x_test: pd.DataFrame) -> None:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    sample = x_test.sample(n=min(500, len(x_test)), random_state=RANDOM_STATE)
    transformed = preprocessor.transform(sample)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    feature_names = [clean_feature_name(name) for name in preprocessor.get_feature_names_out()]
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(transformed)

    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    plt.figure(figsize=(9, 6))
    shap.summary_plot(
        shap_values,
        transformed,
        feature_names=feature_names,
        max_display=15,
        show=False,
    )
    save_current_figure("shap_summary.png")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw_data()
    model = joblib.load(MODEL_PATH)
    x_test, y_test = prepare_test_data()
    y_proba = model.predict_proba(x_test)[:, 1]

    plot_churn_distribution(raw)
    plot_churn_by_contract(raw)
    plot_tenure_distribution(raw)
    plot_monthly_charges(raw)
    plot_model_comparison()
    plot_confusion_matrix(y_test, y_proba)
    plot_roc_curve(y_test, y_proba)
    plot_precision_recall_curve(y_test, y_proba)
    plot_threshold_analysis(build_threshold_analysis(y_test, y_proba))
    plot_permutation_importance(model, x_test, y_test)
    plot_shap_summary(model, x_test)

    print(f"Generated figures in {FIGURES_DIR}")
    print(f"Generated table: {REPORTS_DIR / 'permutation_importance.csv'}")
    print(f"Generated table: {REPORTS_DIR / 'threshold_analysis.csv'}")


if __name__ == "__main__":
    main()

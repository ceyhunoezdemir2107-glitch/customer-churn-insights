from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import clean_telco_data, load_raw_data


MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
PERMUTATION_IMPORTANCE_PATH = REPORTS_DIR / "permutation_importance.csv"
THRESHOLD_ANALYSIS_PATH = REPORTS_DIR / "threshold_analysis.csv"


st.set_page_config(page_title="Customer Churn Insights", layout="wide")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_clean_data() -> pd.DataFrame | None:
    try:
        return clean_telco_data(load_raw_data(PROJECT_ROOT / "data" / "raw" / "telco_customer_churn.csv"))
    except FileNotFoundError:
        return None


@st.cache_data
def load_model_comparison() -> pd.DataFrame | None:
    if not MODEL_COMPARISON_PATH.exists():
        return None
    return pd.read_csv(MODEL_COMPARISON_PATH)


@st.cache_data
def load_permutation_importance() -> pd.DataFrame | None:
    if not PERMUTATION_IMPORTANCE_PATH.exists():
        return None
    return pd.read_csv(PERMUTATION_IMPORTANCE_PATH)


@st.cache_data
def load_threshold_analysis() -> pd.DataFrame | None:
    if not THRESHOLD_ANALYSIS_PATH.exists():
        return None
    return pd.read_csv(THRESHOLD_ANALYSIS_PATH)


def risk_label(probability: float) -> tuple[str, str]:
    if probability >= 0.7:
        return "High", "Customer should be prioritized for retention outreach."
    if probability >= 0.4:
        return "Medium", "Customer should be monitored and considered for targeted offers."
    return "Low", "Customer currently shows limited churn risk."


def build_customer_input() -> pd.DataFrame:
    left, middle, right = st.columns(3)

    with left:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior citizen", [0, 1], format_func=lambda value: "Yes" if value == 1 else "No")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure in months", min_value=0, max_value=72, value=3)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

    with middle:
        phone_service = st.selectbox("Phone service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        online_security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device protection", ["No", "Yes", "No internet service"])

    with right:
        tech_support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming movies", ["Yes", "No", "No internet service"])
        paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly_charges = st.number_input("Monthly charges", min_value=0.0, max_value=150.0, value=85.0, step=1.0)

    total_charges_default = float(tenure) * float(monthly_charges)
    total_charges = st.number_input(
        "Total charges",
        min_value=0.0,
        max_value=10000.0,
        value=round(total_charges_default, 2),
        step=10.0,
    )

    return pd.DataFrame(
        [
            {
                "gender": gender,
                "SeniorCitizen": senior_citizen,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
            }
        ]
    )


def show_metric_cards(data: pd.DataFrame | None, comparison: pd.DataFrame | None) -> None:
    cards = st.columns(4)
    if data is not None:
        cards[0].metric("Customers", f"{len(data):,}")
        cards[1].metric("Churn rate", f"{data['Churn'].mean():.1%}")
    else:
        cards[0].metric("Customers", "n/a")
        cards[1].metric("Churn rate", "n/a")

    if comparison is not None and not comparison.empty:
        best_model = comparison.sort_values("roc_auc", ascending=False).iloc[0]
        cards[2].metric("Best model", str(best_model["model"]).replace("_", " ").title())
        cards[3].metric("Best ROC-AUC", f"{best_model['roc_auc']:.3f}")
    else:
        cards[2].metric("Best model", "n/a")
        cards[3].metric("Best ROC-AUC", "n/a")


def image_or_warning(filename: str, caption: str) -> None:
    path = FIGURES_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Missing report figure: {filename}. Run `python scripts/generate_reports.py`.")


def main() -> None:
    model = load_model()
    data = load_clean_data()
    comparison = load_model_comparison()
    importance = load_permutation_importance()
    threshold_analysis = load_threshold_analysis()

    st.title("Customer Churn Insights")
    st.caption("Streamlit web dashboard for churn prediction, model comparison, explainability, and retention recommendations.")

    show_metric_cards(data, comparison)

    prediction_tab, performance_tab, threshold_tab, insights_tab = st.tabs(
        ["Prediction", "Model Performance", "Threshold Tuning", "Business Insights"]
    )

    with prediction_tab:
        st.subheader("Customer Risk Prediction")
        customer = build_customer_input()
        decision_threshold = st.slider("Decision threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.05)

        if model is None:
            st.error("Model not found. Run `python -m src.train` first.")
        elif st.button("Predict churn risk", type="primary"):
            probability = float(model.predict_proba(customer)[0, 1])
            prediction = int(probability >= decision_threshold)
            label, recommendation = risk_label(probability)

            left, right = st.columns([1, 2])
            left.metric("Churn probability", f"{probability:.1%}")
            left.metric("Risk level", label)
            right.progress(min(max(probability, 0.0), 1.0))
            right.write(recommendation)
            right.write("Predicted class: churn" if prediction == 1 else "Predicted class: no churn")
            right.write(f"Decision threshold: {decision_threshold:.2f}")

            with st.expander("Input customer profile"):
                st.dataframe(customer, use_container_width=True)

    with performance_tab:
        st.subheader("Model Comparison")
        if comparison is not None:
            st.dataframe(
                comparison.sort_values("roc_auc", ascending=False).style.format(
                    {
                        "accuracy": "{:.3f}",
                        "precision": "{:.3f}",
                        "recall": "{:.3f}",
                        "f1": "{:.3f}",
                        "roc_auc": "{:.3f}",
                    }
                ),
                use_container_width=True,
            )
        else:
            st.warning("Missing model comparison. Run `python -m src.train`.")

        image_or_warning("model_roc_auc_comparison.png", "ROC-AUC comparison across trained models")

        col_roc, col_pr = st.columns(2)
        with col_roc:
            image_or_warning("roc_curve.png", "ROC curve for selected model")
        with col_pr:
            image_or_warning("precision_recall_curve.png", "Precision-recall curve for selected model")

        image_or_warning("confusion_matrix.png", "Confusion matrix at default threshold 0.50")

    with threshold_tab:
        st.subheader("Threshold Tuning")
        st.write("The default classification threshold is 0.50, but churn campaigns often need a lower threshold to catch more at-risk customers.")
        if threshold_analysis is not None:
            best_f1 = threshold_analysis.sort_values("f1", ascending=False).iloc[0]
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Best F1 threshold", f"{best_f1['threshold']:.2f}")
            col_b.metric("Precision", f"{best_f1['precision']:.3f}")
            col_c.metric("Recall", f"{best_f1['recall']:.3f}")
            col_d.metric("Predicted churn rate", f"{best_f1['predicted_churn_rate']:.1%}")
            st.dataframe(
                threshold_analysis.sort_values("f1", ascending=False).style.format(
                    {
                        "threshold": "{:.2f}",
                        "precision": "{:.3f}",
                        "recall": "{:.3f}",
                        "f1": "{:.3f}",
                        "predicted_churn_rate": "{:.1%}",
                    }
                ),
                use_container_width=True,
            )
        else:
            st.warning("Missing threshold analysis. Run `python scripts/generate_reports.py`.")
        image_or_warning("threshold_tradeoff.png", "Precision, recall, and F1 across decision thresholds")

    with insights_tab:
        st.subheader("Churn Drivers")
        col_a, col_b = st.columns(2)
        with col_a:
            image_or_warning("churn_distribution.png", "Dataset churn distribution")
            image_or_warning("tenure_distribution_by_churn.png", "Tenure distribution by churn status")
        with col_b:
            image_or_warning("churn_by_contract.png", "Churn rate by contract type")
            image_or_warning("monthly_charges_by_churn.png", "Monthly charges by churn status")

        st.subheader("Explainability")
        if importance is not None:
            st.dataframe(
                importance.head(10).style.format({"importance_mean": "{:.4f}", "importance_std": "{:.4f}"}),
                use_container_width=True,
            )
        else:
            st.warning("Missing permutation importance table. Run `python scripts/generate_reports.py`.")

        col_c, col_d = st.columns(2)
        with col_c:
            image_or_warning("permutation_importance.png", "Top original features by permutation importance")
        with col_d:
            image_or_warning("shap_summary.png", "SHAP summary for encoded model features")

        st.subheader("Retention Actions")
        st.markdown(
            """
- Prioritize short-tenure customers with month-to-month contracts.
- Review high monthly charges for customers with elevated churn probability.
- Promote security, backup, and support services to increase product engagement.
- Tune the decision threshold before launching a campaign, because campaign cost and expected retained revenue determine the optimal cutoff.
"""
        )


if __name__ == "__main__":
    main()

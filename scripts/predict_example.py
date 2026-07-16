import joblib
import pandas as pd


MODEL_PATH = "models/churn_model.joblib"


def main() -> None:
    model = joblib.load(MODEL_PATH)

    new_customer = pd.DataFrame(
        [
            {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 3,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 85.0,
                "TotalCharges": 255.0,
            }
        ]
    )

    prediction = model.predict(new_customer)[0]
    churn_probability = model.predict_proba(new_customer)[0, 1]
    risk_label = "high" if churn_probability >= 0.7 else "medium" if churn_probability >= 0.4 else "low"

    print("Example customer churn prediction")
    print(f"Prediction: {prediction} ({'churn' if prediction == 1 else 'no churn'})")
    print(f"Churn probability: {churn_probability:.2%}")
    print(f"Risk level: {risk_label}")


if __name__ == "__main__":
    main()

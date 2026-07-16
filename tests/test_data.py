import pandas as pd
import pytest

from src.data import clean_telco_data, split_features_target


def test_clean_telco_data_maps_target_and_converts_total_charges():
    raw = pd.DataFrame(
        {
            "customerID": ["A", "B"],
            "tenure": [1, 2],
            "MonthlyCharges": [10.0, 20.0],
            "TotalCharges": ["10.0", " "],
            "Churn": ["No", "Yes"],
        }
    )

    cleaned = clean_telco_data(raw)

    assert "customerID" not in cleaned.columns
    assert cleaned["TotalCharges"].isna().sum() == 0
    assert cleaned["Churn"].tolist() == [0, 1]


def test_split_features_target_requires_churn_column():
    df = pd.DataFrame({"feature": [1, 2]})

    with pytest.raises(ValueError):
        split_features_target(df)

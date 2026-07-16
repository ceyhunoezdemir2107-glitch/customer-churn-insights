import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def infer_feature_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Infer numerical and categorical columns for preprocessing."""
    numeric_features = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = df.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    return numeric_features, categorical_features


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Build a sklearn preprocessor for churn features."""
    numeric_features, categorical_features = infer_feature_types(df)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

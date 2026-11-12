
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class IQRClipper(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        X = np.asarray(X)

        self.q1_ = np.percentile(X, 25, axis=0)
        self.q3_ = np.percentile(X, 75, axis=0)

        iqr = self.q3_ - self.q1_

        self.lower_bound_ = self.q1_ - 1.5 * iqr
        self.upper_bound_ = self.q3_ + 1.5 * iqr

        return self

    def transform(self, X):

        X = np.asarray(X)

        return np.clip(
            X,
            self.lower_bound_,
            self.upper_bound_
        )


def clean_data(data):

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame")

    data = data.copy()

    data = data.drop_duplicates()

    if data.empty:
        raise ValueError("Dataset is empty after cleaning")

    return data


def build_preprocessor(X):

    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("outlier_handler", IQRClipper()),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    transformers = []

    if numeric_features:
        transformers.append(
            ("numeric", numeric_pipeline, numeric_features)
        )

    if categorical_features:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )

    if not transformers:
        raise ValueError("No supported features found in dataset")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    return preprocessor

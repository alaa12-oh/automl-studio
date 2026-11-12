
import pandas as pd


def detect_task(data, target_column):

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame")

    if target_column not in data.columns:
        raise ValueError("Target column does not exist")

    target = data[target_column]

    if target.isnull().any():
        raise ValueError("Target column contains missing values")

    if target.nunique() < 2:
        raise ValueError(
            "Target column must contain at least two unique values"
        )

    # Suggest a task type
    if pd.api.types.is_numeric_dtype(target):
        if target.nunique() == 2:
            suggested_task = "classification"
        else:
            suggested_task = "regression"
    else:
        suggested_task = "classification"

    return suggested_task

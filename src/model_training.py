
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge
)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor
)

from sklearn.svm import SVC, SVR


def get_models(task_type):

    if task_type == "classification":

        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000
            ),

            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ),

            "SVM": SVC(),

            "Gradient Boosting": GradientBoostingClassifier(
                random_state=42
            ),

            "KNN": KNeighborsClassifier(
                n_neighbors=5
            )
        }

    elif task_type == "regression":

        models = {
            "Linear Regression": LinearRegression(),

            "Ridge Regression": Ridge(),

            "Random Forest": RandomForestRegressor(
                n_estimators=100,
                random_state=42
            ),

            "Gradient Boosting": GradientBoostingRegressor(
                random_state=42
            ),

            "SVR": SVR()
        }

    else:
        raise ValueError(
            "Task type must be classification or regression"
        )

    return models


def train_models(X_train, y_train, preprocessor, task_type):

    models = get_models(task_type)

    trained_models = {}
    failed_models = {}

    for model_name, model in models.items():

        try:
            pipeline = Pipeline(steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", clone(model))
            ])

            pipeline.fit(X_train, y_train)

            trained_models[model_name] = pipeline

        except Exception as error:
            failed_models[model_name] = str(error)

    if not trained_models:
        raise RuntimeError(
            "All models failed during training. "
            "Please check your dataset."
        )

    return trained_models, failed_models



from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, KFold

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.svm import SVC, SVR


def get_models(task_type):

    if task_type == "classification":
        return {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(random_state=42),
            "SVM": SVC(),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "KNN": KNeighborsClassifier()
        }

    if task_type == "regression":
        return {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Random Forest": RandomForestRegressor(random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "SVR": SVR()
        }

    raise ValueError("Task type must be classification or regression")


def get_parameters(model_name, max_neighbors=5):

    if model_name == "Logistic Regression":
        return {"model__C": [0.01, 0.1, 1, 10, 100]}

    if model_name == "Linear Regression":
        return {"model__fit_intercept": [True, False]}

    if model_name == "Ridge Regression":
        return {"model__alpha": [0.01, 0.1, 1, 10, 100]}

    if model_name == "Random Forest":
        return {
            "model__n_estimators": [100, 150, 200],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10]
        }

    if model_name == "SVM":
        return {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["linear", "rbf"],
            "model__gamma": ["scale", "auto"]
        }

    if model_name == "Gradient Boosting":
        return {
            "model__n_estimators": [50, 100, 150],
            "model__learning_rate": [0.01, 0.1, 0.2],
            "model__max_depth": [2, 3, 5]
        }

    if model_name == "KNN":
        neighbors = [n for n in [1, 3, 5] if n <= max_neighbors]
        return {"model__n_neighbors": neighbors}

    if model_name == "SVR":
        return {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["linear", "rbf"],
            "model__epsilon": [0.01, 0.1, 0.2]
        }

    return {}


def train_models(X_train, y_train, preprocessor, task_type):

    models = get_models(task_type)
    trained_models = {}
    failed_models = {}

    if task_type == "classification":
        class_counts = y_train.value_counts()
        folds = min(2, int(class_counts.min()))

        if folds < 2:
            raise ValueError("Not enough samples for classification tuning.")

        cv = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=42
        )
        scoring = "f1_weighted"

    elif task_type == "regression":
        folds = min(2, len(X_train))

        if folds < 2:
            raise ValueError("Not enough data for regression tuning.")

        cv = KFold(
            n_splits=folds,
            shuffle=True,
            random_state=42
        )
        scoring = "neg_mean_squared_error"

    else:
        raise ValueError("Task type must be classification or regression")

    max_neighbors = max(
        1,
        int(len(X_train) * (folds - 1) / folds)
    )

    for model_name, model in models.items():
        try:
            pipeline = Pipeline([
                ("preprocessor", clone(preprocessor)),
                ("model", clone(model))
            ])

            search = RandomizedSearchCV(
                pipeline,
                param_distributions=get_parameters(
                    model_name,
                    max_neighbors
                ),
                n_iter=2,
                cv=cv,
                scoring=scoring,
                random_state=42,
                n_jobs=1
            )

            search.fit(X_train, y_train)

            best_model = search.best_estimator_
            best_model.cv_best_score_ = search.best_score_
            best_model.best_params_ = search.best_params_

            trained_models[model_name] = best_model

        except Exception as error:
            failed_models[model_name] = str(error)

    if not trained_models:
        raise RuntimeError(
            "All models failed during training. Please check your dataset."
        )

    return trained_models, failed_models





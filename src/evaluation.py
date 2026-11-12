
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold,
    KFold
)


def evaluate_models(trained_models, X_train, y_train,
                    X_test, y_test, task_type):

    results = {}

    if task_type == "classification":

        class_counts = pd.Series(y_train).value_counts()

        if class_counts.min() < 2:
            raise ValueError(
                "Each class needs at least two samples for cross-validation"
            )

        folds = min(5, int(class_counts.min()))

        cv = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=42
        )

    elif task_type == "regression":

        if len(X_train) < 2:
            raise ValueError(
                "Not enough data for cross-validation"
            )

        folds = min(5, len(X_train))

        cv = KFold(
            n_splits=folds,
            shuffle=True,
            random_state=42
        )

    else:
        raise ValueError(
            "Task type must be classification or regression"
        )

    for model_name, model in trained_models.items():

        if task_type == "classification":

            scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=cv,
                scoring="f1_weighted"
            )

            predictions = model.predict(X_test)

            results[model_name] = {
                "CV F1 Score": scores.mean(),
                "Test Accuracy": accuracy_score(
                    y_test, predictions
                ),
                "Test F1 Score": f1_score(
                    y_test,
                    predictions,
                    average="weighted"
                )
            }

        else:

            scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=cv,
                scoring="neg_mean_squared_error"
            )

            cv_rmse = (-scores.mean()) ** 0.5

            predictions = model.predict(X_test)

            rmse = mean_squared_error(
                y_test, predictions
            ) ** 0.5

            mae = mean_absolute_error(
                y_test, predictions
            )

            r2 = r2_score(y_test, predictions)

            results[model_name] = {
                "CV RMSE": cv_rmse,
                "Test RMSE": rmse,
                "Test MAE": mae,
                "Test R2 Score": r2
            }

    return results


def get_best_model(trained_models, results, task_type):

    if not results:
        raise ValueError("No evaluation results available")

    if task_type == "classification":

        best_model_name = max(
            results,
            key=lambda name: results[name]["CV F1 Score"]
        )

    elif task_type == "regression":

        best_model_name = min(
            results,
            key=lambda name: results[name]["CV RMSE"]
        )

    else:
        raise ValueError(
            "Task type must be classification or regression"
        )

    best_model = trained_models[best_model_name]

    return best_model_name, best_model


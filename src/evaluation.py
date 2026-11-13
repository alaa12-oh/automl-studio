
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


def evaluate_models(trained_models, X_train, y_train,
                    X_test, y_test, task_type):

    results = {}

    for model_name, model in trained_models.items():

        predictions = model.predict(X_test)
        cv_score = getattr(model, "cv_best_score_", None)

        if task_type == "classification":

            if cv_score is None:
                raise ValueError(
                    f"Missing tuning score for {model_name}"
                )

            results[model_name] = {
                "CV F1 Score": cv_score,
                "Test Accuracy": accuracy_score(y_test, predictions),
                "Test F1 Score": f1_score(
                    y_test,
                    predictions,
                    average="weighted"
                )
            }

        elif task_type == "regression":

            if cv_score is None:
                raise ValueError(
                    f"Missing tuning score for {model_name}"
                )

            results[model_name] = {
                "CV RMSE": (-cv_score) ** 0.5,
                "Test RMSE": mean_squared_error(
                    y_test, predictions
                ) ** 0.5,
                "Test MAE": mean_absolute_error(
                    y_test, predictions
                ),
                "Test R2 Score": r2_score(y_test, predictions)
            }

        else:
            raise ValueError(
                "Task type must be classification or regression"
            )

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

    return best_model_name, trained_models[best_model_name]



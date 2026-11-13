
import mlflow

from pathlib import Path


project_folder = Path(__file__).resolve().parent.parent

database_path = project_folder / "mlflow.db"

tracking_uri = f"sqlite:///{database_path.as_posix()}"

mlflow.set_tracking_uri(tracking_uri)


def log_experiment(model_name, model, metrics, task_type):

    mlflow.set_experiment("AutoML Experiments")

    with mlflow.start_run(run_name=model_name):

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("task_type", task_type)

        best_params = getattr(model, "best_params_", {})

        for param_name, param_value in best_params.items():
            mlflow.log_param(f"best_{param_name}", str(param_value))

        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, float(value))





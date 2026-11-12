
import mlflow
import mlflow.sklearn

from pathlib import Path


project_folder = Path(__file__).resolve().parent.parent

database_path = project_folder / "mlflow.db"

tracking_uri = f"sqlite:///{database_path.as_posix()}"

mlflow.set_tracking_uri(tracking_uri)


trusted_types = [
    "numpy.dtype",
    "src.preprocessing.IQRClipper",
    "sklearn.metrics._dist_metrics.EuclideanDistance64",
    "sklearn.neighbors._kd_tree.KDTree"
]


def log_experiment(model_name, model, metrics, task_type):

    mlflow.set_experiment("AutoML Experiments")

    with mlflow.start_run(run_name=model_name):

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("task_type", task_type)

        for metric_name, value in metrics.items():

            mlflow.log_metric(
                metric_name,
                float(value)
            )

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            skops_trusted_types=trusted_types
        )



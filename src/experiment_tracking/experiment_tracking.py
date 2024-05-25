from pathlib import Path

import mlflow
import pandas as pd
from torchinfo import summary

from config import config
from src.model import MLFlowTrackingArgs, TrainArgs


class ExperimentTracking:
    """
    Class for tracking machine learning experiments with MLflow.

    Attributes:
        run_name (str): The name of the run.
        tracking_uri (str): The URI for the MLflow tracking server.
        experiment_name (str): The name of the experiment.
        reports_path (Path): The directory path for storing reports.
    """

    def __init__(self, tracking_args: MLFlowTrackingArgs) -> None:
        """
        Initializes the ExperimentTracking class with the given tracking arguments.

        Args:
            tracking_args (MLFlowTrackingArgs): The tracking arguments.
        """

        self.run_name = tracking_args.run_name
        self.tracking_uri = tracking_args.tracking_uri
        self.experiment_name = tracking_args.experiment_name

        reports_dir = f"{self.experiment_name}_{self.run_name}"
        self.reports_path = Path(config.REPORTS_DIR, reports_dir)
        self.reports_path.mkdir(parents=True, exist_ok=True)

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    def track_run(self, train_loop_fn, train_args: TrainArgs, model):
        """
        Tracks a training run using MLflow.

        Args:
            train_loop_fn (callable): The training loop function.
            train_args (TrainArgs): The training arguments to log as parameters.
            model (): The model to log.
        """

        with open(Path(self.reports_path, "model_summary.txt"), "w") as f:
            NO_PRINT = 0
            f.write(str(summary(model, verbose=NO_PRINT)))

        with mlflow.start_run(run_name=self.run_name):
            mlflow.log_params(train_args.__dict__)

            train_loop_fn(step_logger=self._step_logger)

    def _step_logger(self, epoch, step_mode, step_loss, metrics, model):
        """
        Runs inside the stepper function to log model, artifacts, and metrics
        for every epoch.

        Args:
            epoch (int): The current epoch number.
            step_mode (str): The step mode (e.g., 'train', 'val').
            step_loss (float): The loss value for the current step.
            metrics (dict): The dictionary containing metric names and values.
        """

        mlflow.pytorch.log_model(model, "model")
        mlflow.log_artifacts(str(self.reports_path))
        mlflow.log_metric(f"{step_mode}_loss", f"{step_loss:3f}", step=epoch)

        for metric_name, metric_dict in metrics.items():
            for score_type, score in metric_dict.items():
                if score_type == "mean":
                    mlflow.log_metric(
                        key=f"{step_mode}_{metric_name}_{score_type}",
                        value=f"{score:3f}",
                        step=epoch,
                    )
                else:
                    for class_name, class_score in score.items():
                        mlflow.log_metric(
                            key=f"{step_mode}_{metric_name}_{score_type}_{class_name}",
                            value=f"{class_score:3f}",
                            step=epoch,
                        )


def _view_experiments():
    """NOT YET AVAILABLE FOR USAGE."""

    client = mlflow.tracking.MlflowClient()
    experiments = client.search_experiments()

    for experiment in experiments:
        print(f"{experiment.name} {experiment.experiment_id}\n\n")

        runs = client.search_runs(experiment.experiment_id)

        print(f"number of runs: {len(runs)}")

        if len(runs) == 0:
            continue
        else:
            for run in runs:
                run_info = dict(run.info)
                run_metrics = dict(run.data)["metrics"]
                run_name = run_info["run_name"]

                df = pd.DataFrame(
                    list(run_metrics.items()), columns=["Metric", "Value"]
                )

"""MLflow tracking integration for Voyage Analytics.

This module wraps the official ``mlflow`` PyPI library to track, log and
register the trained models (flight price, gender) plus the recommendation
catalog. It does NOT retrain models.

It is model-agnostic: pass a framework (``xgboost`` or ``sklearn``) and the
correct MLflow ``log_model`` flavour is selected, so a model can be re-loaded
and served via ``mlflow.pyfunc`` / the Model Registry.

Requires a reachable MLflow tracking server (see ``MLFLOW_TRACKING_URI``).
"""

import json
import logging
import os
from typing import Any, Dict, Optional

import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

from config.settings import settings

logger = logging.getLogger(__name__)

#: Registered model name per family.
DEFAULT_MODEL_NAME = "voyage_flight_price"


class ModelTracker:
    """Manages model tracking and registration with MLflow."""

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ):
        self.tracking_uri = tracking_uri or settings.mlflow_tracking_uri
        self.experiment_name = experiment_name or settings.mlflow_experiment_name
        self.client = MlflowClient(self.tracking_uri)

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    # ------------------------------------------------------------------ #
    # Runs
    # ------------------------------------------------------------------ #
    def start_run(self, run_name: str = "production_run") -> "mlflow.ActiveRun":
        """Start a new MLflow run."""
        return mlflow.start_run(run_name=run_name)

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log model metrics."""
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        logger.info("Logged %d metrics", len(metrics))

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log model parameters."""
        for name, value in params.items():
            mlflow.log_param(name, str(value))
        logger.info("Logged %d params", len(params))

    def log_artifact(self, local_path: str, artifact_path: str = None) -> None:
        """Log an artifact (file) to the current run."""
        mlflow.log_artifact(local_path, artifact_path)
        logger.info("Logged artifact: %s", local_path)

    # ------------------------------------------------------------------ #
    # Model logging
    # ------------------------------------------------------------------ #
    def log_model(
        self,
        model_path: str,
        framework: str = "sklearn",
        model_name: str = DEFAULT_MODEL_NAME,
        artifact_path: str = "model",
        registered: bool = True,
        sample_input: Optional[Any] = None,
        skops_trusted_types: Optional[list] = None,
    ) -> Optional[str]:
        """Log and optionally register a serialised model.

        Args:
            model_path: Path to the artifact (``.joblib``/``.pkl``).
            framework: ``xgboost`` or ``sklearn`` — selects the MLflow flavour.
            model_name: Registered model name in the Model Registry.
            artifact_path: Artifact sub-path within the run.
            registered: Whether to register the model in the Registry.
            sample_input: Optional sample input DataFrame/dict to infer a schema.
            skops_trusted_types: Optional list of estimator classes to whitelist
                for the skops-backed sklearn serializer (required for pipelines
                that wrap non-sklearn estimators such as XGboost).

        Returns:
            The registered model URI, or ``None`` if not registered.
        """
        import joblib

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model artifact not found: {model_path}")

        model = joblib.load(model_path)
        signature = None
        if sample_input is not None:
            try:
                signature = infer_signature(model_input=sample_input)
            except Exception as e:  # pragma: no cover - depends on mlflow internals
                logger.warning("Could not infer signature: %s", e)

        if framework == "xgboost":
            mlflow.xgboost.log_model(model, artifact_path=artifact_path, signature=signature)
        else:
            # Use the classic pickle serialization format rather than skops. skops
            # cannot serialize an XGBoost Booster embedded inside a sklearn
            # Pipeline (RepresenterError); pickle handles it cleanly.
            mlflow.sklearn.log_model(
                model,
                artifact_path=artifact_path,
                signature=signature,
                serialization_format="pickle",
                skops_trusted_types=skops_trusted_types,
            )

        logger.info("Model logged under artifact path: %s", artifact_path)

        if registered:
            run_id = mlflow.active_run().info.run_id
            model_uri = f"runs:/{run_id}/{artifact_path}"
            registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
            logger.info(
                "Model registered: %s v%s", registered_model.name, registered_model.version
            )
            return model_uri

        return None

    def log_model_metadata(
        self,
        metadata_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log model metadata to the current run."""
        md: Dict[str, Any] = metadata or {}
        if metadata_path and os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    md.update(json.load(f))
            except Exception:
                logger.warning("Could not read metadata file: %s", metadata_path)

        # Log each field as a param.
        for key, value in md.items():
            mlflow.log_param(key, str(value))

        # Also store the combined JSON as an artifact.
        with open("model_metadata.json", "w", encoding="utf-8") as f:
            json.dump(md, f, indent=2)
        self.log_artifact("model_metadata.json", "metadata")

        logger.info("Logged %d metadata fields", len(md))

    @staticmethod
    def _get_conda_env() -> Dict[str, Any]:
        """Conda env spec used when self-contained packaging is required."""
        return {
            "name": "voyage-analytics-env",
            "channels": ["conda-forge", "defaults"],
            "dependencies": [
                "python=3.11",
                "pip",
                {
                    "pip": [
                        "flask==3.0.3",
                        "joblib==1.4.2",
                        "pydantic==2.7.4",
                        "scikit-learn>=1.5.0",
                        "xgboost==2.0.3",
                        "pandas==2.2.2",
                    ],
                },
            ],
        }


def track_model_from_colab(
    model_path: str,
    framework: str = "xgboost",
    model_name: str = DEFAULT_MODEL_NAME,
    metrics_path: Optional[str] = None,
    metadata_path: Optional[str] = None,
    run_name: str = "colab_training_run",
) -> Dict[str, str]:
    """Track a model trained in Google Colab and register it with MLflow."""
    tracker = ModelTracker()

    with tracker.start_run(run_name=run_name):
        model_uri = tracker.log_model(
            model_path, framework=framework, model_name=model_name, registered=True
        )

        if metrics_path and os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    tracker.log_metrics(json.load(f))
            except Exception:
                logger.warning("Could not load metrics file: %s", metrics_path)

        tracker.log_model_metadata(metadata_path=metadata_path)

        run_id = mlflow.active_run().info.run_id
        return {
            "run_id": run_id,
            "model_uri": model_uri,
            "experiment_name": tracker.experiment_name,
        }


def get_latest_model_version(model_name: str = DEFAULT_MODEL_NAME) -> Optional[int]:
    """Return the latest registered version of a model, or None."""
    client = MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            return None
        return max(int(v.version) for v in versions)
    except Exception:
        return None


def load_registered_model(model_name: str = DEFAULT_MODEL_NAME, version: Optional[int] = None) -> Any:
    """Load a registered model (by version, default latest) from the Registry."""
    stage = f"version {version}" if version else "latest"
    model_uri = f"models:/{model_name}/{stage}"
    return mlflow.pyfunc.load_model(model_uri)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Track a trained model with MLflow")
    parser.add_argument("--model-path", default="artifacts/flight_price_pipeline.joblib")
    parser.add_argument("--framework", choices=["xgboost", "sklearn"], default="xgboost")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--metrics-path", default="artifacts/metrics.json")
    parser.add_argument("--metadata-path", default="artifacts/model_metadata.json")
    parser.add_argument("--run-name", default="colab_training_run")
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()

    print("Starting MLflow tracking...")
    print(f"  Model: {args.model_path}")
    print(f"  Model name: {args.model_name}")

    tracker = ModelTracker()
    print(f"  Experiment: {tracker.experiment_name}")
    print(f"  Tracking URI: {tracker.tracking_uri}")

    with tracker.start_run(run_name=args.run_name):
        tracker.log_model(
            args.model_path,
            framework=args.framework,
            model_name=args.model_name,
            registered=not args.no_register,
        )
        if os.path.exists(args.metrics_path):
            tracker.log_artifact(args.metrics_path, "artifacts")
        if os.path.exists(args.metadata_path):
            tracker.log_artifact(args.metadata_path, "artifacts")
        print(f"  Run ID: {mlflow.active_run().info.run_id}")
        print("Tracking complete")

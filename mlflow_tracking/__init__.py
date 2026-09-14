"""Voyage Analytics MLflow tracking integration.

This package (``mlflow_tracking``) wraps the official ``mlflow`` PyPI library.
It is deliberately NOT named ``mlflow`` so that importing the real library
(``import mlflow``) is never shadowed by a local package, which was the
previous root cause of tracking scripts failing from the project root.
"""

from mlflow_tracking.tracking import (
    ModelTracker,
    track_model_from_colab,
    get_latest_model_version,
)

__all__ = [
    "ModelTracker",
    "track_model_from_colab",
    "get_latest_model_version",
]

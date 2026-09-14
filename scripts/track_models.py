"""Track, log and register all Voyage Analytics models with MLflow.

Logs training metrics, params, metadata and the model artifact for each model
family (flight price, gender) plus the recommendation catalog, and registers
them with the MLflow Model Registry.

Requires the official ``mlflow`` package and a reachable tracking server. Spin
one up with the bundled compose file, e.g.:

    docker compose -f docker/docker-compose.mlflow.yml up -d

or (file-based store, no server needed):

    export MLFLOW_TRACKING_URI=file:./mlruns

Usage:
    python scripts/track_models.py                 # every model
    python scripts/track_models.py --model flight  # just one family
"""

import argparse
import json
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import joblib
import pandas as pd

import mlflow
from mlflow.models.signature import infer_signature

from config.settings import settings

#: model family -> tracking configuration
MODEL_DEFS = {
    "flight": {
        "artifact": settings.model_path,
        "name": "voyage_flight_price",
        "framework": "xgboost",
        "metrics_file": "artifacts/metrics.json",
        "metadata_file": "artifacts/model_metadata.json",
        "sample_input": pd.DataFrame(
            [
                {
                    "from": "Recife (PE)",
                    "to": "Florianopolis (SC)",
                    "flightType": "firstClass",
                    "agency": "FlyingDrops",
                    "time": 1.76,
                    "distance": 676.53,
                    "flight_year": 2019,
                    "flight_month": 9,
                    "flight_day": 26,
                    "flight_dayofweek": 3,
                }
            ]
        ),
    },
    "gender": {
        "artifact": settings.gender_model_path,
        "name": "voyage_gender",
        "framework": "sklearn",
        "metrics_file": "artifacts/metrics.json",
        "metadata_file": "artifacts/model_metadata.json",
        "sample_input": pd.DataFrame([{"first_name": "robert", "age": 33}]),
    },
}

#: Datasets used to (re)build the recommendation catalog.
DATA_DEFS = {
    "catalog": {
        "hotels": settings.hotels_data_path,
        "users": settings.users_data_path,
        "output": settings.hotels_catalog_path,
        "name": "voyage_recommendation",
        "artifact": settings.hotels_catalog_path,
    }
}


def _load_json(path: str) -> dict:
    if path and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _log_metrics(metrics: dict) -> None:
    for k, v in metrics.items():
        try:
            mlflow.log_metric(k, float(v))
        except (TypeError, ValueError):
            mlflow.log_param(k, str(v))


def track_family(name: str) -> dict:
    """Log a single model family to MLflow."""
    if name not in MODEL_DEFS:
        print(f"Unknown model: {name} (choose from {list(MODEL_DEFS)})")
        return {}

    cfg = MODEL_DEFS[name]
    artifact = cfg["artifact"]
    if not os.path.exists(artifact):
        print(f"Skipping {name}: artifact not found at {artifact}")
        return {}

    with mlflow.start_run(run_name=f"{name}_tracking"):
        mlflow.set_tag("task", cfg["name"])
        mlflow.set_tag("framework", cfg["framework"])

        metrics = _load_json(cfg["metrics_file"])
        metadata = _load_json(cfg["metadata_file"])
        _log_metrics(metrics)
        for k, v in metadata.items():
            mlflow.log_param(k, str(v))

        # Always log the raw serialised artifact — this succeeds everywhere.
        mlflow.log_artifact(artifact, artifact_path="raw_artifact")

        # Attempt standard model-flavour logging (sklearn/xgboost). This infers a
        # signature so the model can be served from the Registry. Some
        # environments (e.g. broken torch native libs) make MLflow import torch
        # and fail here; we degrade gracefully to artifact-only tracking.
        logged_uri = None
        try:
            model = joblib.load(artifact)
            signature = None
            try:
                signature = infer_signature(model_input=cfg["sample_input"])
            except Exception as e:
                print(f"  Could not infer signature for {name}: {e}")

            # Both artifacts are sklearn-flavour pickles (the flight model is a
            # sklearn Pipeline wrapping an XGBRegressor), so they are logged via
            # the sklearn flavour. Use the classic pickle serialization format
            # because skops (MLflow's default) cannot serialize an XGBoost
            # Booster embedded in a Pipeline (RepresenterError).
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                signature=signature,
                serialization_format="pickle",
            )
            logged_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            print(f"  Logged {name} model with MLflow flavour ({cfg['framework']})")
        except Exception as e:
            print(
                f"  WARNING: model-flavour logging failed for {name} "
                f"(artifact still logged): {type(e).__name__}: {e}"
            )

        if logged_uri:
            try:
                uri = mlflow.register_model(logged_uri, name=cfg["name"])
                print(f"Registered {cfg['name']} v{uri.version}")
            except Exception as e:
                print(f"  Registration failed for {name}: {e}")

        return {"run_id": mlflow.active_run().info.run_id}


def track_catalog() -> dict:
    """Log the recommendation catalog (JSON) as an artifact in its own run."""
    cfg = DATA_DEFS["catalog"]
    if not os.path.exists(cfg["artifact"]):
        print(f"Skipping catalog: {cfg['artifact']} not found")
        return {}

    with mlflow.start_run(run_name="catalog_tracking"):
        mlflow.set_tag("framework", "content-based")
        mlflow.log_artifact(cfg["artifact"], artifact_path="catalog")
        metadata = _load_json("artifacts/model_metadata.json")
        for k, v in metadata.items():
            mlflow.log_param(k, str(v))
        print("Logged recommendation catalog")
        return {"run_id": mlflow.active_run().info.run_id}


def main():
    parser = argparse.ArgumentParser(description="Track Voyage Analytics models in MLflow")
    parser.add_argument(
        "--model",
        choices=list(MODEL_DEFS) + list(DATA_DEFS) + ["all"],
        default="all",
        help="Which model family to track",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="Override the MLflow tracking URI (default from settings)",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri or settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    targets = (
        list(MODEL_DEFS) + list(DATA_DEFS) if args.model == "all" else [args.model]
    )
    for t in targets:
        if t in DATA_DEFS:
            track_catalog()
        else:
            track_family(t)

    print("Done.")


if __name__ == "__main__":
    main()

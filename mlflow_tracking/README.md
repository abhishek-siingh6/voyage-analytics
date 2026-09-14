# MLflow Tracking (mlflow_tracking)

Voyage Analytics MLflow integration for tracking, logging and registering the
trained models and the recommendation catalog.

## Motivation / naming

The directory is named **`mlflow_tracking`** deliberately. The previous layout
used a local package named `mlflow`, which **shadowed the official PyPI
`mlflow` library** whenever Python ran from the project root (`import mlflow`
resolved to the empty local package). Renaming the wrapper to `mlflow_tracking`
fixes the root cause: `import mlflow` now always refers to the real library.

## Architecture

```
Training scripts
      │  (write artifacts + metrics.json / model_metadata.json)
      ▼
  scripts/track_models.py
      │  (mlflow.sklearn / mlflow.xgboost.log_model + infer_signature)
      ▼
 MLflow Tracking Server ──► Model Registry
      │                          │
      ▼                          ▼
 Flask REST API        mlflow.pyfunc.load_model (serving)
```

## Starting the MLflow Server

The bundled compose file runs the server on **port 5001** (the Flask API owns
port 5000, so a dedicated port avoids the collision):

```bash
docker compose -f docker/docker-compose.mlflow.yml up -d
export MLFLOW_TRACKING_URI=http://localhost:5001
```

Alternatively use a local file-based store (no server needed):

```bash
export MLFLOW_TRACKING_URI=file:./mlruns
```

## Installing the dependency

MLflow is a tracking-only dependency (not needed at runtime):

```bash
pip install -r requirements-dev.txt
```

## Tracking all models

```bash
python scripts/track_models.py                 # flight, gender, catalog
python scripts/track_models.py --model flight  # single family
```

Registered model names: `voyage_flight_price`, `voyage_gender`,
`voyage_recommendation` (catalog logged as an artifact).

## Using the ModelTracker class

```python
from mlflow_tracking.tracking import ModelTracker

tracker = ModelTracker()
with tracker.start_run(run_name="my_run"):
    tracker.log_metrics({"mae": 15.5, "rmse": 22.3, "r2": 0.92})
    tracker.log_params({"algorithm": "xgboost", "n_estimators": 300})
    tracker.log_model("artifacts/flight_price_pipeline.joblib", framework="xgboost")
```

## Loading a registered model for serving

```python
from mlflow_tracking.tracking import load_registered_model
model = load_registered_model("voyage_flight_price")   # latest, or version=N
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server URI |
| `MLFLOW_EXPERIMENT_NAME` | `voyage-flight-price` | Experiment name |

> Note: when using the bundled compose server set
> `MLFLOW_TRACKING_URI=http://localhost:5001`.

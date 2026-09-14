# Scripts

Utility and MLOps scripts for Voyage Analytics.

| Script | Purpose |
|--------|---------|
| `run_local.py` | Run the Flask API locally (optionally creates a dummy model) |
| `test_api.py` | Smoke-test the API endpoints via the Flask test client |
| `train_flight_model.py` | Reproduce the flight-price XGBoost model (from the notebook) |
| `train_gender_model.py` | Train the gender classifier from `users.csv` |
| `build_recommendation_catalog.py` | Build `hotel_catalog.json` from `hotels.csv` + `users.csv` |
| `track_models.py` | Log + register all models with MLflow |
| `validate_artifact.py` | Validate a model artifact loads correctly |

## Run the API

```bash
python run_local.py                 # or: python -m api.app
```

## Train the gender classifier

```bash
python train_gender_model.py
```

Writes `artifacts/gender_model.joblib`, `artifacts/metrics.json` and
`artifacts/model_metadata.json`.

## Reproduce the flight-price model

A faithful, script-ised reproduction of the ML team's Colab notebook. Reads the
bundled `flights.csv` / `hotels.csv` / `users.csv` and writes the model the API
consumes:

```bash
python train_flight_model.py
```

Writes `artifacts/flight_price_pipeline.joblib` plus `artifacts/metrics.json`
and `artifacts/model_metadata.json`. If the artifact is ever missing, this is
how to regenerate it from the bundled datasets.

## Build the recommendation catalog

```bash
python build_recommendation_catalog.py
```

Writes `artifacts/hotel_catalog.json` and `artifacts/model_metadata.json`.

## Track models with MLflow

```bash
# Requires a running MLflow server (see mlflow_tracking/README.md)
python track_models.py                      # all models
python track_models.py --model gender       # single family
python track_models.py --tracking-uri sqlite:///mlflow.db
```

## Notes

- Training scripts default to the bundled datasets under
  `artifacts/data/` but accept `--data-path` / `--hotels` / `--users` overrides.
- All scripts are run from the repository root.

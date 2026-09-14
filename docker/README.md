# Docker

This folder contains the container configuration for Voyage Analytics.

| File | Purpose |
|------|---------|
| `Dockerfile` | Image for the Flask API |
| `docker-compose.mlflow.yml` | Standalone MLflow tracking server |
| `.dockerignore` | Excludes files from the build context |

## Build & run the API

```bash
docker build -f docker/Dockerfile -t voyage-analytics-api .
docker run -p 5000:5000 voyage-analytics-api
```

The image bakes in the model artifacts (whatever is present under
`artifacts/`) so the API is portable.

## Run the MLflow server

```bash
docker compose -f docker/docker-compose.mlflow.yml up -d
```

- Serves MLflow on **host port 5001** (the Flask API owns port 5000).
- Persists the SQLite backend and artifacts in named volumes.
- Set `MLFLOW_TRACKING_URI=http://localhost:5001` to point the CLI/tracking
  scripts at it.

```bash
docker compose -f docker/docker-compose.mlflow.yml down   # stop
```

## Notes

- The API is a runtime container and does not need MLflow installed.
- Model-flavour logging (`mlflow.sklearn`/`xgboost`) requires a healthy
  environment; the MLflow image provides one.

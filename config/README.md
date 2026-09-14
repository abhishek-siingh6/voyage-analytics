# Configuration

All configuration is externalized via environment variables and loaded by
`config/settings.py` into a single `settings` object.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `artifacts/flight_price_pipeline.joblib` | Flight-price model artifact |
| `MODEL_METADATA_PATH` | `artifacts/model_metadata.json` | Optional model metadata |
| `GENDER_MODEL_PATH` | `artifacts/gender_model.joblib` | Gender model artifact |
| `HOTELS_CATALOG_PATH` | `artifacts/hotel_catalog.json` | Recommendation catalog |
| `HOTELS_DATA_PATH` | `artifacts/data/hotels.csv` | Hotels dataset |
| `USERS_DATA_PATH` | `artifacts/data/users.csv` | Users dataset |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow tracking server URI |
| `MLFLOW_EXPERIMENT_NAME` | `voyage-flight-price` | MLflow experiment name |
| `API_HOST` | `0.0.0.0` | API bind address |
| `API_PORT` | `5000` | API port |
| `TESTING` | `false` | Enable testing mode |

## Usage

```python
from config.settings import settings

print(settings.model_path)
print(settings.api_port)
print(settings.mlflow_tracking_uri)
```

## Configuration file

Copy the template and edit:

```bash
cp .env.example .env
```

The settings are loaded once at import time. To reload after changing
environment variables:

```python
import importlib
from config import settings
importlib.reload(settings)
```

## Adding a new setting

1. Add the variable to `.env.example`.
2. Add the property to `config/settings.py`.
3. Use it in code via `settings.VARIABLE_NAME`.

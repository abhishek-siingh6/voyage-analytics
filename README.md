# Voyage Analytics

**Integrating MLOps in Travel — Flight Price · Gender · Hotel Recommendation**

Voyage Analytics is an MLOps capstone that productionizes three ML services for
a travel platform. It trains models from the `flights.csv`, `hotels.csv` and
`users.csv` datasets (from the travel_capstone dataset), exposes them through a
single Flask REST API, tracks them with MLflow, and ships them via Docker and
Kubernetes — all fronted by an interactive Streamlit dashboard.

---

## Features

- **Flight price prediction** — XGBoost regression (route, class, agency, date).
- **Gender classification** — RandomForest classifier trained on `users.csv`.
- **Hotel recommendation** — content-based engine over the real hotel catalog.
- **Smart location autocomplete** — fuzzy matching with edit distance for typos
  and efficient inverted index search.
- **Streamlit dashboard** — interactive predictor, classifier, recommender and
  data insights.
- **MLflow tracking & registry** — metrics, params, artifacts and model
  registration.
- **Docker & Kubernetes** — portable, scalable deployment.
- **Automated tests** — 30+ tests, ~85% coverage.

## Architecture

```
Google Colab (training)
        │
        ▼
 Model Artifacts  (joblib / json)
        │
        ▼
   MLflow  (tracking / registry)
        │
        ▼
 Flask REST API  ◄── Streamlit Dashboard
        │
   ┌────┴────┐
   ▼         ▼
 Docker   Kubernetes
```

## Project Structure

```
voyage-analytics/
├── api/                 # Flask blueprints + endpoint docs
├── artifacts/           # Model artifacts, catalog & datasets (data/)
├── config/              # Settings (README.md)
├── dashboard/           # Streamlit web app (README.md)
│   └── templates/       # HTML templates (if needed)
├── src/
│   ├── features/        # Shared feature engineering
│   ├── model/           # Model loaders (README.md)
│   ├── schemas/         # Pydantic schemas (README.md)
│   └── services/        # Business logic + location search
├── scripts/             # Training / catalog / tracking scripts (README.md)
├── tests/               # Test suite
├── mlflow_tracking/     # MLflow integration (README.md)
├── docker/              # Dockerfile + MLflow compose (README.md)
├── kubernetes/          # K8s manifests (README.md)
└── airflow/             # Orchestration DAG template (README.md)
```

> A list of known issues / open requirements lives in [`Update.md`](./Update.md).

---

## Quick Start

### 1. Prerequisites

- Python **3.10** or **3.11** (recommended for ML)
- `pip`
- Docker (optional — for MLflow server / container deployment)
- `kubectl` (optional — for Kubernetes)

**For Windows users**: Update dataset paths in `.env` to point to `travel_capstone dataset` folder.

### 2. Install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt    # runtime
pip install -r requirements-dev.txt # tests + MLflow (optional)
```

### 3. Configure

```bash
cp .env.example .env
# Edit as needed — defaults work for local development.
```

### 4. Run the API

```bash
python -m api.app                  # or: python scripts/run_local.py
```

The API is served at `http://localhost:5000`.

### 5. Run the tests

```bash
pytest
```

### 6. Run the Streamlit dashboard

```bash
# With the API running (step 4), in a second terminal:
streamlit run dashboard/app.py
```

The dashboard includes:

- **Flight Price Predictor** - Dropdowns for 9 locations with intelligent defaults
- **Gender Classifier** - User gender prediction
- **Hotel Recommender** - Smart location search with fuzzy matching (handles typos)
- **Insights** - Data visualizations and statistics

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | `/api/health`                       | Health check |
| GET  | `/api/model-info`                   | Flight-price model metadata |
| POST | `/api/predict`                      | Predict flight price |
| GET  | `/api/gender/health`                | Gender endpoint health |
| POST | `/api/gender/predict`               | Classify a user's gender |
| GET  | `/api/recommend/health`             | Recommendation endpoint health |
| GET  | `/api/recommend/places`             | List destination cities |
| GET  | `/api/recommend/places/search?q=<query>` | Search places with fuzzy matching |
| POST | `/api/recommend/recommendations`    | Get hotel recommendations |

### Location Search Features

The `/api/recommend/places/search` endpoint uses:

- **Efficient inverted index** - Fast token-based searching
- **Fuzzy matching** - Edit distance algorithm to handle typos (e.g., "Brasillia" → "Brasilia")
- **Relevance ranking** - Exact matches ranked highest
- **Top 5 results** - Returns only the most relevant matches

Request/response schemas: see [`api/README.md`](./api/README.md).

---

## Key Commands

### Build model artifacts

```bash
# Gender classifier (from the bundled users.csv)
python scripts/train_gender_model.py

# Hotel recommendation catalog (from bundled hotels.csv + users.csv)
python scripts/build_recommendation_catalog.py
```

Training scripts write `metrics.json` and `model_metadata.json` for MLflow.

### MLflow tracking & registry

```bash
# Start the tracking server (port 5001 — avoids API port 5000)
docker compose -f docker/docker-compose.mlflow.yml up -d
export MLFLOW_TRACKING_URI=http://localhost:5001

# Log metrics, params, artifacts + register all models
python scripts/track_models.py
```

See [`mlflow_tracking/README.md`](./mlflow_tracking/README.md).

### Docker

```bash
docker build -f docker/Dockerfile -t voyage-analytics-api .
docker run -p 5000:5000 voyage-analytics-api
```

See [`docker/README.md`](./docker/README.md).

### Kubernetes

```bash
kubectl apply -f kubernetes/
```

See [`kubernetes/README.md`](./kubernetes/README.md).

---

## Models

| Model | Dataset | Artifact | Task | Endpoint |
|-------|---------|----------|------|----------|
| Flight price | `flights.csv` | `artifacts/flight_price_pipeline.joblib` | Regression (XGBoost) | `POST /api/predict` |
| Gender | `users.csv` | `artifacts/gender_model.joblib` | Classification | `POST /api/gender/predict` |
| Hotel recommendation | `hotels.csv` + `users.csv` | `artifacts/hotel_catalog.json` | Content-based | `POST /api/recommend/recommendations` |

> The flight-price model is supplied by the ML team and is git-ignored (large
> binary). Place it at `MODEL_PATH` (`artifacts/flight_price_pipeline.joblib`).

## Dataset

The project uses datasets from the `travel_capstone dataset` folder:

| Dataset | Location | Description |
|---------|----------|-------------|
| flights.csv | `travel_capstone dataset/flights.csv` | Flight booking data |
| hotels.csv | `travel_capstone dataset/hotels.csv` | Hotel booking data (9 locations) |
| users.csv | `travel_capstone dataset/users.csv` | User data for gender classification |

**Note**: Update `HOTELS_DATA_PATH` and `USERS_DATA_PATH` in `.env` to point to
the correct dataset location if using the default dataset folder.

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `artifacts/flight_price_pipeline.joblib` | Flight-price model |
| `GENDER_MODEL_PATH` | `artifacts/gender_model.joblib` | Gender model |
| `HOTELS_CATALOG_PATH` | `artifacts/hotel_catalog.json` | Recommendation catalog |
| `HOTELS_DATA_PATH` | `artifacts/data/hotels.csv` | Hotels dataset |
| `USERS_DATA_PATH` | `artifacts/data/users.csv` | Users dataset |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow tracking URI |
| `MLFLOW_EXPERIMENT_NAME` | `voyage-flight-price` | MLflow experiment |
| `API_HOST` | `0.0.0.0` | API bind address |
| `API_PORT` | `5000` | API port |

**Note**: The project uses datasets from the `travel_capstone dataset` folder. When setting up, copy or symlink the datasets to `artifacts/data/`:
- `artifacts/data/hotels.csv` ← `travel_capstone dataset/hotels.csv`
- `artifacts/data/users.csv` ← `travel_capstone dataset/users.csv`

Full reference: [`config/README.md`](./config/README.md).

---

## License

Academic project — Voyage Analytics Capstone.

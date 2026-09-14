# Dashboard (Streamlit)

Interactive front-end for Voyage Analytics. It consumes the Flask API and
renders model results and data insights.

## Pages

- **Flight Price Predictor** — predict flight price via `POST /api/predict`.
- **Gender Classifier** — classify a user's gender via `POST /api/gender/predict`.
- **Hotel Recommender** — get hotel recommendations via
  `POST /api/recommend/recommendations`.
- **Insights** — visualisations from the underlying datasets.

## Run

1. Start the API (see `api/README.md` / repo root README).
2. Launch the dashboard:

```bash
streamlit run dashboard/app.py
```

## Configuration (env vars)

| Variable | Default | Description |
|----------|---------|-------------|
| `API_BASE_URL` | `http://localhost:5000` | Flask API base URL |
| `HOTELS_DATA_PATH` | `artifacts/data/hotels.csv` | Hotels dataset (Insights) |
| `USERS_DATA_PATH` | `artifacts/data/users.csv` | Users dataset (Insights) |
| `HOTELS_CATALOG_PATH` | `artifacts/hotel_catalog.json` | Fallback catalog |

If the raw datasets are missing, the Insights page falls back to the hotel
catalog for a minimal overview.

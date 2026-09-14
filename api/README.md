# API

Flask application exposing the Voyage Analytics ML services. It is built with a
Flask **application factory** (`app.py`) and Blueprints for each service:

| File | Blueprint | Routes |
|------|-----------|--------|
| `app.py` | — (app factory) | registers all blueprints |
| `routes.py` | `/api` | health, model-info, predict |
| `gender_routes.py` | `/api/gender` | health, predict |
| `recommend_routes.py` | `/api/recommend` | health, places, recommendations |

## Run

```bash
python -m api.app                 # serves on http://localhost:5000
```

## Endpoints

### GET /api/health

```json
{ "status": "healthy" }
```

### GET /api/model-info

```json
{ "model_name": "flight_price_regression", "model_version": "1.0", "status": "loaded" }
```

### POST /api/predict — flight price

```json
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
  "flight_dayofweek": 3
}
```

Returns `{ "predicted_price", "model_version", "model_name" }`.

`flightType` must be one of `firstClass`, `economic`, `premium`.

### POST /api/gender/predict — gender classification

```json
{ "user name": "Robert Braun", "age": 33, "company": "4You" }
```

Returns `{ "gender", "probability", "model_version" }`, with `gender` one of
`male`, `female`, `none`.

### GET /api/recommend/places

```json
{ "places": ["Rio de Janeiro (RJ)", "..."], "total": 9 }
```

### POST /api/recommend/recommendations — hotel recommendations

```json
{
  "place": "Rio de Janeiro (RJ)",
  "max_price_per_day": 200,
  "days": 3,
  "company": "4You",
  "top_n": 5
}
```

Returns a ranked list of `{ hotel_name, place, price_per_day, total_cost,
score, reason }`. All fields except `top_n` are optional; an empty body returns
general recommendations.

## Validation & errors

Input is validated with **Pydantic schemas** (see `src/schemas/README.md`).
Errors return `{ "error", "field"?, "code" }` with an appropriate HTTP status:

| Code | HTTP | Meaning |
|------|------|---------|
| `invalid_json` | 400 | Malformed JSON body |
| `validation_failed` | 400 | Missing/invalid field |
| `model_unavailable` / `catalog_unavailable` | 503 | Artifact/catalog missing |
| `prediction_failed` / `recommendation_failed` | 500 | Inference error |

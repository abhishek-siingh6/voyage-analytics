# Schemas

Pydantic schemas for request validation and response formatting, one module
per service.

| File | Contents |
|------|----------|
| `prediction.py` | Flight-price request (`FlightPredictionXgboostInput`) + response models |
| `gender.py` | Gender request (`GenderInput`) + response (`GenderOutput`) |
| `recommendation.py` | Recommendation request + `HotelRecommendation` / response |

## Flight price — request

`FlightPredictionXgboostInput` mirrors the exact columns the XGBoost pipeline
expects. Reserved keyword columns (`from`, `to`, `flightType`) are mapped via
pydantic `alias`, so the JSON payload uses the raw keys.

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

Validation: `time`, `distance` > 0; `flight_month` 1-12; `flight_day` 1-31;
`flight_dayofweek` 0-6; `flightType` in `{firstClass, economic, premium}`.

## Flight price — response

`FlightPredictionOutput`

```json
{ "predicted_price": 451.30, "model_version": "1.0", "model_name": "flight_price_regression" }
```

## Gender — request / response

`GenderInput` (`user name`, `age`, `company`) → `GenderOutput`
(`gender`, `probability`, `model_version`).

```json
{ "user name": "Robert Braun", "age": 33, "company": "4You" }
```
```json
{ "gender": "male", "probability": 0.995, "model_version": "1.0" }
```

## Recommendation

`RecommendationRequest` (all fields optional except `top_n`): `place`,
`max_price_per_day`, `days`, `company`, `top_n` → `RecommendationResponse`
with a ranked `recommendations` list of `HotelRecommendation`
(`hotel_name`, `place`, `price_per_day`, `total_cost`, `score`, `reason`).

## Shared response models

- `HealthCheck` (`status`)
- `ModelInfo` (`model_name`, `model_version`, `status`)
- `ErrorResponse` (`error`, `field?`, `code`)

## Validation / error handling

Invalid input raises a pydantic `ValidationError`, converted by the routes into
an HTTP 400 with the offending field name.

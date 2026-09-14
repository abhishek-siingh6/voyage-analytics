# Services

Business logic layer — one service per model, decoupling the API routes from
model inference.

| File | Purpose |
|------|---------|
| `prediction_service.py` | Runs the flight-price XGBoost pipeline |
| `gender_service.py` | Runs the gender classifier |
| `recommendation_service.py` | Content-based hotel recommender |

## Design

- Services are called by the API Blueprints (`api/README.md`) after request
  validation.
- They load models via the loaders (`src/model/README.md`) and return typed
  Pydantic outputs.
- `recommendation_service.py` reads the hotel catalog
  (`artifacts/hotel_catalog.json`) and ranks hotels by a weighted match score
  (place, budget, company preference).

## Tests

```bash
pytest tests/test_prediction.py tests/test_gender.py tests/test_recommendation.py
```

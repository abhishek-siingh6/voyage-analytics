# Tests

Automated test suite (pytest). Run everything from the repository root with
`pytest` (or `pytest --cov=src --cov=api` for coverage).

| File | Scope |
|------|-------|
| `conftest.py` | Autouse fixture that resets the shared model/catalog caches between tests |
| `test_health.py` | Health endpoint |
| `test_prediction.py` | Flight-price prediction endpoint (skipped if the model artifact is absent) |
| `test_gender.py` | Gender classification endpoint |
| `test_recommendation.py` | Recommendation endpoints |
| `test_model_loader.py` | Model loader caching / error behaviour |
| `fixtures/dummy_model.py` | A `DummyPipeline` for loader tests |

## Notes

- The flight-price tests are skipped when `artifacts/flight_price_pipeline.joblib`
  is missing (it is git-ignored), so CI stays green on a fresh checkout.
- Tests require the gender model and hotel catalog artifacts, which are
  tracked in the repo.

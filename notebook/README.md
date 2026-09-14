# Notebook

This folder holds the ML team's training notebooks.

| File | Purpose |
|------|---------|
| `Copy_of_Untitled.ipynb` | Trains the flight-price regression model in Google Colab |

## What the notebook does

1. Loads `users.csv`, `flights.csv`, `hotels.csv` (from Google Drive).
2. Merges datasets on `userCode` / `travelCode` and engineers date features
   (`flight_year`, `flight_month`, `flight_day`, `flight_dayofweek`).
3. Compares regression baselines (Linear Regression, Random Forest, XGBoost)
   with a grouped train/test split to avoid leakage.
4. Selects the **XGBoost** pipeline and serializes it to
   `flight_price_xgboost.pkl` → handed off as
   `artifacts/flight_price_pipeline.joblib`.

## Note

This notebook is the **ML-team training source** and is intentionally left
unmodified. Production code consumes the resulting artifact, not the notebook.

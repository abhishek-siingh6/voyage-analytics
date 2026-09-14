"""Reproduce the Voyage Analytics flight-price XGBoost model from the notebook.

This script is a straight, script-ised reproduction of the ML team's Colab
notebook (`notebook/Copy_of_Untitled.ipynb`). It performs the exact same data
preparation, grouped train/test split and XGBoost pipeline, so the produced
artifact is equivalent to the notebook's `flight_price_xgboost.pkl` / the
production `artifacts/flight_price_pipeline.joblib`.

It reads the bundled datasets from `artifacts/data/` and writes:
    - artifacts/flight_price_pipeline.joblib  (the model the API consumes)
    - artifacts/metrics.json                  (MAE / RMSE / R2 for MLflow)
    - artifacts/model_metadata.json           (metadata for MLflow)

Usage:
    python scripts/train_flight_model.py
"""

import argparse
import json
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from config.settings import settings

#: Feature columns fed to the model (exactly as in the notebook).
FEATURES = [
    "from",
    "to",
    "flightType",
    "time",
    "distance",
    "agency",
    "flight_year",
    "flight_month",
    "flight_day",
    "flight_dayofweek",
]

CATEGORICAL_FEATURES = ["from", "to", "flightType", "agency"]
NUMERICAL_FEATURES = [
    "time",
    "distance",
    "flight_year",
    "flight_month",
    "flight_day",
    "flight_dayofweek",
]

#: Grouping columns used for the leakage-safe split (as in the notebook).
GROUP_COLS = CATEGORICAL_FEATURES + ["time", "distance"]


def _prepare_data(flights_path: str, hotels_path: str, users_path: str) -> pd.DataFrame:
    """Mirror the notebook's merges + feature engineering."""
    flights = pd.read_csv(flights_path)
    hotels = pd.read_csv(hotels_path)
    users = pd.read_csv(users_path)

    # Notebook renaming.
    hotels.rename(columns={"name": "hotel name"}, inplace=True)
    users.rename(columns={"name": "user name"}, inplace=True)

    travel_df = pd.merge(
        flights, hotels, on=["travelCode", "userCode"], how="inner"
    )
    df = pd.merge(
        travel_df, users, left_on="userCode", right_on="code", how="left"
    )

    # Notebook column renames.
    df.rename(
        columns={
            "price_x": "flight price",
            "price_y": "hotel price per day",
            "date_x": "flight date",
            "date_y": "hotel_checkin date",
            "total": "hotel total cost",
        },
        inplace=True,
    )

    # Datetime + date features.
    df["flight date"] = pd.to_datetime(df["flight date"])
    df["flight_month"] = df["flight date"].dt.month
    df["flight_year"] = df["flight date"].dt.year
    df["flight_day"] = df["flight date"].dt.day
    df["flight_dayofweek"] = df["flight date"].dt.dayofweek
    df.drop(columns=["flight date"], inplace=True)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce the flight-price XGBoost model from the notebook"
    )
    parser.add_argument(
        "--flights", default="artifacts/data/flights.csv", help="Path to flights.csv"
    )
    parser.add_argument(
        "--hotels", default="artifacts/data/hotels.csv", help="Path to hotels.csv"
    )
    parser.add_argument(
        "--users", default="artifacts/data/users.csv", help="Path to users.csv"
    )
    parser.add_argument(
        "--output",
        default="artifacts/flight_price_pipeline.joblib",
        help="Where to write the model artifact",
    )
    args = parser.parse_args()

    for path in (args.flights, args.hotels, args.users):
        if not os.path.exists(path):
            raise SystemExit(f"Dataset not found: {path}")

    print(f"Loading data from {args.flights}, {args.hotels}, {args.users}")
    df = _prepare_data(args.flights, args.hotels, args.users)

    X = df[FEATURES]
    y = df["flight price"]

    # Leakage-safe grouped split on repeating feature combinations (notebook).
    groups = df[GROUP_COLS].astype(str).agg("_".join, axis=1)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("num", "passthrough", NUMERICAL_FEATURES),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("proeprocessor", preprocessor),  # note: typo kept from notebook
            (
                "model",
                XGBRegressor(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print(f"Training on {len(X_train)} rows, testing on {len(X_test)} rows")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    print(f"\nMAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2  : {r2:.4f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    joblib.dump(pipeline, args.output)
    print(f"\nSaved model artifact to: {args.output}")

    # Emit metrics + metadata for MLflow.
    out_dir = os.path.dirname(args.output) or "."
    metrics = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "n_samples": int(len(df)),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote metrics to: {os.path.join(out_dir, 'metrics.json')}")

    metadata = {
        "model_name": "voyage_flight_price",
        "model_version": "1.0",
        "algorithm": "XGBRegressor",
        "framework": "xgboost",
        "task": "regression",
        "features": FEATURES,
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
    }
    with open(os.path.join(out_dir, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote metadata to: {os.path.join(out_dir, 'model_metadata.json')}")


if __name__ == "__main__":
    main()

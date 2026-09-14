"""Train and persist the Voyage Analytics gender classification model.

Builds a classification pipeline that predicts a user's gender from their
profile. Analysis of the real ``users.csv`` (schema: code, company, name,
gender, age; gender in {male, female, none}) showed the **first name** is the
dominant signal (e.g. Robert/John -> male, Mary/Lisa -> female), so the model
feeds character n-grams of the lower-cased first name plus the numeric ``age``
into a scikit-learn Pipeline (TfidfVectorizer + StandardScaler + classifier).

The resulting ``.joblib`` artifact is consumed by the production Flask API.

Usage:
    python scripts/train_gender_model.py \
        --data-path "<dataset>/users.csv" \
        --output artifacts/gender_model.joblib
"""

import argparse
import os
import sys

# Make the project root importable when this script is run directly.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import joblib
import json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.gender_features import _first_name


def _feature_matrix(users: pd.DataFrame) -> pd.DataFrame:
    """Build model features from the raw users dataframe."""
    return pd.DataFrame(
        {
            "first_name": users["name"].map(_first_name),
            "age": users["age"].astype(float),
        }
    )


def main():
    parser = argparse.ArgumentParser(description="Train the gender classification model")
    parser.add_argument(
        "--data-path",
        default="artifacts/data/users.csv",
        help="Path to users.csv",
    )
    parser.add_argument(
        "--output",
        default="artifacts/gender_model.joblib",
        help="Where to write the model artifact",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        raise SystemExit(f"users.csv not found at: {args.data_path}")

    users = pd.read_csv(args.data_path)
    print(f"Loaded dataset from {args.data_path} ({len(users)} rows)")

    needed = {"gender", "name", "age"}
    missing = needed - set(users.columns)
    if missing:
        raise SystemExit(f"users.csv is missing required columns: {sorted(missing)}")

    X = _feature_matrix(users)
    y = users["gender"].astype(str).str.lower()

    preprocessor = ColumnTransformer(
        transformers=[
            ("name", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)), "first_name"),
            ("age", StandardScaler(), ["age"]),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=200, random_state=args.seed, n_jobs=-1
            )),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"\nAccuracy: {acc:.4f}")
    print(f"F1 (macro): {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    joblib.dump(pipeline, args.output)
    print(f"\nSaved model artifact to: {args.output}")

    # Emit metrics + metadata for MLflow tracking.
    out_dir = os.path.dirname(args.output) or "."
    metrics = {
        "accuracy": round(acc, 4),
        "f1_macro": round(f1, 4),
        "n_samples": int(len(users)),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote metrics to: {os.path.join(out_dir, 'metrics.json')}")

    metadata = {
        "model_name": "voyage_gender",
        "model_version": "1.0",
        "algorithm": "RandomForestClassifier",
        "framework": "scikit-learn",
        "task": "classification",
        "data_path": args.data_path,
        "classes": sorted(pipeline.classes_),
    }
    with open(os.path.join(out_dir, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote metadata to: {os.path.join(out_dir, 'model_metadata.json')}")


if __name__ == "__main__":
    main()

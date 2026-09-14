"""Shared feature engineering for the gender classification model.

Kept in one place so the training script and the serving service derive the
exact same feature matrix, avoiding train/serve skew.

Real-world analysis of ``users.csv`` showed that the **first name** is the
dominant predictor of gender (e.g. Robert/John -> male, Mary/Lisa -> female).
We therefore expose the lower-cased first name as free text that the model's
character n-gram vectorizer consumes, alongside the numeric ``age``.
"""

import pandas as pd


def _first_name(user_name: str) -> str:
    cleaned = str(user_name).strip()
    if not cleaned:
        return ""
    return cleaned.split()[0].lower()


def build_gender_features(user_name: str, age: int) -> pd.DataFrame:
    """Build the feature row consumed by the gender model pipeline.

    Args:
        user_name: Full user name (e.g. "Robert Braun").
        age: User's age.

    Returns:
        A single-row DataFrame with columns ``first_name`` and ``age``.
    """
    return pd.DataFrame(
        [
            {
                "first_name": _first_name(user_name),
                "age": float(age),
            }
        ]
    )

"""Service layer for the gender classification model."""

import logging
from typing import Any, Dict, List

from src.features.gender_features import build_gender_features
from src.model.gender_loader import get_gender_model
from src.schemas.gender import GenderInput, GenderOutput

logger = logging.getLogger(__name__)

#: Class labels are normalised to this canonical set on output.
_CANONICAL = ["female", "male", "none"]


class GenderService:
    """Predicts a user's gender from their profile attributes."""

    @classmethod
    def predict(cls, input_data: GenderInput) -> GenderOutput:
        """Generate a gender prediction for a single user."""
        try:
            model = get_gender_model()
            features = build_gender_features(
                input_data.user_name, input_data.age
            )

            classes: List[str] = list(getattr(model, "classes_", _CANONICAL))

            # predict_proba may not exist on every classifier; fall back to
            # predict when unavailable.
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features)[0]
            else:
                proba = None

            if proba is not None and len(proba) == len(classes):
                idx = int(proba.argmax())
                gender = str(classes[idx])
                probability = float(proba[idx])
            else:
                gender = str(model.predict(features)[0])
                probability = 1.0

            # Normalise to the canonical label set.
            if gender not in _CANONICAL:
                gender = "none" if gender not in _CANONICAL else gender

            logger.info(
                "Gender prediction: %s (prob=%.3f)", gender, probability
            )
            return GenderOutput(
                gender=gender,
                probability=round(probability, 4),
                model_version="1.0",
            )
        except Exception as e:
            logger.exception("Gender prediction failed")
            raise RuntimeError(f"Gender prediction failed: {e}") from e

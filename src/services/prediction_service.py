"""Prediction service for Voyage Analytics."""

import logging
from typing import Dict, Any, Optional

import pandas as pd

from src.model.loader import get_model
from src.schemas.prediction import (
    FlightPredictionXgboostInput,
    FlightPredictionOutput,
    ModelInfo,
)
from config.settings import settings

logger = logging.getLogger(__name__)


class PredictionService:
    """Service layer for flight price prediction."""

    _model_info_cache: Optional[Dict[str, Any]] = None

    @classmethod
    def predict(cls, input_data: FlightPredictionXgboostInput) -> float:
        """
        Generate a flight price prediction.

        Args:
            input_data: Validated prediction input schema.

        Returns:
            Predicted flight price as a float.

        Raises:
            RuntimeError: If model loading or prediction fails.
        """
        try:
            model = get_model()
            # Convert input to DataFrame for sklearn pipeline
            input_dict = input_data.model_dump(by_alias=True)
            df = pd.DataFrame([input_dict])

            logger.debug(f"Running prediction with features: {list(df.columns)}")
            logger.debug(f"Input data types: {df.dtypes.to_dict()}")

            prediction = model.predict(df)

            # Extract scalar value from prediction array
            predicted_price = float(prediction[0])

            logger.info(f"Prediction successful: ${predicted_price:.2f}")

            return predicted_price

        except Exception as e:
            logger.exception(f"Prediction failed")
            raise RuntimeError(f"Prediction failed: {e}") from e

    @classmethod
    def get_model_info(cls) -> ModelInfo:
        """
        Get metadata about the loaded model.

        Returns:
            ModelInfo containing model name, version, and status.
        """
        try:
            model = get_model()

            # Try to get metadata from the model object if available
            model_name = getattr(model, "model_name", "flight_price_regression")
            model_version = getattr(model, "model_version", "1.0")

            # If not set on model, try to read from metadata file
            if model_name == "flight_price_regression" and settings.model_metadata_path:
                import json
                import os

                if os.path.exists(settings.model_metadata_path):
                    try:
                        with open(settings.model_metadata_path, "r") as f:
                            metadata = json.load(f)
                            model_name = metadata.get("model_name", model_name)
                            model_version = metadata.get("model_version", model_version)
                    except Exception:
                        pass

            return ModelInfo(
                model_name=model_name,
                model_version=model_version,
                status="loaded",
            )

        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return ModelInfo(
                model_name=None,
                model_version=None,
                status="error",
            )

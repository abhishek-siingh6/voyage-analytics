"""Model loader for Voyage Analytics flight price prediction."""

import joblib
import os
import logging
from typing import Any, Optional

# Compatibility shim for older sklearn pickles
try:
    from sklearn.compose import _column_transformer as ct
    if not hasattr(ct, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass
        ct._RemainderColsList = _RemainderColsList
except Exception:
    pass

from config.settings import settings

logger = logging.getLogger(__name__)


class ModelLoader:
    """Manages loading and caching of the flight price prediction model."""

    _model_instance: Optional[Any] = None

    @classmethod
    def load_model(cls) -> Any:
        """
        Load the flight price prediction model.

        Returns the loaded model pipeline or raises an exception if loading fails.
        The model is cached after first load to avoid repeated file I/O.
        """
        if cls._model_instance is not None:
            logger.debug("Returning cached model")
            return cls._model_instance

        model_path = settings.model_path

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model artifact not found at: {model_path}. "
                "Ensure the model has been trained and copied to the artifacts directory. "
                "See artifacts/README.md for instructions."
            )

        try:
            logger.info(f"Loading model from: {model_path}")
            cls._model_instance = joblib.load(model_path)
            logger.info("Model loaded successfully")
            return cls._model_instance
        except Exception as e:
            cls._model_instance = None
            raise RuntimeError(f"Failed to load model: {e}") from e

    @classmethod
    def unload_model(cls) -> None:
        """Clear the cached model instance."""
        cls._model_instance = None
        logger.info("Model cache cleared")

    @classmethod
    def get_model(cls) -> Any:
        """
        Get the loaded model, loading it if necessary.

        Returns:
            The loaded model pipeline object.
        """
        if cls._model_instance is None:
            return cls.load_model()
        return cls._model_instance


def load_model() -> Any:
    """
    Convenience function to load the model.

    Returns:
        The loaded model pipeline object.
    """
    return ModelLoader.load_model()


def get_model() -> Any:
    """
    Convenience function to get or load the model.

    Returns:
        The loaded model pipeline object.
    """
    return ModelLoader.get_model()

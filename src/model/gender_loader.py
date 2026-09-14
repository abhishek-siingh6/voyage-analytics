"""Model loader for the Voyage Analytics gender classification model."""

import joblib
import logging
import os
from typing import Any, Optional

from config.settings import settings

logger = logging.getLogger(__name__)


class GenderModelLoader:
    """Loads and caches the gender classification model."""

    _model_instance: Optional[Any] = None

    #: Class labels the model was trained with (used to normalise outputs).
    _classes: Optional[list] = None

    @classmethod
    def load_model(cls) -> Any:
        """Load (and cache) the gender model artifact."""
        if cls._model_instance is not None:
            return cls._model_instance

        model_path = settings.gender_model_path

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Gender model artifact not found at: {model_path}. "
                "Run `python scripts/train_gender_model.py` to create it."
            )

        try:
            logger.info("Loading gender model from: %s", model_path)
            cls._model_instance = joblib.load(model_path)
            cls._classes = list(getattr(cls._model_instance, "classes_", []))
            logger.info("Gender model loaded successfully")
            return cls._model_instance
        except Exception as e:
            cls._model_instance = None
            cls._classes = None
            raise RuntimeError(f"Failed to load gender model: {e}") from e

    @classmethod
    def get_model(cls) -> Any:
        """Get the loaded gender model, loading it if necessary."""
        if cls._model_instance is None:
            return cls.load_model()
        return cls._model_instance

    @classmethod
    def unload_model(cls) -> None:
        """Clear the cached gender model instance."""
        cls._model_instance = None
        cls._classes = None


def get_gender_model() -> Any:
    """Convenience accessor for the gender model."""
    return GenderModelLoader.get_model()

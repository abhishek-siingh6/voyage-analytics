"""Configuration settings for Voyage Analytics."""

import os
from typing import Optional


class Settings:
    """Application settings loaded from environment variables."""

    def __init__(self):
        # Model settings
        self.model_path: str = os.getenv(
            "MODEL_PATH", "artifacts/flight_price_pipeline.joblib"
        )
        self.model_metadata_path: Optional[str] = os.getenv(
            "MODEL_METADATA_PATH", "artifacts/model_metadata.json"
        )

        # Gender classification model
        self.gender_model_path: str = os.getenv(
            "GENDER_MODEL_PATH", "artifacts/gender_model.joblib"
        )

        # Recommendation engine data
        self.hotels_data_path: str = os.getenv(
            "HOTELS_DATA_PATH", "artifacts/data/hotels.csv"
        )
        self.users_data_path: str = os.getenv(
            "USERS_DATA_PATH", "artifacts/data/users.csv"
        )
        self.hotels_catalog_path: str = os.getenv(
            "HOTELS_CATALOG_PATH", "artifacts/hotel_catalog.json"
        )

        # MLflow settings
        self.mlflow_tracking_uri: str = os.getenv(
            "MLFLOW_TRACKING_URI", "http://localhost:5000"
        )
        self.mlflow_experiment_name: str = os.getenv(
            "MLFLOW_EXPERIMENT_NAME", "voyage-flight-price"
        )

        # API settings
        self.api_host: str = os.getenv("API_HOST", "0.0.0.0")
        self.api_port: int = int(os.getenv("API_PORT", "5000"))

        # Testing settings
        self.testing_mode: bool = os.getenv("TESTING", "false").lower() == "true"


# Global settings instance
settings = Settings()

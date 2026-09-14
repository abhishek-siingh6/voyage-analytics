"""Dummy model for development/testing.

WARNING: This is a MOCK model for development purposes only.
DO NOT use this model in production.

To use the real model, replace this file with the trained
flight_price_pipeline.joblib from Google Colab.
"""

from typing import Any, List


class DummyPipeline:
    """Dummy model pipeline that simulates flight price prediction."""

    def __init__(self):
        self.model_name = "flight_price_regression"
        self.model_version = "1.0"
        self.algorithm = "DummyRegressor (development only)"
        self.training_date = "2024-01-01"

    def predict(self, X: Any) -> List[float]:
        """
        Make dummy predictions based on input features.

        Args:
            X: Input features (dict or list of dicts)

        Returns:
            List of predicted prices (dummy values)
        """
        if isinstance(X, dict):
            X = [X]

        predictions = []

        for item in X:
            # Simple dummy calculation based on flight duration
            base_price = 50.0
            duration = item.get("flight_duration", 5) * 20
            distance = item.get("distance", 1000) * 0.15
            class_multiplier = {"economy": 1.0, "business": 1.8, "first": 2.5}.get(
                item.get("class_type", "economy"), 1.0
            )

            price = (base_price + duration + distance) * class_multiplier
            predictions.append(price)

        return predictions

    def save(self, path: str) -> None:
        """Save dummy model (for testing)."""
        import joblib
        joblib.dump(self, path)


def create_dummy_model(path: str) -> None:
    """
    Create and save a dummy model for testing.

    Args:
        path: Path where the dummy model will be saved
    """
    pipeline = DummyPipeline()
    pipeline.save(path)
    print(f"Dummy model saved to: {path}")


if __name__ == "__main__":
    # Create dummy model in test fixtures directory
    import os
    os.makedirs("tests/fixtures", exist_ok=True)
    create_dummy_model("tests/fixtures/dummy_model.joblib")

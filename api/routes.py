"""Flask API routes for Voyage Analytics."""

import logging
from typing import Dict, Any

from flask import Blueprint, jsonify, request, Response

from src.schemas.prediction import (
    FlightPredictionXgboostInput,
    FlightPredictionOutput,
    HealthCheck,
    ModelInfo,
    ErrorResponse,
)
from src.services.prediction_service import PredictionService

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__)


def json_response(
    data: Dict[str, Any], status: int = 200
) -> Response:
    """Create a JSON response with proper headers."""
    return jsonify(data), status


def error_response(
    message: str, field: str = None, code: str = "error", status: int = 400
) -> Response:
    """Create an error response."""
    response = ErrorResponse(error=message, field=field, code=code)
    return json_response(response.model_dump(), status)


@api_bp.route("/health", methods=["GET"])
def health_check() -> Response:
    """Health check endpoint."""
    logger.debug("Health check requested")
    return json_response(HealthCheck(status="healthy").model_dump())


@api_bp.route("/model-info", methods=["GET"])
def get_model_info() -> Response:
    """Get information about the loaded model."""
    logger.debug("Model info requested")
    try:
        model_info = PredictionService.get_model_info()
        return json_response(model_info.model_dump())
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        return error_response(
            message="Failed to retrieve model information",
            code="model_unavailable",
            status=503,
        )


@api_bp.route("/predict", methods=["POST"])
def predict() -> Response:
    """Predict flight price endpoint."""
    logger.info("Prediction request received")

    # Parse and validate request body
    try:
        data = request.get_json(force=True)
        if data is None:
            return error_response(
                message="Invalid JSON payload",
                code="invalid_json",
                status=400,
            )
    except Exception as e:
        logger.error(f"Failed to parse JSON: {e}")
        return error_response(
            message="Invalid JSON payload",
            code="invalid_json",
            status=400,
        )

    # Validate input schema
    try:
        input_data = FlightPredictionXgboostInput(**data)
    except Exception as e:
        logger.error(f"Validation error: {e}")
        # Extract the first offending field from a pydantic ValidationError
        field = None
        loc = getattr(e, "errors", None)
        if callable(loc):
            try:
                errs = e.errors()
                if errs and "loc" in errs[0] and errs[0]["loc"]:
                    # loc may be a tuple ("from",); use last element
                    field = str(errs[0]["loc"][-1])
            except Exception:
                field = None
        return error_response(
            message="Invalid input data",
            field=field,
            code="validation_failed",
            status=400,
        )

    # Generate prediction
    try:
        predicted_price = PredictionService.predict(input_data)

        # Get model info for response
        model_info = PredictionService.get_model_info()

        output = FlightPredictionOutput(
            predicted_price=predicted_price,
            model_version=model_info.model_version,
            model_name=model_info.model_name,
        )

        logger.info(f"Prediction successful: ${predicted_price:.2f}")
        return json_response(output.model_dump())

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return error_response(
            message="Prediction failed",
            code="prediction_failed",
            status=500,
        )


def register_routes(app) -> None:
    """Register API routes with the Flask app."""
    app.register_blueprint(api_bp, url_prefix="/api")

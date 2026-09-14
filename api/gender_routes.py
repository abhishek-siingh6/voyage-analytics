"""Flask routes for the gender classification endpoint."""

import logging
from typing import Any, Dict, Optional

from flask import Blueprint, jsonify, request, Response

from src.schemas.gender import GenderInput, GenderOutput
from src.services.gender_service import GenderService

logger = logging.getLogger(__name__)

gender_bp = Blueprint("gender", __name__)


def _json(data: Dict[str, Any], status: int = 200) -> Response:
    return jsonify(data), status


def _error(
    message: str,
    code: str = "error",
    status: int = 400,
    field: Optional[str] = None,
) -> Response:
    payload: Dict[str, Any] = {"error": message, "code": code}
    if field:
        payload["field"] = field
    return _json(payload, status)


@gender_bp.route("/health", methods=["GET"])
def gender_health() -> Response:
    """Health check specific to the gender endpoint."""
    return _json({"status": "healthy"})


@gender_bp.route("/predict", methods=["POST"])
def predict_gender() -> Response:
    """Predict a user's gender from their profile attributes."""
    logger.info("Gender prediction request received")

    try:
        data = request.get_json(force=True)
        if data is None:
            return _error("Invalid JSON payload", code="invalid_json", status=400)
    except Exception as e:
        logger.error("Failed to parse JSON: %s", e)
        return _error("Invalid JSON payload", code="invalid_json", status=400)

    try:
        input_data = GenderInput(**data)
    except Exception as e:
        logger.error("Gender validation error: %s", e)
        field: Optional[str] = None
        errors_fn = getattr(e, "errors", None)
        if callable(errors_fn):
            try:
                errs = e.errors()
                if errs and errs[0].get("loc"):
                    field = str(errs[0]["loc"][-1])
            except Exception:
                field = None
        return _error(
            "Invalid input data",
            field=field,
            code="validation_failed",
            status=400,
        )

    try:
        output: GenderOutput = GenderService.predict(input_data)
        return _json(output.model_dump())
    except FileNotFoundError as e:
        logger.error("Gender model unavailable: %s", e)
        return _error(
            "Gender model artifact not found",
            code="model_unavailable",
            status=503,
        )
    except Exception as e:
        logger.error("Gender prediction failed: %s", e)
        return _error("Gender prediction failed", code="prediction_failed", status=500)


def register_gender_routes(app) -> None:
    """Register gender routes with the Flask app."""
    app.register_blueprint(gender_bp, url_prefix="/api/gender")

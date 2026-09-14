"""Flask routes for the travel recommendation endpoint."""

import logging
from typing import Any, Dict, Optional

from flask import Blueprint, jsonify, request, Response

from src.schemas.recommendation import RecommendationRequest, RecommendationResponse
from src.services.recommendation_service import RecommendationService
from src.services.location_search_service import get_location_search_service

logger = logging.getLogger(__name__)

recommend_bp = Blueprint("recommend", __name__)


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


@recommend_bp.route("/health", methods=["GET"])
def recommend_health() -> Response:
    """Health check for the recommendation endpoint."""
    return _json({"status": "healthy"})


@recommend_bp.route("/places", methods=["GET"])
def list_places() -> Response:
    """List all destination cities available for recommendations."""
    try:
        catalog = RecommendationService._catalog()
        places = catalog.data.get("places", [])
        return _json({"places": places, "total": len(places)})
    except Exception as e:
        logger.error("Failed to list places: %s", e)
        return _error(
            "Recommendation catalog unavailable",
            code="catalog_unavailable",
            status=503,
        )


@recommend_bp.route("/places/search", methods=["GET"])
def search_places() -> Response:
    """Search places matching user input with fuzzy matching using inverted index."""
    query = request.args.get("q", "").strip().lower()
    if not query:
        return _error("Missing query parameter 'q'", code="missing_query", status=400)
    
    try:
        search_service = get_location_search_service()
        matches = search_service.search(query, top_n=5)
        
        if matches:
            return _json({"matches": matches, "found": True})
        else:
            return _json({"matches": [], "found": False, "message": "not in database"})
    except Exception as e:
        logger.error("Failed to search places: %s", e)
        return _error(
            "Failed to search places",
            code="search_failed",
            status=500,
        )


@recommend_bp.route("/recommendations", methods=["POST"])
def get_recommendations() -> Response:
    """Return ranked hotel recommendations for the given preferences."""
    logger.info("Recommendation request received")

    try:
        data = request.get_json(force=True) or {}
    except Exception as e:
        logger.error("Failed to parse JSON: %s", e)
        return _error("Invalid JSON payload", code="invalid_json", status=400)

    # Empty body (no filters) is valid and returns general recommendations.
    try:
        req = RecommendationRequest(**data)
    except Exception as e:
        logger.error("Recommendation validation error: %s", e)
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
            "Invalid input data", field=field, code="validation_failed", status=400
        )

    try:
        output: RecommendationResponse = RecommendationService.recommend(req)
        return _json(output.model_dump())
    except FileNotFoundError as e:
        logger.error("Recommendation catalog unavailable: %s", e)
        return _error(
            "Recommendation catalog not found",
            code="catalog_unavailable",
            status=503,
        )
    except Exception as e:
        logger.error("Recommendation failed: %s", e)
        return _error("Recommendation failed", code="recommendation_failed", status=500)


def register_recommend_routes(app) -> None:
    """Register recommendation routes with the Flask app."""
    app.register_blueprint(recommend_bp, url_prefix="/api/recommend")

"""Content-based travel recommendation service for Voyage Analytics.

Recommends hotels from the real hotel catalog (derived from ``hotels.csv``)
ranked by how well they match a user's stated preferences plus historical
company booking behaviour. Each hotel serves exactly one place at a fixed
price per day, so place + budget are the primary signals.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from config.settings import settings
from src.schemas.recommendation import (
    HotelRecommendation,
    RecommendationRequest,
    RecommendationResponse,
)

logger = logging.getLogger(__name__)

#: Tunable scoring weights. Keep them simple and explainable.
_WEIGHTS = {
    "place": 50.0,      # exact place match
    "budget": 30.0,     # how well price fits the requested budget
    "company": 20.0,    # historical company preference
}


class _Catalog:
    """Thin cache wrapper around the recommendation catalog JSON."""

    _instance: Optional["_Catalog"] = None

    def __init__(self, path: str):
        self.path = path
        with open(path, encoding="utf-8") as f:
            self.data = json.load(f)

    @classmethod
    def load(cls) -> "_Catalog":
        if cls._instance is None:
            path = settings.hotels_catalog_path
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Recommendation catalog not found at: {path}. "
                    "Run `python scripts/build_recommendation_catalog.py` to create it."
                )
            cls._instance = cls(path)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None


class RecommendationService:
    """Ranks hotels against a set of user preferences."""

    @staticmethod
    def _score_hotel(
        hotel: Dict[str, Any],
        place: Optional[str],
        max_price: Optional[float],
        days: Optional[int],
        company: Optional[str],
    ) -> tuple:
        scores: Dict[str, float] = {}
        reasons: List[str] = []
        name = hotel["name"]
        hp = float(hotel["price_per_day"])
        hplace = hotel["place"]

        # 1) Place
        if place:
            if hplace.lower() == place.lower():
                scores["place"] = _WEIGHTS["place"]
                reasons.append("Exact place match")
            else:
                scores["place"] = 0.0
        else:
            scores["place"] = _WEIGHTS["place"] * 0.5
            reasons.append("No place preference (broad match)")

        # 2) Budget
        if max_price is not None:
            if hp <= max_price:
                # Cheaper hotels get a small bonus; cap at full weight.
                ratio = hp / max_price if max_price else 1.0
                scores["budget"] = _WEIGHTS["budget"] * (0.4 + 0.6 * ratio)
                reasons.append(f"Within budget (${hp:.2f}/night)")
            else:
                scores["budget"] = _WEIGHTS["budget"] * 0.1
                reasons.append(f"Above budget (${hp:.2f} > ${max_price:.2f})")
        else:
            # No budget: neutral score.
            scores["budget"] = _WEIGHTS["budget"] * 0.7
            reasons.append("No budget constraint")

        # 3) Company preference
        if company:
            prefs = RecommendationService._catalog().data.get(
                "company_preferences", {}
            ).get(company, {})
            pref = float(prefs.get(name, 0.0))
            scores["company"] = _WEIGHTS["company"] * (pref / 0.1340)  # ~max weight
            scores["company"] = min(scores["company"], _WEIGHTS["company"])
            if pref > 0:
                reasons.append(f"Popular with {company}")
            else:
                reasons.append("No company booking history")
        else:
            scores["company"] = _WEIGHTS["company"] * 0.5  # neutral
            reasons.append("No personalisation (company not provided)")

        total_score = min(100.0, sum(scores.values()))
        return round(total_score, 2), "; ".join(reasons)

    @staticmethod
    def _catalog() -> _Catalog:
        return _Catalog.load()

    @classmethod
    def recommend(cls, req: RecommendationRequest) -> RecommendationResponse:
        catalog = cls._catalog()
        hotels = [
            {
                "name": name,
                "place": meta["place"],
                "price_per_day": meta["price_per_day"],
            }
            for name, meta in catalog.data["hotels"].items()
        ]
        
        # Filter hotels by place if specified - place is priority filter
        if req.place:
            hotels = [h for h in hotels if h["place"].lower() == req.place.lower()]
        
        results: List[Dict[str, Any]] = []
        for hotel in hotels:
            score, reason = cls._score_hotel(
                hotel,
                req.place,
                req.max_price_per_day,
                req.days,
                req.company,
            )
            total_cost = None
            if req.days is not None:
                total_cost = round(hotel["price_per_day"] * req.days, 2)
            results.append(
                {
                    "hotel_name": hotel["name"],
                    "place": hotel["place"],
                    "price_per_day": hotel["price_per_day"],
                    "total_cost": total_cost,
                    "score": score,
                    "reason": reason,
                }
            )

        # Rank by score desc, tie-break by price asc.
        results.sort(key=lambda r: (-r["score"], r["price_per_day"]))

        # Return ALL matching hotels (not limited to top_n when filtering by place)
        # top_n is only used when no place filter is specified
        if req.place:
            top = results  # Return all matching hotels
        else:
            top = results[: req.top_n]
            
        recommendations = [HotelRecommendation(**r) for r in top]

        return RecommendationResponse(
            recommendations=recommendations,
            total=len(recommendations),
            filters={
                "place": req.place,
                "max_price_per_day": req.max_price_per_day,
                "days": req.days,
                "company": req.company,
            },
        )

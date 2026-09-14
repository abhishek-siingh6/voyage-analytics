"""Input/output schemas for the travel recommendation engine."""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RecommendationRequest(BaseModel):
    """User preferences used to rank hotel recommendations."""

    place: Optional[str] = Field(
        None, description="Destination city (e.g. 'Rio de Janeiro (RJ)')"
    )
    max_price_per_day: Optional[float] = Field(
        None, gt=0, description="Maximum acceptable price per night"
    )
    days: Optional[int] = Field(
        None, ge=1, le=30, description="Length of stay in days (for total cost)"
    )
    company: Optional[str] = Field(
        None, min_length=1, description="User's company (used for personalisation)"
    )
    top_n: int = Field(5, ge=1, le=20, description="Number of recommendations to return")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "place": "Rio de Janeiro (RJ)",
                "max_price_per_day": 200,
                "days": 3,
                "company": "4You",
                "top_n": 5,
            }
        }
    )

    @field_validator("place", "company")
    @classmethod
    def strip_optional_str(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        s = v.strip()
        return s or None


class HotelRecommendation(BaseModel):
    """A single recommended hotel."""

    hotel_name: str = Field(..., description="Hotel name (e.g. 'Hotel A')")
    place: str = Field(..., description="City the hotel is located in")
    price_per_day: float = Field(..., description="Price per night")
    total_cost: Optional[float] = Field(
        None, description="Estimated total cost (price x days) if days provided"
    )
    score: float = Field(..., description="Match score (0-100)")
    reason: str = Field(..., description="Human-readable explanation of the score")


class RecommendationResponse(BaseModel):
    """Container for a ranked list of hotel recommendations."""

    recommendations: List[HotelRecommendation] = Field(
        ..., description="Ranked hotel recommendations"
    )
    total: int = Field(..., description="Number of recommendations returned")
    filters: dict = Field(default_factory=dict, description="Applied filter summary")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "recommendations": [
                    {
                        "hotel_name": "Hotel CB",
                        "place": "Rio de Janeiro (RJ)",
                        "price_per_day": 165.99,
                        "total_cost": 497.97,
                        "score": 100.0,
                        "reason": "Exact place match",
                    }
                ],
                "total": 1,
                "filters": {"place": "Rio de Janeiro (RJ)"},
            }
        }
    )

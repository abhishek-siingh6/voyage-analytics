"""Input schema for flight price prediction using the XGBoost pipeline.

This schema matches the exact columns expected by the trained pipeline
saved in `artifacts/flight_price_pipeline.joblib`.
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional


class FlightPredictionXgboostInput(BaseModel):
    """Schema matching the XGBoost model's input features.

    Column names that collide with reserved keywords (``from`` / ``to``) are
    mapped from the wire format via pydantic aliases.
    """

    model_config = ConfigDict(
        # Accept both the raw JSON keys ("from"/"to"/"flightType") and the
        # field names (from_location/to_location/flight_type).
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "from": "Recife (PE)",
                "to": "Florianopolis (SC)",
                "flightType": "firstClass",
                "agency": "FlyingDrops",
                "time": 1.76,
                "distance": 676.53,
                "flight_year": 2019,
                "flight_month": 9,
                "flight_day": 26,
                "flight_dayofweek": 3,
            }
        },
    )

    from_location: str = Field(
        ..., alias="from", description="Origin city (e.g., 'Recife (PE)')"
    )
    to_location: str = Field(
        ..., alias="to", description="Destination city (e.g., 'Florianopolis (SC)')"
    )
    flight_type: str = Field(
        ..., alias="flightType", description="Flight class: firstClass, economic, premium"
    )
    agency: str = Field(..., description="Travel agency name")
    time: float = Field(..., gt=0, description="Flight duration in hours")
    distance: float = Field(..., gt=0, description="Distance in km")
    flight_year: int = Field(..., description="Year of flight (e.g., 2019)")
    flight_month: int = Field(..., ge=1, le=12, description="Month of flight (1-12)")
    flight_day: int = Field(..., ge=1, le=31, description="Day of month (1-31)")
    flight_dayofweek: int = Field(..., ge=0, le=6, description="Day of week (0=Monday)")

    @field_validator("flight_type")
    @classmethod
    def validate_flight_type(cls, v: str) -> str:
        allowed = {"firstClass", "economic", "premium"}
        if v not in allowed:
            raise ValueError(f"flightType must be one of {allowed}")
        return v


class FlightPredictionOutput(BaseModel):
    """Schema for flight price prediction output."""

    predicted_price: float = Field(..., description="Predicted flight price")
    model_version: Optional[str] = Field(
        None, description="Version of the model used"
    )
    model_name: Optional[str] = Field(None, description="Name of the model used")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predicted_price": 450.75,
                "model_version": "1.0",
                "model_name": "flight_price_regression",
            }
        }
    )


class HealthCheck(BaseModel):
    """Schema for health check response."""

    status: str = Field(..., description="Health status")

    model_config = ConfigDict(json_schema_extra={"example": {"status": "healthy"}})


class ModelInfo(BaseModel):
    """Schema for model info response."""

    model_name: Optional[str] = Field(None, description="Name of the loaded model")
    model_version: Optional[str] = Field(None, description="Version of the model")
    status: str = Field(..., description="Model loading status")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_name": "flight_price_regression",
                "model_version": "1.0",
                "status": "loaded",
            }
        }
    )


class ErrorResponse(BaseModel):
    """Schema for error responses."""

    error: str = Field(..., description="Error message")
    field: Optional[str] = Field(None, description="Field that caused the error")
    code: str = Field(..., description="Error code")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "Validation failed",
                "field": "departure_hour",
                "code": "invalid_value",
            }
        }
    )

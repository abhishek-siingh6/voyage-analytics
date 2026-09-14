"""Input/output schemas for the gender classification model."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class GenderInput(BaseModel):
    """Input schema for a single gender prediction.

    Mirrors the ``users.csv`` schema from the Colab notebook: the classifier
    infers gender primarily from ``age`` and ``company`` plus name-derived
    features built internally from ``user name``.
    """

    user_name: str = Field(
        ..., alias="user name", min_length=1, description="Full name of the user"
    )
    age: int = Field(..., ge=0, le=120, description="Age of the user")
    company: str = Field(..., min_length=1, description="Company the user belongs to")

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "user name": "Roy Braun",
                "age": 21,
                "company": "4You",
            }
        },
    )

    @field_validator("user_name")
    @classmethod
    def clean_name(cls, v: str) -> str:
        return v.strip()


class GenderOutput(BaseModel):
    """Output schema for a gender prediction."""

    gender: str = Field(..., description="Predicted gender (male/female/none)")
    probability: float = Field(
        ..., description="Confidence of the prediction (0.0-1.0)"
    )
    model_version: Optional[str] = Field(None, description="Version of the model")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "gender": "male",
                "probability": 0.87,
                "model_version": "1.0",
            }
        }
    )

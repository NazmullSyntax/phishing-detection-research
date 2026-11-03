"""
Pydantic schemas for API request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Literal


class PredictRequest(BaseModel):
    """Request body for /predict endpoint."""
    url: str = Field(
        ...,
        min_length=4,
        max_length=2048,
        description="URL to analyze (phishing or legitimate)",
        examples=["https://www.google.com", "http://paypal.verify-login.xyz/account"],
    )


class PredictResponse(BaseModel):
    """Response from /predict endpoint."""
    url: str
    prediction: Literal["phishing", "legitimate"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    phishing_probability: float = Field(..., ge=0.0, le=1.0)
    latency_ms: float = Field(..., ge=0.0)
    model_version: str


class HealthResponse(BaseModel):
    """Response from /health endpoint."""
    status: Literal["healthy", "unhealthy"]
    model_loaded: bool
    model_version: str


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: str
"""
Pydantic schemas for API request/response.
v2: Added 'uncertain' prediction class + risk_level + threshold_used.
"""
from pydantic import BaseModel, Field
from typing import Literal, Optional


class PredictRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    url: str = Field(
        ...,
        min_length=4,
        max_length=2048,
        description="URL to analyze",
        examples=["https://www.google.com", "http://paypal.verify-login.xyz/account"],
    )


class PredictResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    url: str
    prediction: Literal["phishing", "legitimate", "uncertain"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    phishing_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: Literal["safe", "medium", "high"]
    latency_ms: float = Field(..., ge=0.0)
    model_version: str
    threshold_used: Optional[float] = None


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    status: Literal["healthy", "unhealthy"]
    model_loaded: bool
    model_version: str
    threshold: Optional[float] = None


class ErrorResponse(BaseModel):
    error: str
    detail: str
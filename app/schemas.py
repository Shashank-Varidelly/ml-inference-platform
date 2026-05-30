from pydantic import BaseModel, Field, field_validator
from typing import Literal

class TransactionRequest(BaseModel):
    amount: float = Field(..., gt=0, description="Transaction amount in USD")
    hour_of_day: float = Field(..., ge=0, le=23, description="Hour of the day (0-23)")
    txn_last_24h: float = Field(..., ge=0, description="Number of transactions in last 24h")
    velocity_score: float = Field(..., ge=0, le=1, description="Transaction velocity score (0-1)")
    location_match: float = Field(..., ge=0, le=1, description="Location match score (0-1)")
    credit_score: float = Field(..., ge=300, le=850, description="Credit score (300-850)")
    account_age_years: float = Field(..., ge=0, description="Account age in years")
    foreign_ratio: float = Field(..., ge=0, le=1, description="Foreign transaction ratio (0-1)")
    model_version: Literal["v1", "v2", "auto"] = Field(
        default="auto",
        description="Model version: v1 (logistic), v2 (random forest), auto (A/B routing)"
    )

    model_config = {"json_schema_extra": {
        "example": {
            "amount": 250.00,
            "hour_of_day": 2,
            "txn_last_24h": 12,
            "velocity_score": 0.85,
            "location_match": 0.15,
            "credit_score": 590,
            "account_age_years": 0.3,
            "foreign_ratio": 0.7,
            "model_version": "auto"
        }
    }}


class PredictionResponse(BaseModel):
    fraud_probability: float = Field(..., description="Probability of fraud (0-1)")
    is_fraud: bool = Field(..., description="True if predicted as fraud")
    risk_level: Literal["low", "medium", "high"] = Field(..., description="Risk classification")
    model_version_used: str = Field(..., description="Which model served this prediction")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    confidence: float = Field(..., description="Model confidence score")


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict
    uptime_seconds: float
    total_predictions: int


class MetricsSummary(BaseModel):
    total_predictions: int
    fraud_detected: int
    fraud_rate: float
    avg_latency_ms: float
    p99_latency_ms: float
    v1_requests: int
    v2_requests: int

"""
ML Inference Platform — FastAPI entry point.

Routes:
  POST /predict        — fraud prediction (A/B routed)
  GET  /health         — service health + model status
  GET  /metrics        — Prometheus-format metrics
  GET  /metrics/summary — JSON summary for dashboard
  GET  /docs           — auto Swagger UI (free from FastAPI)
"""

import time
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from app.schemas import TransactionRequest, PredictionResponse, HealthResponse, MetricsSummary
from app.model_registry import registry
from app.metrics import metrics

app = FastAPI(
    title="ML Inference Platform",
    description="Real-time fraud detection with A/B model routing and Prometheus metrics.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRAUD_THRESHOLD = 0.5
FEATURE_ORDER = [
    "amount", "hour_of_day", "txn_last_24h",
    "velocity_score", "location_match",
    "credit_score", "account_age_years", "foreign_ratio"
]

def risk_level(prob: float) -> str:
    if prob < 0.3:
        return "low"
    elif prob < 0.6:
        return "medium"
    return "high"


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(request: TransactionRequest):
    """
    Predict fraud probability for a transaction.
    Set model_version='auto' for A/B routing between v1 and v2.
    """
    features = np.array([[getattr(request, f) for f in FEATURE_ORDER]])

    t0 = time.perf_counter()
    try:
        fraud_prob, version_used = registry.predict(features, version=request.model_version)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    latency_ms = (time.perf_counter() - t0) * 1000

    is_fraud = fraud_prob >= FRAUD_THRESHOLD
    metrics.record(latency_ms, is_fraud, version_used)

    return PredictionResponse(
        fraud_probability=round(fraud_prob, 4),
        is_fraud=is_fraud,
        risk_level=risk_level(fraud_prob),
        model_version_used=version_used,
        latency_ms=round(latency_ms, 3),
        confidence=round(max(fraud_prob, 1 - fraud_prob), 4),
    )


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    """Service health check — call this from your load balancer."""
    s = metrics.summary()
    return HealthResponse(
        status="ok",
        models_loaded=registry.loaded_versions(),
        uptime_seconds=s["uptime_seconds"],
        total_predictions=s["total_predictions"],
    )


@app.get("/metrics", response_class=PlainTextResponse, tags=["ops"])
async def prometheus_metrics():
    """Prometheus scrape endpoint. Point Prometheus at /metrics."""
    return PlainTextResponse(metrics.prometheus_text(), media_type="text/plain")


@app.get("/metrics/summary", response_model=MetricsSummary, tags=["ops"])
async def metrics_summary():
    """JSON metrics summary — used by the Streamlit dashboard."""
    return MetricsSummary(**{k: v for k, v in metrics.summary().items()
                             if k in MetricsSummary.model_fields})

"""
Tests for the ML inference API.
Run: pytest tests/ -v
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Mock the model registry before importing app
with patch("app.model_registry.ModelRegistry._load_models"):
    from app.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "amount": 50.0,
    "hour_of_day": 14,
    "txn_last_24h": 2,
    "velocity_score": 0.1,
    "location_match": 0.95,
    "credit_score": 720,
    "account_age_years": 5.0,
    "foreign_ratio": 0.05,
    "model_version": "auto"
}

FRAUD_PAYLOAD = {
    "amount": 500.0,
    "hour_of_day": 2,
    "txn_last_24h": 15,
    "velocity_score": 0.9,
    "location_match": 0.1,
    "credit_score": 500,
    "account_age_years": 0.2,
    "foreign_ratio": 0.8,
    "model_version": "v2"
}


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "uptime_seconds" in data
        assert "total_predictions" in data

    def test_health_has_models_loaded(self):
        response = client.get("/health")
        data = response.json()
        assert isinstance(data["models_loaded"], dict)


class TestPredictEndpoint:
    def test_predict_returns_200(self):
        with patch("app.main.registry.predict", return_value=(0.05, "v1")):
            response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 200

    def test_predict_response_schema(self):
        with patch("app.main.registry.predict", return_value=(0.05, "v1")):
            response = client.post("/predict", json=VALID_PAYLOAD)
        data = response.json()
        required_fields = [
            "fraud_probability", "is_fraud", "risk_level",
            "model_version_used", "latency_ms", "confidence"
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_low_prob_returns_legit(self):
        with patch("app.main.registry.predict", return_value=(0.05, "v1")):
            response = client.post("/predict", json=VALID_PAYLOAD)
        data = response.json()
        assert data["is_fraud"] is False
        assert data["risk_level"] == "low"

    def test_high_prob_returns_fraud(self):
        with patch("app.main.registry.predict", return_value=(0.95, "v2")):
            response = client.post("/predict", json=FRAUD_PAYLOAD)
        data = response.json()
        assert data["is_fraud"] is True
        assert data["risk_level"] == "high"

    def test_latency_is_positive(self):
        with patch("app.main.registry.predict", return_value=(0.3, "v1")):
            response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.json()["latency_ms"] > 0

    def test_fraud_prob_in_range(self):
        with patch("app.main.registry.predict", return_value=(0.72, "v2")):
            response = client.post("/predict", json=VALID_PAYLOAD)
        prob = response.json()["fraud_probability"]
        assert 0.0 <= prob <= 1.0

    def test_invalid_credit_score_rejected(self):
        bad_payload = {**VALID_PAYLOAD, "credit_score": 100}  # below 300
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422

    def test_invalid_amount_rejected(self):
        bad_payload = {**VALID_PAYLOAD, "amount": -50}
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422

    def test_missing_field_rejected(self):
        incomplete = {"amount": 100.0, "hour_of_day": 12}
        response = client.post("/predict", json=incomplete)
        assert response.status_code == 422

    def test_model_version_v1_explicit(self):
        with patch("app.main.registry.predict", return_value=(0.1, "v1")) as mock:
            client.post("/predict", json={**VALID_PAYLOAD, "model_version": "v1"})
            mock.assert_called_once()
            _, kwargs = mock.call_args
            assert kwargs.get("version") == "v1" or mock.call_args[0][1] == "v1"


class TestMetricsEndpoint:
    def test_metrics_returns_text(self):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "ml_predictions_total" in response.text

    def test_metrics_summary_schema(self):
        response = client.get("/metrics/summary")
        assert response.status_code == 200
        data = response.json()
        assert "total_predictions" in data
        assert "fraud_rate" in data
        assert "p99_latency_ms" in data


class TestRiskLevels:
    @pytest.mark.parametrize("prob,expected_risk", [
        (0.1, "low"),
        (0.29, "low"),
        (0.3, "medium"),
        (0.59, "medium"),
        (0.6, "high"),
        (0.99, "high"),
    ])
    def test_risk_levels(self, prob, expected_risk):
        with patch("app.main.registry.predict", return_value=(prob, "v1")):
            response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.json()["risk_level"] == expected_risk

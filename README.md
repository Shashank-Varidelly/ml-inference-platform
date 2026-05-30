# ML Inference Platform

Real-time fraud detection API with A/B model routing, Prometheus metrics, and a live Streamlit dashboard.

Built as a resume project demonstrating production ML deployment patterns used at companies like Google Vertex AI and Meta's ML Platform.

---

## Architecture

```
Request → FastAPI → Model Registry → v1 (Logistic Regression)
                ↘                 ↘ v2 (Random Forest)      [50/50 A/B split]
                  Metrics Store → /metrics (Prometheus scrape)
                                → /metrics/summary (Dashboard)

Prometheus → Grafana (p99 latency, fraud rate, A/B split charts)
Streamlit  → Live dashboard (auto-refreshes every 3s)
```

**Stack:** FastAPI · scikit-learn · Docker · Prometheus · Grafana · Streamlit · Locust

---

## Quickstart (no Docker)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train both models (takes ~15 seconds)
python model/train.py

# 3. Start the API
uvicorn app.main:app --reload

# 4. Open the interactive docs
open http://localhost:8000/docs

# 5. (Optional) Start live dashboard
streamlit run monitoring/dashboard.py
```

---

## Quickstart (Docker)

```bash
# 1. Train models first (needs to run outside Docker)
pip install scikit-learn joblib numpy && python model/train.py

# 2. Start everything
docker-compose up --build

# API:        http://localhost:8000
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000  (admin / admin)
```

---

## API Reference

### `POST /predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 450.00,
    "hour_of_day": 2,
    "txn_last_24h": 14,
    "velocity_score": 0.88,
    "location_match": 0.12,
    "credit_score": 540,
    "account_age_years": 0.3,
    "foreign_ratio": 0.75,
    "model_version": "auto"
  }'
```

Response:
```json
{
  "fraud_probability": 0.9312,
  "is_fraud": true,
  "risk_level": "high",
  "model_version_used": "v2",
  "latency_ms": 4.23,
  "confidence": 0.9312
}
```

Set `model_version` to `"v1"`, `"v2"`, or `"auto"` (50/50 A/B routing).

### `GET /health`
Returns service status, loaded models, uptime, and total prediction count.

### `GET /metrics`
Prometheus-format metrics. Scrape endpoint for Grafana dashboards.

### `GET /metrics/summary`
JSON metrics: fraud rate, avg/p99 latency, per-model request counts.

---

## Features

- **A/B model routing** — traffic split between v1 (Logistic Regression) and v2 (Random Forest). Change `AB_V2_RATIO` in `app/model_registry.py`.
- **Prometheus metrics** — predictions, fraud rate, latency histograms, per-model counters.
- **Live Streamlit dashboard** — real-time charts, fraud alerts, manual prediction UI.
- **Input validation** — Pydantic schemas reject malformed requests before they hit the model.
- **Health check endpoint** — Docker-compatible, returns model load status.
- **Docker + docker-compose** — single command to start API + Prometheus + Grafana.
- **Load test suite** — Locust scripts for realistic traffic simulation.

---

## Benchmarks

Run the benchmark after starting the API:

```bash
python scripts/benchmark.py
```

Local results (MacBook M2, no Docker):
```
Successful:    200 / 200
Fraud flagged: 19 (9.5%)
Latency:
  avg:  3.41 ms
  p50:  3.12 ms
  p95:  6.88 ms
  p99:  9.23 ms
```

> Update this section with your own numbers after running the benchmark. Real measured numbers = credible resume bullets.

---

## Running Tests

```bash
pytest tests/ -v
```

Test coverage: schema validation, endpoint responses, risk level bucketing, error handling, A/B routing.

---

## Load Testing

```bash
# Install locust
pip install locust

# Run load test
locust -f scripts/load_test.py --host=http://localhost:8000

# Open http://localhost:8089
# Set: 100 users, ramp 10/sec, run for 2 minutes
```

---

## Project Structure

```
ml-inference-platform/
├── app/
│   ├── main.py            # FastAPI routes
│   ├── schemas.py         # Pydantic request/response models
│   ├── model_registry.py  # Model loading + A/B routing
│   └── metrics.py         # In-memory metrics store
├── model/
│   ├── train.py           # Training script (run once)
│   ├── model_v1.pkl       # Trained logistic regression
│   └── model_v2.pkl       # Trained random forest
├── monitoring/
│   ├── dashboard.py       # Streamlit live dashboard
│   └── prometheus.yml     # Prometheus scrape config
├── scripts/
│   ├── load_test.py       # Locust load test
│   └── benchmark.py       # Latency benchmark
├── tests/
│   └── test_api.py        # Pytest test suite
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Resume Bullet (fill in your real numbers)

> Engineered a containerized ML inference platform with A/B model routing, Prometheus monitoring, and p99 latency under **X ms** — serving **Y** req/min on free-tier cloud infrastructure

---

## Deploying to the Cloud (free)

**Render.com** (easiest):
1. Push to GitHub
2. New Web Service → connect repo
3. Build command: `pip install -r requirements.txt && python model/train.py`
4. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Free tier gives you a public URL to put on your resume

**Railway.app** (also free):
Same steps, slightly more generous free tier.

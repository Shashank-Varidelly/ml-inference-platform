"""
Quick latency benchmark — no Locust needed.
Run AFTER the API is up: python scripts/benchmark.py

Sends 200 requests and prints avg, p50, p95, p99 latency.
Copy these numbers into your README and resume bullet.
"""

import requests
import time
import numpy as np
import random

API_URL = "http://localhost:8000"
N_REQUESTS = 200

def make_payload():
    fraud = random.random() < 0.1
    if fraud:
        return {
            "amount": round(random.gauss(400, 100), 2),
            "hour_of_day": random.choice([1, 2, 3]),
            "txn_last_24h": random.randint(12, 20),
            "velocity_score": round(random.uniform(0.7, 0.95), 3),
            "location_match": round(random.uniform(0.05, 0.2), 3),
            "credit_score": random.randint(400, 560),
            "account_age_years": round(random.uniform(0.1, 0.8), 2),
            "foreign_ratio": round(random.uniform(0.5, 0.9), 3),
            "model_version": "auto"
        }
    return {
        "amount": round(random.gauss(55, 15), 2),
        "hour_of_day": random.randint(9, 18),
        "txn_last_24h": random.randint(1, 4),
        "velocity_score": round(random.uniform(0.05, 0.15), 3),
        "location_match": round(random.uniform(0.85, 1.0), 3),
        "credit_score": random.randint(680, 800),
        "account_age_years": round(random.uniform(3, 12), 1),
        "foreign_ratio": round(random.uniform(0.0, 0.08), 3),
        "model_version": "auto"
    }

print(f"Benchmarking {API_URL}/predict with {N_REQUESTS} requests...\n")

# Check API is up
try:
    requests.get(f"{API_URL}/health", timeout=3).raise_for_status()
except Exception as e:
    print(f"API not reachable: {e}\nStart with: uvicorn app.main:app --reload")
    exit(1)

latencies = []
errors = 0
fraud_count = 0

for i in range(N_REQUESTS):
    t0 = time.perf_counter()
    try:
        r = requests.post(f"{API_URL}/predict", json=make_payload(), timeout=5)
        r.raise_for_status()
        result = r.json()
        latencies.append((time.perf_counter() - t0) * 1000)
        if result["is_fraud"]:
            fraud_count += 1
    except Exception:
        errors += 1
    
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{N_REQUESTS} requests sent...")

lats = np.array(latencies)
print(f"\n{'='*40}")
print(f"  Results ({N_REQUESTS} requests)")
print(f"{'='*40}")
print(f"  Successful:    {len(latencies)}")
print(f"  Errors:        {errors}")
print(f"  Fraud flagged: {fraud_count} ({fraud_count/len(latencies)*100:.1f}%)")
print(f"\n  Latency (end-to-end, ms):")
print(f"    avg:  {np.mean(lats):.2f} ms")
print(f"    p50:  {np.percentile(lats, 50):.2f} ms")
print(f"    p95:  {np.percentile(lats, 95):.2f} ms")
print(f"    p99:  {np.percentile(lats, 99):.2f} ms")
print(f"    max:  {np.max(lats):.2f} ms")
print(f"\nPaste your p99 into your resume bullet!")

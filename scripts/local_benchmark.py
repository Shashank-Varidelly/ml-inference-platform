"""
Local benchmark — measures pure inference latency with no network.
Run from inside the project folder: python scripts/local_benchmark.py
"""
import sys
import os
sys.path.insert(0, os.getcwd())

import time
import numpy as np
import random

from app.model_registry import registry

N = 500
WARMUP = 20  # warmup calls to avoid cold-start skew

def make_features(fraud=False):
    if fraud:
        return np.array([[
            random.gauss(400, 100), random.choice([1,2,3]),
            random.randint(12,20), random.uniform(0.7,0.95),
            random.uniform(0.05,0.2), random.randint(400,560),
            random.uniform(0.1,0.8), random.uniform(0.5,0.9)
        ]])
    return np.array([[
        random.gauss(55,15), random.randint(9,18),
        random.randint(1,4), random.uniform(0.05,0.15),
        random.uniform(0.85,1.0), random.randint(680,800),
        random.uniform(3,12), random.uniform(0.0,0.08)
    ]])

# Warmup — loads model into memory, not measured
print(f"Warming up ({WARMUP} calls)...")
for _ in range(WARMUP):
    registry.predict(make_features(), version="auto")

# Now measure
print(f"Running {N} direct inference calls (no network)...\n")

latencies = []
fraud_count = 0

for i in range(N):
    is_fraud = random.random() < 0.1
    features = make_features(is_fraud)
    t0 = time.perf_counter()
    prob, version = registry.predict(features, version="auto")
    latencies.append((time.perf_counter() - t0) * 1000)
    if prob >= 0.5:
        fraud_count += 1

lats = np.array(latencies)
print(f"{'='*45}")
print(f"  Pure inference latency ({N} requests)")
print(f"{'='*45}")
print(f"  Fraud flagged: {fraud_count} ({fraud_count/N*100:.1f}%)")
print(f"  avg:  {np.mean(lats):.3f} ms")
print(f"  p50:  {np.percentile(lats,50):.3f} ms")
print(f"  p95:  {np.percentile(lats,95):.3f} ms")
print(f"  p99:  {np.percentile(lats,99):.3f} ms")
print(f"  max:  {np.max(lats):.3f} ms")
print(f"\n  Resume bullet:")
print(f"  p99 latency: {np.percentile(lats,99):.1f} ms")

"""
Lightweight in-memory metrics store.
Prometheus scrapes /metrics, Streamlit dashboard reads /metrics/summary.
"""

import time
import numpy as np
from collections import deque
from threading import Lock
from dataclasses import dataclass, field
from typing import Deque

@dataclass
class MetricsStore:
    _lock: Lock = field(default_factory=Lock)
    start_time: float = field(default_factory=time.time)

    # Counters
    total_predictions: int = 0
    fraud_detected: int = 0
    v1_requests: int = 0
    v2_requests: int = 0

    # Rolling window of latencies (last 1000 requests)
    latencies_ms: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))

    def record(self, latency_ms: float, is_fraud: bool, model_version: str):
        with self._lock:
            self.total_predictions += 1
            self.latencies_ms.append(latency_ms)
            if is_fraud:
                self.fraud_detected += 1
            if model_version == "v1":
                self.v1_requests += 1
            else:
                self.v2_requests += 1

    def summary(self) -> dict:
        with self._lock:
            lats = list(self.latencies_ms)
            return {
                "total_predictions": self.total_predictions,
                "fraud_detected": self.fraud_detected,
                "fraud_rate": self.fraud_detected / max(self.total_predictions, 1),
                "avg_latency_ms": round(float(np.mean(lats)), 2) if lats else 0.0,
                "p99_latency_ms": round(float(np.percentile(lats, 99)), 2) if len(lats) >= 10 else 0.0,
                "v1_requests": self.v1_requests,
                "v2_requests": self.v2_requests,
                "uptime_seconds": round(time.time() - self.start_time, 1),
            }

    def prometheus_text(self) -> str:
        s = self.summary()
        lines = [
            "# HELP ml_predictions_total Total number of predictions served",
            "# TYPE ml_predictions_total counter",
            f"ml_predictions_total {s['total_predictions']}",
            "",
            "# HELP ml_fraud_detected_total Total fraud predictions",
            "# TYPE ml_fraud_detected_total counter",
            f"ml_fraud_detected_total {s['fraud_detected']}",
            "",
            "# HELP ml_latency_avg_ms Average inference latency in ms",
            "# TYPE ml_latency_avg_ms gauge",
            f"ml_latency_avg_ms {s['avg_latency_ms']}",
            "",
            "# HELP ml_latency_p99_ms P99 inference latency in ms",
            "# TYPE ml_latency_p99_ms gauge",
            f"ml_latency_p99_ms {s['p99_latency_ms']}",
            "",
            "# HELP ml_model_requests_total Requests per model version",
            "# TYPE ml_model_requests_total counter",
            f'ml_model_requests_total{{version="v1"}} {s["v1_requests"]}',
            f'ml_model_requests_total{{version="v2"}} {s["v2_requests"]}',
        ]
        return "\n".join(lines)


# Singleton
metrics = MetricsStore()

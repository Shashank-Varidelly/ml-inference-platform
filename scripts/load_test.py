"""
Load test the inference API with Locust.

Install: pip install locust
Run:     locust -f scripts/load_test.py --host=http://localhost:8000

Then open http://localhost:8089 to control the test.
Target: 100 users, ramp 10/sec → aim for p99 < 100ms
"""

import random
from locust import HttpUser, task, between

def random_legit_transaction():
    return {
        "amount": round(random.gauss(50, 20), 2),
        "hour_of_day": random.randint(8, 20),
        "txn_last_24h": random.randint(1, 5),
        "velocity_score": round(random.uniform(0.05, 0.2), 3),
        "location_match": round(random.uniform(0.8, 1.0), 3),
        "credit_score": random.randint(680, 800),
        "account_age_years": round(random.uniform(2, 15), 1),
        "foreign_ratio": round(random.uniform(0.0, 0.1), 3),
        "model_version": "auto"
    }

def random_fraud_transaction():
    return {
        "amount": round(random.gauss(400, 150), 2),
        "hour_of_day": random.choice([0, 1, 2, 3, 22, 23]),
        "txn_last_24h": random.randint(10, 25),
        "velocity_score": round(random.uniform(0.7, 1.0), 3),
        "location_match": round(random.uniform(0.0, 0.3), 3),
        "credit_score": random.randint(300, 550),
        "account_age_years": round(random.uniform(0.0, 1.0), 2),
        "foreign_ratio": round(random.uniform(0.5, 1.0), 3),
        "model_version": "auto"
    }


class FraudDetectionUser(HttpUser):
    wait_time = between(0.1, 0.5)  # 2-10 req/sec per user

    @task(9)
    def predict_legit(self):
        """90% legitimate transactions."""
        self.client.post(
            "/predict",
            json=random_legit_transaction(),
            name="/predict [legit]"
        )

    @task(1)
    def predict_fraud(self):
        """10% fraudulent transactions."""
        self.client.post(
            "/predict",
            json=random_fraud_transaction(),
            name="/predict [fraud]"
        )

    @task(1)
    def health_check(self):
        self.client.get("/health", name="/health")

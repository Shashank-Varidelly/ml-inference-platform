"""
Train two fraud detection models (v1 and v2) on synthetic credit card data.
Run this once before starting the API: python model/train.py
"""

import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

RANDOM_SEED = 42
N_SAMPLES = 10000
FRAUD_RATE = 0.02  # 2% fraud — realistic for credit card data

def generate_dataset(n_samples=N_SAMPLES, fraud_rate=FRAUD_RATE, seed=RANDOM_SEED):
    """Generate synthetic credit card transaction data."""
    rng = np.random.RandomState(seed)
    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    # Legitimate transactions
    legit = np.column_stack([
        rng.normal(50, 20, n_legit),       # amount (mean $50)
        rng.normal(12, 4, n_legit),         # hour of day
        rng.normal(3, 1, n_legit),          # transactions last 24h
        rng.normal(0.1, 0.05, n_legit),     # velocity score
        rng.normal(0.9, 0.05, n_legit),     # location match score
        rng.normal(720, 50, n_legit),       # credit score
        rng.normal(5, 2, n_legit),          # account age (years)
        rng.normal(0.05, 0.02, n_legit),    # foreign transaction ratio
    ])

    # Fraudulent transactions — different distribution
    fraud = np.column_stack([
        rng.normal(300, 100, n_fraud),      # higher amounts
        rng.uniform(0, 24, n_fraud),        # random hour (odd hours)
        rng.normal(15, 5, n_fraud),         # high velocity
        rng.normal(0.8, 0.1, n_fraud),      # high velocity score
        rng.normal(0.2, 0.1, n_fraud),      # low location match
        rng.normal(580, 80, n_fraud),       # lower credit score
        rng.normal(0.5, 0.3, n_fraud),      # newer accounts
        rng.normal(0.6, 0.2, n_fraud),      # more foreign transactions
    ])

    X = np.vstack([legit, fraud])
    y = np.array([0] * n_legit + [1] * n_fraud)

    # Shuffle
    idx = rng.permutation(len(y))
    return X[idx], y[idx]

FEATURE_NAMES = [
    "amount", "hour_of_day", "txn_last_24h",
    "velocity_score", "location_match", "credit_score",
    "account_age_years", "foreign_ratio"
]

def train_model_v1(X_train, y_train):
    """v1: Logistic Regression — fast, interpretable baseline."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_SEED
        ))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def train_model_v2(X_train, y_train):
    """v2: Random Forest — higher accuracy, used for A/B comparison."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            max_depth=8,
            random_state=RANDOM_SEED,
            n_jobs=-1
        ))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"{'='*40}")
    print(classification_report(y_test, y_pred, target_names=["legit", "fraud"]))
    print(f"  ROC-AUC: {auc:.4f}")
    return auc

if __name__ == "__main__":
    print("Generating dataset...")
    X, y = generate_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    print(f"Fraud rate in train: {y_train.mean():.2%}")

    print("\nTraining v1 (Logistic Regression)...")
    model_v1 = train_model_v1(X_train, y_train)
    auc_v1 = evaluate(model_v1, X_test, y_test, "Model v1 — Logistic Regression")

    print("\nTraining v2 (Random Forest)...")
    model_v2 = train_model_v2(X_train, y_train)
    auc_v2 = evaluate(model_v2, X_test, y_test, "Model v2 — Random Forest")

    os.makedirs("model", exist_ok=True)
    joblib.dump(model_v1, "model/model_v1.pkl")
    joblib.dump(model_v2, "model/model_v2.pkl")

    print(f"\nSaved model_v1.pkl (AUC={auc_v1:.4f}) and model_v2.pkl (AUC={auc_v2:.4f})")
    print("Run: uvicorn app.main:app --reload")

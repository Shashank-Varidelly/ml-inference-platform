"""
Live monitoring dashboard — run alongside the API:
  streamlit run monitoring/dashboard.py

Shows real-time metrics: predictions/sec, fraud rate, latency, A/B split.
Auto-refreshes every 3 seconds.
"""

import time
import requests
import streamlit as st
import pandas as pd
import numpy as np

API_URL = "http://localhost:8000"
REFRESH_INTERVAL = 3  # seconds

st.set_page_config(
    page_title="ML Inference Monitor",
    page_icon="🔍",
    layout="wide"
)

st.title("ML Inference Platform — Live Dashboard")
st.caption(f"Auto-refreshes every {REFRESH_INTERVAL}s · API: {API_URL}")

# ── Fetch metrics ──────────────────────────────────────────────
def fetch_metrics():
    try:
        r = requests.get(f"{API_URL}/metrics/summary", timeout=2)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return None

def fetch_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        r.raise_for_status()
        return r.json()
    except:
        return None

# ── State: rolling history ─────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

metrics = fetch_metrics()
health = fetch_health()

if metrics is None:
    st.error("Cannot reach API at http://localhost:8000. Make sure it's running: uvicorn app.main:app --reload")
    st.stop()

# Store point in history
st.session_state.history.append({
    "time": time.strftime("%H:%M:%S"),
    "total": metrics["total_predictions"],
    "fraud_rate": round(metrics["fraud_rate"] * 100, 2),
    "avg_latency": metrics["avg_latency_ms"],
    "p99_latency": metrics["p99_latency_ms"],
})
if len(st.session_state.history) > 60:
    st.session_state.history.pop(0)

# ── Status banner ──────────────────────────────────────────────
if health:
    models = health.get("models_loaded", {})
    model_str = " | ".join([f"{k}: {v}" for k, v in models.items()])
    st.success(f"API healthy · Models: {model_str} · Uptime: {health['uptime_seconds']}s")

# ── Top metrics ────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total predictions", f"{metrics['total_predictions']:,}")
col2.metric("Fraud detected", f"{metrics['fraud_detected']:,}")
col3.metric("Fraud rate", f"{metrics['fraud_rate']*100:.1f}%")
col4.metric("Avg latency", f"{metrics['avg_latency_ms']:.1f} ms")
col5.metric("P99 latency", f"{metrics['p99_latency_ms']:.1f} ms")

st.divider()

# ── Charts ─────────────────────────────────────────────────────
hist_df = pd.DataFrame(st.session_state.history)

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Fraud rate over time (%)")
    if len(hist_df) > 1:
        st.line_chart(hist_df.set_index("time")["fraud_rate"], use_container_width=True)
    else:
        st.info("Send some requests to see the chart fill in...")

with col_right:
    st.subheader("Latency over time (ms)")
    if len(hist_df) > 1:
        st.line_chart(
            hist_df.set_index("time")[["avg_latency", "p99_latency"]],
            use_container_width=True
        )
    else:
        st.info("Send some requests to see the chart fill in...")

# ── A/B model split ────────────────────────────────────────────
st.subheader("A/B model traffic split")
v1 = metrics["v1_requests"]
v2 = metrics["v2_requests"]
total = v1 + v2
if total > 0:
    ab_df = pd.DataFrame({
        "Model": ["v1 (Logistic Regression)", "v2 (Random Forest)"],
        "Requests": [v1, v2],
        "Share (%)": [round(v1/total*100, 1), round(v2/total*100, 1)]
    })
    st.dataframe(ab_df, use_container_width=True, hide_index=True)
    st.progress(v2 / total if total > 0 else 0, text=f"v2 traffic: {v2/total*100:.0f}%")
else:
    st.info("No requests yet. Use the load test script or /docs to send predictions.")

# ── Manual predict ─────────────────────────────────────────────
st.divider()
st.subheader("Try a prediction")
with st.expander("Send a test transaction"):
    c1, c2, c3, c4 = st.columns(4)
    amount       = c1.number_input("Amount ($)", value=250.0, min_value=1.0)
    hour         = c2.slider("Hour of day", 0, 23, 2)
    txn_24h      = c3.number_input("Txns last 24h", value=12, min_value=0)
    velocity     = c4.slider("Velocity score", 0.0, 1.0, 0.85)
    c5, c6, c7, c8 = st.columns(4)
    location     = c5.slider("Location match", 0.0, 1.0, 0.15)
    credit_score = c6.number_input("Credit score", value=590, min_value=300, max_value=850)
    acct_age     = c7.number_input("Account age (yrs)", value=0.3, min_value=0.0)
    foreign      = c8.slider("Foreign ratio", 0.0, 1.0, 0.70)
    model_ver    = st.selectbox("Model version", ["auto", "v1", "v2"])

    if st.button("Predict"):
        payload = {
            "amount": amount, "hour_of_day": hour, "txn_last_24h": txn_24h,
            "velocity_score": velocity, "location_match": location,
            "credit_score": credit_score, "account_age_years": acct_age,
            "foreign_ratio": foreign, "model_version": model_ver
        }
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
            result = resp.json()
            risk_colors = {"low": "green", "medium": "orange", "high": "red"}
            color = risk_colors.get(result.get("risk_level", "low"), "gray")
            st.markdown(f"""
            **Fraud probability:** `{result['fraud_probability']:.2%}`  
            **Decision:** {'🚨 FRAUD' if result['is_fraud'] else '✅ Legitimate'}  
            **Risk level:** :{color}[{result['risk_level'].upper()}]  
            **Model used:** `{result['model_version_used']}`  
            **Latency:** `{result['latency_ms']:.2f} ms`
            """)
        except Exception as e:
            st.error(f"Request failed: {e}")

# ── Auto refresh ───────────────────────────────────────────────
time.sleep(REFRESH_INTERVAL)
st.rerun()

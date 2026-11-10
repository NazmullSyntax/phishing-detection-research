"""
Streamlit UI v2: shows prediction, confidence, risk level, threshold, feature explanation.
"""
import requests
import streamlit as st
import pandas as pd
from datetime import datetime

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Phishing URL Detector v2",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Phishing URL Detector")
st.markdown("Analyze a URL using a **robust XGBoost model** with tuned threshold.")

# Sidebar
with st.sidebar:
    st.header("Model Info")
    try:
        health = requests.get(f"{API_URL}/health", timeout=2).json()
        if health.get("model_loaded"):
            st.success(f"API healthy")
            st.json({
                "model": health.get("model_version"),
                "threshold": health.get("threshold"),
            })
        else:
            st.error("API unhealthy")
    except Exception:
        st.error("API unreachable")

    st.divider()
    st.caption(
        "This tool is for **research/educational purposes** only. "
        "Model confidence is not a security guarantee."
    )
    st.caption(f"Session: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# Session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# Input
st.subheader("Enter URL to Analyze")
url = st.text_input(
    "URL",
    placeholder="e.g., https://accounts.google.com or http://paypal.verify-login.xyz",
    label_visibility="collapsed",
)

col1, col2 = st.columns([1, 4])
with col1:
    analyze = st.button("🔍 Analyze", type="primary", use_container_width=True)

if analyze and url.strip():
    with st.spinner("Analyzing..."):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json={"url": url.strip()},
                timeout=5,
            )

            if response.status_code == 200:
                result = response.json()

                # Add to history
                st.session_state.history.insert(0, {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "url": url.strip()[:60],
                    "prediction": result["prediction"],
                    "confidence": f"{result['confidence']*100:.1f}%",
                    "risk": result["risk_level"],
                })
                st.session_state.history = st.session_state.history[:10]

                # Result display
                st.divider()
                st.subheader("Result")

                col_a, col_b = st.columns([1, 1])
                with col_a:
                    if result["prediction"] == "phishing":
                        st.error(f"### 🚨 PHISHING")
                    elif result["prediction"] == "uncertain":
                        st.warning(f"### ⚠️ UNCERTAIN")
                    else:
                        st.success(f"### ✅ LEGITIMATE")

                    st.caption(f"Risk Level: **{result['risk_level'].upper()}**")

                with col_b:
                    st.metric("Confidence", f"{result['confidence']*100:.1f}%")

                # Probabilities
                st.markdown("**Probability Breakdown**")
                col_p, col_l = st.columns(2)
                with col_p:
                    st.metric("Phishing", f"{result['phishing_probability']*100:.1f}%")
                with col_l:
                    st.metric("Legitimate", f"{(1 - result['phishing_probability'])*100:.1f}%")

                st.progress(result["phishing_probability"])

                # Details
                st.divider()
                st.subheader("Details")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Latency", f"{result['latency_ms']:.2f} ms")
                with col2:
                    thresh_display = f"{result['threshold_used']*100:.0f}%" if result['threshold_used'] else "N/A"
                    st.metric("Threshold (Alert if >)", thresh_display)
                with col3:
                    st.metric("Model", result["model_version"].split("+")[0])
                with col4:
                    st.metric("URL Length", len(result["url"]))

                # Warnings
                if result["prediction"] == "uncertain":
                    st.info(
                        "⚠️ **Uncertain prediction** — the model's confidence is in the "
                        "borderline range. Treat this as low-confidence; manually verify."
                    )
                if "whitelist" in result["model_version"]:
                    st.info("✅ This domain is on the **trusted list**. ML prediction was skipped.")

            elif response.status_code == 400:
                st.error(f"❌ Invalid input: {response.json().get('detail', 'Unknown error')}")
            else:
                st.error(f"❌ API error (HTTP {response.status_code})")

        except requests.exceptions.RequestException as e:
            st.error(f"❌ Could not reach API: {e}")
            st.info("Make sure FastAPI is running at `http://127.0.0.1:8000`")

elif analyze:
    st.warning("Please enter a URL first.")

# History
if st.session_state.history:
    st.divider()
    st.subheader("Recent Analyses")
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()

# Footer
st.divider()
st.caption(
    "⚠️ **Disclaimer:** For research/educational use only. "
    "Model confidence is not a guarantee of URL safety."
)
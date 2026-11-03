"""
Day 9: Streamlit UI for phishing URL detection.
Calls FastAPI backend at http://127.0.0.1:8000/predict.
"""
import time
import requests
import streamlit as st
import pandas as pd
from datetime import datetime

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Header ----------
st.title("🛡️ Phishing URL Detector")
st.markdown(
    "Analyze a URL using a **robust XGBoost model** trained against adversarial attacks."
)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("About")
    st.markdown(
        """
        This application is part of a research project:

        **Adversarially Robust and Deployment-Ready Phishing URL Detection**

        **Model:** XGBoost (adversarial training)
        **Clean Test Accuracy:** 96.78%
        **Subdomain Attack Accuracy:** 96.56%

        **API Endpoint:** `POST /predict`
        """
    )
    st.divider()
    st.caption(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Check API health
    try:
        health = requests.get(f"{API_URL}/health", timeout=2).json()
        if health.get("model_loaded"):
            st.success("✅ API healthy")
        else:
            st.error("❌ API unhealthy")
    except Exception:
        st.error("❌ API unreachable")

# ---------- Session state ----------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------- Input ----------
st.subheader("Enter URL to Analyze")
url = st.text_input(
    "URL",
    placeholder="e.g., https://www.google.com or http://paypal.verify-login.xyz/account",
    label_visibility="collapsed",
)

col1, col2 = st.columns([1, 3])
with col1:
    analyze = st.button("🔍 Analyze", type="primary", use_container_width=True)

# ---------- Analyze ----------
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
                    "url": url.strip(),
                    "prediction": result["prediction"],
                    "confidence": result["confidence"],
                })
                st.session_state.history = st.session_state.history[:10]

                # ---------- Result card ----------
                st.divider()
                st.subheader("Result")

                col_a, col_b = st.columns([1, 1])
                with col_a:
                    if result["prediction"] == "phishing":
                        st.error(f"### 🚨 {result['prediction'].upper()}")
                    else:
                        st.success(f"### ✅ {result['prediction'].upper()}")

                with col_b:
                    st.metric(
                        "Confidence",
                        f"{result['confidence']*100:.1f}%",
                        help="Model's confidence in its prediction (not a security guarantee)"
                    )

                # ---------- Probability bar ----------
                st.markdown("**Probability Breakdown**")
                col_p, col_l = st.columns(2)
                with col_p:
                    st.metric("Phishing", f"{result['phishing_probability']*100:.1f}%")
                with col_l:
                    st.metric("Legitimate", f"{(1 - result['phishing_probability'])*100:.1f}%")

                st.progress(result["phishing_probability"])

                # ---------- Details ----------
                st.divider()
                st.subheader("Details")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Latency", f"{result['latency_ms']:.2f} ms")
                with col2:
                    st.metric("Model", result["model_version"])
                with col3:
                    st.metric("URL Length", len(result["url"]))

                # ---------- Warnings ----------
                if result["confidence"] < 0.7:
                    st.warning(
                        "⚠️ Low confidence — treat this prediction with caution. "
                        "Model confidence is not a guarantee of security."
                    )

                if len(url) < 30:
                    st.info(
                        "ℹ️ Short URLs (e.g., shortened URLs) are harder to classify. "
                        "See the project thesis (Section 5) for details."
                    )

            elif response.status_code == 400:
                st.error(f"❌ Invalid input: {response.json().get('detail', 'Unknown error')}")
            else:
                st.error(f"❌ API error (HTTP {response.status_code})")

        except requests.exceptions.RequestException as e:
            st.error(f"❌ Could not reach API: {e}")
            st.info("💡 Make sure the FastAPI server is running at `http://127.0.0.1:8000`")

elif analyze:
    st.warning("Please enter a URL first.")

# ---------- History ----------
if st.session_state.history:
    st.divider()
    st.subheader("Recent Analyses")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True,
    )
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()

# ---------- Footer ----------
st.divider()
st.caption(
    "⚠️ **Disclaimer:** This tool is for research and educational purposes only. "
    "Model confidence is not a guarantee of URL safety. Always verify URLs through "
    "official channels before interacting with them."
)
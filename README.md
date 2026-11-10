# 🛡️ Adversarially Robust and Deployment-Ready Phishing URL Detection

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/license-Academic-lightgrey.svg)](LICENSE)

A complete research project investigating **adversarial robustness** and **deployment readiness** of ML-based phishing URL detectors.

---

## 📖 Overview

Phishing is one of the most prevalent cybersecurity threats. ML models achieve 95-99% accuracy on clean benchmarks, but **do they actually work under attack? And can they be deployed in real-time?**

This thesis answers both questions with:
- **Empirical adversarial robustness study** (5 attack types)
- **Adversarial training mitigation** (+49.81% recovery)
- **Production deployment evaluation** (5.3ms latency, 280 req/sec)

---

## 🎯 Key Findings

| Metric | Value |
|--------|-------|
| Dataset | 253,098 URLs (Kaggle) |
| Balanced sample | 50,000 (25K phishing + 25K legitimate) |
| Features | 26 lexical features |
| Best model | XGBoost (97.40% accuracy, F1=0.9719) |
| **Subdomain attack** | **47.23%** (-49.81%) 🚨 |
| **Adversarial training recovery** | **+49.81%** 🏆 |
| Median API latency | **5.30 ms** |
| Peak throughput | **279.8 req/sec** |
| Error rate under load | **0.00%** |

**🚨 Core insight:** ML phishing detectors achieve 97% clean accuracy but collapse to **~47% under subdomain manipulation**. Adversarial training recovers subdomain accuracy to **96.56%** with negligible trade-off.

---

## 📚 Research Questions

- **RQ1:** How accurately can ML models classify phishing vs legitimate URLs?
- **RQ2:** How much does URL manipulation reduce detection performance?
- **RQ3:** Which robustness techniques improve detection against manipulated URLs?
- **RQ4:** Can the selected model operate efficiently through a real-time API?
- **RQ5:** What trade-offs exist between accuracy, robustness, and efficiency?

---

## 🏗️ Project Structure

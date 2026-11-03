# Adversarially Robust and Deployment-Ready Phishing URL Detection

A research project investigating (1) how much ML-based phishing URL detectors degrade under adversarial URL manipulation, and (2) how selected models perform when deployed as a real-time API.

## 🎯 Research Questions

- **RQ1:** How accurately can ML models classify phishing vs legitimate URLs?
- **RQ2:** How much does URL manipulation reduce detection performance?
- **RQ3:** Which robustness techniques improve detection against manipulated URLs?
- **RQ4:** Can the selected model operate efficiently through a real-time API?
- **RQ5:** What trade-offs exist between accuracy, robustness, and computational efficiency?

## 🧩 Project Structure

    research/       - Research proposal, literature review, gap analysis
    data/           - Raw and processed datasets
    notebooks/      - Jupyter notebooks for exploration
    src/            - Production source code
    api/            - FastAPI backend
    app/            - Streamlit frontend
    tests/          - Automated tests
    experiments/    - Experiment logs and configs
    results/        - Figures and tables
    thesis/         - Thesis chapters

## ⚙️ Setup

    python -m venv venv
    .\venv\Scripts\Activate.ps1
    pip install -r requirements.txt

## 📊 Status

- [x] Day 1: Research foundation + environment setup
- [ ] Day 2: Dataset collection
- [ ] Day 3: Data cleaning + EDA
- [ ] Day 4: Feature engineering
- [ ] Day 5: Model development
- [ ] Day 6: Adversarial attacks
- [ ] Day 7: Robustness improvement
- [ ] Day 8: FastAPI backend
- [ ] Day 9: Streamlit frontend
- [ ] Day 10: Deployment testing
- [ ] Day 11: Thesis writing
- [ ] Day 12: Final polish

## 📄 License

For academic and educational use.
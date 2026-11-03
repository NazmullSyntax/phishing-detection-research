# Research Proposal

## Title
Adversarially Robust and Deployment-Ready Phishing URL Detection Using Machine Learning:
A Study of URL Manipulation and Real-World Performance

## 1. Background

Phishing is one of the most common cyberattacks. Attackers create URLs that mimic legitimate sites (banks, e-commerce, government) and trick users into revealing credentials or downloading malware. Modern phishing URLs use:
- Character substitution (paypal.com -> paypa1.com)
- Unicode homographs (Cyrillic 'а' instead of Latin 'a')
- Misleading subdomains (apple.com.verify-login.xyz)
- URL encoding and obfuscation
- URL shorteners

Machine learning (ML) methods achieve high accuracy on clean datasets but may fail when attackers manipulate URLs. Furthermore, models that perform well in Jupyter notebooks may not meet latency, throughput, and resource constraints in real deployment.

## 2. Research Problems

### Problem 1: Adversarial Robustness
Can ML-based phishing URL detection maintain reliable performance when attackers manipulate URLs?

### Problem 2: Real-World Deployment Testing
Can the selected model operate efficiently in a practical application considering prediction latency, throughput, CPU consumption, memory consumption, and classification performance?

## 3. Research Objectives
1. Develop and compare multiple ML models for phishing URL detection.
2. Evaluate model vulnerability against different adversarial URL manipulations.
3. Investigate and implement robustness improvement methods.
4. Develop a deployable phishing detection application.
5. Evaluate the application under realistic testing conditions.
6. Compare the baseline and robust models.
7. Publish all code, experiments, and findings on GitHub.

## 4. Research Questions
- RQ1: Accuracy of ML models on phishing vs legit URL classification?
- RQ2: Performance degradation under URL manipulation?
- RQ3: Which robustness techniques improve detection?
- RQ4: Efficiency of the deployed API (latency, throughput, CPU, RAM)?
- RQ5: Trade-offs among accuracy, robustness, efficiency?

## 5. Scope

- **In scope:** URL-string-based features only (no live crawling of malicious sites).
- **Out of scope:** DNS/Domain-age features requiring live queries (kept as optional extension).
- **Datasets:** Publicly available (Kaggle, PhishTank, OpenPhish, Tranco).

## 6. Methodology Summary

1. Collect and prepare ~50,000 URLs (phishing + legitimate).
2. Prevent data leakage with stratified and domain-grouped splits.
3. Extract lexical URL features.
4. Train 5 ML models and select a baseline by a defined protocol.
5. Build adversarial attack framework (5 attack types).
6. Measure baseline degradation.
7. Apply adversarial training + augmentation + robust normalization.
8. Deploy via FastAPI + Streamlit.
9. Measure latency, throughput, CPU, RAM under load.
10. Compare baseline vs robust models across accuracy, robustness, and efficiency.

## 7. Expected Contributions

1. Empirical evidence of how much state-of-practice ML detectors degrade under URL manipulation.
2. Evidence of how much adversarial training recovers robustness.
3. Evidence of real-world deployability (latency, throughput, resource usage).
4. Reproducible open-source codebase.

## 8. Risks and Limitations

- Some datasets change over time (PhishTank URLs expire).
- Live crawling is avoided for safety; only URL-string features used.
- Local deployment testing does not equal production deployment.
- Attack coverage is limited to defined attack types.

## 9. Timeline (10 working days)

See README.md status checklist.
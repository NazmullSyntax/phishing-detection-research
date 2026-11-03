# Phishing Detection Research

A research project for studying and evaluating phishing detection methods.

## Project structure

- `research/` — proposal, literature review, and research gaps
- `data/` — raw and processed datasets
- `notebooks/` — exploratory analysis, experiments, and plots
- `src/` — reusable Python code for data, features, models, adversarial testing, and evaluation
- `api/` — FastAPI backend
- `app/` — Streamlit frontend
- `tests/` — automated tests
- `experiments/` — experiment configurations and logs
- `results/` — figures, tables, and reports
- `thesis/` — thesis chapters

## Getting started

Use Python 3.10 or newer. Create and activate a virtual environment, then install
the project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Keep sensitive or licensed datasets out of version control. Document dataset
sources and preprocessing steps before using data in experiments.

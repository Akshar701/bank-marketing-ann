# Contributing to Bank Marketing ANN & ML Project

Welcome to the **Bank Marketing Subscription Prediction** repository! We are excited to collaborate with you. This guide provides step-by-step instructions for teammates to set up the environment, run experiments, build features, and submit Pull Requests.

---

## 🧭 Table of Contents
1. [Collaboration Workflow](#1-collaboration-workflow)
2. [Local Environment Setup](#2-local-environment-setup)
3. [Running via Google Colab](#3-running-via-google-colab)
4. [Running the Application & Pipelines](#4-running-the-application--pipelines)
5. [Code Quality & Testing](#5-code-quality--testing)
6. [Pull Request (PR) Checklist](#6-pull-request-pr-checklist)

---

## 1. Collaboration Workflow

We follow standard GitHub Flow:
- **`main` Branch:** Protected, always stable, matches the report benchmarks.
- **Feature/Experiment Branches:** All work should be done in separate feature branches.

### Branch Naming Conventions
- `feature/<feature-name>`: New model architecture, new Streamlit feature, or data pipeline improvement.
  - *Example:* `feature/tabnet-model`, `feature/shap-waterfall`
- `exp/<experiment-name>`: Hyperparameter exploration or ablation study.
  - *Example:* `exp/focal-loss-tuning`, `exp/smote-comparison`
- `fix/<bug-name>`: Bug fixes or documentation corrections.
  - *Example:* `fix/threshold-rounding-issue`

### Recommended Git Commands
```bash
# 1. Fetch latest changes from main
git checkout main
git pull origin main

# 2. Create and switch to your feature branch
git checkout -b feature/my-new-feature

# 3. Commit your work with descriptive messages
git add .
git commit -m "feat: implement focal loss for class imbalance"

# 4. Push branch to GitHub
git push -u origin feature/my-new-feature

# 5. Open a Pull Request on GitHub for team review!
```

---

## 2. Local Environment Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### Quick Setup Steps
```bash
# 1. Clone the repository
git clone https://github.com/Akshar701/bank-marketing-ann.git
cd bank-marketing-ann

# 2. Create a virtual environment (venv or conda)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on Keras 3 Backend:**  
> This project uses Keras 3 with the PyTorch backend. The environment variable `KERAS_BACKEND=torch` is automatically configured in code, but you can also export it in your shell:
> ```bash
> export KERAS_BACKEND=torch
> ```

---

## 3. Running via Google Colab

If you do not want to set up Python locally, you can run everything in Google Colab with one click:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Akshar701/bank-marketing-ann/blob/main/notebooks/colab_quickstart.ipynb)

- **Quickstart Demo:** `notebooks/colab_quickstart.ipynb`
- **Full EDA & Benchmark Notebook:** `notebooks/eda_and_experiments.ipynb`

Colab automatically clones the repo and loads pretrained model weights.

---

## 4. Running the Application & Pipelines

### Launch the Streamlit Analytics Web Dashboard
```bash
streamlit run app.py
```
Access the web dashboard in your browser at `http://localhost:8501`.

### Run Model Training Pipeline
To retrain baseline, medium, advanced ANNs, and ML baselines:
```bash
python src/train.py
```

### Run Model Evaluation & Metric Generation
```bash
python src/evaluate.py
```

### Run Hyperparameter Tuning with Optuna
```bash
python src/tuning.py
```

### Run Model Explainability (SHAP & Permutation Importance)
```bash
python src/explainability.py
```

---

## 5. Code Quality & Testing

Before submitting a Pull Request, ensure that all automated tests pass:

```bash
# Run the test suite
pytest tests/ -v
```

Our tests verify:
- ✅ Data loading & schema compliance (16 attributes + binary target `y`).
- ✅ Leakage prevention (preprocessors fit strictly on train split).
- ✅ Model predictions & threshold application (single customer & batch CSV).

---

## 6. Pull Request (PR) Checklist

When opening a Pull Request:
1. Provide a concise summary of the change in the PR description.
2. Link any related GitHub issue if applicable.
3. Ensure no data leakage occurs (never fit scalers/encoders on validation/test data).
4. Run `pytest` locally to confirm all tests pass.
5. Request a review from at least one teammate before merging into `main`.

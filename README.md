# Bank Marketing Subscription Prediction: Deep Artificial Neural Networks & Classical ML Benchmark

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Keras 3](https://img.shields.io/badge/Keras-3.0+-red.svg)](https://keras.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-orange.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Akshar701/bank-marketing-ann/blob/main/notebooks/colab_quickstart.ipynb)
[![Tests Passing](https://img.shields.io/badge/pytest-14%20passed-green.svg)](https://pytest.org/)

A production-grade, end-to-end Machine Learning and Deep Learning system built on the official [UCI Bank Marketing Benchmark](https://archive.ics.uci.edu/dataset/222/bank+marketing) (`bank-full.csv`, 45,211 rows). This project rigorously benchmarks traditional machine learning algorithms against multiple Artificial Neural Network (ANN) architectures, investigates ex-post target leakage, addresses severe class imbalance, optimizes decision thresholds, evaluates probability calibration, provides model explainability (SHAP & Permutation Importance), and deploys an interactive financial analytics Streamlit dashboard.

---

## 🤝 Teammate Collaboration & Google Colab Quickstart

Welcome teammates! You can run and collaborate on this project in two ways:

### Option A: 1-Click Cloud Execution (Google Colab)
No local setup required. Click below to open interactive notebooks with pretrained models and live customer inference:
- **🚀 [Colab Quickstart Notebook](https://colab.research.google.com/github/Akshar701/bank-marketing-ann/blob/main/notebooks/colab_quickstart.ipynb):** 2-minute walkthrough of model predictions, leakage comparison, and benchmark figures.
- **📊 [Full EDA & Experimentation Notebook](https://colab.research.google.com/github/Akshar701/bank-marketing-ann/blob/main/notebooks/eda_and_experiments.ipynb):** Complete exploratory data analysis and architecture benchmarking.

### Option B: Local Setup & Dashboard
```bash
# 1. Clone repository
git clone https://github.com/Akshar701/bank-marketing-ann.git
cd bank-marketing-ann

# 2. Create virtual environment & install requirements
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Launch the Streamlit Web Application
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

> 💡 **Contributing:** See our [CONTRIBUTING.md](CONTRIBUTING.md) guide for branch naming conventions, PR guidelines, and running tests.


---

## Architecture & System Pipeline

```mermaid
flowchart TD
    subgraph Data Layer
        A[UCI bank-full.csv<br/>45,211 rows, ';' separated] --> B[Data Loader & Schema Validator]
        B --> C[Stratified Split<br/>70% Train | 15% Val | 15% Test]
    end

    subgraph Preprocessing Pipeline
        C --> D[BankPreprocessor<br/>Fitted strictly on Train partition]
        D --> E1[Numeric Scaler<br/>StandardScaler]
        D --> E2[Categorical Encoder<br/>OneHotEncoder handle_unknown]
        D --> E3[Class Weight Calculator<br/>W0=0.57, W1=4.27]
        E1 --> F[51-Dim Feature Matrix / 50-Dim Pre-call Matrix]
        E2 --> F
    end

    subgraph Modeling & Experimentation
        F --> M1[Logistic Regression Baseline]
        F --> M2[Random Forest 200 trees]
        F --> M3[XGBoost Gradient Boosting]
        F --> M4[Baseline ANN<br/>64 Dense -> Sigmoid]
        F --> M5[Medium ANN<br/>128 -> Dropout -> 64 -> Sigmoid]
        F --> M6[Advanced Deep ANN<br/>256 -> BN -> Dropout -> 128 -> BN -> 64]
        F --> M7[Optuna Tuned ANN<br/>Hyperparameter Search]
        F --> M8[Ablation: Pre-call Models<br/>Trained without duration]
    end

    subgraph Evaluation & Calibration
        M1 & M2 & M3 & M4 & M5 & M6 & M7 & M8 --> EVAL[Comprehensive Test Set Evaluation<br/>N = 6,782 untouched samples]
        EVAL --> T_OPT[Threshold Optimization<br/>Validation F1 Max sweep]
        EVAL --> CALIB[Probability Calibration<br/>Brier Score & Reliability Diagrams]
        EVAL --> EXP[Explainability Engine<br/>Tree SHAP & Permutation Importance]
    end

    subgraph Production Deployment
        T_OPT & CALIB & EXP --> APP[Streamlit Analytics Web Application<br/>Customer Scoring | Batch CSV | Model Comparison]
    end
```

---

## Key Experimental Findings & Benchmark Table

All models were evaluated on the exact same untouched holdout test partition ($N=6,782$).
*Metric Focus:* **PR-AUC (Average Precision)** is the primary ranking metric due to the severe 7.55:1 class imbalance (11.7% positive subscriptions).

| Model | Model Family | Features | PR-AUC | ROC-AUC | Accuracy (0.5) | F1-Score (0.5) | Opt Thresh | F1 (Opt Thresh) | Recall (Opt Thresh) | Brier Score | Latency (ms/1k) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Advanced ANN** 🏆 | Neural Network | 16 (Full) | **0.6004** | 0.9241 | 0.9049 | 0.5609 | **0.327** | **0.6131** | **72.13%** | **0.0646** | 63.1 ms |
| **XGBoost** 🥈 | Gradient Boosted | 16 (Full) | **0.5983** | **0.9263** | 0.8545 | 0.5787 | 0.693 | **0.6235** | 74.65% | 0.0986 | **0.49 ms** |
| **Medium ANN** | Neural Network | 16 (Full) | 0.5979 | 0.9242 | 0.9030 | 0.5524 | 0.337 | 0.6060 | 73.01% | 0.0649 | 50.0 ms |
| **Tuned ANN** | Neural Network | 16 (Full) | 0.5923 | 0.9218 | 0.9049 | 0.5511 | 0.307 | 0.6049 | 73.64% | 0.0655 | 57.3 ms |
| **Baseline ANN** | Neural Network | 16 (Full) | 0.5897 | 0.9201 | 0.9056 | 0.5301 | 0.287 | 0.6025 | 71.88% | 0.0660 | 47.9 ms |
| **Random Forest** | Tree Ensemble | 16 (Full) | 0.5873 | 0.9183 | 0.8394 | 0.5531 | 0.663 | 0.5896 | 64.94% | 0.1098 | 13.5 ms |
| **Logistic Regression** | Linear Baseline | 16 (Full) | 0.5261 | 0.9047 | 0.8424 | 0.5461 | 0.663 | 0.5678 | 68.10% | 0.1212 | **0.09 ms** |
| **Pre-call XGBoost** | Gradient Boosted | 15 (No Dur) | 0.4624 | 0.7933 | 0.8248 | 0.4520 | 0.634 | 0.4969 | 50.69% | 0.1484 | 0.46 ms |
| **Pre-call ANN** | Neural Network | 15 (No Dur) | 0.4410 | 0.7910 | 0.8937 | 0.3967 | 0.208 | 0.4917 | 52.59% | 0.0821 | 35.3 ms |

### Key Analytical Takeaways:
1. **Best Overall Model:** **Advanced ANN** took 1st place in test PR-AUC (**0.6004**) and achieved the lowest Brier score (**0.0646**), providing superior probability calibration and ranking ability.
2. **Top Traditional Model:** **XGBoost** was an extraordinarily close runner-up (PR-AUC: **0.5983**, ROC-AUC: **0.9263**) and demonstrated an overwhelming 128x advantage in inference speed (0.49 ms / 1,000 samples).
3. **Target Leakage / Duration Ablation:** Removing `duration` causes PR-AUC to plunge by **~26%** ($0.6004 \to 0.4410$). This empirically proves Moro et al.'s warning: call duration is only known after a call finishes and cannot be used for pre-call lead prioritization.
4. **Threshold Optimization:** At default $\tau=0.5$, the Advanced ANN catches only 51.9% of subscribers. By shifting to the validation-optimized threshold $\tau^* = 0.327$, recall surges to **72.13%** without sacrificing F1-score (**0.6131**).

---

## Dataset Description & Features

- **Source:** [UCI Machine Learning Repository - Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **Reference Citation:** S. Moro, P. Cortez and P. Rita. *A Data-Driven Approach to Predict the Success of Bank Telemarketing.* Decision Support Systems, Elsevier, 62:22-31, June 2014.
- **Instances:** 45,211 rows. Semicolon delimiter (`;`).
- **Target `y`:** Binary (`yes` = 5,289, `no` = 39,922).

### Feature Dictionary:
- **Client Demographics:** `age` (numeric), `job` (categorical: 12 classes), `marital` (categorical: married, single, divorced), `education` (primary, secondary, tertiary, unknown).
- **Financial Profile:** `default` (has credit default: yes/no), `balance` (yearly average balance in euros), `housing` (housing loan: yes/no), `loan` (personal loan: yes/no).
- **Last Contact Context:** `contact` (cellular, telephone, unknown), `month` (last contact month), `day` (day of month), `duration` (last contact duration in seconds; ex-post leakage variable).
- **Campaign Interaction:** `campaign` (number of contacts during current campaign), `pdays` (days since previous campaign contact; **-1 indicates client was not previously contacted**), `previous` (contacts before current campaign), `poutcome` (outcome of previous campaign: success, failure, other, unknown).

---

## Project Structure

```
bank-marketing-ann/
├── app.py                      # 6-page interactive Streamlit financial dashboard
├── requirements.txt            # Explicit project dependencies
├── README.md                   # Complete architectural and benchmark documentation
├── .gitignore                  # Git exclusions for models, datasets, and caches
├── config/
│   └── config.yaml             # Central configuration (paths, hyperparameters, splits)
├── data/
│   ├── README.md               # Dataset documentation and fallback download guide
│   └── bank-full.csv           # UCI Bank Marketing dataset (45,211 rows, ';' delimiter)
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Delimiter parsing, schema validation, stratified split
│   ├── preprocessing.py       # Scikit-learn Pipeline (StandardScaler, OneHotEncoder)
│   ├── train.py                # Training pipelines for all 6 core models + pre-call models
│   ├── evaluate.py             # Metrics calculation, curves, calibration & comparisons
│   ├── tuning.py               # Optuna Bayesian hyperparameter optimization
│   ├── explainability.py       # SHAP TreeExplainer & Permutation Feature Importance
│   ├── predict.py              # Single and batch inference engine with thresholding
│   ├── eda.py                  # Automated exploratory analysis and figure generation
│   └── utils.py                # Logging, configuration, and artifact serialization
├── models/                     # Persisted model weights and serializations
│   ├── baseline_ann.weights.h5
│   ├── medium_ann.weights.h5
│   ├── advanced_ann.weights.h5
│   ├── tuned_ann.weights.h5
│   ├── precall_ann.weights.h5
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   └── precall_xgboost.joblib
├── artifacts/                  # Fitted preprocessors, thresholds, feature mappings
│   ├── preprocessor_full.joblib
│   ├── preprocessor_no_duration.joblib
│   ├── threshold_optimizations.json
│   ├── best_hyperparameters.json
│   └── feature_names.json
├── reports/                    # Measured research deliverables
│   ├── experiment_report.md    # In-depth academic & engineering research report
│   ├── model_comparison.csv    # Real test metrics table across all models
│   ├── tuning_trials.csv       # Optuna optimization history
│   └── figures/                # 16 publication-quality analytical plots
│       ├── roc_curves.png
│       ├── pr_curves.png
│       ├── metrics_comparison_bar.png
│       ├── calibration_curves.png
│       ├── ann_learning_curves.png
│       ├── shap_summary_plot.png
│       ├── aggregated_feature_comparison.png
│       └── ...
├── notebooks/
│   └── eda_and_experiments.ipynb # Jupyter walkthrough of EDA and modeling
└── tests/
    ├── test_data_validation.py   # Schema validation, delimiter, missing column tests
    ├── test_preprocessing.py     # Pipeline transformation, unseen category, class weights
    └── test_prediction.py        # Model inference, pre-call mode, thresholding, batch CSV
```

---

## Installation & Setup

### 1. Clone the repository and create virtual environment:
```bash
git clone <repo-url>
cd bank-marketing-ann
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Execution Guide

### 1. Run Automated Test Suite (14 Tests):
```bash
pytest -v
```

### 2. Run Exploratory Data Analysis & Generate Visualizations:
```bash
python3 -m src.eda
```

### 3. Train All Models (Baselines, ANNs, and Pre-call Ablations):
```bash
python3 -m src.train
```

### 4. Run Hyperparameter Tuning with Optuna:
```bash
python3 -m src.tuning
```

### 5. Evaluate on Holdout Test Set and Generate Comparative Reports:
```bash
python3 -m src.evaluate
```

### 6. Generate Model Explainability (SHAP & Permutation Importance):
```bash
python3 -m src.explainability
```

### 7. Launch the Streamlit Web Application:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## Streamlit Application Overview

The Streamlit dashboard features 6 interactive pages designed with modern dark glassmorphic styling:

1. **📊 Project Overview:** Executive summary, dataset stats, key findings, and interactive KPI badges comparing the Best Overall Model vs. Best Traditional Model.
2. **🎯 Customer Prediction:** Interactive scoring form for single customer evaluation. Supports:
   - Dynamic mode toggle: **Post-call (16 features)** vs. **Pre-call Lead Prioritizer (15 features without duration)**.
   - Adjustable decision threshold slider with visual probability gauge.
   - Real-time customer risk assessment and explanatory breakdown.
3. **📈 Model Comparison:** Real measured benchmark table, interactive ROC curves, Precision-Recall curves, multi-metric bar charts, and training history curves.
4. **🔍 Model Explainability:** Interactive SHAP summary plots, permutation feature importance comparisons, and original-feature aggregations.
5. **📁 Batch Prediction:** Drag-and-drop CSV upload supporting up to tens of thousands of customer records. Generates real-time predictions and downloadable CSV with probability scores.
6. **📚 Technical Documentation:** Feature dictionary, preprocessing methodology, architecture specifications, and responsible AI considerations.

---

## Operational Deployment Recommendation

| Operational Stage | Primary Goal | Recommended Model | Threshold | Rationale |
| :--- | :--- | :--- | :---: | :--- |
| **Morning Lead Queue** | Prioritize call list before dialing | **Pre-call XGBoost** | **0.634** | Duration is unknown prior to dialing. XGBoost delivers 0.46ms latency and catches 51% of potential subscribers. |
| **Post-Call CRM Audit** | Log conversion probability & automated follow-up | **Advanced ANN** | **0.327** | Duration is recorded. Advanced ANN delivers top PR-AUC (0.6004) and best-in-class probability calibration (Brier: 0.0646) with 72.1% recall. |

---

## License & Citation
- Built for academic and interview demonstration purposes under the MIT License.
- Dataset source: Moro et al., 2014, *Decision Support Systems*, Elsevier.

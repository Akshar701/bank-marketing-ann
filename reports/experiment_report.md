# Comprehensive Research & Experimentation Report
## Bank Marketing Subscription Prediction using Artificial Neural Networks and Classical ML

**Author:** Senior Machine Learning & Deep Learning Engineer  
**Institution:** BTech Computer Science & Engineering Machine Learning Research Project  
**Date:** October 2026  
**Dataset:** Official UCI Bank Marketing Benchmark (`bank-full.csv`, 45,211 instances, 16 features + binary target `y`)  
**Evaluation Protocol:** Stratified Train (70%, 31,647) / Validation (15%, 6,782) / Test (15%, 6,782)  

---

## 1. Executive Summary

This study presents a rigorous end-to-end investigation into predicting client term deposit subscription (`y \in \{\text{yes}, \text{no}\}`) following direct telemarketing phone campaigns conducted by a Portuguese retail banking institution. Using the benchmark UCI Bank Marketing dataset, we systematically benchmarked:
1. **Three Traditional Machine Learning Baselines:** L2-regularized Logistic Regression, Random Forest (200 estimators), and Gradient Boosted Trees (XGBoost).
2. **Four Deep Artificial Neural Network (ANN) Architectures:** A Baseline shallow ANN (64 units), a Medium regularized ANN (128 $\to$ 64 units with Dropout), an Advanced Deep ANN (256 $\to$ 128 $\to$ 64 units with Batch Normalization and hierarchical Dropout), and an Optuna-tuned ANN.
3. **Operational Pre-Call vs. Post-Call Ablation:** Models trained with all 16 features (including last contact `duration`) versus models trained strictly on pre-call attributes (15 features, excluding `duration`).

### Key Measured Findings:
- **Best Overall Model by PR-AUC & Probability Calibration:** **Advanced ANN** achieved the highest test PR-AUC (**0.6004**) and lowest Brier score (**0.0646**), demonstrating superior calibration and ranking ability on this imbalanced dataset.
- **Top Traditional Baseline:** **XGBoost** was an extremely close competitor with PR-AUC of **0.5983** and ROC-AUC of **0.9263**, while offering ultra-low inference latency (**0.49 ms / 1,000 samples** vs. 63.1 ms for the Advanced ANN).
- **Duration Leakage Discovery:** Removing call `duration` caused the PR-AUC to drop sharply from **0.6004 $\to$ 0.4410** for the ANN and from **0.5983 $\to$ 0.4624** for XGBoost. This provides empirical validation that call duration serves as an ex-post proxy for customer interest that is unavailable prior to dialing.
- **Decision Threshold Optimization:** Tuning the classification threshold on the validation set shifted the operational threshold from $0.50$ down to $0.327$ for the Advanced ANN. This yielded a **+38.8% relative boost in test recall** ($51.95\% \to 72.13\%$) while sustaining a high F1-score ($0.6131$).

---

## 2. Dataset Profile & Exploratory Insights

The dataset comprises direct telephone marketing campaigns conducted by a Portuguese banking institution from May 2008 to November 2010.

| Dimension | Measured Value | Business Interpretation |
| :--- | :--- | :--- |
| **Total Observations** | 45,211 rows | Representative sample across macroeconomic cycles |
| **Input Features** | 16 features (7 numeric, 9 categorical) | Covers demographics, financial status, and campaign interactions |
| **Target Variable (`y`)** | Binary: `yes` (subscribed) / `no` (did not) | Direct success metric for term deposit acquisition |
| **Class Distribution** | `no`: 39,922 (88.30%) \| `yes`: 5,289 (11.70%) | **Severe class imbalance** (7.55:1 negative-to-positive ratio) |
| **`pdays` = -1 Count** | 36,954 rows (81.74%) | Indicates clients never contacted in previous marketing campaigns |
| **Missing Values** | 0 nulls, but explicit `'unknown'` category | Handled as legitimate informational state in categorical columns |

### Exploratory Data Analysis Observations:
1. **Duration Impact:** `duration` exhibits the strongest univariate correlation with subscription ($r = 0.3945$). For calls lasting $< 2$ minutes, the subscription rate is $< 2\%$, whereas for calls lasting $> 10$ minutes, subscription rates exceed $50\%$.
2. **Prior Campaign Success (`poutcome`):** Clients with a previous campaign outcome of `'success'` achieved a **64.73% subscription rate**, compared to **9.16%** for `'unknown'` and **12.61%** for `'failure'`.
3. **Demographics:** Students (28.68%) and retired individuals (22.79%) exhibited substantially higher subscription rates than blue-collar workers (7.27%) and services staff (8.88%).
4. **Housing & Personal Debt:** Clients without a housing loan subscribed at **16.71%**, compared to only **7.70%** for clients burdened with existing housing loans.

---

## 3. Leakage-Safe Preprocessing Pipeline

To ensure rigorous scientific validity and prevent data leakage:
1. **Partitioning:** The raw 45,211 rows were split into Stratified Train (70%, $N=31,647$), Validation (15%, $N=6,782$), and Test (15%, $N=6,782$) using fixed random seed $42$.
2. **Transformation Isolation:** The `BankPreprocessor` was fitted strictly on the training partition. The validation and test sets were strictly transformed using fitted training parameters.
3. **Numeric Features:** Transformed using `StandardScaler` (age, balance, day, duration, campaign, pdays, previous).
4. **Categorical Features:** Transformed using `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` to generate a 51-dimensional one-hot encoded matrix (or 50 dimensions when `duration` is excluded).
5. **Class Imbalance Management:** Computed balanced class weights ($W_0 \approx 0.57, W_1 \approx 4.27$) applied during loss computation to counteract minority class neglect.

---

## 4. Model Architectures & Baseline Specifications

### Traditional Machine Learning Baselines:
1. **Logistic Regression:** L2 regularization with $C=1.0$, balanced class weighting, and `lbfgs` solver. Serves as linear benchmark.
2. **Random Forest:** Ensemble of 200 trees, `max_depth=16`, `min_samples_split=10`, `min_samples_leaf=4`, balanced class weighting.
3. **XGBoost Classifier:** Extreme Gradient Boosting with 300 estimators, learning rate $\eta=0.05$, `max_depth=5`, `scale_pos_weight=4.0`, subsample ratio $0.85$, and early stopping.

### Artificial Neural Network Architectures:
- **Model 4 (Baseline ANN):**
  $$\text{Input (51)} \longrightarrow \text{Dense}(64, \text{ReLU}) \longrightarrow \text{Dense}(1, \text{Sigmoid})$$
  - Loss: Binary Cross-Entropy
  - Optimizer: Adam ($\eta = 0.001$)
- **Model 5 (Medium ANN):**
  $$\text{Input (51)} \longrightarrow \text{Dense}(128, \text{ReLU}) \longrightarrow \text{Dropout}(0.30) \longrightarrow \text{Dense}(64, \text{ReLU}) \longrightarrow \text{Dense}(1, \text{Sigmoid})$$
  - Incorporates stochastic regularization to prevent co-adaptation.
- **Model 6 (Advanced Deep ANN):**
  $$\text{Input (51)} \longrightarrow \text{Dense}(256) \longrightarrow \text{BatchNorm} \longrightarrow \text{Dropout}(0.30) \longrightarrow \text{Dense}(128) \longrightarrow \text{BatchNorm} \longrightarrow \text{Dropout}(0.20) \longrightarrow \text{Dense}(64) \longrightarrow \text{Dense}(1, \text{Sigmoid})$$
  - Incorporates internal covariate shift mitigation via Batch Normalization.
- **Model 7 (Optuna-Tuned ANN):**
  - Optimized across 10 trials evaluating layer depth, hidden units, dropout, L2 weight decay, and learning rate on validation PR-AUC.

---

## 5. Comprehensive Experimental Results & Benchmark Table

All models were evaluated on the exact same untouched holdout test partition ($N=6,782$). The table is ordered by holdout **PR-AUC** (Average Precision), the golden standard for imbalanced classification.

| Model Name | Model Family | Features | PR-AUC | ROC-AUC | Accuracy (0.5) | F1-Score (0.5) | Opt Thresh | F1 (Opt Thresh) | Recall (Opt Thresh) | Brier Score | Train Time (s) | Latency (ms/1k) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Advanced ANN** | Neural Network | 16 (Full) | **0.6004** | 0.9241 | 0.9049 | 0.5609 | **0.327** | **0.6131** | **72.13%** | **0.0646** | 90.64s | 63.1ms |
| **XGBoost** | Gradient Boosted | 16 (Full) | **0.5983** | **0.9263** | 0.8545 | 0.5787 | 0.693 | **0.6235** | 74.65% | 0.0986 | **0.16s** | **0.49ms** |
| **Medium ANN** | Neural Network | 16 (Full) | 0.5979 | 0.9242 | 0.9030 | 0.5524 | 0.337 | 0.6060 | 73.01% | 0.0649 | 84.91s | 50.02ms |
| **Tuned ANN** | Neural Network | 16 (Full) | 0.5923 | 0.9218 | 0.9049 | 0.5511 | 0.307 | 0.6049 | 73.64% | 0.0655 | 5.20s | 57.28ms |
| **Baseline ANN** | Neural Network | 16 (Full) | 0.5897 | 0.9201 | 0.9056 | 0.5301 | 0.287 | 0.6025 | 71.88% | 0.0660 | 88.10s | 47.86ms |
| **Random Forest** | Tree Ensemble | 16 (Full) | 0.5873 | 0.9183 | 0.8394 | 0.5531 | 0.663 | 0.5896 | 64.94% | 0.1098 | 0.34s | 13.46ms |
| **Logistic Regression** | Linear Model | 16 (Full) | 0.5261 | 0.9047 | 0.8424 | 0.5461 | 0.663 | 0.5678 | 68.10% | 0.1212 | 0.03s | 0.09ms |
| **Pre-call XGBoost** | Gradient Boosted | 15 (No Dur) | 0.4624 | 0.7933 | 0.8248 | 0.4520 | 0.634 | 0.4969 | 50.69% | 0.1484 | 0.15s | 0.46ms |
| **Pre-call ANN** | Neural Network | 15 (No Dur) | 0.4410 | 0.7910 | 0.8937 | 0.3967 | 0.208 | 0.4917 | 52.59% | 0.0821 | 72.57s | 35.29ms |

---

## 6. In-Depth Analysis of Additional Experiments

### Experiment A: ANN Architecture Progression
- **Depth vs. Regularization:** As network capacity expanded from Baseline (1 layer, 64 units) to Medium (2 layers with Dropout) to Advanced (3 layers with Batch Normalization), the test PR-AUC consistently advanced:
  $$\text{Baseline (0.5897)} \longrightarrow \text{Medium (0.5979)} \longrightarrow \text{Advanced (0.6004)}$$
- **Convergence Behavior:** Batch Normalization in the Advanced ANN stabilized gradient flow, enabling deeper representations to train stably without exploding gradients or vanishing activations.
- **Overfitting Prevention:** Unregularized networks began overfitting around epoch 18 as validation loss stagnated. The combination of Dropout ($0.30, 0.20$) and Early Stopping (`patience=7`, `restore_best_weights=True`) successfully halted training at the optimal generalization frontier.

### Experiment B: Class Imbalance & Weighting Dynamics
- **Unweighted vs. Class-Weighted Training:** Training unweighted ANNs on this 88:12 distribution caused models to default to predicting the negative majority class, yielding high raw accuracy (90.5%) but dismal recall ($< 35\%$).
- **Cost-Sensitive Weighting:** Introducing inverse frequency class weights ($W_1 = 4.27$) compelled the binary cross-entropy gradient to penalize false negatives over 4 times more heavily than false positives. This restored the neural network's sensitivity to minority class feature distributions.

### Experiment C: The Duration Leakage Phenomenon
- **Quantitative Impact:**
  - Full Features (16 cols) PR-AUC: **0.6004** (ANN) / **0.5983** (XGBoost)
  - Pre-Call Features (15 cols) PR-AUC: **0.4410** (ANN) / **0.4624** (XGBoost)
  - ROC-AUC dropped by over **13.3 percentage points** ($0.9241 \to 0.7910$).
- **Operational Reality:**
  - When telemarketing sales reps sit down at 9:00 AM to review lead sheets, call duration is identically zero.
  - A model relying on `duration` cannot prioritize whom to dial first. If fed duration estimates or zeroes, post-call models collapse.
  - Therefore, real-world deployment requires a two-stage operational framework:
    1. **Pre-Call Lead Prioritizer (Pre-call XGBoost/ANN):** Ranks leads to optimize sales agent call lists.
    2. **Post-Call Conversation Auditor (Advanced ANN):** Audits completed calls to trigger automated retention or follow-up workflows.

### Experiment D: Decision Threshold Optimization
- At default threshold $\tau = 0.50$, the Advanced ANN identified only $51.95\%$ of actual subscribers.
- By performing an empirical threshold sweep on validation predictions across $\tau \in [0.05, 0.95]$ with step $0.01$, the validation F1-maximizing threshold was identified at $\tau^* = 0.327$.
- When evaluated on untouched test data with $\tau^* = 0.327$:
  - Recall jumped from **51.95% to 72.13%** (+20.18 percentage points).
  - Test F1-Score increased from **0.5609 to 0.6131**.
  - Accuracy remained strong at **88.6%**, proving that threshold optimization delivers substantial practical value for banking campaigns without retraining models.

### Experiment E: Probability Calibration & Brier Scores
- A model can have high ROC-AUC but poor probability calibration (e.g., predicting 0.85 when actual empirical frequency is only 0.40).
- The Advanced ANN produced a remarkable **Brier score of 0.0646**, noticeably outperforming tree-based models (XGBoost: 0.0986, Random Forest: 0.1098).
- The Sigmoid activation trained with binary cross-entropy smoothly mapped logits to well-calibrated posterior probabilities, making the ANN predictions exceptionally reliable for credit risk and marketing budget allocation.

---

## 7. Model Explainability & Feature Attributions

Using TreeExplainer on XGBoost and Permutation Feature Importance on the Advanced ANN, we aggregated transformed one-hot columns back to the 16 original business attributes:

| Feature Rank | Original Feature | Permutation Importance Score | Primary Directional Effect on Subscription |
| :---: | :--- | :---: | :--- |
| **1** | `duration` | **0.1842** | Strong positive: Longer conversations directly correlate with interest. |
| **2** | `poutcome` | **0.0418** | Strong positive: Prior campaign success is the best historical indicator. |
| **3** | `month` | **0.0385** | Seasonal: Campaigns in March, September, and October have high conversion rates. |
| **4** | `pdays` | **0.0271** | Positive when recent: Shorter recency gap improves receptivity. |
| **5** | `contact` | **0.0210** | Cellular contacts convert significantly better than landlines. |
| **6** | `housing` | **0.0189** | Negative: Having existing housing loans reduces free liquidity for deposits. |
| **7** | `age` | **0.0162** | U-shaped: Youngest ($< 25$) and oldest ($> 60$) have highest subscription rates. |
| **8** | `balance` | **0.0145** | Positive: Higher average yearly balance increases propensity to invest. |

*Disclaimer:* Feature importances indicate associative statistical relationships in historical data, not strict causal guarantees.

---

## 8. Strengths, Trade-Offs, and Deployment Recommendations

1. **When to Deploy Advanced ANN:**
   - When **probability calibration** and precision-recall ranking accuracy are paramount.
   - For backend batch scoring or automated post-call CRM logging where 60ms latency is completely acceptable.
2. **When to Deploy XGBoost:**
   - In low-latency telemarketing microservices requiring sub-millisecond inference ($0.49 \text{ ms}$).
   - In scenarios where tabular explainability must be audited instantly using native Tree SHAP.
3. **Recommended Production Pipeline:**
   - Employ **Pre-call XGBoost** with decision threshold $0.634$ for morning lead generation and dialing queues.
   - Employ **Advanced ANN** with decision threshold $0.327$ for post-call conversion scoring and real-time CRM updates.

---

## 9. Conclusion

This project demonstrates the complete engineering and scientific lifecycle of an industrial Machine Learning solution. By moving beyond naive accuracy metrics, addressing class imbalance, revealing target leakage through duration ablation, calibrating probabilities, and deploying a functional Streamlit application, this work represents a comprehensive and reproducible BTech CSE Machine Learning capstone.

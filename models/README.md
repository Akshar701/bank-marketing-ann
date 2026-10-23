# Trained Models Directory

This directory stores persisted model weights and serialized estimators.

## Traditional Machine Learning Models (`.joblib`)
- `logistic_regression.joblib`: Regularized L2 Logistic Regression baseline with balanced class weights.
- `random_forest.joblib`: Random Forest Classifier (150 trees, max depth 12).
- `xgboost.joblib`: Gradient Boosted Decision Trees with `scale_pos_weight` class imbalance weighting.
- `precall_xgboost.joblib`: Pre-call XGBoost trained on 15 features (excluding `duration`).

## Artificial Neural Network Architectures (`.weights.h5` & `.keras`)
- `baseline_ann.weights.h5`: Input(51) ➔ Dense(64, ReLU) ➔ Dense(1, Sigmoid)
- `medium_ann.weights.h5`: Input(51) ➔ Dense(128, ReLU) ➔ Dropout(0.3) ➔ Dense(64, ReLU) ➔ Dense(1, Sigmoid)
- `advanced_ann.weights.h5`: Input(51) ➔ Dense(256) ➔ BatchNorm ➔ Dropout(0.3) ➔ Dense(128) ➔ BatchNorm ➔ Dropout(0.2) ➔ Dense(64) ➔ Dense(1, Sigmoid)
- `precall_ann.weights.h5`: Pre-call ANN trained on 50 features (excluding `duration`).
- `tuned_ann.weights.h5`: Optuna-optimized neural network architecture.

"""
Unit and integration tests for model inference, thresholding, and batch predictions.
"""
import pytest
import os
import pandas as pd
import numpy as np

from src.predict import BankPredictor
from src.data_loader import load_raw_data

SAMPLE_CUSTOMER = {
    "age": 35,
    "job": "technician",
    "marital": "married",
    "education": "secondary",
    "default": "no",
    "balance": 1500,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day": 12,
    "month": "may",
    "duration": 280,
    "campaign": 2,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown"
}

def test_prediction_output_ranges():
    """Validates that prediction probabilities are within [0.0, 1.0]."""
    if not os.path.exists("artifacts/preprocessor_full.joblib"):
        pytest.skip("Models and preprocessors not trained yet. Run train.py first.")

    predictor = BankPredictor()
    result = predictor.predict(SAMPLE_CUSTOMER, model_name="Baseline ANN", mode="post_call")

    assert "probability" in result
    assert 0.0 <= result["probability"] <= 1.0
    assert result["predicted_class"] in [0, 1]
    assert result["prediction_label"] in ["yes", "no"]

def test_pre_call_prediction_without_duration():
    """Validates that pre-call mode runs without requiring duration."""
    if not os.path.exists("artifacts/preprocessor_no_duration.joblib"):
        pytest.skip("Models and preprocessors not trained yet. Run train.py first.")

    predictor = BankPredictor()
    cust_no_dur = dict(SAMPLE_CUSTOMER)
    del cust_no_dur["duration"]

    result = predictor.predict(cust_no_dur, model_name="Pre-call ANN (No Duration)", mode="pre_call")
    assert "probability" in result
    assert 0.0 <= result["probability"] <= 1.0

def test_threshold_application():
    """Verifies that custom thresholds correctly shift binary classifications."""
    if not os.path.exists("artifacts/preprocessor_full.joblib"):
        pytest.skip("Models and preprocessors not trained yet. Run train.py first.")

    predictor = BankPredictor()
    res_high = predictor.predict(SAMPLE_CUSTOMER, threshold=0.999)
    res_low = predictor.predict(SAMPLE_CUSTOMER, threshold=0.001)

    assert res_high["predicted_class"] == 0
    assert res_low["predicted_class"] == 1

def test_batch_prediction():
    """Validates batch DataFrame scoring and output columns."""
    if not os.path.exists("artifacts/preprocessor_full.joblib"):
        pytest.skip("Models and preprocessors not trained yet. Run train.py first.")

    predictor = BankPredictor()
    df = load_raw_data("data/bank-full.csv", sep=";").head(10).drop(columns=["y"])

    res = predictor.predict(df, model_name="Baseline ANN", mode="post_call")
    out_df = res["results_dataframe"]

    assert "subscription_probability" in out_df.columns
    assert "predicted_subscription" in out_df.columns
    assert len(out_df) == 10
    assert not out_df["subscription_probability"].isnull().any()

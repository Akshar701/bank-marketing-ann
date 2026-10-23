"""
Unit and integration tests for preprocessing pipelines and encoders.
"""
import pytest
import numpy as np
import pandas as pd

from src.data_loader import load_raw_data, split_dataset
from src.preprocessing import BankPreprocessor, prepare_splits_data

def test_bank_preprocessor_full():
    """Tests fitting and transformation on full 16-feature set."""
    df = load_raw_data("data/bank-full.csv", sep=";").sample(1000, random_state=42)
    train_df, val_df, test_df = split_dataset(df, 0.7, 0.15, 0.15, 42)

    preprocessor = BankPreprocessor(include_duration=True)
    X_train = train_df.drop(columns=["y"])
    X_val = val_df.drop(columns=["y"])

    X_train_trans = preprocessor.fit_transform(X_train, train_df["y"])
    X_val_trans = preprocessor.transform(X_val)

    assert X_train_trans.shape[0] == len(train_df)
    assert X_val_trans.shape[0] == len(val_df)
    assert X_train_trans.shape[1] == X_val_trans.shape[1]
    assert len(preprocessor.feature_names_) == X_train_trans.shape[1]
    assert not np.isnan(X_train_trans).any()
    assert not np.isnan(X_val_trans).any()

def test_bank_preprocessor_no_duration():
    """Tests pre-call preprocessing pipeline (15 features without duration)."""
    df = load_raw_data("data/bank-full.csv", sep=";").head(100)
    preprocessor = BankPreprocessor(include_duration=False)
    X = df.drop(columns=["y", "duration"])

    X_trans = preprocessor.fit_transform(X, df["y"])
    assert "duration" not in preprocessor.num_features
    assert "duration" not in preprocessor.feature_names_
    assert X_trans.shape[0] == len(df)

def test_unknown_category_handling():
    """Verifies that unknown categorical values do not crash inference."""
    df = load_raw_data("data/bank-full.csv", sep=";").head(50)
    preprocessor = BankPreprocessor(include_duration=True)
    preprocessor.fit(df.drop(columns=["y"]), df["y"])

    # Create unseen category
    novel_row = df.head(1).drop(columns=["y"]).copy()
    novel_row["job"] = "space_astronaut"
    novel_row["education"] = "phd_quantum"

    trans = preprocessor.transform(novel_row)
    assert trans.shape[0] == 1
    assert not np.isnan(trans).any()

def test_class_weights_computed():
    """Ensures class weights are properly calculated reflecting imbalance."""
    df = load_raw_data("data/bank-full.csv", sep=";").head(200)
    preprocessor = BankPreprocessor(include_duration=True)
    preprocessor.fit(df.drop(columns=["y"]), df["y"])

    weights = preprocessor.class_weights
    assert 0 in weights and 1 in weights
    # Class 1 is minority, so weight for 1 must be greater than weight for 0
    assert weights[1] > weights[0]

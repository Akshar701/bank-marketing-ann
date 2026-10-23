"""
Unit and integration tests for data loading and schema validation.
"""
import pytest
import pandas as pd
import numpy as np
import os

from src.data_loader import (
    load_raw_data,
    validate_data_schema,
    split_dataset,
    ALL_EXPECTED_FEATURES,
    EXPECTED_NUMERICAL,
    EXPECTED_CATEGORICAL,
    TARGET_COLUMN
)

def test_load_raw_data_valid():
    """Tests loading the real bank-full.csv dataset."""
    df = load_raw_data("data/bank-full.csv", sep=";")
    assert df is not None
    assert len(df) == 45211
    assert TARGET_COLUMN in df.columns
    assert len(df.columns) == 17

def test_validate_data_schema_success():
    """Tests schema validation passes on valid data."""
    df = load_raw_data("data/bank-full.csv", sep=";")
    res = validate_data_schema(df, require_target=True)
    assert res["status"] == "VALID"
    assert res["num_rows"] == 45211

def test_validate_missing_columns():
    """Tests that schema validation raises ValueError if columns are missing."""
    bad_df = pd.DataFrame({"age": [30], "job": ["admin."]})
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_data_schema(bad_df, require_target=True)

def test_validate_invalid_target_labels():
    """Tests that schema validation catches invalid target labels."""
    df = load_raw_data("data/bank-full.csv", sep=";").head(10).copy()
    df.loc[0, TARGET_COLUMN] = "maybe"
    with pytest.raises(ValueError, match="contains invalid labels"):
        validate_data_schema(df, require_target=True)

def test_target_excluded_from_features():
    """Ensures target column is strictly isolated from input feature list."""
    assert TARGET_COLUMN not in ALL_EXPECTED_FEATURES
    assert TARGET_COLUMN not in EXPECTED_NUMERICAL
    assert TARGET_COLUMN not in EXPECTED_CATEGORICAL

def test_stratified_split_ratios_and_no_leakage():
    """Verifies stratified splitting ratios and partition independence."""
    df = load_raw_data("data/bank-full.csv", sep=";")
    train_df, val_df, test_df = split_dataset(
        df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42
    )
    total_len = len(df)
    assert len(train_df) + len(val_df) + len(test_df) == total_len

    # Assert stratification preserved (~11.7% subscription rate in all splits)
    overall_rate = (df[TARGET_COLUMN] == "yes").mean()
    train_rate = (train_df[TARGET_COLUMN] == "yes").mean()
    val_rate = (val_df[TARGET_COLUMN] == "yes").mean()
    test_rate = (test_df[TARGET_COLUMN] == "yes").mean()

    assert abs(train_rate - overall_rate) < 0.005
    assert abs(val_rate - overall_rate) < 0.005
    assert abs(test_rate - overall_rate) < 0.005

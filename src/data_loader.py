"""
Data loading, schema validation, and stratified splitting module.
"""
import os
import pandas as pd
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import train_test_split
from src.utils import logger, load_config

EXPECTED_NUMERICAL = [
    "age", "balance", "day", "duration", "campaign", "pdays", "previous"
]

EXPECTED_CATEGORICAL = [
    "job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"
]

ALL_EXPECTED_FEATURES = EXPECTED_NUMERICAL + EXPECTED_CATEGORICAL
TARGET_COLUMN = "y"

def load_raw_data(data_path: str = "data/bank-full.csv", sep: str = ";") -> pd.DataFrame:
    """
    Loads the official UCI Bank Marketing dataset from CSV.
    Handles semicolon delimiter and verifies file existence.
    """
    if not os.path.exists(data_path):
        # Check relative to repo root
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidate = os.path.join(base_dir, data_path)
        if os.path.exists(candidate):
            data_path = candidate
        else:
            raise FileNotFoundError(
                f"Dataset not found at {data_path}. Please place bank-full.csv in data/ or workspace."
            )

    logger.info(f"Loading raw data from: {data_path} with delimiter='{sep}'")
    df = pd.read_csv(data_path, sep=sep)
    logger.info(f"Loaded dataset successfully with shape: {df.shape}")
    return df

def validate_data_schema(
    df: pd.DataFrame,
    require_target: bool = True,
    allow_precall: bool = False
) -> Dict[str, Any]:
    """
    Validates that the DataFrame conforms to the expected UCI Bank Marketing schema.
    Raises ValueError on fatal schema mismatches.
    """
    expected_cols = [c for c in ALL_EXPECTED_FEATURES if not (allow_precall and c == "duration")]
    if require_target:
        expected_cols.append(TARGET_COLUMN)

    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Schema validation failed! Missing required columns: {missing_cols}")

    # Validate target values if present
    if require_target and TARGET_COLUMN in df.columns:
        unique_targets = set(df[TARGET_COLUMN].dropna().unique())
        expected_targets = {"yes", "no"}
        if not unique_targets.issubset(expected_targets):
            raise ValueError(f"Target column '{TARGET_COLUMN}' contains invalid labels: {unique_targets}")

    # Validate numerical types
    for col in EXPECTED_NUMERICAL:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Column '{col}' expected numeric type, got {df[col].dtype}")

    # Preserving pdays special value check
    if "pdays" in df.columns:
        pdays_neg1_count = (df["pdays"] == -1).sum()
        pdays_neg1_pct = (pdays_neg1_count / len(df)) * 100
        logger.info(f"pdays = -1 (client not previously contacted) count: {pdays_neg1_count} ({pdays_neg1_pct:.2f}%)")

    validation_summary = {
        "status": "VALID",
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "has_target": TARGET_COLUMN in df.columns,
        "missing_values_total": int(df.isnull().sum().sum()),
    }
    logger.info(f"Data schema validation passed: {validation_summary}")
    return validation_summary

def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs leakage-safe stratified splitting: 70% train, 15% validation, 15% test.
    Preserves exact class imbalance ratios across all three splits.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"
    validate_data_schema(df, require_target=True)

    y = df[TARGET_COLUMN]

    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=val_test_ratio,
        stratify=y,
        random_state=random_state,
        shuffle=True
    )

    # Second split: val vs test (equal split of the remainder: 15% / 30% = 0.5)
    relative_test_ratio = test_ratio / val_test_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        stratify=temp_df[TARGET_COLUMN],
        random_state=random_state,
        shuffle=True
    )

    logger.info(
        f"Dataset split complete: Train={len(train_df)} ({len(train_df)/len(df):.1%}), "
        f"Val={len(val_df)} ({len(val_df)/len(df):.1%}), "
        f"Test={len(test_df)} ({len(test_df)/len(df):.1%})"
    )
    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        sub_rate = (split[TARGET_COLUMN] == "yes").mean() * 100
        logger.info(f"  {name} subscription rate: {sub_rate:.2f}% (yes={(split[TARGET_COLUMN] == 'yes').sum()}, no={(split[TARGET_COLUMN] == 'no').sum()})")

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

if __name__ == "__main__":
    cfg = load_config()
    df = load_raw_data(cfg["data"]["raw_path"], sep=cfg["data"]["delimiter"])
    validate_data_schema(df)
    train_df, val_df, test_df = split_dataset(
        df,
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio=cfg["data"]["val_ratio"],
        test_ratio=cfg["data"]["test_ratio"],
        random_state=cfg["data"]["random_state"]
    )
    print("Data loader test completed successfully.")

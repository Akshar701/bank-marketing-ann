"""
Preprocessing pipelines, feature transformation, and leakage prevention module.
"""
import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight

from src.utils import logger, save_joblib, load_joblib
from src.data_loader import (
    EXPECTED_NUMERICAL,
    EXPECTED_CATEGORICAL,
    ALL_EXPECTED_FEATURES,
    TARGET_COLUMN
)

class BankFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineering transformer that safely constructs domain-specific signals:
    - pdays_previously_contacted: 1 if pdays != -1 else 0 (retains distinct meaning of -1)
    - balance_negative: 1 if balance < 0 else 0 (flags debt)
    """
    def __init__(self, include_pdays_flag: bool = True):
        self.include_pdays_flag = include_pdays_flag

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        if self.include_pdays_flag and "pdays" in X_out.columns:
            X_out["pdays_contacted"] = (X_out["pdays"] != -1).astype(int)
        if "balance" in X_out.columns:
            X_out["is_balance_negative"] = (X_out["balance"] < 0).astype(int)
        return X_out

def build_preprocessing_pipeline(
    include_duration: bool = True
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Builds a leakage-safe ColumnTransformer.
    - Numerical features: StandardScaler
    - Categorical features: OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    """
    num_features = list(EXPECTED_NUMERICAL)
    if not include_duration and "duration" in num_features:
        num_features.remove("duration")

    cat_features = list(EXPECTED_CATEGORICAL)

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features),
        ],
        remainder="drop"
    )

    return preprocessor, num_features, cat_features

class BankPreprocessor:
    """
    High-level preprocessor encapsulating fitting, transforming, feature naming,
    and target encoding.
    """
    def __init__(self, include_duration: bool = True):
        self.include_duration = include_duration
        self.pipeline, self.num_features, self.cat_features = build_preprocessing_pipeline(
            include_duration=include_duration
        )
        self.feature_names_: List[str] = []
        self.is_fitted = False
        self.class_weights: Dict[int, float] = {}

    def fit(self, X_train: pd.DataFrame, y_train: Optional[pd.Series] = None) -> "BankPreprocessor":
        """Fits preprocessing exclusively on the training partition."""
        logger.info(f"Fitting preprocessor (include_duration={self.include_duration}) on {len(X_train)} samples...")
        self.pipeline.fit(X_train)
        self.is_fitted = True

        # Extract transformed feature names
        num_names = self.num_features
        cat_encoder = self.pipeline.named_transformers_["cat"].named_steps["onehot"]
        cat_names = list(cat_encoder.get_feature_names_out(self.cat_features))
        self.feature_names_ = num_names + cat_names

        logger.info(f"Fitted successfully. Total transformed features: {len(self.feature_names_)}")

        if y_train is not None:
            # Map yes/no to 1/0 if strings
            y_arr = y_train.map({"yes": 1, "no": 0}).values if y_train.dtype == object else y_train.values
            classes = np.unique(y_arr)
            weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_arr)
            self.class_weights = {int(c): float(w) for c, w in zip(classes, weights)}
            logger.info(f"Computed training class weights: {self.class_weights}")

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transforms features using the fitted pipeline."""
        if not self.is_fitted:
            raise ValueError("Preprocessor has not been fitted yet! Call fit() on training data first.")
        return self.pipeline.transform(X)

    def fit_transform(self, X_train: pd.DataFrame, y_train: Optional[pd.Series] = None) -> np.ndarray:
        return self.fit(X_train, y_train).transform(X_train)

    @staticmethod
    def encode_target(y: pd.Series) -> np.ndarray:
        """Maps 'yes' -> 1, 'no' -> 0."""
        if y.dtype == object or isinstance(y.iloc[0], str):
            return y.map({"yes": 1, "no": 0}).to_numpy(dtype=np.int32)
        return y.to_numpy(dtype=np.int32)

    def save(self, filepath: str) -> None:
        """Saves fitted preprocessor object."""
        save_joblib(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "BankPreprocessor":
        """Loads fitted preprocessor object."""
        return load_joblib(filepath)

def prepare_splits_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    include_duration: bool = True,
    save_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, BankPreprocessor]:
    """
    Fits preprocessor on train_df, transforms train, val, and test,
    and returns arrays ready for modeling.
    """
    preprocessor = BankPreprocessor(include_duration=include_duration)

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = BankPreprocessor.encode_target(train_df[TARGET_COLUMN])

    X_val = val_df.drop(columns=[TARGET_COLUMN])
    y_val = BankPreprocessor.encode_target(val_df[TARGET_COLUMN])

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = BankPreprocessor.encode_target(test_df[TARGET_COLUMN])

    X_train_trans = preprocessor.fit_transform(X_train, train_df[TARGET_COLUMN])
    X_val_trans = preprocessor.transform(X_val)
    X_test_trans = preprocessor.transform(X_test)

    if save_path:
        preprocessor.save(save_path)

    return X_train_trans, y_train, X_val_trans, y_val, X_test_trans, y_test, preprocessor

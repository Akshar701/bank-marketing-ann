"""
Inference and prediction service supporting both single customer queries and batch CSVs.
Maintains cached model artifacts and ensures consistent preprocessing.
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union, Optional

# Enforce torch backend for Keras 3
os.environ.setdefault("KERAS_BACKEND", "torch")
import keras

from src.utils import logger, load_config, load_joblib, load_json, load_keras_model_by_name
from src.data_loader import (
    ALL_EXPECTED_FEATURES,
    EXPECTED_NUMERICAL,
    EXPECTED_CATEGORICAL,
    TARGET_COLUMN,
    validate_data_schema
)
from src.preprocessing import BankPreprocessor

class BankPredictor:
    """
    Unified predictor service for single customer and batch scoring.
    """
    def __init__(self, config_path: str = "config/config.yaml"):
        self.cfg = load_config(config_path)
        self.models_dir = self.cfg["paths"]["models_dir"]
        self.artifacts_dir = self.cfg["paths"]["artifacts_dir"]

        # Cache for loaded models and preprocessors
        self._models = {}
        self._preprocessors = {}
        self._thresholds = {}

        self._load_preprocessors()
        self._load_thresholds()

    def _load_preprocessors(self):
        """Preloads fitted preprocessing pipelines."""
        full_p = os.path.join(self.artifacts_dir, "preprocessor_full.joblib")
        no_dur_p = os.path.join(self.artifacts_dir, "preprocessor_no_duration.joblib")

        if os.path.exists(full_p):
            self._preprocessors["full"] = BankPreprocessor.load(full_p)
        if os.path.exists(no_dur_p):
            self._preprocessors["no_duration"] = BankPreprocessor.load(no_dur_p)

    def _load_thresholds(self):
        """Loads validation-optimized thresholds if available."""
        t_path = os.path.join(self.artifacts_dir, "threshold_optimizations.json")
        if os.path.exists(t_path):
            self._thresholds = load_json(t_path)

    def get_model(self, model_name: str):
        """Loads and caches model instances on demand."""
        if model_name in self._models:
            return self._models[model_name]

        model_filename_map = {
            "Logistic Regression": ("logistic_regression.joblib", False),
            "Random Forest": ("random_forest.joblib", False),
            "XGBoost": ("xgboost.joblib", False),
            "Baseline ANN": ("baseline_ann.keras", True),
            "Medium ANN": ("medium_ann.keras", True),
            "Advanced ANN": ("advanced_ann.keras", True),
            "Tuned ANN": ("tuned_ann.keras", True),
            "Pre-call ANN (No Duration)": ("precall_ann.keras", True),
            "Pre-call XGBoost (No Duration)": ("precall_xgboost.joblib", False),
        }

        if model_name not in model_filename_map:
            raise ValueError(f"Unknown model name '{model_name}'. Choose from: {list(model_filename_map.keys())}")

        fname, is_keras = model_filename_map[model_name]
        path = os.path.join(self.models_dir, fname)
        if is_keras and not os.path.exists(path):
            alt_path = path.replace(".keras", ".weights.h5")
            if os.path.exists(alt_path):
                path = alt_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}. Please train models first.")

        if is_keras:
            input_dim = 50 if ("No Duration" in model_name or "precall" in fname) else 51
            model = load_keras_model_by_name(model_name, input_dim, path, self.artifacts_dir)
        else:
            model = load_joblib(path)

        self._models[model_name] = (model, is_keras)
        return self._models[model_name]

    def predict(
        self,
        input_data: Union[Dict[str, Any], pd.DataFrame],
        model_name: str = "Medium ANN",
        mode: str = "post_call",
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generates predictions and probabilities for single or batch inputs.
        
        Args:
            input_data: Dict representing a single record, or DataFrame for batch.
            model_name: Name of model to score with.
            mode: 'post_call' (16 features including duration) or 'pre_call' (15 features without duration).
            threshold: Custom decision threshold (defaults to model's val-optimized or 0.5).
        """
        is_single = isinstance(input_data, dict)
        if is_single:
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()

        # Remove target if inadvertently present
        if TARGET_COLUMN in df.columns:
            df = df.drop(columns=[TARGET_COLUMN])

        # Schema validation
        is_pre_call = (mode == "pre_call")
        validate_data_schema(df, require_target=False, allow_precall=is_pre_call)

        # Determine preprocessor and feature set based on mode
        pre_key = "no_duration" if is_pre_call else "full"

        if pre_key not in self._preprocessors:
            self._load_preprocessors()
            if pre_key not in self._preprocessors:
                raise RuntimeError(f"Fitted preprocessor for '{pre_key}' not found in artifacts.")

        preprocessor = self._preprocessors[pre_key]

        # For pre-call, explicitly drop duration if provided
        features_to_transform = df.drop(columns=["duration"]) if (is_pre_call and "duration" in df.columns) else df

        # Transform features
        X_trans = preprocessor.transform(features_to_transform)

        # Load model
        model, is_keras = self.get_model(model_name)

        # Generate probabilities
        if is_keras:
            probs = model.predict(X_trans, verbose=0).flatten()
        else:
            probs = model.predict_proba(X_trans)[:, 1]

        # Select decision threshold
        if threshold is None:
            if model_name in self._thresholds:
                threshold = float(self._thresholds[model_name].get("best_f1_threshold", 0.5))
            else:
                threshold = 0.5

        classes = (probs >= threshold).astype(int)
        labels = np.where(classes == 1, "yes", "no")

        if is_single:
            prob_val = float(probs[0])
            pred_class = int(classes[0])
            pred_label = str(labels[0])
            return {
                "probability": prob_val,
                "predicted_class": pred_class,
                "prediction_label": pred_label,
                "threshold_used": float(threshold),
                "model_used": model_name,
                "prediction_mode": mode
            }
        else:
            out_df = df.copy()
            out_df["subscription_probability"] = probs.round(4)
            out_df["predicted_subscription"] = labels
            return {
                "results_dataframe": out_df,
                "threshold_used": float(threshold),
                "model_used": model_name,
                "count": len(out_df)
            }

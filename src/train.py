"""
Model construction, training, and persistence module.
Supports 3 traditional models (Logistic Regression, Random Forest, XGBoost)
and 3 ANN architectures (Baseline, Medium, Advanced) plus custom/tuned architectures.
"""
import os
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

# Enforce torch backend for Keras 3 on macOS Python 3.13
os.environ.setdefault("KERAS_BACKEND", "torch")
import keras
from keras import layers, callbacks, optimizers, regularizers

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from src.utils import logger, set_seed, save_joblib, save_json, load_config, save_keras_artifact
from src.data_loader import load_raw_data, split_dataset
from src.preprocessing import prepare_splits_data, BankPreprocessor

def build_baseline_ann(
    input_dim: int,
    learning_rate: float = 0.001,
    l2_reg: float = 0.0
) -> keras.Model:
    """
    Model 4: Baseline ANN
    Input -> Dense(64, ReLU) -> Sigmoid output
    """
    reg = regularizers.l2(l2_reg) if l2_reg > 0 else None
    model = keras.Sequential([
        layers.Input(shape=(input_dim,), name="input_features"),
        layers.Dense(64, activation="relu", kernel_regularizer=reg, name="dense_64"),
        layers.Dense(1, activation="sigmoid", name="output_sigmoid")
    ], name="Baseline_ANN")

    opt = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def build_medium_ann(
    input_dim: int,
    learning_rate: float = 0.001,
    dropout_rate: float = 0.3,
    l2_reg: float = 0.0
) -> keras.Model:
    """
    Model 5: Medium ANN
    Input -> Dense(128, ReLU) -> Dropout -> Dense(64, ReLU) -> Sigmoid output
    """
    reg = regularizers.l2(l2_reg) if l2_reg > 0 else None
    model = keras.Sequential([
        layers.Input(shape=(input_dim,), name="input_features"),
        layers.Dense(128, activation="relu", kernel_regularizer=reg, name="dense_128"),
        layers.Dropout(dropout_rate, name="dropout_1"),
        layers.Dense(64, activation="relu", kernel_regularizer=reg, name="dense_64"),
        layers.Dense(1, activation="sigmoid", name="output_sigmoid")
    ], name="Medium_ANN")

    opt = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def build_advanced_ann(
    input_dim: int,
    learning_rate: float = 0.0008,
    dropout_rates: Tuple[float, float] = (0.3, 0.2),
    use_batch_norm: bool = True,
    l2_reg: float = 0.0
) -> keras.Model:
    """
    Model 6: Advanced ANN
    Input -> Dense(256, ReLU) -> BatchNorm -> Dropout -> Dense(128, ReLU) -> BatchNorm -> Dropout -> Dense(64, ReLU) -> Sigmoid output
    """
    reg = regularizers.l2(l2_reg) if l2_reg > 0 else None
    layer_list = [layers.Input(shape=(input_dim,), name="input_features")]

    # Block 1: 256
    layer_list.append(layers.Dense(256, activation="relu", kernel_regularizer=reg, name="dense_256"))
    if use_batch_norm:
        layer_list.append(layers.BatchNormalization(name="bn_1"))
    layer_list.append(layers.Dropout(dropout_rates[0], name="dropout_1"))

    # Block 2: 128
    layer_list.append(layers.Dense(128, activation="relu", kernel_regularizer=reg, name="dense_128"))
    if use_batch_norm:
        layer_list.append(layers.BatchNormalization(name="bn_2"))
    layer_list.append(layers.Dropout(dropout_rates[1], name="dropout_2"))

    # Block 3: 64
    layer_list.append(layers.Dense(64, activation="relu", kernel_regularizer=reg, name="dense_64"))

    # Output
    layer_list.append(layers.Dense(1, activation="sigmoid", name="output_sigmoid"))

    model = keras.Sequential(layer_list, name="Advanced_ANN")
    opt = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def train_keras_ann(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 35,
    batch_size: int = 64,
    class_weight: Optional[Dict[int, float]] = None,
    checkpoint_path: Optional[str] = None,
    patience: int = 8
) -> Tuple[keras.Model, Dict[str, list], float]:
    """Trains a Keras ANN with EarlyStopping and ReduceLROnPlateau."""
    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(2, patience // 2),
            min_lr=1e-5,
            verbose=1
        )
    ]

    start_time = time.time()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=cb_list,
        verbose=1
    )
    training_time = time.time() - start_time

    # Clean history dictionary (convert np types to native floats for JSON serialization)
    clean_history = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    return model, clean_history, training_time

def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: dict
) -> Tuple[LogisticRegression, float]:
    """Model 1: Logistic Regression."""
    logger.info("Training Logistic Regression...")
    lr_cfg = cfg.get("models", {}).get("logistic_regression", {})
    start = time.time()
    model = LogisticRegression(
        C=lr_cfg.get("C", 1.0),
        max_iter=lr_cfg.get("max_iter", 1000),
        class_weight=lr_cfg.get("class_weight", "balanced"),
        solver=lr_cfg.get("solver", "lbfgs"),
        random_state=cfg.get("data", {}).get("random_state", 42)
    )
    model.fit(X_train, y_train)
    training_time = time.time() - start
    logger.info(f"Logistic Regression trained in {training_time:.2f}s")
    return model, training_time

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: dict
) -> Tuple[RandomForestClassifier, float]:
    """Model 2: Random Forest."""
    logger.info("Training Random Forest...")
    rf_cfg = cfg.get("models", {}).get("random_forest", {})
    start = time.time()
    model = RandomForestClassifier(
        n_estimators=rf_cfg.get("n_estimators", 150),
        max_depth=rf_cfg.get("max_depth", 12),
        min_samples_split=rf_cfg.get("min_samples_split", 5),
        min_samples_leaf=rf_cfg.get("min_samples_leaf", 2),
        class_weight=rf_cfg.get("class_weight", "balanced"),
        random_state=cfg.get("data", {}).get("random_state", 42),
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    training_time = time.time() - start
    logger.info(f"Random Forest trained in {training_time:.2f}s")
    return model, training_time

def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: dict,
    class_ratio: float
) -> Tuple[xgb.XGBClassifier, float]:
    """Model 3: XGBoost."""
    logger.info("Training XGBoost...")
    xgb_cfg = cfg.get("models", {}).get("xgboost", {})
    start = time.time()
    model = xgb.XGBClassifier(
        n_estimators=xgb_cfg.get("n_estimators", 150),
        max_depth=xgb_cfg.get("max_depth", 6),
        learning_rate=xgb_cfg.get("learning_rate", 0.08),
        subsample=xgb_cfg.get("subsample", 0.8),
        colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
        scale_pos_weight=class_ratio,
        eval_metric="logloss",
        random_state=cfg.get("data", {}).get("random_state", 42),
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    training_time = time.time() - start
    logger.info(f"XGBoost trained in {training_time:.2f}s")
    return model, training_time

def train_all_models(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Executes training of all 6 mandatory models plus pre-call ablation models.
    Persists trained weights and metadata.
    """
    cfg = load_config(config_path)
    set_seed(cfg["data"]["random_state"])

    # 1. Load and Split
    df = load_raw_data(cfg["data"]["raw_path"], sep=cfg["data"]["delimiter"])
    train_df, val_df, test_df = split_dataset(
        df,
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio=cfg["data"]["val_ratio"],
        test_ratio=cfg["data"]["test_ratio"],
        random_state=cfg["data"]["random_state"]
    )

    models_dir = cfg["paths"]["models_dir"]
    artifacts_dir = cfg["paths"]["artifacts_dir"]
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    # 2. Preprocess full 16-feature set
    preprocessor_full_path = os.path.join(artifacts_dir, "preprocessor_full.joblib")
    (
        X_train_full, y_train,
        X_val_full, y_val,
        X_test_full, y_test,
        preprocessor_full
    ) = prepare_splits_data(
        train_df, val_df, test_df,
        include_duration=True,
        save_path=preprocessor_full_path
    )

    input_dim_full = X_train_full.shape[1]
    class_weights = preprocessor_full.class_weights
    class_ratio = class_weights[1] / class_weights[0]  # negative_count / positive_count

    # 3. Preprocess 15-feature set (pre-call, without duration)
    preprocessor_no_dur_path = os.path.join(artifacts_dir, "preprocessor_no_duration.joblib")
    (
        X_train_no_dur, _,
        X_val_no_dur, _,
        X_test_no_dur, _,
        preprocessor_no_dur
    ) = prepare_splits_data(
        train_df, val_df, test_df,
        include_duration=False,
        save_path=preprocessor_no_dur_path
    )
    input_dim_no_dur = X_train_no_dur.shape[1]

    # Save feature names list
    save_json(
        {"features_full": preprocessor_full.feature_names_, "features_no_duration": preprocessor_no_dur.feature_names_},
        os.path.join(artifacts_dir, "feature_names.json")
    )

    training_metadata = {}
    training_histories = {}

    # --- Model 1: Logistic Regression ---
    lr_model, lr_time = train_logistic_regression(X_train_full, y_train, cfg)
    lr_path = os.path.join(models_dir, "logistic_regression.joblib")
    save_joblib(lr_model, lr_path)
    training_metadata["Logistic Regression"] = {"time": lr_time, "path": lr_path}

    # --- Model 2: Random Forest ---
    rf_model, rf_time = train_random_forest(X_train_full, y_train, cfg)
    rf_path = os.path.join(models_dir, "random_forest.joblib")
    save_joblib(rf_model, rf_path)
    training_metadata["Random Forest"] = {"time": rf_time, "path": rf_path}

    # --- Model 3: XGBoost ---
    xgb_model, xgb_time = train_xgboost(X_train_full, y_train, cfg, class_ratio)
    xgb_path = os.path.join(models_dir, "xgboost.joblib")
    save_joblib(xgb_model, xgb_path)
    training_metadata["XGBoost"] = {"time": xgb_time, "path": xgb_path}

    # --- Model 4: Baseline ANN ---
    logger.info("Training Baseline ANN (Input -> Dense(64, ReLU) -> Sigmoid)...")
    base_ann = build_baseline_ann(
        input_dim=input_dim_full,
        learning_rate=cfg["models"]["baseline_ann"]["learning_rate"]
    )
    base_ann_path = os.path.join(models_dir, "baseline_ann.keras")
    base_ann, base_hist, base_time = train_keras_ann(
        base_ann, X_train_full, y_train, X_val_full, y_val,
        epochs=cfg["models"]["baseline_ann"]["epochs"],
        batch_size=cfg["models"]["baseline_ann"]["batch_size"],
        class_weight=class_weights,
        checkpoint_path=base_ann_path,
        patience=cfg["training"]["early_stopping_patience"]
    )
    save_keras_artifact(base_ann, base_ann_path)
    training_metadata["Baseline ANN"] = {"time": base_time, "path": base_ann_path}
    training_histories["Baseline ANN"] = base_hist

    # --- Model 5: Medium ANN ---
    logger.info("Training Medium ANN (Input -> Dense(128) -> Dropout -> Dense(64) -> Sigmoid)...")
    med_ann = build_medium_ann(
        input_dim=input_dim_full,
        learning_rate=cfg["models"]["medium_ann"]["learning_rate"],
        dropout_rate=cfg["models"]["medium_ann"]["dropout"]
    )
    med_ann_path = os.path.join(models_dir, "medium_ann.keras")
    med_ann, med_hist, med_time = train_keras_ann(
        med_ann, X_train_full, y_train, X_val_full, y_val,
        epochs=cfg["models"]["medium_ann"]["epochs"],
        batch_size=cfg["models"]["medium_ann"]["batch_size"],
        class_weight=class_weights,
        checkpoint_path=med_ann_path,
        patience=cfg["training"]["early_stopping_patience"]
    )
    save_keras_artifact(med_ann, med_ann_path)
    training_metadata["Medium ANN"] = {"time": med_time, "path": med_ann_path}
    training_histories["Medium ANN"] = med_hist

    # --- Model 6: Advanced ANN ---
    logger.info("Training Advanced ANN (Input -> Dense(256) -> BN -> Dropout -> Dense(128) -> BN -> Dropout -> Dense(64) -> Sigmoid)...")
    adv_ann = build_advanced_ann(
        input_dim=input_dim_full,
        learning_rate=cfg["models"]["advanced_ann"]["learning_rate"],
        dropout_rates=tuple(cfg["models"]["advanced_ann"]["dropout"]),
        use_batch_norm=cfg["models"]["advanced_ann"]["batch_norm"]
    )
    adv_ann_path = os.path.join(models_dir, "advanced_ann.keras")
    adv_ann, adv_hist, adv_time = train_keras_ann(
        adv_ann, X_train_full, y_train, X_val_full, y_val,
        epochs=cfg["models"]["advanced_ann"]["epochs"],
        batch_size=cfg["models"]["advanced_ann"]["batch_size"],
        class_weight=class_weights,
        checkpoint_path=adv_ann_path,
        patience=cfg["training"]["early_stopping_patience"]
    )
    save_keras_artifact(adv_ann, adv_ann_path)
    training_metadata["Advanced ANN"] = {"time": adv_time, "path": adv_ann_path}
    training_histories["Advanced ANN"] = adv_hist

    # --- Duration Ablation Models (Pre-call Models without duration) ---
    logger.info("Training Pre-call Models (Without Duration)...")
    # 1. Pre-call Medium ANN
    pre_ann = build_medium_ann(
        input_dim=input_dim_no_dur,
        learning_rate=cfg["models"]["medium_ann"]["learning_rate"],
        dropout_rate=cfg["models"]["medium_ann"]["dropout"]
    )
    pre_ann_path = os.path.join(models_dir, "precall_ann.keras")
    pre_ann, pre_hist, pre_time = train_keras_ann(
        pre_ann, X_train_no_dur, y_train, X_val_no_dur, y_val,
        epochs=cfg["models"]["medium_ann"]["epochs"],
        batch_size=cfg["models"]["medium_ann"]["batch_size"],
        class_weight=class_weights,
        checkpoint_path=pre_ann_path,
        patience=cfg["training"]["early_stopping_patience"]
    )
    save_keras_artifact(pre_ann, pre_ann_path)
    training_metadata["Pre-call ANN (No Duration)"] = {"time": pre_time, "path": pre_ann_path}
    training_histories["Pre-call ANN"] = pre_hist

    # 2. Pre-call XGBoost
    pre_xgb, pre_xgb_time = train_xgboost(X_train_no_dur, y_train, cfg, class_ratio)
    pre_xgb_path = os.path.join(models_dir, "precall_xgboost.joblib")
    save_joblib(pre_xgb, pre_xgb_path)
    training_metadata["Pre-call XGBoost (No Duration)"] = {"time": pre_xgb_time, "path": pre_xgb_path}

    # Save histories & metadata
    save_json(training_histories, os.path.join(artifacts_dir, "training_histories.json"))
    save_json(training_metadata, os.path.join(artifacts_dir, "training_metadata.json"))

    logger.info("All 6 mandatory models + ablation models trained and persisted successfully!")
    return {
        "metadata": training_metadata,
        "histories": training_histories,
        "preprocessor_full": preprocessor_full,
        "preprocessor_no_dur": preprocessor_no_dur
    }

if __name__ == "__main__":
    train_all_models()

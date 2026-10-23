"""
Hyperparameter tuning for the Artificial Neural Network using Optuna.
Optimizes ANN architecture, regularization, learning rate, batch size, and class weighting
strictly on the validation set using PR-AUC as the objective.
"""
import os
import optuna
import pandas as pd
import numpy as np
from typing import Dict, Any

# Enforce torch backend for Keras 3
os.environ.setdefault("KERAS_BACKEND", "torch")
import keras
from keras import layers, optimizers, callbacks, regularizers
from sklearn.metrics import average_precision_score

from src.utils import logger, set_seed, save_json, load_config, save_keras_artifact
from src.data_loader import load_raw_data, split_dataset
from src.preprocessing import prepare_splits_data, BankPreprocessor

def build_dynamic_ann(
    input_dim: int,
    num_layers: int,
    units_list: list,
    dropout_list: list,
    use_batch_norm: bool,
    learning_rate: float,
    l2_reg: float
) -> keras.Model:
    """Constructs dynamic ANN based on trial parameters."""
    reg = regularizers.l2(l2_reg) if l2_reg > 0 else None
    layer_list = [layers.Input(shape=(input_dim,), name="input_features")]

    for i in range(num_layers):
        u = units_list[i]
        d = dropout_list[i]
        layer_list.append(layers.Dense(u, activation="relu", kernel_regularizer=reg, name=f"dense_{i+1}_{u}"))
        if use_batch_norm:
            layer_list.append(layers.BatchNormalization(name=f"bn_{i+1}"))
        if d > 0:
            layer_list.append(layers.Dropout(d, name=f"dropout_{i+1}"))

    layer_list.append(layers.Dense(1, activation="sigmoid", name="output_sigmoid"))

    model = keras.Sequential(layer_list, name="Tuned_ANN_Trial")
    opt = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def run_ann_hyperparameter_tuning(
    n_trials: int = 12,
    config_path: str = "config/config.yaml"
) -> Dict[str, Any]:
    """Runs Optuna hyperparameter tuning strictly on validation set."""
    cfg = load_config(config_path)
    set_seed(cfg["data"]["random_state"])

    df = load_raw_data(cfg["data"]["raw_path"], sep=cfg["data"]["delimiter"])
    train_df, val_df, test_df = split_dataset(
        df,
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio=cfg["data"]["val_ratio"],
        test_ratio=cfg["data"]["test_ratio"],
        random_state=cfg["data"]["random_state"]
    )

    artifacts_dir = cfg["paths"]["artifacts_dir"]
    reports_dir = cfg["paths"]["reports_dir"]
    models_dir = cfg["paths"]["models_dir"]

    # Preprocess
    preprocessor = BankPreprocessor.load(os.path.join(artifacts_dir, "preprocessor_full.joblib"))
    X_train = preprocessor.transform(train_df.drop(columns=["y"]))
    y_train = BankPreprocessor.encode_target(train_df["y"])
    X_val = preprocessor.transform(val_df.drop(columns=["y"]))
    y_val = BankPreprocessor.encode_target(val_df["y"])

    input_dim = X_train.shape[1]
    class_weights = preprocessor.class_weights

    # Silence verbose Optuna logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        num_layers = trial.suggest_int("num_layers", 1, 3)
        units_list = []
        dropout_list = []
        for i in range(num_layers):
            units_list.append(trial.suggest_categorical(f"units_l{i+1}", [32, 64, 128, 256]))
            dropout_list.append(trial.suggest_float(f"dropout_l{i+1}", 0.0, 0.4, step=0.1))

        use_batch_norm = trial.suggest_categorical("use_batch_norm", [True, False])
        learning_rate = trial.suggest_float("learning_rate", 3e-4, 3e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        l2_reg = trial.suggest_float("l2_reg", 1e-5, 1e-2, log=True)
        use_class_weight = trial.suggest_categorical("use_class_weight", [True, False])

        cw = class_weights if use_class_weight else None

        model = build_dynamic_ann(
            input_dim=input_dim,
            num_layers=num_layers,
            units_list=units_list,
            dropout_list=dropout_list,
            use_batch_norm=use_batch_norm,
            learning_rate=learning_rate,
            l2_reg=l2_reg
        )

        early_stop = callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=0
        )

        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=20,
            batch_size=batch_size,
            class_weight=cw,
            callbacks=[early_stop],
            verbose=0
        )

        val_preds = model.predict(X_val, verbose=0).flatten()
        val_pr_auc = average_precision_score(y_val, val_preds)
        return float(val_pr_auc)

    logger.info(f"Starting Optuna hyperparameter optimization ({n_trials} trials)...")
    study = optuna.create_study(direction="maximize", study_name="ann_optimization")
    study.optimize(objective, n_trials=n_trials)

    best_trial = study.best_trial
    logger.info(f"Best Optuna Trial #{best_trial.number}: Val PR-AUC = {best_trial.value:.4f}")
    logger.info(f"Best Parameters: {best_trial.params}")

    # Save trials dataframe
    trials_df = study.trials_dataframe()
    trials_csv_path = os.path.join(reports_dir, "tuning_trials.csv")
    trials_df.to_csv(trials_csv_path, index=False)
    logger.info(f"Saved all tuning trials to: {trials_csv_path}")

    # Save best parameters
    best_params = {
        "best_trial_number": int(best_trial.number),
        "best_val_pr_auc": float(best_trial.value),
        "params": best_trial.params
    }
    save_json(best_params, os.path.join(artifacts_dir, "best_hyperparameters.json"))

    # Retrain final Tuned ANN with best parameters
    logger.info("Retraining final Tuned ANN using optimal hyperparameters...")
    p = best_trial.params
    num_layers = p["num_layers"]
    units = [p[f"units_l{i+1}"] for i in range(num_layers)]
    dropouts = [p[f"dropout_l{i+1}"] for i in range(num_layers)]

    tuned_model = build_dynamic_ann(
        input_dim=input_dim,
        num_layers=num_layers,
        units_list=units,
        dropout_list=dropouts,
        use_batch_norm=p["use_batch_norm"],
        learning_rate=p["learning_rate"],
        l2_reg=p["l2_reg"]
    )

    cw = class_weights if p["use_class_weight"] else None
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1
    )

    tuned_model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=35,
        batch_size=p["batch_size"],
        class_weight=cw,
        callbacks=[early_stop],
        verbose=1
    )

    tuned_path = os.path.join(models_dir, "tuned_ann.keras")
    save_keras_artifact(tuned_model, tuned_path)
    logger.info(f"Saved Tuned ANN to {tuned_path}")

    return best_params

if __name__ == "__main__":
    run_ann_hyperparameter_tuning(n_trials=10)

"""
Utility functions for configuration, reproducibility, serialization, and logging.
"""
import os
import random
import logging
import yaml
import json
import joblib
import numpy as np

def setup_logger(name: str = "bank_marketing", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a standard logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger

logger = setup_logger()

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Loads configuration YAML file safely."""
    if not os.path.isabs(config_path):
        # Resolve relative to project root or current working directory
        cwd = os.getcwd()
        candidate = os.path.join(cwd, config_path)
        if os.path.exists(candidate):
            config_path = candidate
        else:
            # Check relative to this file's parent directory
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            candidate2 = os.path.join(base_dir, config_path)
            if os.path.exists(candidate2):
                config_path = candidate2

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def set_seed(seed: int = 42) -> None:
    """Enforces deterministic random seed across random, numpy, and torch/keras."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def save_json(data: dict, filepath: str) -> None:
    """Saves dictionary as formatted JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved JSON artifact to {filepath}")

def load_json(filepath: str) -> dict:
    """Loads dictionary from JSON."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def save_joblib(obj, filepath: str) -> None:
    """Saves object using joblib."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    joblib.dump(obj, filepath)
    logger.info(f"Saved Joblib artifact to {filepath}")

def load_joblib(filepath: str):
    """Loads object using joblib."""
    return joblib.load(filepath)

def save_keras_artifact(model, filepath: str) -> str:
    """Saves Keras weights safely to HDF5 avoiding macOS Python 3.13 abseil mutex deadlock."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    w_path = filepath.replace(".keras", ".weights.h5") if filepath.endswith(".keras") else filepath
    model.save_weights(w_path)
    logger.info(f"Saved Keras weights artifact to {w_path}")
    return w_path

def load_keras_artifact(model, filepath: str):
    """Loads Keras model weights safely."""
    w_path = filepath.replace(".keras", ".weights.h5") if filepath.endswith(".keras") else filepath
    if not os.path.exists(w_path):
        if os.path.exists(filepath):
            w_path = filepath
        else:
            raise FileNotFoundError(f"Keras weights artifact not found at {w_path}")
    model.load_weights(w_path)
    return model

def load_keras_model_by_name(model_name: str, input_dim: int, model_path: str, artifacts_dir: str = "artifacts"):
    """Instantiates the exact Keras architecture and loads persisted weights safely."""
    if "Baseline" in model_name:
        from src.train import build_baseline_ann
        model = build_baseline_ann(input_dim)
    elif "Medium" in model_name or "Pre-call ANN" in model_name:
        from src.train import build_medium_ann
        model = build_medium_ann(input_dim)
    elif "Advanced" in model_name:
        from src.train import build_advanced_ann
        model = build_advanced_ann(input_dim)
    elif "Tuned" in model_name:
        from src.tuning import build_dynamic_ann
        hp_path = os.path.join(artifacts_dir, "best_hyperparameters.json")
        hp = load_json(hp_path)["params"]
        n_l = hp["num_layers"]
        model = build_dynamic_ann(
            input_dim=input_dim,
            num_layers=n_l,
            units_list=[hp[f"units_l{i+1}"] for i in range(n_l)],
            dropout_list=[hp[f"dropout_l{i+1}"] for i in range(n_l)],
            use_batch_norm=hp["use_batch_norm"],
            learning_rate=hp["learning_rate"],
            l2_reg=hp["l2_reg"]
        )
    else:
        from src.train import build_medium_ann
        model = build_medium_ann(input_dim)

    load_keras_artifact(model, model_path)
    return model

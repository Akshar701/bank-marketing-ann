"""
Model explainability module.
Implements SHAP global and local explanations, permutation importance,
and feature aggregation back to original input columns.
"""
import os
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional

# Enforce torch backend for Keras 3
os.environ.setdefault("KERAS_BACKEND", "torch")
import keras

from sklearn.inspection import permutation_importance
from src.utils import logger, load_config, load_joblib, load_json, save_json, load_keras_model_by_name
from src.data_loader import load_raw_data, split_dataset, ALL_EXPECTED_FEATURES
from src.preprocessing import BankPreprocessor

def compute_permutation_importance(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    is_keras: bool = False,
    n_repeats: int = 5
) -> pd.DataFrame:
    """Computes model-agnostic permutation feature importance."""
    logger.info("Computing permutation feature importance...")
    
    # Custom scoring wrapper for Keras model
    if is_keras:
        class KerasScorerWrapper:
            def __init__(self, keras_m):
                self.m = keras_m
                self.classes_ = np.array([0, 1])
            def fit(self, X, y=None):
                return self
            def predict(self, X):
                return (self.m.predict(X, verbose=0).flatten() >= 0.5).astype(int)
            def score(self, X, y):
                preds = self.predict(X)
                return float(np.mean(preds == y))
        scorer_model = KerasScorerWrapper(model)
    else:
        scorer_model = model

    perm = permutation_importance(
        scorer_model, X_val, y_val,
        n_repeats=3,
        random_state=42,
        n_jobs=1
    )

    df_perm = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std
    }).sort_values(by="importance_mean", ascending=False).reset_index(drop=True)

    return df_perm

def aggregate_importance_to_original_features(
    feature_importances: pd.DataFrame,
    original_features: List[str] = ALL_EXPECTED_FEATURES
) -> pd.DataFrame:
    """
    Aggregates one-hot encoded dummy feature importances back to their original column names.
    e.g., job_admin., job_blue-collar -> job
    """
    agg_scores = {orig: 0.0 for orig in original_features}

    for _, row in feature_importances.iterrows():
        feat = row["feature"]
        val = abs(row["importance_mean"])
        matched = False
        for orig in original_features:
            if feat == orig or feat.startswith(orig + "_"):
                agg_scores[orig] += val
                matched = True
                break
        if not matched:
            agg_scores["other"] = agg_scores.get("other", 0.0) + val

    df_agg = pd.DataFrame([
        {"feature": k, "aggregated_importance": v}
        for k, v in agg_scores.items() if v > 0
    ]).sort_values(by="aggregated_importance", ascending=False).reset_index(drop=True)

    return df_agg

def generate_shap_explanations(
    xgb_model,
    X_sample: np.ndarray,
    feature_names: List[str],
    figures_dir: str
) -> Tuple[np.ndarray, shap.Explainer]:
    """
    Generates TreeExplainer SHAP values for XGBoost (fast and exact)
    and saves global summary and bar plots.
    """
    os.makedirs(figures_dir, exist_ok=True)
    logger.info("Computing SHAP values using TreeExplainer...")
    
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_sample)

    # Global summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False, max_display=15)
    plt.title("SHAP Global Summary Plot (XGBoost)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "shap_summary_plot.png"), dpi=300)
    plt.close()

    # Global bar plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False, max_display=15)
    plt.title("SHAP Global Feature Importance (Mean |SHAP|)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "shap_importance_bar.png"), dpi=300)
    plt.close()

    logger.info("Saved SHAP summary and importance plots.")
    return shap_values, explainer

def explain_single_instance(
    explainer: shap.Explainer,
    instance_trans: np.ndarray,
    feature_names: List[str],
    top_n: int = 8
) -> List[Dict[str, Any]]:
    """
    Explains a single test prediction using SHAP local contributions.
    Returns ranked feature contributions.
    """
    shap_vals = explainer.shap_values(instance_trans.reshape(1, -1))[0]
    contributions = []

    for name, val, sv in zip(feature_names, instance_trans, shap_vals):
        contributions.append({
            "feature": name,
            "transformed_value": round(float(val), 3),
            "shap_value": round(float(sv), 4),
            "direction": "Increases Subscription" if sv > 0 else "Decreases Subscription"
        })

    # Sort by absolute SHAP impact
    contributions = sorted(contributions, key=lambda x: abs(x["shap_value"]), reverse=True)
    return contributions[:top_n]

def run_explainability_pipeline(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Runs end-to-end explainability suite and exports reports."""
    cfg = load_config(config_path)
    models_dir = cfg["paths"]["models_dir"]
    artifacts_dir = cfg["paths"]["artifacts_dir"]
    figures_dir = cfg["paths"]["figures_dir"]
    reports_dir = cfg["paths"]["reports_dir"]

    df = load_raw_data(cfg["data"]["raw_path"], sep=cfg["data"]["delimiter"])
    _, val_df, _ = split_dataset(
        df,
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio=cfg["data"]["val_ratio"],
        test_ratio=cfg["data"]["test_ratio"],
        random_state=cfg["data"]["random_state"]
    )

    pre_full = BankPreprocessor.load(os.path.join(artifacts_dir, "preprocessor_full.joblib"))
    feature_names = pre_full.feature_names_

    X_val = pre_full.transform(val_df.drop(columns=["y"]))
    y_val = BankPreprocessor.encode_target(val_df["y"])

    # Load XGBoost for SHAP
    xgb_model = load_joblib(os.path.join(models_dir, "xgboost.joblib"))
    # Load Best ANN for Permutation Importance
    ann_model = load_keras_model_by_name("Medium ANN", X_val.shape[1], os.path.join(models_dir, "medium_ann.keras"), artifacts_dir)

    # 1. SHAP with background sample
    sample_indices = np.random.choice(len(X_val), size=min(400, len(X_val)), replace=False)
    X_sample = X_val[sample_indices]
    shap_vals, explainer = generate_shap_explanations(xgb_model, X_sample, feature_names, figures_dir)

    # 2. Permutation Importance for ANN
    perm_ann = compute_permutation_importance(ann_model, X_sample, y_val[sample_indices], feature_names, is_keras=True)
    perm_ann.to_csv(os.path.join(reports_dir, "ann_permutation_importance.csv"), index=False)

    # 3. Permutation Importance for XGBoost
    perm_xgb = compute_permutation_importance(xgb_model, X_sample, y_val[sample_indices], feature_names, is_keras=False)
    perm_xgb.to_csv(os.path.join(reports_dir, "xgb_permutation_importance.csv"), index=False)

    # 4. Aggregated Feature Importance (Original 16 features)
    agg_xgb = aggregate_importance_to_original_features(perm_xgb)
    agg_ann = aggregate_importance_to_original_features(perm_ann)

    agg_merged = pd.merge(agg_xgb, agg_ann, on="feature", suffixes=("_xgb", "_ann"))
    agg_merged.to_csv(os.path.join(reports_dir, "aggregated_feature_importance.csv"), index=False)

    # Plot Aggregated Comparison
    plt.figure(figsize=(10, 6))
    x = np.arange(len(agg_merged))
    width = 0.35
    plt.bar(x - width/2, agg_merged["aggregated_importance_xgb"], width, label="XGBoost", color="#2b5c8f")
    plt.bar(x + width/2, agg_merged["aggregated_importance_ann"], width, label="Medium ANN", color="#d95f02")
    plt.xticks(x, agg_merged["feature"], rotation=40, ha="right")
    plt.ylabel("Aggregated Permutation Importance")
    plt.title("Original Feature Importance: Tree-Based vs Neural Network", fontsize=13, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "aggregated_feature_comparison.png"), dpi=300)
    plt.close()

    logger.info("Explainability analysis complete. Artifacts and plots persisted.")
    return {
        "status": "SUCCESS",
        "top_features_xgb": agg_xgb.head(5).to_dict(orient="records"),
        "top_features_ann": agg_ann.head(5).to_dict(orient="records")
    }

if __name__ == "__main__":
    run_explainability_pipeline()

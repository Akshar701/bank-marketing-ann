"""
Model evaluation, comparison, metric calculation, and visualization module.
Calculates Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Balanced Accuracy,
Specificity, Brier score, inference latency, and generates publication-quality plots.
"""
import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, List, Optional

# Enforce torch backend for Keras 3 on macOS
os.environ.setdefault("KERAS_BACKEND", "torch")
import keras

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, brier_score_loss, roc_curve, precision_recall_curve,
    classification_report
)
from sklearn.calibration import calibration_curve

from src.utils import logger, load_config, load_joblib, load_json, save_json, load_keras_model_by_name
from src.data_loader import load_raw_data, split_dataset
from src.preprocessing import prepare_splits_data, BankPreprocessor

def calculate_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    training_time: float = 0.0,
    inference_time_ms: float = 0.0
) -> Dict[str, Any]:
    """Computes all required classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    return {
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1-Score": float(f1),
        "ROC-AUC": float(roc_auc),
        "PR-AUC": float(pr_auc),
        "Balanced Accuracy": float(bal_acc),
        "Specificity": float(specificity),
        "Brier Score": float(brier),
        "Threshold": float(threshold),
        "Training Time (s)": float(training_time),
        "Inference Latency (ms/1k)": float(inference_time_ms),
        "Confusion Matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
    }

def measure_inference_speed(model, X_sample: np.ndarray, is_keras: bool, n_runs: int = 5) -> float:
    """Measures inference time in milliseconds per 1,000 samples."""
    # Warmup
    _ = model.predict(X_sample[:100], verbose=0) if is_keras else model.predict_proba(X_sample[:100])

    subset = X_sample[:min(1000, len(X_sample))]
    times = []
    for _ in range(n_runs):
        start = time.time()
        if is_keras:
            _ = model.predict(subset, verbose=0)
        else:
            _ = model.predict_proba(subset)
        times.append(time.time() - start)

    avg_sec = np.mean(times)
    # Scale to 1,000 samples
    latency_per_1k_ms = (avg_sec / len(subset)) * 1000.0 * 1000.0
    return float(latency_per_1k_ms)

def optimize_decision_threshold(
    y_val: np.ndarray,
    val_probs: np.ndarray,
    cost_matrix: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Finds optimal decision thresholds on validation data:
    1. Max F1 Threshold
    2. Minimum Business Cost Threshold
    """
    thresholds = np.linspace(0.01, 0.99, 100)
    best_f1 = -1.0
    best_f1_thresh = 0.5

    fn_cost = cost_matrix.get("false_negative_cost", 100.0) if cost_matrix else 100.0
    fp_cost = cost_matrix.get("false_positive_cost", 15.0) if cost_matrix else 15.0
    tp_benefit = cost_matrix.get("true_positive_benefit", 200.0) if cost_matrix else 200.0

    best_profit = -np.inf
    best_cost_thresh = 0.5

    f1_curve = []
    cost_curve = []

    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        f1_curve.append(f1)
        if f1 > best_f1:
            best_f1 = f1
            best_f1_thresh = t

        cm = confusion_matrix(y_val, preds)
        tn, fp, fn, tp = cm.ravel()
        # Net value = (TP * benefit) - (FP * cost) - (FN * cost)
        profit = (tp * tp_benefit) - (fp * fp_cost) - (fn * fn_cost)
        cost_curve.append(profit)
        if profit > best_profit:
            best_profit = profit
            best_cost_thresh = t

    return {
        "best_f1_threshold": float(best_f1_thresh),
        "best_f1_val_score": float(best_f1),
        "best_business_threshold": float(best_cost_thresh),
        "thresholds": thresholds.tolist(),
        "f1_scores": [float(x) for x in f1_curve],
        "business_values": [float(x) for x in cost_curve]
    }

def generate_comparison_plots(
    model_results: Dict[str, Dict[str, Any]],
    y_test: np.ndarray,
    figures_dir: str
) -> None:
    """Generates all comparison figures: ROC curves, PR curves, Metric charts, and Confusion Matrices."""
    os.makedirs(figures_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#1f78b4"]

    # 1. ROC Curves
    plt.figure(figsize=(9, 7))
    for i, (name, res) in enumerate(model_results.items()):
        fpr, tpr, _ = roc_curve(y_test, res["test_probs"])
        auc_val = res["metrics"]["ROC-AUC"]
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", lw=2, color=colors[i % len(colors)])
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.6, label="Random Chance")
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Recall)", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC) Comparison", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "roc_curves_comparison.png"), dpi=300)
    plt.close()

    # 2. Precision-Recall Curves
    plt.figure(figsize=(9, 7))
    no_skill = (y_test == 1).mean()
    plt.plot([0, 1], [no_skill, no_skill], "k--", lw=1.5, alpha=0.6, label=f"No Skill ({no_skill:.3f})")
    for i, (name, res) in enumerate(model_results.items()):
        prec, rec, _ = precision_recall_curve(y_test, res["test_probs"])
        pr_auc_val = res["metrics"]["PR-AUC"]
        plt.plot(rec, prec, label=f"{name} (PR-AUC = {pr_auc_val:.3f})", lw=2, color=colors[i % len(colors)])
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall (PR) Curves Comparison (Target Imbalance)", fontsize=14, fontweight="bold")
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "pr_curves_comparison.png"), dpi=300)
    plt.close()

    # 3. Bar Chart of Key Metrics (PR-AUC, ROC-AUC, F1, Recall)
    metric_keys = ["PR-AUC", "ROC-AUC", "F1-Score", "Recall", "Balanced Accuracy"]
    models = list(model_results.keys())
    data_matrix = [[model_results[m]["metrics"][k] for k in metric_keys] for m in models]

    df_plot = pd.DataFrame(data_matrix, index=models, columns=metric_keys)
    plt.figure(figsize=(12, 6))
    df_plot.plot(kind="bar", figsize=(12, 6), width=0.8, colormap="tab10")
    plt.title("Core Performance Metrics by Model (Evaluated at Default Threshold 0.5)", fontsize=14, fontweight="bold")
    plt.ylabel("Score", fontsize=12)
    plt.ylim(0.0, 1.05)
    plt.xticks(rotation=25, ha="right", fontsize=10)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "model_metrics_comparison.png"), dpi=300)
    plt.close()

    # 4. Confusion Matrices Grid
    n_models = len(models)
    cols = 3
    rows = int(np.ceil(n_models / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes = axes.flatten() if n_models > 1 else [axes]

    for idx, (name, res) in enumerate(model_results.items()):
        cm = confusion_matrix(y_test, (res["test_probs"] >= res["metrics"]["Threshold"]).astype(int))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[idx],
            xticklabels=["No", "Yes"], yticklabels=["No", "Yes"]
        )
        axes[idx].set_title(f"{name}\n(F1: {res['metrics']['F1-Score']:.3f}, Rec: {res['metrics']['Recall']:.3f})", fontsize=11)
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")

    # Hide extra unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "confusion_matrices.png"), dpi=300)
    plt.close()

    # 5. Calibration Curve Plot
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for i, (name, res) in enumerate(model_results.items()):
        frac_pos, mean_pred = calibration_curve(y_test, res["test_probs"], n_bins=10)
        plt.plot(mean_pred, frac_pos, "s-", label=f"{name} (Brier: {res['metrics']['Brier Score']:.3f})", color=colors[i % len(colors)])
    plt.xlabel("Mean Predicted Probability", fontsize=12)
    plt.ylabel("Fraction of Positives", fontsize=12)
    plt.title("Reliability Calibration Curves", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "calibration_curves.png"), dpi=300)
    plt.close()

    logger.info("Generated all comparison plots successfully!")

def plot_learning_curves(histories_dict: Dict[str, dict], figures_dir: str) -> None:
    """Generates training history curves for all trained ANN architectures."""
    os.makedirs(figures_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for name, hist in histories_dict.items():
        if "loss" in hist and "val_loss" in hist:
            epochs = range(1, len(hist["loss"]) + 1)
            ax1.plot(epochs, hist["loss"], label=f"{name} (Train)", linestyle="--", alpha=0.7)
            ax1.plot(epochs, hist["val_loss"], label=f"{name} (Val)", lw=2)

        if "pr_auc" in hist and "val_pr_auc" in hist:
            epochs = range(1, len(hist["pr_auc"]) + 1)
            ax2.plot(epochs, hist["pr_auc"], label=f"{name} (Train)", linestyle="--", alpha=0.7)
            ax2.plot(epochs, hist["val_pr_auc"], label=f"{name} (Val)", lw=2)

    ax1.set_title("ANN Training vs Validation Binary Cross-Entropy Loss", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_title("ANN Training vs Validation PR-AUC", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("PR-AUC")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "ann_learning_curves.png"), dpi=300)
    plt.close()
    logger.info("Saved ANN learning curves plot.")

def evaluate_all_models(config_path: str = "config/config.yaml") -> pd.DataFrame:
    """
    Loads all trained models and preprocessed test data,
    evaluates them comprehensively, and writes reports/model_comparison.csv.
    """
    cfg = load_config(config_path)
    models_dir = cfg["paths"]["models_dir"]
    artifacts_dir = cfg["paths"]["artifacts_dir"]
    reports_dir = cfg["paths"]["reports_dir"]
    figures_dir = cfg["paths"]["figures_dir"]

    # 1. Load data
    df = load_raw_data(cfg["data"]["raw_path"], sep=cfg["data"]["delimiter"])
    train_df, val_df, test_df = split_dataset(
        df,
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio=cfg["data"]["val_ratio"],
        test_ratio=cfg["data"]["test_ratio"],
        random_state=cfg["data"]["random_state"]
    )

    pre_full = BankPreprocessor.load(os.path.join(artifacts_dir, "preprocessor_full.joblib"))
    pre_no_dur = BankPreprocessor.load(os.path.join(artifacts_dir, "preprocessor_no_duration.joblib"))

    X_val_full = pre_full.transform(val_df.drop(columns=["y"]))
    X_test_full = pre_full.transform(test_df.drop(columns=["y"]))
    y_val = BankPreprocessor.encode_target(val_df["y"])
    y_test = BankPreprocessor.encode_target(test_df["y"])

    X_val_no_dur = pre_no_dur.transform(val_df.drop(columns=["y"]))
    X_test_no_dur = pre_no_dur.transform(test_df.drop(columns=["y"]))

    metadata_path = os.path.join(artifacts_dir, "training_metadata.json")
    training_meta = load_json(metadata_path) if os.path.exists(metadata_path) else {}

    model_registry = [
        ("Logistic Regression", "logistic_regression.joblib", False, False),
        ("Random Forest", "random_forest.joblib", False, False),
        ("XGBoost", "xgboost.joblib", False, False),
        ("Baseline ANN", "baseline_ann.keras", True, False),
        ("Medium ANN", "medium_ann.keras", True, False),
        ("Advanced ANN", "advanced_ann.keras", True, False),
        ("Pre-call ANN (No Duration)", "precall_ann.keras", True, True),
        ("Pre-call XGBoost (No Duration)", "precall_xgboost.joblib", False, True),
    ]

    # Check if Tuned ANN exists
    if os.path.exists(os.path.join(models_dir, "tuned_ann.weights.h5")) or os.path.exists(os.path.join(models_dir, "tuned_ann.keras")):
        model_registry.append(("Tuned ANN", "tuned_ann.weights.h5", True, False))

    evaluation_records = []
    plot_data = {}
    threshold_optimizations = {}

    for model_name, filename, is_keras, is_no_duration in model_registry:
        model_path = os.path.join(models_dir, filename)
        # Check both .keras and .weights.h5
        if not os.path.exists(model_path):
            alt_path = model_path.replace(".keras", ".weights.h5") if filename.endswith(".keras") else model_path.replace(".weights.h5", ".keras")
            if os.path.exists(alt_path):
                model_path = alt_path
            else:
                logger.warning(f"Model file not found: {model_path}. Skipping.")
                continue

        logger.info(f"Evaluating {model_name}...")
        X_val = X_val_no_dur if is_no_duration else X_val_full
        X_test = X_test_no_dur if is_no_duration else X_test_full

        if is_keras:
            model = load_keras_model_by_name(model_name, X_val.shape[1], model_path, artifacts_dir)
            val_probs = model.predict(X_val, verbose=0).flatten()
            test_probs = model.predict(X_test, verbose=0).flatten()
        else:
            model = load_joblib(model_path)
            val_probs = model.predict_proba(X_val)[:, 1]
            test_probs = model.predict_proba(X_test)[:, 1]

        t_time = training_meta.get(model_name, {}).get("time", 0.0)
        inf_lat = measure_inference_speed(model, X_test, is_keras)

        # 1. Standard evaluation at default threshold 0.5
        metrics_05 = calculate_metrics(y_test, test_probs, threshold=0.5, training_time=t_time, inference_time_ms=inf_lat)

        # 2. Threshold optimization using validation set only
        thresh_opt = optimize_decision_threshold(y_val, val_probs, cfg.get("cost_matrix"))
        threshold_optimizations[model_name] = thresh_opt

        opt_thresh = thresh_opt["best_f1_threshold"]
        # Apply validation-chosen threshold to untouched test set
        metrics_opt = calculate_metrics(y_test, test_probs, threshold=opt_thresh, training_time=t_time, inference_time_ms=inf_lat)

        record = {
            "Model": model_name,
            "Type": "Artificial Neural Network" if is_keras else "Traditional ML",
            "Features": "15 (No Duration)" if is_no_duration else "16 (Full)",
            "Accuracy (0.5)": round(metrics_05["Accuracy"], 4),
            "Precision (0.5)": round(metrics_05["Precision"], 4),
            "Recall (0.5)": round(metrics_05["Recall"], 4),
            "F1-Score (0.5)": round(metrics_05["F1-Score"], 4),
            "ROC-AUC": round(metrics_05["ROC-AUC"], 4),
            "PR-AUC": round(metrics_05["PR-AUC"], 4),
            "Balanced Acc (0.5)": round(metrics_05["Balanced Accuracy"], 4),
            "Specificity (0.5)": round(metrics_05["Specificity"], 4),
            "Brier Score": round(metrics_05["Brier Score"], 4),
            "Val-Optimized Thresh": round(opt_thresh, 3),
            "F1-Score (Opt Thresh)": round(metrics_opt["F1-Score"], 4),
            "Recall (Opt Thresh)": round(metrics_opt["Recall"], 4),
            "Precision (Opt Thresh)": round(metrics_opt["Precision"], 4),
            "Training Time (s)": round(t_time, 2),
            "Inference Latency (ms/1k)": round(inf_lat, 2)
        }
        evaluation_records.append(record)

        plot_data[model_name] = {
            "metrics": metrics_05,
            "metrics_opt": metrics_opt,
            "test_probs": test_probs,
            "is_keras": is_keras
        }

    # Build comparison DataFrame sorted by PR-AUC descending
    df_comparison = pd.DataFrame(evaluation_records)
    df_comparison = df_comparison.sort_values(by="PR-AUC", ascending=False).reset_index(drop=True)

    csv_path = os.path.join(reports_dir, "model_comparison.csv")
    df_comparison.to_csv(csv_path, index=False)
    logger.info(f"Saved full model comparison table to: {csv_path}")

    # Save threshold optimizations
    save_json(threshold_optimizations, os.path.join(artifacts_dir, "threshold_optimizations.json"))

    # Generate comparison figures
    generate_comparison_plots(plot_data, y_test, figures_dir)

    # Plot learning curves if histories exist
    hist_path = os.path.join(artifacts_dir, "training_histories.json")
    if os.path.exists(hist_path):
        histories = load_json(hist_path)
        plot_learning_curves(histories, figures_dir)

    # Log summary of winners
    top_overall = df_comparison.iloc[0]
    top_ann = df_comparison[df_comparison["Type"] == "Artificial Neural Network"].iloc[0]
    top_trad = df_comparison[df_comparison["Type"] == "Traditional ML"].iloc[0]

    logger.info("=" * 60)
    logger.info("MODEL EVALUATION SUMMARY (UNTOUCHED TEST SET)")
    logger.info(f"Top Overall Model: {top_overall['Model']} (PR-AUC: {top_overall['PR-AUC']}, ROC-AUC: {top_overall['ROC-AUC']})")
    logger.info(f"Top ANN Model:     {top_ann['Model']} (PR-AUC: {top_ann['PR-AUC']}, ROC-AUC: {top_ann['ROC-AUC']})")
    logger.info(f"Top Traditional:   {top_trad['Model']} (PR-AUC: {top_trad['PR-AUC']}, ROC-AUC: {top_trad['ROC-AUC']})")
    logger.info("=" * 60)

    return df_comparison

if __name__ == "__main__":
    evaluate_all_models()

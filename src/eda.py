"""
Exploratory Data Analysis (EDA) module.
Generates comprehensive visualizations, leakage analyses, and statistical summaries.
Saves publication-quality figures to reports/figures/.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import logger, load_config
from src.data_loader import (
    load_raw_data,
    EXPECTED_NUMERICAL,
    EXPECTED_CATEGORICAL,
    TARGET_COLUMN
)

def run_eda(config_path: str = "config/config.yaml") -> dict:
    """Executes full EDA suite and saves figures to reports/figures/."""
    cfg = load_config(config_path)
    figures_dir = cfg["paths"]["figures_dir"]
    os.makedirs(figures_dir, exist_ok=True)

    df = load_raw_data(cfg["data"]["raw_path"], sep=cfg["data"]["delimiter"])
    logger.info(f"Starting EDA on dataset of shape: {df.shape}")

    sns.set_theme(style="whitegrid")
    palette = ["#4c72b0", "#dd8452"]

    # 1. Target Distribution & Imbalance
    plt.figure(figsize=(6, 5))
    ax = sns.countplot(data=df, x="y", palette=palette, hue="y", legend=False)
    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height:,}\n({height/total:.1%})",
                    (p.get_x() + p.get_width() / 2., height / 2),
                    ha="center", va="center", fontsize=11, color="white", weight="bold")
    plt.title("Target Distribution (Bank Term Deposit Subscription)", fontsize=13, fontweight="bold")
    plt.xlabel("Subscribed ('y')", fontsize=11)
    plt.ylabel("Count", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "target_distribution.png"), dpi=300)
    plt.close()

    # 2. Numerical Distributions
    num_cols = ["age", "balance", "duration", "campaign", "pdays", "previous"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        sns.histplot(data=df, x=col, hue="y", kde=True, bins=30, ax=axes[i], palette=palette)
        axes[i].set_title(f"Distribution of {col}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "numerical_distributions.png"), dpi=300)
    plt.close()

    # 3. Outlier Boxplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        sns.boxplot(data=df, x="y", y=col, ax=axes[i], palette=palette, hue="y", legend=False)
        axes[i].set_title(f"Boxplot of {col} by Class", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "outliers_boxplots.png"), dpi=300)
    plt.close()

    # 4. Categorical Subscription Rates
    cat_cols = ["job", "education", "marital", "contact", "poutcome", "month"]
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    axes = axes.flatten()
    for i, col in enumerate(cat_cols):
        sub_rates = df.groupby(col)["y"].apply(lambda s: (s == "yes").mean() * 100).sort_values(ascending=False)
        sns.barplot(x=sub_rates.values, y=sub_rates.index, ax=axes[i], color="#2b5c8f")
        axes[i].set_title(f"Subscription Rate by {col.capitalize()} (%)", fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Subscription Rate (%)")
        for p in axes[i].patches:
            width = p.get_width()
            axes[i].annotate(f"{width:.1f}%", (width + 0.5, p.get_y() + p.get_height() / 2),
                             va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "categorical_subscriptions.png"), dpi=300)
    plt.close()

    # 5. Correlation Heatmap (Numerical)
    plt.figure(figsize=(8, 6))
    df_num = df[EXPECTED_NUMERICAL].copy()
    df_num["y_numeric"] = (df["y"] == "yes").astype(int)
    corr = df_num.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, cbar=True, square=True)
    plt.title("Correlation Matrix of Numerical Features & Target", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "correlation_matrix.png"), dpi=300)
    plt.close()

    # 6. Duration Leakage Deep-Dive
    plt.figure(figsize=(10, 5))
    df["duration_min"] = df["duration"] / 60.0
    bins = [0, 1, 3, 5, 10, 20, 60]
    labels = ["<1 min", "1-3 min", "3-5 min", "5-10 min", "10-20 min", ">20 min"]
    df["duration_bin"] = pd.cut(df["duration_min"], bins=bins, labels=labels, include_lowest=True)
    dur_sub = df.groupby("duration_bin", observed=False)["y"].apply(lambda s: (s == "yes").mean() * 100)

    ax = dur_sub.plot(kind="bar", color="#d95f02", width=0.6)
    plt.title("Subscription Rate vs Call Duration (Illustrating Target Leakage)", fontsize=13, fontweight="bold")
    plt.ylabel("Subscription Rate (%)", fontsize=11)
    plt.xlabel("Call Duration Interval", fontsize=11)
    plt.xticks(rotation=0)
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(f"{h:.1f}%", (p.get_x() + p.get_width()/2., h + 1), ha="center", fontsize=10, weight="bold")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "duration_leakage_analysis.png"), dpi=300)
    plt.close()

    # 7. Campaign Dynamics: Previous Contacts & pdays
    plt.figure(figsize=(9, 5))
    pdays_contacted = df[df["pdays"] != -1]
    pdays_uncontacted = df[df["pdays"] == -1]

    comp_df = pd.DataFrame({
        "Customer Group": ["Previously Contacted (pdays != -1)", "New Lead (pdays == -1)"],
        "Count": [len(pdays_contacted), len(pdays_uncontacted)],
        "Subscription Rate (%)": [
            (pdays_contacted["y"] == "yes").mean() * 100,
            (pdays_uncontacted["y"] == "yes").mean() * 100
        ]
    })
    ax = sns.barplot(data=comp_df, x="Customer Group", y="Subscription Rate (%)", palette=["#2b5c8f", "#7570b3"])
    plt.title("Subscription Rate by Previous Contact History", fontsize=13, fontweight="bold")
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(f"{h:.1f}%", (p.get_x() + p.get_width()/2., h + 0.5), ha="center", fontsize=11, weight="bold")
    plt.ylim(0, 30)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "campaign_relationships.png"), dpi=300)
    plt.close()

    eda_summary = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "target_yes_count": int((df["y"] == "yes").sum()),
        "target_no_count": int((df["y"] == "no").sum()),
        "imbalance_ratio": float((df["y"] == "no").sum() / (df["y"] == "yes").sum()),
        "missing_cells": int(df.isnull().sum().sum()),
        "unknown_job_count": int((df["job"] == "unknown").sum()),
        "unknown_education_count": int((df["education"] == "unknown").sum()),
        "pdays_neg1_pct": float((df["pdays"] == -1).mean() * 100),
    }

    logger.info(f"EDA completed successfully: {eda_summary}")
    return eda_summary

if __name__ == "__main__":
    run_eda()

# src/generate_performance_plots.py
"""
Generate standard ML figures for the loan default project.

Figures:
1) Before/After SMOTE class distribution
2) ROC-AUC + Precision-Recall curves
3) SHAP summary plot

Usage examples:
  python -m src.generate_performance_plots --plot smote
  python -m src.generate_performance_plots --plot curves
  python -m src.generate_performance_plots --plot shap
  python -m src.generate_performance_plots --plot all
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, precision_recall_curve,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import PROCESSED_DATA_PATH, RANDOM_STATE, TARGET_COLUMN

OUTPUTS_DIR = os.path.join(Path(__file__).resolve().parent.parent, "outputs")


def sanitize_columns(columns) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for col in columns:
        c = re.sub(r"[\[\]<>]", "_", str(col).strip())
        c = re.sub(r"\s+", "_", c)
        c = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in c)
        if c in seen:
            seen[c] += 1
            c = f"{c}_{seen[c]}"
        else:
            seen[c] = 0
        out.append(c)
    return out


def _load_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(PROCESSED_DATA_PATH)
    y  = df[TARGET_COLUMN]
    X  = df.drop(columns=[TARGET_COLUMN])
    X  = pd.get_dummies(X, drop_first=True)
    X.columns = sanitize_columns(X.columns)
    return X.astype("float32"), y


def _train_quick_model(X_train, y_train) -> XGBClassifier:
    # FIX: Removed deprecated use_label_encoder kwarg (removed in XGBoost ≥ 2.0)
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def plot_smote_distribution(X, y) -> None:
    """Bar chart showing class balance before and after SMOTE."""
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        print("imblearn not installed — skipping SMOTE plot")
        return

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    smote = SMOTE(random_state=RANDOM_STATE)
    _, y_res = smote.fit_resample(X_train, y_train)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (label, series) in zip(axes, [("Before SMOTE", y_train), ("After SMOTE", y_res)]):
        counts = series.value_counts().sort_index()
        ax.bar(["Repay (0)", "Default (1)"], counts.values, color=["#22c55e", "#ef4444"])
        ax.set_title(label)
        ax.set_ylabel("Count")
        for i, v in enumerate(counts.values):
            ax.text(i, v + 50, str(v), ha="center", va="bottom", fontsize=9)
    fig.suptitle("Class Distribution — SMOTE Resampling", fontsize=13)
    plt.tight_layout()

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    out = os.path.join(OUTPUTS_DIR, "smote_class_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SMOTE distribution plot saved → {out}")


def plot_roc_pr_curves(X, y) -> None:
    """ROC-AUC and Precision-Recall curves on the test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    model  = _train_quick_model(X_train, y_train)
    probs  = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _   = roc_curve(y_test, probs)
    prec, rec, _  = precision_recall_curve(y_test, probs)
    roc_auc       = roc_auc_score(y_test, probs)
    avg_prec      = average_precision_score(y_test, probs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(fpr, tpr, color="#3b82f6", lw=2, label=f"ROC-AUC = {roc_auc:.4f}")
    ax1.plot([0, 1], [0, 1], "k--", lw=1)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve — XGBoost")
    ax1.legend(loc="lower right")

    ax2.plot(rec, prec, color="#f59e0b", lw=2, label=f"Avg Precision = {avg_prec:.4f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve — XGBoost")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    out = os.path.join(OUTPUTS_DIR, "roc_pr_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"ROC + PR curves saved → {out}")


def plot_shap_summary(X, y) -> None:
    """SHAP feature importance summary plot."""
    try:
        import shap
    except ImportError:
        print("shap not installed — skipping SHAP summary plot")
        return

    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    model   = _train_quick_model(X_train, y_train)
    X_samp  = X_test.sample(min(500, len(X_test)), random_state=42)

    explainer  = shap.TreeExplainer(model)
    sv         = explainer.shap_values(X_samp)
    if isinstance(sv, list):
        sv = sv[1]

    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X_samp, show=False)
    plt.tight_layout()

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    out = os.path.join(OUTPUTS_DIR, "shap_summary_plot.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary plot saved → {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate performance plots")
    parser.add_argument(
        "--plot",
        choices=["smote", "curves", "shap", "all"],
        default="all",
        help="Which plot(s) to generate",
    )
    args = parser.parse_args()

    X, y = _load_data()

    if args.plot in ("smote", "all"):
        plot_smote_distribution(X, y)
    if args.plot in ("curves", "all"):
        plot_roc_pr_curves(X, y)
    if args.plot in ("shap", "all"):
        plot_shap_summary(X, y)


if __name__ == "__main__":
    main()

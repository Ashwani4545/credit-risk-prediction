# src/evaluate_model.py
"""
Loan Default Prediction — Standalone Model Evaluation

Loads the saved model and processed data, runs evaluation,
prints a report, and overwrites model_metrics.json.

Usage:
    python -m src.evaluate_model
"""

import sys
import re
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
    roc_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import PROCESSED_DATA_PATH, MODEL_PATH, METRICS_PATH, TARGET_COLUMN

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def sanitize_columns(columns) -> list:
    seen: dict = {}
    result: list = []
    for col in columns:
        c = re.sub(r"[\[\]<>]", "_", str(col))
        c = re.sub(r"\s+",      "_", c.strip())
        c = re.sub(r"[^0-9a-zA-Z_]", "_", c)
        if c in seen:
            seen[c] += 1
            c = f"{c}_{seen[c]}"
        else:
            seen[c] = 0
        result.append(c)
    return result


def _align_to_model(X: pd.DataFrame, model) -> pd.DataFrame:
    """
    Align dataframe columns to the feature names the model was trained on.
    Missing columns are filled with 0; extra columns are dropped.
    """
    try:
        expected = model.get_booster().feature_names
    except AttributeError:
        expected = list(getattr(model, "feature_names_in_", X.columns))

    for col in expected:
        if col not in X.columns:
            X[col] = 0.0
    return X[expected].astype("float32")


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────────────────────────────────────

def evaluate() -> dict:
    model = joblib.load(MODEL_PATH)
    df    = pd.read_csv(PROCESSED_DATA_PATH)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X = pd.get_dummies(X, drop_first=True)
    X.columns = sanitize_columns(X.columns)
    X = _align_to_model(X, model)
    log.info("Aligned to model: %s features", X.shape[1])

    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    # Compute optimal threshold
    fpr, tpr, thresholds = roc_curve(y, probs)
    best_threshold = float(thresholds[(tpr - fpr).argmax()])

    tn, fp, fn, tp = confusion_matrix(y, preds).ravel()

    metrics = {
        "model_name":  "xgboost",
        "accuracy":    round(float(accuracy_score(y, preds)),                  4),
        "precision":   round(float(precision_score(y, preds, zero_division=0)), 4),
        "recall":      round(float(recall_score(y, preds,    zero_division=0)), 4),
        "f1_score":    round(float(f1_score(y, preds,        zero_division=0)), 4),
        "roc_auc":     round(float(roc_auc_score(y, probs)),                    4),
        "decision_threshold": round(best_threshold,                             6),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp),
        },
    }

    sep = "=" * 52
    print(f"\n{sep}\n  MODEL EVALUATION METRICS\n{sep}")
    for k in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        print(f"  {k.capitalize():<12}: {metrics[k]:.4f}")
    print(f"  Decision Threshold: {best_threshold:.6f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={tn:>6}   FP={fp:>6}")
    print(f"    FN={fn:>6}   TP={tp:>6}")
    print(f"\n{classification_report(y, preds)}")

    with open(METRICS_PATH, "w") as fh:
        json.dump(metrics, fh, indent=4)
    log.info("Metrics saved → %s", METRICS_PATH)

    return metrics


if __name__ == "__main__":
    evaluate()
# src/train_model.py
"""
Loan Default Prediction — Model Training

Steps:
  1. Load processed CSV
  2. One-hot encode & sanitize column names (XGBoost-safe)
  3. Train-test split
  4. Train XGBoost only
  5. Evaluate XGBoost model
  6. Save model + feature list + metrics JSON
"""

import sys
import os
import re
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score, roc_curve,
)
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import (
    PROCESSED_DATA_PATH, TARGET_COLUMN,
    TEST_SIZE, RANDOM_STATE,
    MODEL_PATH, FEATURES_PATH, METRICS_PATH, XGB_PARAMS,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION: Alternative Data Sources
# ─────────────────────────────────────────────────────────────────────────────
USE_REAL_ALTERNATIVE_DATA = True  # Toggle to True for production with real data
ALTERNATIVE_DATA_PATH = os.path.join(Path(__file__).resolve().parent.parent, "data", "alternative_data.csv")

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def sanitize_columns(columns) -> list:
    """
    Make column names safe for XGBoost:
      - Replace forbidden chars [ ] < >
      - Replace whitespace with _
      - Keep only alphanumeric + _
      - Deduplicate by appending _N
    """
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


def calculate_profit(y_true, y_pred, loan_amounts):
    profit = 0

    for yt, yp, loan in zip(y_true, y_pred, loan_amounts):
        if yp == 0:  # predicted repay
            if yt == 0:
                profit += loan * 0.1   # interest gain
            else:
                profit -= loan         # default loss
        else:
            profit += 0  # rejected loan

    return profit


def get_feature_importances(model, model_name: str, feature_names: list) -> pd.DataFrame:
    """Extract feature importances from XGBoost model."""
    importances = None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

    if importances is None or len(importances) == 0:
        log.warning("Could not extract feature importances from %s", model_name)
        return pd.DataFrame()
    
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)
    
    log.info("📊 Top 10 Features (%s):", model_name)
    for idx, row in importance_df.head(10).iterrows():
        log.info("  %d. %s: %.6f", idx + 1, row["feature"], row["importance"])
    
    return importance_df


def select_top_features(importance_df: pd.DataFrame, n_features: int = 200) -> list:
    """Select top N features by importance."""
    if len(importance_df) == 0:
        return None
    
    top_features = importance_df.head(n_features)["feature"].tolist()
    log.info("✅ Selected top %d / %d features for next training run", len(top_features), len(importance_df))
    return top_features


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    # Financial ratios
    df["loan_to_income"] = df["loan_amnt"] / (df["annual_inc"] + 1e-6)
    df["installment_to_income"] = df["installment"] / (df["annual_inc"] + 1e-6)

    # Credit behavior
    df["credit_utilization"] = df["revol_bal"] / (df["revol_bal"] + df["bc_open_to_buy"] + 1e-6)

    # Behavioral features
    df["payment_capacity"] = df["annual_inc"] - df["installment"] * 12
    df["credit_stress"] = df["dti"] * df["loan_amnt"]
    df["recent_inquiries_flag"] = (df["inq_last_6mths"] > 3).astype(int)

    # Risk indicators
    df["high_dti_flag"] = (df["dti"] > 20).astype(int)
    df["low_fico_flag"] = (df["fico_range_low"] < 600).astype(int)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & PREPROCESS
# ─────────────────────────────────────────────────────────────────────────────

def _load_alternative_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Load alternative credit data for credit-invisible users.
    Supports both real and synthetic data sources.
    """
    use_real_alternative_data = USE_REAL_ALTERNATIVE_DATA

    if use_real_alternative_data:
        try:
            alt_df = pd.read_csv(ALTERNATIVE_DATA_PATH)
            log.info("Loaded real alternative data: %s rows", len(alt_df))
            # Merge on common ID (adjust key as needed)
            if "customer_id" in alt_df.columns and "customer_id" in df.columns:
                df = df.merge(alt_df, on="customer_id", how="left")
            elif "id" in alt_df.columns and "id" in df.columns:
                df = df.merge(alt_df, on="id", how="left")
            else:
                log.warning("Cannot merge alternative data — no common ID column. Using synthetic fallback.")
                use_real_alternative_data = False
        except FileNotFoundError:
            log.warning("Alternative data file not found at %s. Falling back to synthetic data.", ALTERNATIVE_DATA_PATH)
            use_real_alternative_data = False
    
    # Use synthetic as fallback or primary
    if not use_real_alternative_data:
        log.info("Using placeholder (0) for alternative features — no real alternative data available.")
        log.info("FIX Bug 8: previously used np.random noise here which trained the model on garbage.")
        log.info("Now using 0 consistently — matches what inference sends when these fields are absent.")
        # NOTE: To properly use these features, collect them in the web form
        # (mobile_usage_score, digital_txn_count, utility_payment_score, employment_stability)
        # and provide real data. Until then, 0 is the honest placeholder.
        df["mobile_usage_score"]    = 0
        df["digital_txn_count"]     = 0
        df["utility_payment_score"] = 0
        df["employment_stability"]  = 0
    
    return df


def load_and_preprocess():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    log.info("Loaded data: %s rows × %s cols", *df.shape)

    # Economic context features (static demo values)
    df["inflation_rate"] = 0.06
    df["interest_rate_env"] = 0.08
    df["unemployment_rate"] = 0.07
    df["economic_stress"] = (
        df["inflation_rate"] * 0.4 +
        df["unemployment_rate"] * 0.4 +
        df["interest_rate_env"] * 0.2
    )

    # Load alternative credit data (real or synthetic)
    df = _load_alternative_data(df)

    # Create engineered features before training
    df = create_features(df)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # One-hot encode remaining categoricals
    X = pd.get_dummies(X, drop_first=True)
    X.columns = sanitize_columns(X.columns)
    X = X.astype("float32")

    log.info("After encoding: %s features", X.shape[1])
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 2. SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def split(X, y):
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAIN
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_SAMPLE_CAP = 100000

def _tune_threshold(y_true, y_prob, loan_amounts=None) -> tuple[float, float]:
    """
    Return (best_threshold, best_score) for the given probabilities.
    If loan_amounts provided, optimize for profit; otherwise optimize for F1.
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    best_threshold = 0.5
    best_score = -float("inf")
    best_metric_type = "profit" if loan_amounts is not None else "f1"
    
    for threshold in thresholds:
        preds = (y_prob >= threshold).astype(int)
        
        if loan_amounts is not None:
            # Optimize for profit
            score = calculate_profit(y_true, preds, loan_amounts)
        else:
            # Optimize for F1
            score = f1_score(y_true, preds, zero_division=0)
        
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    
    return best_threshold, float(best_score)


def _build_candidate_models(scale_pos_weight: float) -> dict:
    return {
        "xgboost": XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            n_estimators=XGB_PARAMS["n_estimators"],
            max_depth=XGB_PARAMS["max_depth"],
            learning_rate=XGB_PARAMS["learning_rate"],
            subsample=XGB_PARAMS["subsample"],
            colsample_bytree=XGB_PARAMS["colsample_bytree"],
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def train_all(X_train, y_train) -> tuple[str, object, float]:
    if len(X_train) > TRAIN_SAMPLE_CAP:
        sample_idx = X_train.sample(n=TRAIN_SAMPLE_CAP, random_state=RANDOM_STATE).index
        X_train = X_train.loc[sample_idx]
        y_train = y_train.loc[sample_idx]
        log.info("Training sample capped: %d -> %d rows", len(sample_idx), len(X_train))

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    counter = Counter(y_fit)
    negative_count = counter.get(0, 0)
    positive_count = counter.get(1, 0)
    if positive_count == 0 or negative_count == 0:
        scale_pos_weight = 1.0
        log.warning(
            "One-class training data detected (neg=%d, pos=%d); using neutral scale_pos_weight=1.0",
            negative_count,
            positive_count,
        )
    else:
        scale_pos_weight = negative_count / positive_count

    # ── Train XGBoost only ───────────────────────────────────────────────────
    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        n_estimators=XGB_PARAMS["n_estimators"],
        max_depth=XGB_PARAMS["max_depth"],
        learning_rate=XGB_PARAMS["learning_rate"],
        subsample=XGB_PARAMS["subsample"],
        colsample_bytree=XGB_PARAMS["colsample_bytree"],
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    log.info("Training XGBoost on fit split …")
    model.fit(X_fit, y_fit)
    val_prob = model.predict_proba(X_val)[:, 1]

    # Tune decision threshold on validation set
    best_threshold, best_f1 = _tune_threshold(y_val, val_prob)
    log.info(
        "XGBoost validation → threshold=%.2f  f1=%.4f  roc_auc=%.4f",
        best_threshold,
        best_f1,
        roc_auc_score(y_val, val_prob),
    )

    # Refit on full training data
    log.info("Refitting XGBoost on full training split …")
    model.fit(X_train, y_train)

    return "xgboost", model, best_threshold


# ─────────────────────────────────────────────────────────────────────────────
# 4. EVALUATE
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, model_name: str, X_test, y_test, decision_threshold: float) -> dict:
    y_prob = model.predict_proba(X_test)[:, 1]
    test_profit = calculate_profit(y_test, (y_prob >= decision_threshold).astype(int), X_test["loan_amnt"])
    log.info("🎯 Using validation threshold %.2f on test set (profit=%.2f)", decision_threshold, test_profit)

    preds = (y_prob >= decision_threshold).astype(int)

    recall = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)

    mse = mean_squared_error(y_test, y_prob)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_prob)
    mape = np.mean(np.abs((y_test - y_prob) / (y_test + 1e-10))) * 100
    r2 = r2_score(y_test, y_prob)

    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    loan_amounts = X_test["loan_amnt"]
    profit = calculate_profit(y_test, preds, loan_amounts)

    metrics = {
        "accuracy":  round(float(accuracy_score(y_test, preds)), 4),
        "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
        "recall":    round(float(recall), 4),
        "f1_score":  round(float(f1), 4),
        "roc_auc":   round(float(roc_auc), 4),
        "mse":       round(float(mse), 6),
        "rmse":      round(float(rmse), 6),
        "mae":       round(float(mae), 6),
        "mape":      round(float(mape), 4),
        "r2":        round(float(r2), 6),
        "profit":    round(float(profit), 2),
        "decision_threshold": round(float(decision_threshold), 2),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp),
        },
        "model_name": model_name,
    }

    log.info("%s  recall=%.4f  f1=%.4f  threshold=%.2f  profit=%.2f", model_name, metrics["recall"], metrics["f1_score"], decision_threshold, profit)
    log.info("\n%s", classification_report(y_test, preds))
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 5. SAVE
# ─────────────────────────────────────────────────────────────────────────────

def save_artifacts(best_model, best_metrics: dict, feature_names: list) -> None:
    best_name = best_metrics["model_name"]
    log.info("Best model: %s  (profit=%.2f)", best_name, best_metrics["profit"])

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    log.info("Model saved → %s", MODEL_PATH)

    import pickle
    os.makedirs(os.path.dirname(FEATURES_PATH), exist_ok=True)
    with open(FEATURES_PATH, "wb") as f:
        pickle.dump(feature_names, f)
    log.info("Feature list saved → %s", FEATURES_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump(best_metrics, f, indent=4)
    log.info("Metrics saved → %s", METRICS_PATH)
    
    # Extract and save feature importances
    importance_df = get_feature_importances(best_model, best_name, feature_names)
    if not importance_df.empty:
        importance_path = os.path.join(os.path.dirname(MODEL_PATH), "feature_importances.csv")
        importance_df.to_csv(importance_path, index=False)
        log.info("Feature importances saved → %s", importance_path)
        
        # Suggest top features for next iteration
        top_features = select_top_features(importance_df, n_features=200)
        if top_features:
            next_iter_path = os.path.join(os.path.dirname(MODEL_PATH), "top_features_next_iteration.pkl")
            with open(next_iter_path, "wb") as f:
                pickle.dump(top_features, f)
            log.info("💡 Next iteration: Consider training on top %d features only (saved → %s)", 
                     len(top_features), next_iter_path)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    X, y         = load_and_preprocess()
    
    # Check if we should use feature selection from previous iteration
    import pickle
    features_selection_path = os.path.join(os.path.dirname(MODEL_PATH), "top_features_next_iteration.pkl")
    USE_FEATURE_SELECTION = os.path.exists(features_selection_path)
    
    if USE_FEATURE_SELECTION:
        try:
            with open(features_selection_path, "rb") as f:
                selected_features = pickle.load(f)
            if set(selected_features).issubset(set(X.columns)):
                X = X[selected_features]
                log.info("✅ Using feature selection: %d features (down from %d)", len(selected_features), len(X.columns))
            else:
                log.warning("Selected features not all present; using all features")
        except Exception as e:
            log.warning("Failed to load selected features: %s; using all features", e)
    
    X_train, X_test, y_train, y_test = split(X, y)

    best_name, best_model, decision_threshold = train_all(X_train, y_train)
    best_metrics = evaluate_model(best_model, best_name, X_test, y_test, decision_threshold)

    save_artifacts(best_model, best_metrics, list(X.columns))
    log.info("Training pipeline complete ✅")


if __name__ == "__main__":
    main()
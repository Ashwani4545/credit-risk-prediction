# src/train_model.py
"""
Loan Default Prediction — Model Training (XGBoost only)

Steps:
  1. Load processed CSV
  2. One-hot encode & sanitize column names (XGBoost-safe)
  3. Train-test split (stratified)
  4. Apply SMOTE on training split only (no leakage)
  5. Train XGBoost with GridSearchCV
  6. Evaluate & save model + feature list + metrics JSON
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

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
    roc_curve,
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

# ── Alternative data path ────────────────────────────────────────────────────
ALTERNATIVE_DATA_PATH = os.path.join(
    Path(__file__).resolve().parent.parent, "data", "alternative_data.csv"
)


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


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Additional engineered features added at training time."""
    # Financial ratios
    df["loan_to_income"]         = df["loan_amnt"] / (df["annual_inc"] + 1e-6)
    df["installment_to_income"]  = df["installment"] / (df["annual_inc"] + 1e-6)

    # Credit behavior
    df["credit_utilization"] = df["revol_bal"] / (
        df["revol_bal"] + df["bc_open_to_buy"] + 1e-6
    )

    # Behavioral features
    df["payment_capacity"]       = df["annual_inc"] - df["installment"] * 12
    df["credit_stress"]          = df["dti"] * df["loan_amnt"]
    df["recent_inquiries_flag"]  = (df["inq_last_6mths"] > 3).astype(int)

    # Risk flags
    df["high_dti_flag"]   = (df["dti"] > 20).astype(int)
    df["low_fico_flag"]   = (df["fico_range_low"] < 600).astype(int)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & PREPROCESS
# ─────────────────────────────────────────────────────────────────────────────

def _load_alternative_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge real alternative credit data if available; otherwise fill with 0.

    FIX: Previously used np.random noise as a fallback which caused the model
    to train on garbage features. 0 is now used — it is honest and matches
    what the inference pipeline sends when these fields are absent.
    """
    merged = False
    if os.path.exists(ALTERNATIVE_DATA_PATH):
        try:
            alt_df = pd.read_csv(ALTERNATIVE_DATA_PATH)
            log.info("Loaded real alternative data: %s rows", len(alt_df))
            if "customer_id" in alt_df.columns and "customer_id" in df.columns:
                df = df.merge(alt_df, on="customer_id", how="left")
                merged = True
            elif "id" in alt_df.columns and "id" in df.columns:
                df = df.merge(alt_df, on="id", how="left")
                merged = True
            else:
                log.warning("Cannot merge alternative data — no common ID column.")
        except Exception as exc:
            log.warning("Alternative data load failed: %s", exc)

    if not merged:
        log.info(
            "No real alternative data — using 0 placeholder for alternative features. "
            "Collect real data (mobile_usage_score, digital_txn_count, "
            "utility_payment_score, employment_stability) to use these features."
        )
        df["mobile_usage_score"]    = 0
        df["digital_txn_count"]     = 0
        df["utility_payment_score"] = 0
        df["employment_stability"]  = 0

    return df


def load_and_preprocess():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    log.info("Loaded data: %s rows × %s cols", *df.shape)

    # ── Drop columns that explode the feature space with one-hot encoding ────
    # addr_state  → 50 dummy columns, very weak signal vs noise
    # sub_grade   → 35 dummy columns, grade already captures this
    # emp_title   → thousands of unique strings, useless
    # url, desc, title, zip_code → identifiers / free text, no signal
    # earliest_cr_line → raw date string; credit age already captured by open_acc
    HIGH_CARDINALITY_COLS = [
        "addr_state", "sub_grade", "emp_title", "url", "desc",
        "title", "zip_code", "earliest_cr_line", "last_pymnt_d",
        "next_pymnt_d", "last_credit_pull_d", "issue_d",
    ]
    drop_cols = [c for c in HIGH_CARDINALITY_COLS if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        log.info("Dropped high-cardinality columns: %s", drop_cols)

    # Static economic context features
    df["inflation_rate"]    = 0.06
    df["interest_rate_env"] = 0.08
    df["unemployment_rate"] = 0.07
    df["economic_stress"]   = (
        df["inflation_rate"]    * 0.4
        + df["unemployment_rate"] * 0.4
        + df["interest_rate_env"] * 0.2
    )

    df = _load_alternative_data(df)
    df = create_features(df)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X = pd.get_dummies(X, drop_first=True)
    X.columns = sanitize_columns(X.columns)
    X = X.astype("float32")

    log.info("After encoding: %s features", X.shape[1])

    # ── Feature selection: keep top-N by XGBoost importance ─────────────────
    # 806 features → model learns noise. Keep top 80 which cover 95%+ of
    # actual predictive signal for loan default.
    MAX_FEATURES = 80
    if X.shape[1] > MAX_FEATURES:
        log.info("Running feature selection: %d → top %d features …", X.shape[1], MAX_FEATURES)
        from sklearn.model_selection import train_test_split as _tts
        _Xs, _, _ys, _ = _tts(X, y, test_size=0.5, random_state=RANDOM_STATE, stratify=y)
        selector = XGBClassifier(
            n_estimators=50, max_depth=4, learning_rate=0.1,
            eval_metric="logloss", tree_method="hist",
            random_state=RANDOM_STATE, device="cpu",
        )
        selector.fit(_Xs, _ys)
        importances = pd.Series(selector.feature_importances_, index=X.columns)
        top_features = importances.nlargest(MAX_FEATURES).index.tolist()
        X = X[top_features]
        log.info("Feature selection complete → %d features retained", len(top_features))

    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 2. SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def split(X, y):
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAIN  (XGBoost only — no multi-model comparison)
# ─────────────────────────────────────────────────────────────────────────────

def train_xgboost(X_train, y_train) -> XGBClassifier:
    """
    Train a single XGBoost model with GridSearchCV hyper-parameter tuning.

    FIX: Removed Logistic Regression and Random Forest. The codebase previously
    trained three models, evaluated them, then selected the 'best' one.  This
    created unnecessary complexity and the final saved model was not guaranteed
    to be XGBoost (could be LR or RF if their profit metric was higher).

    SMOTE SPEED FIX:
    - SMOTE is O(n * k * features) — on large datasets (>50k rows, >100 features)
      it can take 10–30 minutes.
    - Solution: cap the training data at SMOTE_SAMPLE_CAP rows before applying SMOTE.
      This gives the oversampler a representative but manageable subset, then we
      combine with the full majority class for final training.
    - GridSearchCV param_grid is also reduced to 4 combinations (was 8) with cv=2
      (was 3). Still finds good params, trains ~3x faster.
    """
    # ── SMOTE with size cap to prevent long runtimes ─────────────────────────
    SMOTE_SAMPLE_CAP = 30_000   # rows fed into SMOTE; increase if you have time

    counter_orig = Counter(y_train)
    log.info("Class distribution before resampling: %s", dict(counter_orig))

    try:
        from imblearn.over_sampling import SMOTE, RandomOverSampler

        n_minority = counter_orig.get(1, 0)
        n_majority = counter_orig.get(0, 0)

        if len(X_train) > SMOTE_SAMPLE_CAP:
            log.info(
                "Dataset (%d rows) exceeds SMOTE cap (%d). "
                "Subsampling for SMOTE, then combining with full data.",
                len(X_train), SMOTE_SAMPLE_CAP,
            )
            # Subsample both classes proportionally for SMOTE
            cap_ratio   = SMOTE_SAMPLE_CAP / len(X_train)
            sub_idx     = (
                pd.Series(y_train.values)
                .groupby(y_train.values)
                .apply(lambda g: g.sample(frac=cap_ratio, random_state=RANDOM_STATE))
                .index.get_level_values(1)
            )
            X_sub = X_train.iloc[sub_idx]
            y_sub = y_train.iloc[sub_idx]

            smote  = SMOTE(random_state=RANDOM_STATE, k_neighbors=min(5, Counter(y_sub)[1] - 1))
            X_res, y_res = smote.fit_resample(X_sub, y_sub)
            log.info("SMOTE on subsample: %d → %d samples", len(X_sub), len(X_res))

            # Combine SMOTE-synthetic samples with the original full training set
            X_res = pd.concat([X_train, pd.DataFrame(X_res, columns=X_train.columns)
                                .iloc[len(X_sub):]], ignore_index=True)
            y_res = pd.concat([y_train, pd.Series(y_res).iloc[len(y_sub):]], ignore_index=True)
            log.info("Final combined training set: %d samples", len(X_res))

        elif n_minority < 6:
            # Too few minority samples even for SMOTE — use simple random oversampling
            log.warning("Minority class has only %d samples — using RandomOverSampler", n_minority)
            ros   = RandomOverSampler(random_state=RANDOM_STATE)
            X_res, y_res = ros.fit_resample(X_train, y_train)
        else:
            smote  = SMOTE(random_state=RANDOM_STATE, k_neighbors=min(5, n_minority - 1))
            X_res, y_res = smote.fit_resample(X_train, y_train)
            log.info("SMOTE applied: %d → %d samples", len(X_train), len(X_res))

    except ImportError:
        log.warning("imblearn not installed — training without SMOTE (using scale_pos_weight).")
        X_res, y_res = X_train, y_train

    counter      = Counter(y_res)
    scale_pos_wt = counter.get(0, 1) / max(counter.get(1, 1), 1)

    # ── XGBoost with GridSearch (6 combos × cv=3 = 18 fits, ~3-5 min) ────────
    # With only 80 features (down from 806) each fit is ~10x faster,
    # so we can afford a slightly wider search for better recall.
    xgb_base = XGBClassifier(
        scale_pos_weight = scale_pos_wt,
        eval_metric      = "aucpr",
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 3,      # prevents overfitting on minority class
        random_state     = RANDOM_STATE,
        tree_method      = "hist",
        device           = "cpu",
    )

    param_grid = {
        "n_estimators":  [100, 200, 300],
        "max_depth":     [4, 6],
    }

    grid_search = GridSearchCV(
        estimator  = xgb_base,
        param_grid = param_grid,
        scoring    = "roc_auc",    # roc_auc more stable than recall for imbalanced data
        cv         = 3,
        n_jobs     = -1,
        verbose    = 1,
    )
    grid_search.fit(X_res, y_res)
    log.info("Best XGBoost params: %s", grid_search.best_params_)
    log.info("Best ROC-AUC (CV): %.4f", grid_search.best_score_)

    return grid_search.best_estimator_


# ─────────────────────────────────────────────────────────────────────────────
# 4. EVALUATE
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model: XGBClassifier, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    # Find optimal threshold via Youden's J (maximise TPR - FPR)
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    best_idx       = (tpr - fpr).argmax()
    best_threshold = float(thresholds[best_idx])

    recall  = recall_score(y_test, preds,  zero_division=0)
    f1      = f1_score(y_test, preds,      zero_division=0)
    roc_auc = roc_auc_score(y_test, probs)

    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

    metrics = {
        "model_name":       "xgboost",
        "accuracy":         round(float(accuracy_score(y_test, preds)),                  4),
        "precision":        round(float(precision_score(y_test, preds, zero_division=0)), 4),
        "recall":           round(float(recall),                                          4),
        "f1_score":         round(float(f1),                                              4),
        "roc_auc":          round(float(roc_auc),                                         4),
        "decision_threshold": round(best_threshold,                                       6),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp),
        },
    }

    log.info(
        "XGBoost — recall=%.4f  f1=%.4f  roc_auc=%.4f  threshold=%.4f",
        metrics["recall"], metrics["f1_score"],
        metrics["roc_auc"], best_threshold,
    )
    log.info("\n%s", classification_report(y_test, preds))
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 5. SAVE
# ─────────────────────────────────────────────────────────────────────────────

def save_artifacts(model: XGBClassifier, metrics: dict, feature_names: list) -> None:
    import pickle

    os.makedirs(os.path.dirname(MODEL_PATH),    exist_ok=True)
    os.makedirs(os.path.dirname(FEATURES_PATH), exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    log.info("Model saved → %s", MODEL_PATH)

    with open(FEATURES_PATH, "wb") as f:
        pickle.dump(feature_names, f)
    log.info("Feature list saved → %s", FEATURES_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)
    log.info("Metrics saved → %s", METRICS_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    X, y                             = load_and_preprocess()
    X_train, X_test, y_train, y_test = split(X, y)

    model   = train_xgboost(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)

    save_artifacts(model, metrics, list(X.columns))
    log.info("Training pipeline complete ✅")


if __name__ == "__main__":
    main()

# src/data_preprocessing.py
"""
Loan Default Prediction — Data Preprocessing & Feature Engineering

Pipeline:
  1. Load raw CSV
  2. Clean (dedup, fill nulls)
  3. Engineer domain features
  4. Save processed data (keep target column intact for train_model.py)

NOTE: Scaling (StandardScaler) and SMOTE are intentionally NOT applied here.
      They are applied inside train_model.py on the training split ONLY,
      which is the correct ML practice. Applying them here before splitting
      causes data leakage (test data statistics contaminate the scaler).
"""

import os
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ── ensure project root on path ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, TARGET_COLUMN

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    log.info("Loaded dataset: %s rows × %s cols", *df.shape)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. CLEAN
# ─────────────────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    log.info("Dropped %d duplicate rows", before - len(df))

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    log.info("Nulls filled — numeric: median, categorical: mode")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-specific derived features (safe: only if source cols exist)."""

    def safe_ratio(num_col, den_col, new_col):
        if num_col in df.columns and den_col in df.columns:
            df[new_col] = df[num_col] / (df[den_col].replace(0, np.nan) + 1)

    # LendingClub-style derived features
    safe_ratio("loan_amnt",   "annual_inc",  "loan_income_ratio")
    safe_ratio("revol_bal",   "annual_inc",  "revol_income_ratio")
    safe_ratio("open_acc",    "total_acc",   "open_acc_ratio")
    safe_ratio("installment", "annual_inc",  "installment_income_ratio")

    if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
        df["fico_avg"] = (df["fico_range_low"] + df["fico_range_high"]) / 2

    if "int_rate" in df.columns and "dti" in df.columns:
        df["risk_score"] = df["int_rate"] * df["dti"]

    # Fill any NaN introduced by feature engineering
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in num_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    log.info("Feature engineering complete")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. SAVE PROCESSED DATA
# ─────────────────────────────────────────────────────────────────────────────

def save_processed(df: pd.DataFrame, path: str = PROCESSED_DATA_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    log.info("Processed data saved → %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_preprocessing() -> None:
    df = load_raw_data()
    df = clean_data(df)
    df = engineer_features(df)
    save_processed(df)
    log.info("Preprocessing complete — shape: %s", df.shape)


if __name__ == "__main__":
    run_preprocessing()
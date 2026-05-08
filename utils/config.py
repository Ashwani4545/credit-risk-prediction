# utils/config.py
"""
Central configuration for Next-Gen Credit Risk Prediction System.
Refactored for Dynamic Risk Thresholding and Macro-Aware Learning.
"""

import os

# ── BASE DIRECTORY ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── DATA PATHS ───────────────────────────────────────────────────────────────
RAW_DATA_PATH       = os.path.join(BASE_DIR, "data", "raw",       "loan_dataset.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_data.csv")

# ── MODEL ARTIFACTS (Glass-Box EBM) ──────────────────────────────────────────
MODEL_PATH    = os.path.join(BASE_DIR, "models", "loan_ebm_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "utils",  "model_features.pkl")
METRICS_PATH  = os.path.join(BASE_DIR, "model_metrics.json")

# ── AUDIT & GOVERNANCE PATHS ─────────────────────────────────────────────────
HISTORY_PATH   = os.path.join(BASE_DIR, "outputs", "prediction_history.json")
AUDIT_LOG_PATH = os.path.join(BASE_DIR, "logs", "audit_log.json")
FAIRNESS_PATH  = os.path.join(BASE_DIR, "outputs", "fairness_report.txt")

# ── TARGET & SENSITIVE ATTRIBUTES ───────────────────────────────────────────
TARGET_COLUMN    = "loan_status"
SENSITIVE_COLUMN = "addr_state" # Used for real-time Fairness Auditing

# ── TRAIN / TEST SPLIT (Stratified for Imbalance) ──────────────────────────
TEST_SIZE    = 0.20
RANDOM_STATE = 42

# ── DYNAMIC RISK THRESHOLDING LOGIC ──────────────────────────────────────────
# Base thresholds for "Standard" economic conditions.
DEFAULT_THRESHOLDS = [
    (0.40, "LOW RISK",       "#22c55e"),
    (0.60, "MEDIUM RISK",    "#f59e0b"),
    (0.80, "HIGH RISK",      "#f97316"),
    (1.01, "VERY HIGH RISK", "#ef4444"),
]

def get_risk_level(probability: float, macro_stress: float = 0.0) -> dict:
    """
    Returns risk label and color adjusted by live Macro Telemetry.
    As economic stress increases, risk boundaries become more conservative.
    """
    # Shift thresholds downward as macro-economic stress increases
    # e.g., if stress is 0.1, a 0.3 probability becomes "MEDIUM RISK"
    adjustment = macro_stress * 0.5
    
    for base_threshold, label, color in DEFAULT_THRESHOLDS:
        adjusted_threshold = base_threshold - adjustment
        if probability < adjusted_threshold:
            return {"label": label, "color": color, "adjusted": adjustment > 0}
            
    return {"label": "VERY HIGH RISK", "color": "#ef4444", "adjusted": adjustment > 0}

# ── RECALL-FOCUSED OPTIMIZATION PARAMS ───────────────────────────────────────
EBM_PARAMS = {
    "learning_rate": 0.02,
    "max_bins": 256,
    "interactions": 10,
    "random_state": RANDOM_STATE,
}
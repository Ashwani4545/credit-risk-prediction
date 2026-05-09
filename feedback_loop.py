# feedback_loop.py
"""
Feedback loop — builds a training dataset from prediction history
and appends it to the processed data file for the next retraining cycle.

IMPORTANT CAVEAT:
  The 'ground truth' label here is the model's own prediction, NOT the
  actual repayment outcome (which would only be known months later).
  This is a simulation of continuous learning for demonstration purposes.
  In production, replace `prediction_numeric` with real outcome labels
  once they become available.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.config import HISTORY_PATH, PROCESSED_DATA_PATH

log = logging.getLogger(__name__)

FEATURES = [
    "loan_amnt", "int_rate", "annual_inc", "fico_range_low",
    "dti", "open_acc", "revol_bal", "total_acc",
]


def build_feedback_dataset() -> pd.DataFrame | None:
    """
    Load prediction history and build a lightweight feedback DataFrame.
    Returns None if fewer than 100 records exist.
    """
    try:
        import json
        with open(HISTORY_PATH) as f:
            history = json.load(f)
    except Exception as exc:
        log.warning("Cannot read prediction history: %s", exc)
        return None

    if len(history) < 100:
        log.info("Not enough history (%d < 100) for feedback dataset", len(history))
        return None

    rows = []
    for rec in history:
        raw = rec.get("raw_input", {})
        row = {}
        for feat in FEATURES:
            # Values may be stored at top level or inside raw_input
            val = rec.get(feat, raw.get(feat))
            try:
                row[feat] = float(val) if val is not None else None
            except (TypeError, ValueError):
                row[feat] = None
        # FIX: Use prediction_numeric (0/1 int) not the string label.
        #      Old code used rec["prediction"] which was "Repay"/"Default" string
        #      and then tried to map it — but newer records already store
        #      prediction_numeric directly.
        pred = rec.get("prediction_numeric")
        if pred is None:
            label_map = {"Repay": 0, "Default": 1, "Review": 1}
            pred = label_map.get(str(rec.get("prediction", "")), None)
        row["loan_status"] = pred
        rows.append(row)

    df = pd.DataFrame(rows).dropna()
    log.info("Feedback dataset built: %d rows", len(df))
    return df


def update_training_data(feedback_df: pd.DataFrame) -> bool:
    """Append feedback rows to the processed CSV used for retraining."""
    if feedback_df is None or feedback_df.empty:
        return False

    try:
        base_path = Path(PROCESSED_DATA_PATH)
        if base_path.exists():
            base_df  = pd.read_csv(base_path)
            combined = pd.concat([base_df, feedback_df], ignore_index=True)
        else:
            combined = feedback_df

        combined.to_csv(base_path, index=False)
        log.info("Training data updated with %d new rows → %s", len(feedback_df), base_path)
        return True
    except Exception as exc:
        log.error("update_training_data failed: %s", exc)
        return False

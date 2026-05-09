# utils/preprocessor.py
"""
Utility: validate saved model artefacts and print a summary.

Usage:
    python -m utils.preprocessor
"""

import json
import logging
import pickle
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import MODEL_PATH, FEATURES_PATH, METRICS_PATH, PROCESSED_DATA_PATH

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def validate_artefacts() -> bool:
    ok = True

    # ── Model ────────────────────────────────────────────────────────────────
    try:
        model = joblib.load(MODEL_PATH)
        n_feat = (
            len(model.get_booster().feature_names)
            if hasattr(model, "get_booster")
            else len(getattr(model, "feature_names_in_", []))
        )
        log.info("Model:    ✅  %s  (%d training features)", MODEL_PATH, n_feat)
    except Exception as exc:
        log.error("Model:    ❌  %s", exc)
        ok = False

    # ── Feature list ─────────────────────────────────────────────────────────
    try:
        with open(FEATURES_PATH, "rb") as f:
            features = pickle.load(f)
        log.info("Features: ✅  %d features  (%s)", len(features), FEATURES_PATH)
    except Exception as exc:
        log.error("Features: ❌  %s", exc)
        ok = False

    # ── Metrics ──────────────────────────────────────────────────────────────
    try:
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
        log.info(
            "Metrics:  ✅  accuracy=%.4f  roc_auc=%.4f  recall=%.4f",
            metrics.get("accuracy", 0),
            metrics.get("roc_auc",  0),
            metrics.get("recall",   0),
        )
    except Exception as exc:
        log.error("Metrics:  ❌  %s", exc)
        ok = False

    # ── Processed data ───────────────────────────────────────────────────────
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
        log.info("Data:     ✅  %d rows × %d cols  (%s)", *df.shape, PROCESSED_DATA_PATH)
    except Exception as exc:
        log.error("Data:     ❌  %s", exc)
        ok = False

    return ok


if __name__ == "__main__":
    success = validate_artefacts()
    sys.exit(0 if success else 1)

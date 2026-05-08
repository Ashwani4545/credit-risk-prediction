# src/drift_detection.py
"""
Loan Default Prediction â€” Feature Drift Monitoring (PSI-based)

Population Stability Index (PSI) thresholds:
  PSI < 0.10  â†’ No Drift
  PSI < 0.25  â†’ Moderate Drift
  PSI â‰¥ 0.25  â†’ High Drift

Usage:
    python -m src.drift_detection
"""

import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import PROCESSED_DATA_PATH, HISTORY_PATH

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# â”€â”€ CONFIGURATION â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
PSI_LOW    = 0.10
PSI_MEDIUM = 0.25
NUM_BINS   = 10

# Numeric features present in the LendingClub processed dataset
FEATURE_COLUMNS = [
    "loan_amnt", "int_rate",    "installment", "annual_inc",
    "dti",       "fico_range_low", "open_acc", "revol_bal", "total_acc",
]

STATUS_COLOR = {
    "No Drift":       "#22c55e",
    "Moderate Drift": "#f59e0b",
    "High Drift":     "#ef4444",
}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# PSI
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = NUM_BINS) -> float:
    """Compute Population Stability Index between two distributions."""

    def _as_numeric(arr: np.ndarray) -> np.ndarray:
        values = pd.to_numeric(pd.Series(arr), errors="coerce").dropna().to_numpy(dtype=float)
        return values

    def _scale(arr: np.ndarray) -> np.ndarray:
        arr = _as_numeric(arr)
        if arr.size == 0:
            return arr
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo + 1e-9)

    expected = _scale(expected)
    actual   = _scale(actual)

    if expected.size == 0 or actual.size == 0:
        return 0.0

    breakpoints = np.linspace(0, 1, bins + 1)

    def _pct(arr):
        counts = np.histogram(arr, bins=breakpoints)[0]
        pct    = counts / len(arr)
        return np.where(pct == 0, 1e-4, pct)          # avoid log(0)

    e_pct = _pct(expected)
    a_pct = _pct(actual)

    return float(np.sum((e_pct - a_pct) * np.log(e_pct / a_pct)))


def interpret_psi(value: float) -> str:
    if value < PSI_LOW:
        return "No Drift"
    if value < PSI_MEDIUM:
        return "Moderate Drift"
    return "High Drift"


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# DETECTOR
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class DriftDetector:

    def __init__(self, reference: pd.DataFrame, current: pd.DataFrame):
        self.reference = reference
        self.current   = current

    def run(self) -> dict:
        """Compute PSI for every monitored feature."""
        results: dict = {}
        for col in FEATURE_COLUMNS:
            if col not in self.reference.columns or col not in self.current.columns:
                log.warning("Column '%s' not found in data â€” skipping.", col)
                continue
            psi    = round(calculate_psi(self.reference[col].to_numpy(), self.current[col].to_numpy()), 4)
            status = interpret_psi(psi)
            results[col] = {"psi_value": psi, "drift_status": status}
        return results

    @staticmethod
    def overall_alert(results: dict) -> str:
        has_high = any(v["drift_status"] == "High Drift" for v in results.values())
        return "ðŸš¨ ALERT: Significant Feature Drift Detected" if has_high else "âœ… System Stable â€” No Significant Drift"


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# REPORTING
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def save_report(results: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = output_dir / "drift_report.csv"
    pd.DataFrame(results).T.to_csv(csv_path)
    log.info("Drift CSV â†’ %s", csv_path)

    # Bar chart
    features = list(results.keys())
    psi_vals = [results[f]["psi_value"]  for f in features]
    colors   = [STATUS_COLOR[results[f]["drift_status"]] for f in features]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(features, psi_vals, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.axhline(PSI_LOW,    color="#3b82f6", linestyle="--", linewidth=1.5,
               label=f"Low threshold ({PSI_LOW})")
    ax.axhline(PSI_MEDIUM, color="#ef4444", linestyle="--", linewidth=1.5,
               label=f"High threshold ({PSI_MEDIUM})")
    ax.set_xlabel("Feature", fontsize=12, fontweight="bold")
    ax.set_ylabel("PSI",     fontsize=12, fontweight="bold")
    ax.set_title("Feature Drift Detection (PSI)", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()

    png_path = output_dir / "drift_report.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Drift chart â†’ %s", png_path)


def detect_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> tuple[dict, bool]:
    detector = DriftDetector(reference_df, current_df)
    results  = detector.run()

    # Check if any feature has high drift
    drift_detected = any(
        v["drift_status"] == "High Drift"
        for v in results.values()
    )

    return results, drift_detected


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# MAIN
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_monitoring(data_path: str = PROCESSED_DATA_PATH) -> dict:
    log.info("Loading reference data from %s â€¦", data_path)
    reference = pd.read_csv(data_path)

    # Load current data from prediction history
    log.info("Loading current data from history %s â€¦", HISTORY_PATH)
    try:
        with open(HISTORY_PATH) as f:
            history_list = json.load(f)
        if not history_list:
            log.warning("No history data available, using split from reference")
            split = int(len(reference) * 0.70)
            current = reference.iloc[split:]
        else:
            # Extract raw_input as current data
            current = pd.DataFrame([entry["raw_input"] for entry in history_list])
            log.info("Using %d historical predictions as current data", len(current))
    except (FileNotFoundError, json.JSONDecodeError):
        log.warning("History file not found or invalid, using split from reference")
        split = int(len(reference) * 0.70)
        current = reference.iloc[split:]

    results, drift_detected = detect_drift(reference, current)

    log.info("\nFeature Drift Report:")
    for feat, res in results.items():
        log.info("  %-22s  PSI=%.4f  â†’ %s", feat, res["psi_value"], res["drift_status"])

    output_dir = Path(__file__).resolve().parent.parent / "outputs"
    save_report(results, output_dir)

    if drift_detected:
        log.warning("âš ï¸ High drift detected! Consider retraining the model.")

    detector = DriftDetector(reference, current)  # For overall_alert
    log.info("\n%s", detector.overall_alert(results))
    return results


def run_monitoring_summary(data_path: str = PROCESSED_DATA_PATH) -> dict:
    """Return a compact dashboard summary for the Flask app."""

    try:
        results = run_monitoring(data_path=data_path)
    except Exception:
        log.exception("Monitoring summary failed")
        return {"status": "unavailable", "features": []}

    if not results:
        return {"status": "unavailable", "features": []}

    features = [
        {
            "feature": feature,
            "psi": data["psi_value"],
            "status": data["drift_status"],
        }
        for feature, data in results.items()
    ]

    if any(item["status"] == "High Drift" for item in features):
        status = "ALERT"
    elif any(item["status"] == "Moderate Drift" for item in features):
        status = "WARNING"
    else:
        status = "OK"

    return {"status": status, "features": features}


if __name__ == "__main__":
    run_monitoring()


# monitoring/drift_detection.py
"""
Loan Default Prediction — Feature Drift Monitoring (PSI-based)

Population Stability Index (PSI) thresholds:
  PSI < 0.10  → No Drift
  PSI < 0.25  → Moderate Drift
  PSI ≥ 0.25  → High Drift

Usage:
    python -m monitoring.drift_detection
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

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"

PSI_LOW    = 0.10
PSI_MEDIUM = 0.25
NUM_BINS   = 10

FEATURE_COLUMNS = [
    "loan_amnt", "int_rate", "installment", "annual_inc",
    "dti",       "fico_range_low", "open_acc", "revol_bal", "total_acc",
]

STATUS_COLOR = {
    "No Drift":       "#22c55e",
    "Moderate Drift": "#f59e0b",
    "High Drift":     "#ef4444",
}


# ── PSI ───────────────────────────────────────────────────────────────────────

def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = NUM_BINS) -> float:
    """Compute Population Stability Index between two distributions."""

    def _scale(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo + 1e-9)

    expected = _scale(expected)
    actual   = _scale(actual)
    bp       = np.linspace(0, 1, bins + 1)

    def _pct(arr):
        counts = np.histogram(arr, bins=bp)[0]
        pct    = counts / len(arr)
        return np.where(pct == 0, 1e-4, pct)

    e_pct = _pct(expected)
    a_pct = _pct(actual)
    return float(np.sum((e_pct - a_pct) * np.log(e_pct / a_pct)))


def interpret_psi(value: float) -> str:
    if value < PSI_LOW:
        return "No Drift"
    if value < PSI_MEDIUM:
        return "Moderate Drift"
    return "High Drift"


# ── Main detection function ───────────────────────────────────────────────────

def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[list[dict], bool]:
    """
    Compare feature distributions between reference and current data.

    Returns:
        results   — list of {feature, psi, status} dicts
        drift_flag — True if any feature has High Drift
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS

    results: list[dict] = []
    drift_flag = False

    for col in feature_cols:
        if col not in reference.columns or col not in current.columns:
            log.warning("Column '%s' missing from reference or current data — skipping", col)
            continue

        ref_vals = reference[col].dropna().values
        cur_vals = current[col].dropna().values

        if len(ref_vals) < 10 or len(cur_vals) < 10:
            log.warning("Too few values for '%s' — skipping PSI", col)
            continue

        psi    = calculate_psi(ref_vals, cur_vals)
        status = interpret_psi(psi)
        results.append({"feature": col, "psi": round(psi, 4), "status": status})

        if status == "High Drift":
            drift_flag = True
        log.info("%-20s  PSI=%.4f  → %s", col, psi, status)

    return results, drift_flag


# ── Reporting helpers ─────────────────────────────────────────────────────────

def save_drift_report(results: list[dict]) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    df  = pd.DataFrame(results)
    out = OUTPUTS_DIR / "drift_report.csv"
    df.to_csv(out, index=False)
    log.info("Drift report saved → %s", out)


def plot_drift_report(results: list[dict]) -> None:
    if not results:
        return
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    features = [r["feature"] for r in results]
    psi_vals = [r["psi"]     for r in results]
    colors   = [STATUS_COLOR.get(r["status"], "#6b7280") for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(features, psi_vals, color=colors)
    ax.axvline(PSI_LOW,    color="#f59e0b", linestyle="--", linewidth=1, label=f"Moderate ({PSI_LOW})")
    ax.axvline(PSI_MEDIUM, color="#ef4444", linestyle="--", linewidth=1, label=f"High ({PSI_MEDIUM})")
    ax.set_xlabel("PSI")
    ax.set_title("Feature Drift Report (PSI)")
    ax.legend()
    plt.tight_layout()

    out = OUTPUTS_DIR / "drift_report.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Drift chart saved → %s", out)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    reference_df = pd.read_csv(PROCESSED_DATA_PATH).iloc[:10_000]

    try:
        with open(HISTORY_PATH) as f:
            history = json.load(f)
        rows = []
        for rec in history:
            raw = rec.get("raw_input", {})
            row = {col: rec.get(col, raw.get(col)) for col in FEATURE_COLUMNS}
            rows.append(row)
        current_df = pd.DataFrame(rows)
    except Exception as exc:
        log.error("Cannot build current data from history: %s", exc)
        sys.exit(1)

    results, drift_flag = detect_drift(reference_df, current_df)
    save_drift_report(results)
    plot_drift_report(results)

    if drift_flag:
        log.warning("🚨 SIGNIFICANT DRIFT DETECTED — consider retraining the model")
    else:
        log.info("✅ No significant drift detected")

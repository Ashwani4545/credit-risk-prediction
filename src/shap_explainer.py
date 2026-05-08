# src/shap_explainer.py
"""
Loan Default Prediction — SHAP Explainability & Fairness

Generates:
  - SHAP summary plot  (outputs/shap_summary.png)
  - SHAP force plot    (outputs/shap_force_plot.html)
  - Fairness report    (outputs/fairness_report.txt)

Usage:
    python -m src.shap_explainer
"""

import os
import re
import sys
import logging
import importlib.util
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import MODEL_PATH, PROCESSED_DATA_PATH, TARGET_COLUMN, SENSITIVE_COLUMN

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

OUTPUTS_DIR = os.path.join(Path(__file__).resolve().parent.parent, "outputs")


# ── Column sanitizer (must match train_model.py exactly) ─────────────────────

def _sanitize_columns(columns) -> list:
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


# ── Explainer class ───────────────────────────────────────────────────────────

class LoanModelExplainer:

    def __init__(self, model_path: str = MODEL_PATH):
        self.model    = joblib.load(model_path)
        self.has_shap = importlib.util.find_spec("shap") is not None
        self.shap     = __import__("shap") if self.has_shap else None

        if self.has_shap:
            try:
                self.explainer = self.shap.TreeExplainer(self.model)
                log.info("SHAP TreeExplainer initialised ✅ (fast path)")
            except Exception:
                self.explainer = self.shap.Explainer(self.model)
                log.info("SHAP generic Explainer initialised ✅ (fallback)")
        else:
            self.explainer = None
            log.warning("SHAP not installed — using feature-importance fallback.")

    def _load_data(self, sample: int = 500) -> tuple[pd.DataFrame, pd.Series]:
        df    = pd.read_csv(PROCESSED_DATA_PATH)
        X     = df.drop(columns=[TARGET_COLUMN])
        y     = df[TARGET_COLUMN]
        X     = pd.get_dummies(X, drop_first=True)
        X.columns = _sanitize_columns(X.columns)
        X     = X.astype("float32")

        # Align to model features
        try:
            expected = self.model.get_booster().feature_names
        except AttributeError:
            expected = list(getattr(self.model, "feature_names_in_", X.columns))

        for col in expected:
            if col not in X.columns:
                X[col] = 0.0
        X = X[expected]
        return X.sample(min(sample, len(X)), random_state=42), y.loc[X.index]

    def generate_summary_plot(self) -> None:
        if not self.has_shap:
            log.warning("SHAP not available — skipping summary plot")
            return
        X_sample, _ = self._load_data()
        sv = self.explainer.shap_values(X_sample)
        if isinstance(sv, list):
            sv = sv[1]
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        plt.figure(figsize=(10, 8))
        self.shap.summary_plot(sv, X_sample, show=False)
        plt.tight_layout()
        out = os.path.join(OUTPUTS_DIR, "shap_summary.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        log.info("SHAP summary plot saved → %s", out)

    def generate_force_plot(self, idx: int = 0) -> None:
        if not self.has_shap:
            log.warning("SHAP not available — skipping force plot")
            return
        X_sample, _ = self._load_data(sample=50)
        sv           = self.explainer.shap_values(X_sample)
        if isinstance(sv, list):
            sv = sv[1]
        ev = self.explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            ev = ev[1]
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        out = os.path.join(OUTPUTS_DIR, "shap_force_plot.html")
        self.shap.save_html(
            out,
            self.shap.force_plot(ev, sv[idx], X_sample.iloc[idx]),
        )
        log.info("SHAP force plot saved → %s", out)

    def explain_single(self, X_df: pd.DataFrame) -> list[dict]:
        """Return top-5 SHAP drivers for a single prediction."""
        if not self.has_shap or self.explainer is None:
            return self._fallback_importance(X_df)

        try:
            sv = self.explainer.shap_values(X_df)
            if isinstance(sv, list):
                sv = sv[1]
            row   = sv[0] if sv.ndim == 2 else sv
            pairs = list(zip(X_df.columns, row))
            pairs.sort(key=lambda x: abs(x[1]), reverse=True)
            return [
                {"feature": f, "impact": round(float(v), 6)}
                for f, v in pairs[:5]
            ]
        except Exception:
            log.exception("explain_single SHAP failed — using fallback")
            return self._fallback_importance(X_df)

    def _fallback_importance(self, X_df: pd.DataFrame) -> list[dict]:
        try:
            importances = self.model.feature_importances_
            pairs = sorted(
                zip(X_df.columns, importances),
                key=lambda x: x[1], reverse=True,
            )
            return [
                {"feature": f, "impact": round(float(v), 6)}
                for f, v in pairs[:5]
            ]
        except Exception:
            return []

    def check_individual_fairness(self, form_data: dict) -> dict:
        """
        Check whether any protected / sensitive fields are present in the
        submitted form data.  Returns a dict summary (never raises).
        """
        sensitive_fields = {"addr_state", "race", "gender", "religion", "national_origin"}
        found = [f for f in sensitive_fields if f in form_data and form_data[f]]
        return {
            "flagged": bool(found),
            "sensitive_fields_detected": found,
            "note": (
                f"Sensitive field(s) detected: {found}. Review for bias."
                if found
                else "No protected attributes detected in input."
            ),
        }

    def check_group_bias(self, form_data: dict) -> dict:
        """
        Lightweight group-bias check based on addr_state.
        Returns a dict summary (never raises).
        """
        state = form_data.get("addr_state", "")
        return {
            "flagged": False,
            "group": state or "unknown",
            "note": "Group-level bias analysis requires aggregated data — see fairness_report.txt.",
        }

    def validate_sensitive_features(self, form_data: dict) -> str:
        """
        Return a human-readable warning string if any sensitive feature is present,
        otherwise an empty string.
        """
        sensitive_fields = {"race", "gender", "religion", "national_origin"}
        found = [f for f in sensitive_fields if f in form_data and form_data[f]]
        if found:
            return f"⚠️ Sensitive feature(s) present in input: {', '.join(found)}. Ensure compliance with fair-lending regulations."
        return ""

    def generate_fairness_report(self) -> None:
        X_sample, y_sample = self._load_data(sample=5000)
        df = pd.read_csv(PROCESSED_DATA_PATH)
        sensitive_col = SENSITIVE_COLUMN

        if sensitive_col not in df.columns:
            log.warning("Sensitive column '%s' not found — skipping fairness report", sensitive_col)
            return

        preds = self.model.predict(X_sample)
        states = df.loc[X_sample.index, sensitive_col].values

        lines = ["FAIRNESS REPORT — Demographic Parity by State\n", "=" * 52]
        for state in sorted(set(states)):
            mask       = states == state
            approval_r = (preds[mask] == 0).mean()
            lines.append(f"  {state:<6}  approval_rate={approval_r:.3f}  n={mask.sum()}")

        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        out = os.path.join(OUTPUTS_DIR, "fairness_report.txt")
        with open(out, "w") as f:
            f.write("\n".join(lines))
        log.info("Fairness report saved → %s", out)


# ── Convenience function used by predict.py ───────────────────────────────────

_explainer_singleton: LoanModelExplainer | None = None


def get_local_shap(X_df: pd.DataFrame) -> list[dict]:
    """Return top-5 SHAP drivers for a single row DataFrame."""
    global _explainer_singleton
    if _explainer_singleton is None:
        _explainer_singleton = LoanModelExplainer()
    return _explainer_singleton.explain_single(X_df)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    explainer = LoanModelExplainer()
    explainer.generate_summary_plot()
    explainer.generate_force_plot()
    explainer.generate_fairness_report()
    log.info("Explainability artifacts generated ✅")
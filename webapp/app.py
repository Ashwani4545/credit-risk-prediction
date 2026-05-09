# webapp/app.py
"""
Credit Risk Prediction — Flask Application

Routes:
  GET  /             → Loan assessment form
  POST /predict      → Run model, save to history, show result
  GET  /dashboard    → Model metrics + confusion matrix
  GET  /history      → All past predictions
  GET  /reports      → Individual borrower reports
  GET  /reports/<id> → Single report detail
  GET  /api/metrics  → JSON metrics for dashboard charts
  GET  /api/history  → JSON history for AJAX
  GET  /health       → Healthcheck
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, render_template, request

# ── Retrain helper ───────────────────────────────────────────────────────────
try:
    from .retrain import retrain_model
except ImportError:
    from retrain import retrain_model

# ── Project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import (
    MODEL_PATH, FEATURES_PATH, METRICS_PATH, HISTORY_PATH,
    get_risk_level, PROCESSED_DATA_PATH,
)
from feedback_loop import build_feedback_dataset, update_training_data
from governance import log_decision
from monitoring.drift_detection import detect_drift
from src.shap_explainer import LoanModelExplainer

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP: load model artefacts
# ─────────────────────────────────────────────────────────────────────────────

def _load_model():
    try:
        m = joblib.load(MODEL_PATH)
        log.info("Model loaded ✅  (%s)", MODEL_PATH)
        return m
    except Exception as exc:
        log.error("Model load failed: %s", exc)
        return None


def _load_features() -> list:
    try:
        with open(FEATURES_PATH, "rb") as f:
            feats = pickle.load(f)
        log.info("Feature list loaded — %d features", len(feats))
        return list(feats)
    except Exception as exc:
        log.error("Feature load failed: %s — run src/train_model.py first", exc)
        return []


def _load_metrics() -> dict:
    defaults = {
        "accuracy": 0.0, "precision": 0.0, "recall": 0.0,
        "f1_score": 0.0, "roc_auc": 0.0,
        "confusion_matrix": {"tn": 0, "fp": 0, "fn": 0, "tp": 0},
    }
    try:
        with open(METRICS_PATH) as f:
            data = json.load(f)
        return {
            "accuracy":  float(data.get("accuracy",  0)),
            "precision": float(data.get("precision", 0)),
            "recall":    float(data.get("recall",    0)),
            "f1_score":  float(data.get("f1_score",  0)),
            "roc_auc":   float(data.get("roc_auc",   0)),
            "confusion_matrix": {
                "tn": int(data.get("confusion_matrix", {}).get("tn", 0)),
                "fp": int(data.get("confusion_matrix", {}).get("fp", 0)),
                "fn": int(data.get("confusion_matrix", {}).get("fn", 0)),
                "tp": int(data.get("confusion_matrix", {}).get("tp", 0)),
            },
        }
    except FileNotFoundError:
        log.warning("model_metrics.json not found — run src/evaluate_model.py")
        return defaults
    except Exception as exc:
        log.error("Metrics load error: %s", exc)
        return defaults


def _load_threshold() -> float:
    """Load the decision threshold saved by train_model.py."""
    try:
        with open(METRICS_PATH) as f:
            data = json.load(f)
        t = data.get("decision_threshold")
        if t is not None:
            return float(t)
    except Exception:
        pass
    return 0.5


MODEL         = _load_model()
MODEL_FEATURES = _load_features()
METRICS       = _load_metrics()
REFERENCE_DATA = pd.read_csv(PROCESSED_DATA_PATH).iloc[:10_000]
EXPLAINER     = LoanModelExplainer()


def reload_model() -> None:
    """Reload model, features, metrics and SHAP explainer after retraining."""
    global MODEL, MODEL_FEATURES, METRICS, EXPLAINER
    MODEL          = _load_model()
    MODEL_FEATURES = _load_features()
    METRICS        = _load_metrics()
    EXPLAINER      = LoanModelExplainer()
    log.info("🔄 Model + SHAP explainer reloaded after retraining")


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION HISTORY
# ─────────────────────────────────────────────────────────────────────────────

def _load_history() -> list:
    try:
        with open(HISTORY_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_history(records: list) -> None:
    Path(HISTORY_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "w") as f:
        json.dump(records, f, indent=2, default=str)


def _append_to_history(record: dict) -> None:
    history = _load_history()
    history.insert(0, record)
    _save_history(history[:500])


def should_retrain() -> bool:
    history = _load_history()
    return len(history) >= 100 and len(history) % 100 == 0


def should_check_drift() -> bool:
    history = _load_history()
    return len(history) >= 50 and len(history) % 10 == 0


def get_current_data() -> pd.DataFrame | None:
    history = _load_history()
    if len(history) < 50:
        return None
    cols = [
        "loan_amnt", "int_rate", "installment", "annual_inc",
        "dti", "fico_range_low", "open_acc", "revol_bal", "total_acc",
    ]
    rows = []
    for rec in history:
        raw = rec.get("raw_input", {})
        rows.append({
            "loan_amnt":      float(rec.get("loan_amnt", 0) or 0),
            "int_rate":       float(rec.get("int_rate",  raw.get("int_rate",  0)) or 0),
            "installment":    float(rec.get("installment", raw.get("installment", 0)) or 0),
            "annual_inc":     float(rec.get("annual_inc",  0) or 0),
            "dti":            float(rec.get("dti",         raw.get("dti",         0)) or 0),
            "fico_range_low": float(rec.get("fico",        raw.get("fico_range_low", 0)) or 0),
            "open_acc":       float(rec.get("open_acc",    raw.get("open_acc",    0)) or 0),
            "revol_bal":      float(rec.get("revol_bal",   raw.get("revol_bal",   0)) or 0),
            "total_acc":      float(rec.get("total_acc",   raw.get("total_acc",   0)) or 0),
        })
    return pd.DataFrame(rows)[cols].dropna()


# ─────────────────────────────────────────────────────────────────────────────
# COLUMN SANITIZER  (must match train_model.py exactly)
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


# ─────────────────────────────────────────────────────────────────────────────
# INPUT PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

_NUMERIC_FIELDS = {
    "loan_amnt", "int_rate", "installment", "annual_inc", "dti",
    "fico_range_low", "fico_range_high", "open_acc", "revol_bal",
    "revol_util", "total_acc", "delinq_2yrs", "inq_last_6mths",
    "pub_rec", "pub_rec_bankruptcies", "collections_12_mths_ex_med",
    "acc_now_delinq", "tot_coll_amt", "tot_cur_bal", "avg_cur_bal",
    "bc_open_to_buy", "bc_util", "num_actv_bc_tl", "num_rev_accts",
    "percent_bc_gt_75",
}

_CATEGORICAL_FIELDS = [
    "term", "grade", "sub_grade", "emp_length",
    "home_ownership", "verification_status", "purpose",
    "addr_state", "initial_list_status",
]


def _create_features_live(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror the feature engineering done in train_model.py and data_preprocessing.py."""
    df["loan_to_income"]           = df["loan_amnt"]    / (df["annual_inc"] + 1e-6)
    df["installment_to_income"]    = df["installment"]  / (df["annual_inc"] + 1e-6)
    df["credit_utilization"]       = df["revol_bal"]    / (df["revol_bal"] + df["bc_open_to_buy"] + 1e-6)
    df["payment_capacity"]         = df["annual_inc"]   - df["installment"] * 12
    df["credit_stress"]            = df["dti"]          * df["loan_amnt"]
    df["recent_inquiries_flag"]    = (df["inq_last_6mths"] > 3).astype(int)
    df["high_dti_flag"]            = (df["dti"] > 20).astype(int)
    df["low_fico_flag"]            = (df["fico_range_low"] < 600).astype(int)
    # Features from data_preprocessing.engineer_features()
    df["loan_income_ratio"]        = df["loan_amnt"]    / (df["annual_inc"].replace(0, np.nan) + 1)
    df["revol_income_ratio"]       = df["revol_bal"]    / (df["annual_inc"].replace(0, np.nan) + 1)
    df["open_acc_ratio"]           = df["open_acc"]     / (df["total_acc"].replace(0, np.nan) + 1)
    df["installment_income_ratio"] = df["installment"]  / (df["annual_inc"].replace(0, np.nan) + 1)
    df["fico_avg"]                 = (df["fico_range_low"] + df["fico_range_high"]) / 2
    df["risk_score"]               = df["int_rate"] * df["dti"]
    # Static economic features
    df["inflation_rate"]    = 0.06
    df["interest_rate_env"] = 0.08
    df["unemployment_rate"] = 0.07
    df["economic_stress"]   = 0.06 * 0.4 + 0.07 * 0.4 + 0.08 * 0.2
    # Alternative data placeholder (0 matches training default)
    df["mobile_usage_score"]    = 0
    df["digital_txn_count"]     = 0
    df["utility_payment_score"] = 0
    df["employment_stability"]  = 0
    return df


def preprocess_input(form_data: dict) -> pd.DataFrame:
    """Convert raw form POST data into a 1-row DataFrame aligned to model features."""
    if not MODEL_FEATURES:
        raise RuntimeError("Model feature list is empty — run src/train_model.py first.")

    row = {feat: 0.0 for feat in MODEL_FEATURES}

    for field in _NUMERIC_FIELDS:
        if field in row:
            try:
                row[field] = max(float(form_data.get(field, 0) or 0), 0.0)
            except (ValueError, TypeError):
                row[field] = 0.0

    for cat in _CATEGORICAL_FIELDS:
        value = form_data.get(cat, "")
        if not value:
            continue
        for col_name in (f"{cat}_{value}", f"{cat}__{value}"):
            if col_name in row:
                row[col_name] = 1.0
                break

    df = pd.DataFrame([row])
    df = _create_features_live(df)
    df = df.reindex(columns=MODEL_FEATURES, fill_value=0.0).astype("float32")
    return df


def _validate_input(form_data: dict) -> list[str]:
    errors: list[str] = []
    try:
        if float(form_data.get("loan_amnt", 0) or 0) < 500:
            errors.append("Loan amount must be at least $500.")
    except ValueError:
        errors.append("Loan amount is not a valid number.")
    try:
        if float(form_data.get("annual_inc", 0) or 0) <= 0:
            errors.append("Annual income must be greater than 0.")
    except ValueError:
        errors.append("Annual income is not a valid number.")
    try:
        fico = float(form_data.get("fico_range_low", 300) or 300)
        if not (300 <= fico <= 850):
            errors.append("FICO score must be between 300 and 850.")
    except ValueError:
        errors.append("FICO score is not a valid number.")
    return errors


# ─────────────────────────────────────────────────────────────────────────────
# FINANCIAL CALCULATIONS
# ─────────────────────────────────────────────────────────────────────────────

def _calculate_lgd(fico: float) -> float:
    """Loss Given Default based on FICO score tiers."""
    if fico >= 750: return 0.15
    if fico >= 700: return 0.25
    if fico >= 650: return 0.35
    if fico >= 600: return 0.45
    return 0.55


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────

_REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def _generate_risk_report(record: dict) -> str:
    lines = [
        "===== Loan Risk Report =====",
        f"Borrower:          {record.get('borrower', 'Anonymous')}",
        f"Loan Amount:       ${record.get('loan_amnt', 0):,.2f}",
        f"Annual Income:     ${record.get('annual_inc', 0):,.2f}",
        f"",
        f"Probability of Default (PD): {record.get('probability', 0):.2f}%",
        f"Risk Level:        {record.get('risk_level', 'N/A')}",
        f"Decision:          {record.get('decision', 'N/A')}",
        f"",
        "Key Risk Drivers (SHAP):",
    ]
    for feat in record.get("top_features", []):
        lines.append(f"  - {feat.get('feature', '')}: {feat.get('shap_value', feat.get('impact', 0)):.4f}")
    return "\n".join(lines)


def _save_report(report_text: str, record_id: str) -> str:
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _REPORTS_DIR / f"{record_id}.txt"
    with open(path, "w") as f:
        f.write(report_text)
    return str(path)


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if MODEL is None:
        return jsonify({"error": "Model not loaded — run src/train_model.py first."}), 503

    form_data = request.form.to_dict()

    errors = _validate_input(form_data)
    if errors:
        return render_template("index.html", errors=errors, form_data=form_data)

    try:
        input_df    = preprocess_input(form_data)
        explanation = EXPLAINER.explain_single(input_df)

        prob      = float(MODEL.predict_proba(input_df)[0][1])
        threshold = _load_threshold()

        loan_amount = float(form_data.get("loan_amnt", 0) or 0)
        annual_inc  = float(form_data.get("annual_inc", 0) or 0)
        fico        = float(form_data.get("fico_range_low", 0) or 0)
        int_rate    = float(form_data.get("int_rate", 0) or 0) / 100.0

        override_triggered = annual_inc > 0 and loan_amount > 5 * annual_inc

        lgd            = _calculate_lgd(fico)
        ead            = loan_amount
        expected_loss  = prob * lgd * ead
        expected_profit = (loan_amount * (1 - prob) * int_rate) - (loan_amount * prob * lgd)

        # ── Risk classification ──────────────────────────────────────────────
        if override_triggered:
            risk_label   = "HIGH RISK (OVERRIDE)"
            verdict      = "Default"
            show_warning = True
        else:
            risk_info    = get_risk_level(prob)
            risk_label_v = risk_info["label"]
            if risk_label_v == "LOW RISK":
                risk_label, verdict, show_warning = "LOW RISK", "Repay", False
            elif risk_label_v == "MEDIUM RISK":
                risk_label, verdict, show_warning = "MEDIUM RISK", "Review", True
            else:
                risk_label, verdict, show_warning = risk_label_v, "Default", True

        risk_color_map = {
            "LOW RISK":              "#22c55e",
            "MEDIUM RISK":           "#f59e0b",
            "HIGH RISK":             "#f97316",
            "HIGH RISK (OVERRIDE)":  "#dc2626",
            "VERY HIGH RISK":        "#ef4444",
        }
        message   = "Default Risk Detected — Review Recommended" if show_warning else "Safe Borrower — No Immediate Risk"
        risk_note = "📌 Credit Invisible — evaluated using alternative data" if fico == 0 else "Standard credit evaluation"

        record = {
            "id":               str(uuid.uuid4()),
            "timestamp":        datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "borrower":         form_data.get("borrower_name", "Anonymous"),
            "loan_amnt":        loan_amount,
            "int_rate":         float(form_data.get("int_rate",    0) or 0),
            "installment":      float(form_data.get("installment", 0) or 0),
            "annual_inc":       annual_inc,
            "dti":              float(form_data.get("dti",         0) or 0),
            "fico":             fico,
            "open_acc":         float(form_data.get("open_acc",    0) or 0),
            "revol_bal":        float(form_data.get("revol_bal",   0) or 0),
            "total_acc":        float(form_data.get("total_acc",   0) or 0),
            "purpose":          form_data.get("purpose", ""),
            "grade":            form_data.get("grade", ""),
            "prediction":       verdict,
            "prediction_numeric": 1 if verdict == "Default" else 0,
            "decision":         verdict,
            "probability":      round(prob * 100, 2),
            "PD":               round(prob, 4),
            "LGD":              round(lgd, 2),
            "EAD":              round(ead, 2),
            "expected_loss":    round(expected_loss, 2),
            "expected_profit":  round(expected_profit, 2),
            "model_version":    "v1.0",
            "decision_threshold": threshold,
            "risk_level":       risk_label,
            "show_warning":     show_warning,
            "message":          message,
            "color":            risk_color_map.get(risk_label, "#6b7280"),
            "risk_note":        risk_note,
            "top_features":     explanation,
            "raw_input":        form_data,
        }

        report_text = _generate_risk_report(record)
        record["report_path"] = _save_report(report_text, record["id"])

        _append_to_history(record)
        log_decision(record)

        # Feedback loop — every 100th prediction
        if should_retrain():
            feedback_data = build_feedback_dataset()
            if feedback_data is not None:
                update_training_data(feedback_data)
            retrain_model()
            reload_model()

        # Drift detection — every 10th prediction
        if should_check_drift():
            current_data = get_current_data()
            if current_data is not None:
                _, drift_flag = detect_drift(REFERENCE_DATA, current_data)
                if drift_flag:
                    log.warning("🚨 DRIFT DETECTED — triggering retraining")
                    retrain_model()
                    reload_model()

        return render_template(
            "result.html",
            risk            = risk_label,
            show_warning    = show_warning,
            prob            = prob,
            verdict         = verdict,
            top_features    = explanation,
            expected_loss   = round(expected_loss, 2),
            expected_profit = round(expected_profit, 2),
            borrower        = form_data.get("borrower_name", "Anonymous"),
            loan_amnt       = loan_amount,
            pd_value        = round(prob, 4),
            lgd             = round(lgd, 2),
            message         = message,
            risk_note       = risk_note,
        )

    except Exception as exc:
        log.exception("Prediction error")
        return render_template("index.html", errors=[f"Prediction failed: {exc}"], form_data=form_data)


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", metrics=METRICS)


@app.route("/history")
def history():
    records = _load_history()
    return render_template("history.html", records=records)


@app.route("/reports")
def reports():
    records = _load_history()
    return render_template("reports.html", records=records)


@app.route("/reports/<record_id>")
def report_detail(record_id: str):
    records = _load_history()
    record  = next((r for r in records if r.get("id") == record_id), None)
    if record is None:
        abort(404)
    return render_template("report_detail.html", record=record)


# ── JSON APIs ─────────────────────────────────────────────────────────────────

@app.route("/api/metrics")
def api_metrics():
    return jsonify(METRICS)


@app.route("/api/history")
def api_history():
    q       = request.args.get("q", "").lower()
    records = _load_history()
    if q:
        records = [
            r for r in records
            if q in r.get("borrower",   "").lower()
            or q in r.get("purpose",    "").lower()
            or q in r.get("risk_level", "").lower()
        ]
    return jsonify(records)


@app.route("/health")
def health():
    return jsonify({
        "status":       "ok",
        "model_loaded": MODEL is not None,
        "features":     len(MODEL_FEATURES),
    })


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)

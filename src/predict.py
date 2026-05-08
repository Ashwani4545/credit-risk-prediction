"""Loan default inference entrypoint used by the Flask app.

The predictor mirrors the training-time preprocessing pipeline, aligns the
incoming applicant data to the saved model feature list, and returns a rich
result payload for both the HTML UI and the JSON API.
"""

from __future__ import annotations

import json
import logging
import math
import re
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from governance import log_decision
from utils.config import (
    FEATURES_PATH,
    HISTORY_PATH,
    METRICS_PATH,
    MODEL_PATH,
    PROCESSED_DATA_PATH,
    get_risk_level,
)
from src.data_preprocessing import engineer_features
from src.train_model import create_features, sanitize_columns
from src.shap_explainer import get_local_shap

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


TERM_MAP = {
    "0": "36 months",
    "1": "60 months",
    0: "36 months",
    1: "60 months",
    "36": "36 months",
    "60": "60 months",
    "36 months": "36 months",
    "60 months": "60 months",
}

PURPOSE_MAP = {
    "0": "car",
    "1": "credit_card",
    "2": "debt_consolidation",
    "3": "education",
    "4": "home_improvement",
    "5": "major_purchase",
    "6": "small_business",
    "7": "medical",
    0: "car",
    1: "credit_card",
    2: "debt_consolidation",
    3: "education",
    4: "home_improvement",
    5: "major_purchase",
    6: "small_business",
    7: "medical",
}

HOME_OWNERSHIP_MAP = {
    "0": "OTHER",
    "1": "RENT",
    "2": "MORTGAGE",
    "3": "OWN",
    0: "OTHER",
    1: "RENT",
    2: "MORTGAGE",
    3: "OWN",
    "other": "OTHER",
    "rent": "RENT",
    "mortgage": "MORTGAGE",
    "own": "OWN",
}

VERIFICATION_MAP = {
    "0": "Not Verified",
    "1": "Verified",
    "2": "Source Verified",
    0: "Not Verified",
    1: "Verified",
    2: "Source Verified",
}


@lru_cache(maxsize=1)
def _load_reference_frame() -> pd.DataFrame:
    reference = pd.read_csv(PROCESSED_DATA_PATH)
    if "loan_status" in reference.columns:
        reference = reference.drop(columns=["loan_status"])
    return reference


@lru_cache(maxsize=1)
def _load_model():
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def _load_feature_names() -> list[str]:
    try:
        import pickle

        with open(FEATURES_PATH, "rb") as f:
            return list(pickle.load(f))
    except Exception:
        model = _load_model()
        if hasattr(model, "get_booster"):
            names = model.get_booster().feature_names
            if names:
                return list(names)
        return list(getattr(model, "feature_names_in_", []))


@lru_cache(maxsize=1)
def _load_threshold() -> float:
    try:
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
        threshold = metrics.get("decision_threshold")
        if threshold is not None:
            return float(threshold)
    except Exception:
        pass
    return 0.5


@lru_cache(maxsize=1)
def _reference_defaults() -> dict[str, Any]:
    reference = _load_reference_frame()
    defaults: dict[str, Any] = {}
    for column in reference.columns:
        series = reference[column]
        if pd.api.types.is_numeric_dtype(series):
            defaults[column] = float(series.median())
        else:
            modes = series.mode(dropna=True)
            defaults[column] = modes.iloc[0] if not modes.empty else ""
    return defaults


def _coerce_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    if isinstance(value, str):
        cleaned = value.replace(",", "").replace("%", "").strip()
        if cleaned == "":
            return float(default)
        try:
            return float(cleaned)
        except ValueError:
            return float(default)
    try:
        if pd.isna(value):
            return float(default)
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_term(value: Any) -> str:
    return TERM_MAP.get(value, TERM_MAP.get(str(value).strip(), str(value).strip() or "36 months"))


def _normalize_purpose(value: Any) -> str:
    return PURPOSE_MAP.get(value, PURPOSE_MAP.get(str(value).strip(), str(value).strip() or "debt_consolidation"))


def _normalize_home_ownership(value: Any) -> str:
    normalized = HOME_OWNERSHIP_MAP.get(value, HOME_OWNERSHIP_MAP.get(str(value).strip().lower(), str(value).strip()))
    return str(normalized).upper()


def _normalize_verification_status(value: Any, fallback: Any) -> str:
    if value is None or str(value).strip() == "":
        return str(fallback)
    normalized = VERIFICATION_MAP.get(value, VERIFICATION_MAP.get(str(value).strip(), value))
    return str(normalized)


def _normalize_emp_length(value: Any, fallback: Any = "2 years") -> str:
    if value is None or str(value).strip() == "":
        return str(fallback)

    text = str(value).strip().lower()
    if text in {"10+", "10+ years", "10", "10 years"}:
        return "10+ years"
    if text in {"< 1", "< 1 year", "less than 1", "less than 1 year", "0", "0 years"}:
        return "< 1 year"
    if re.fullmatch(r"\d+", text):
        years = int(text)
        if years <= 0:
            return "< 1 year"
        if years >= 10:
            return "10+ years"
        return f"{years} years"
    return str(value).strip()


def _grade_from_fico(fico: float) -> str:
    if fico >= 740:
        return "A"
    if fico >= 700:
        return "B"
    if fico >= 660:
        return "C"
    if fico >= 620:
        return "D"
    if fico >= 580:
        return "E"
    if fico >= 550:
        return "F"
    return "G"


def _sub_grade_from_fico(fico: float) -> str:
    grade = _grade_from_fico(fico)
    if grade == "A":
        band = max(1, min(5, 6 - int((fico - 700) // 10)))
    elif grade == "B":
        band = max(1, min(5, 6 - int((fico - 660) // 10)))
    elif grade == "C":
        band = max(1, min(5, 6 - int((fico - 620) // 10)))
    elif grade == "D":
        band = max(1, min(5, 6 - int((fico - 580) // 10)))
    elif grade == "E":
        band = max(1, min(5, 6 - int((fico - 550) // 10)))
    else:
        band = 5
    return f"{grade}{band}"


def _amortized_installment(loan_amount: float, annual_rate: float, term_months: int) -> float:
    if loan_amount <= 0 or term_months <= 0:
        return 0.0
    monthly_rate = annual_rate / 100.0 / 12.0
    if monthly_rate <= 0:
        return loan_amount / term_months
    numerator = monthly_rate * (1 + monthly_rate) ** term_months
    denominator = (1 + monthly_rate) ** term_months - 1
    return loan_amount * numerator / denominator


def _build_raw_input(input_data: Mapping[str, Any]) -> dict[str, Any]:
    defaults = _reference_defaults()
    raw = dict(defaults)

    alias_map = {
        "income": "annual_inc",
        "loan_amount": "loan_amnt",
        "credit_score": "fico_range_low",
        "fico": "fico_range_low",
        "borrower_name": "borrower",
        "borrower": "borrower",
    }

    for key, value in input_data.items():
        canonical_key = alias_map.get(key, key)
        raw[canonical_key] = value

    annual_inc = _coerce_float(raw.get("annual_inc"), defaults.get("annual_inc", 0.0))
    loan_amnt = _coerce_float(raw.get("loan_amnt"), defaults.get("loan_amnt", 0.0))
    fico_low = _coerce_float(raw.get("fico_range_low"), defaults.get("fico_range_low", 0.0))
    int_rate = _coerce_float(raw.get("int_rate"), defaults.get("int_rate", 0.0))
    dti = _coerce_float(raw.get("dti"), defaults.get("dti", 0.0))

    raw["annual_inc"] = annual_inc
    raw["loan_amnt"] = loan_amnt
    raw["fico_range_low"] = fico_low
    raw["fico_range_high"] = _coerce_float(raw.get("fico_range_high"), fico_low + 4.0 if fico_low else defaults.get("fico_range_high", fico_low))
    raw["int_rate"] = int_rate
    raw["dti"] = dti
    raw["revol_util"] = _coerce_float(raw.get("revol_util"), defaults.get("revol_util", 0.0))
    raw["open_acc"] = _coerce_float(raw.get("open_acc"), defaults.get("open_acc", 0.0))
    raw["delinq_2yrs"] = _coerce_float(raw.get("delinq_2yrs"), defaults.get("delinq_2yrs", 0.0))
    raw["inq_last_6mths"] = _coerce_float(raw.get("inq_last_6mths"), defaults.get("inq_last_6mths", 0.0))
    raw["pub_rec"] = _coerce_float(raw.get("pub_rec"), defaults.get("pub_rec", 0.0))

    if raw.get("term") is None or str(raw.get("term")).strip() == "":
        raw["term"] = defaults.get("term", "36 months")
    else:
        raw["term"] = _normalize_term(raw.get("term"))

    raw["purpose"] = _normalize_purpose(raw.get("purpose"))
    raw["home_ownership"] = _normalize_home_ownership(raw.get("home_ownership"))
    raw["verification_status"] = _normalize_verification_status(raw.get("verification_status"), defaults.get("verification_status", "Not Verified"))
    raw["emp_length"] = _normalize_emp_length(raw.get("emp_length"), defaults.get("emp_length", "2 years"))

    raw["grade"] = str(raw.get("grade") or _grade_from_fico(fico_low))
    raw["sub_grade"] = str(raw.get("sub_grade") or _sub_grade_from_fico(fico_low))

    if not raw.get("installment"):
        term_months = 60 if str(raw["term"]).strip() == "60 months" else 36
        raw["installment"] = round(_amortized_installment(loan_amnt, int_rate, term_months), 2)
    else:
        raw["installment"] = _coerce_float(raw.get("installment"), defaults.get("installment", 0.0))

    raw["revol_bal"] = _coerce_float(raw.get("revol_bal"), defaults.get("revol_bal", 0.0))
    raw["total_acc"] = _coerce_float(raw.get("total_acc"), defaults.get("total_acc", 0.0))
    raw["bc_open_to_buy"] = _coerce_float(raw.get("bc_open_to_buy"), defaults.get("bc_open_to_buy", 0.0))
    raw["bc_util"] = _coerce_float(raw.get("bc_util"), defaults.get("bc_util", 0.0))
    raw["tot_cur_bal"] = _coerce_float(raw.get("tot_cur_bal"), defaults.get("tot_cur_bal", 0.0))
    raw["avg_cur_bal"] = _coerce_float(raw.get("avg_cur_bal"), defaults.get("avg_cur_bal", 0.0))
    raw["num_actv_bc_tl"] = _coerce_float(raw.get("num_actv_bc_tl"), defaults.get("num_actv_bc_tl", 0.0))
    raw["num_rev_accts"] = _coerce_float(raw.get("num_rev_accts"), defaults.get("num_rev_accts", 0.0))
    raw["percent_bc_gt_75"] = _coerce_float(raw.get("percent_bc_gt_75"), defaults.get("percent_bc_gt_75", 0.0))
    raw["pub_rec_bankruptcies"] = _coerce_float(raw.get("pub_rec_bankruptcies"), defaults.get("pub_rec_bankruptcies", 0.0))
    raw["collections_12_mths_ex_med"] = _coerce_float(raw.get("collections_12_mths_ex_med"), defaults.get("collections_12_mths_ex_med", 0.0))
    raw["acc_now_delinq"] = _coerce_float(raw.get("acc_now_delinq"), defaults.get("acc_now_delinq", 0.0))
    raw["tot_coll_amt"] = _coerce_float(raw.get("tot_coll_amt"), defaults.get("tot_coll_amt", 0.0))
    raw["revol_util"] = _coerce_float(raw.get("revol_util"), defaults.get("revol_util", 0.0))

    raw["earliest_cr_line"] = str(raw.get("earliest_cr_line") or defaults.get("earliest_cr_line", "Jan-00"))
    raw["addr_state"] = str(raw.get("addr_state") or defaults.get("addr_state", "CA"))

    return raw


def _prepare_model_frame(raw_input: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = _build_raw_input(raw_input)
    frame = pd.DataFrame([raw])

    frame = engineer_features(frame)
    frame = create_features(frame)

    # Apply any remaining defaults for columns created during training.
    for column, default_value in _reference_defaults().items():
        if column not in frame.columns:
            frame[column] = default_value

    encoded = pd.get_dummies(frame, drop_first=True)
    encoded.columns = sanitize_columns(encoded.columns)
    encoded = encoded.astype("float32")

    feature_names = _load_feature_names()
    encoded = encoded.reindex(columns=feature_names, fill_value=0.0)
    return encoded, raw


def _build_advice(probability: float, raw_input: Mapping[str, Any], prediction: int, override: bool) -> list[str]:
    advice: list[str] = []

    annual_inc = _coerce_float(raw_input.get("annual_inc"), 0.0)
    loan_amnt = _coerce_float(raw_input.get("loan_amnt"), 0.0)
    dti = _coerce_float(raw_input.get("dti"), 0.0)
    fico = _coerce_float(raw_input.get("fico_range_low"), 0.0)
    revol_util = _coerce_float(raw_input.get("revol_util"), 0.0)

    if override:
        advice.append("Loan amount is more than 5x annual income, so the business override flagged this application.")
    if probability >= 0.6:
        advice.append("Recommend manual underwriting review before approval.")
    if dti >= 20:
        advice.append("Debt-to-income ratio is elevated; reducing monthly obligations would improve affordability.")
    if fico < 600:
        advice.append("Credit score is below the preferred threshold; consider credit repair or additional collateral.")
    if revol_util >= 75:
        advice.append("Revolving credit utilization is high; lowering card balances could reduce risk.")
    if annual_inc > 0 and loan_amnt / annual_inc >= 1:
        advice.append("Requested loan size is large relative to income; verify repayment capacity carefully.")
    if prediction == 0 and probability < 0.4:
        advice.append("Profile is within the lower-risk band, but standard verification still applies.")

    if not advice:
        advice.append("No additional adverse risk signals were detected beyond the model output.")

    return advice


def _append_history(record: dict[str, Any]) -> None:
    history_path = Path(HISTORY_PATH)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(history_path) as f:
            history = json.load(f)
        if not isinstance(history, list):
            history = []
    except Exception:
        history = []

    history.insert(0, record)

    with open(history_path, "w") as f:
        json.dump(history[:1000], f, indent=2)


def predict(input_data: Mapping[str, Any]) -> dict[str, Any]:
    """Predict loan default risk for a single applicant payload."""

    model = _load_model()
    model_frame, raw_input = _prepare_model_frame(input_data)

    probability = float(model.predict_proba(model_frame)[0][1])
    threshold = _load_threshold()
    override = _coerce_float(raw_input.get("loan_amnt"), 0.0) > 5 * max(_coerce_float(raw_input.get("annual_inc"), 0.0), 1.0)

    prediction = 1 if override or probability >= threshold else 0
    risk_level = get_risk_level(probability)

    shap_values = []
    try:
        shap_values = get_local_shap(model_frame)
    except Exception:
        log.exception("Local SHAP explanation failed; returning empty explanation list")

    advice = _build_advice(probability, raw_input, prediction, override)

    result = {
        "prediction": int(prediction),
        "prediction_label": "Default" if prediction == 1 else "Repay",
        "default_probability": probability,
        "threshold_used": threshold,
        "risk_level": risk_level,
        "override_applied": bool(override),
        "shap_values": shap_values,
        "advice": advice,
        "raw_input": raw_input,
        "model_features": _load_feature_names(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    history_record = {
        "timestamp": result["timestamp"],
        "raw_input": raw_input,
        "prediction": result["prediction_label"],
        "prediction_numeric": result["prediction"],
        "default_probability": round(probability, 6),
        "threshold_used": threshold,
        "risk_level": risk_level["label"],
        "override_applied": bool(override),
        "advice": advice,
        "shap_values": shap_values,
    }

    _append_history(history_record)

    try:
        log_decision(history_record)
    except Exception:
        log.exception("Failed to write governance log; continuing without audit persistence")

    return result


__all__ = ["predict"]

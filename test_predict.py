# test_predict.py
"""
Smoke-test: run a single prediction without Flask or a trained model artefact.
Useful during CI / local dev to verify the preprocessing pipeline does not crash.

Usage:
    python test_predict.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _run_smoke_test() -> None:
    print("=" * 56)
    print("  Credit Risk Prediction — Pipeline Smoke Test")
    print("=" * 56)

    # ── Load model ───────────────────────────────────────────────────────────
    from utils.config import MODEL_PATH, FEATURES_PATH, METRICS_PATH

    for label, path in [
        ("Model",    MODEL_PATH),
        ("Features", FEATURES_PATH),
        ("Metrics",  METRICS_PATH),
    ]:
        if not Path(path).exists():
            print(f"SKIP: {label} not found at {path} — run src/train_model.py first.")
            return

    # ── Sample applicant ─────────────────────────────────────────────────────
    sample = {
        "loan_amnt":        15_000,
        "int_rate":         12.5,
        "installment":      335.0,
        "annual_inc":       75_000,
        "dti":              18.5,
        "fico_range_low":   690,
        "fico_range_high":  694,
        "open_acc":         10,
        "total_acc":        22,
        "revol_bal":        8_200,
        "revol_util":       35,
        "bc_open_to_buy":   5_000,
        "delinq_2yrs":      0,
        "inq_last_6mths":   1,
        "pub_rec":          0,
        "term":             "36 months",
        "grade":            "B",
        "sub_grade":        "B3",
        "emp_length":       "5 years",
        "home_ownership":   "RENT",
        "verification_status": "Verified",
        "purpose":          "debt_consolidation",
        "addr_state":       "CA",
        "borrower_name":    "Test Applicant",
    }

    print("\nApplicant:")
    for k, v in sample.items():
        print(f"  {k:<25}: {v}")
    print()

    # ── Run prediction ───────────────────────────────────────────────────────
    from src.predict import predict
    result = predict(sample)

    print(f"Prediction        : {result['prediction_label']}")
    print(f"Default probability: {result['default_probability']:.4f}")
    print(f"Risk level        : {result['risk_level']['label']}")
    print(f"Threshold used    : {result['threshold_used']}")
    print(f"Override applied  : {result['override_applied']}")

    if result.get("shap_values"):
        print("\nTop SHAP features:")
        for feat in result["shap_values"]:
            print(f"  {feat['feature']:<35} {feat['shap_value']:+.6f}")

    if result.get("advice"):
        print("\nAdvice:")
        for a in result["advice"]:
            print(f"  • {a}")

    print("\n✅ Smoke test passed")


if __name__ == "__main__":
    _run_smoke_test()

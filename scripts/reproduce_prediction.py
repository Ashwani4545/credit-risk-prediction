# scripts/reproduce_prediction.py
"""
Reproduce a prediction from the audit log by record ID.

Usage:
    python scripts/reproduce_prediction.py --id <record_id>
    python scripts/reproduce_prediction.py --latest
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.predict import predict
from utils.config import HISTORY_PATH


def _load_history() -> list:
    try:
        with open(HISTORY_PATH) as f:
            return json.load(f)
    except Exception as exc:
        print(f"ERROR: Cannot read history — {exc}")
        sys.exit(1)


def reproduce(record_id: str | None = None, latest: bool = False) -> None:
    history = _load_history()
    if not history:
        print("ERROR: History is empty.")
        sys.exit(1)

    if latest:
        record = history[0]
    elif record_id:
        record = next((r for r in history if r.get("id") == record_id), None)
        if record is None:
            print(f"ERROR: Record {record_id!r} not found in history.")
            sys.exit(1)
    else:
        print("ERROR: Provide --id <id> or --latest")
        sys.exit(1)

    raw_input = record.get("raw_input", {})
    if not raw_input:
        print("ERROR: No raw_input saved in this record.")
        sys.exit(1)

    print(f"\nReproducing prediction for record: {record.get('id', 'unknown')}")
    print(f"Original timestamp : {record.get('timestamp', 'unknown')}")
    print(f"Original prediction: {record.get('prediction', 'unknown')}")
    print(f"Original prob      : {record.get('default_probability', record.get('probability', 'unknown'))}")
    print("-" * 50)

    result = predict(raw_input)

    print(f"Reproduced prediction : {result['prediction_label']}")
    print(f"Reproduced probability: {result['default_probability']:.6f}")
    print(f"Risk level            : {result['risk_level']['label']}")
    print(f"Threshold used        : {result['threshold_used']}")

    if result.get("shap_values"):
        print("\nTop SHAP features:")
        for feat in result["shap_values"]:
            print(f"  {feat['feature']:<35} {feat['shap_value']:+.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce a stored prediction")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--id",     type=str, help="Record ID to reproduce")
    group.add_argument("--latest", action="store_true", help="Reproduce the most recent prediction")
    args = parser.parse_args()

    reproduce(record_id=args.id, latest=args.latest)

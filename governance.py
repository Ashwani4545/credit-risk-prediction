# governance.py
"""
Governance — append every prediction decision to an append-only audit log.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

log = logging.getLogger(__name__)

# FIX: Use absolute path derived from this file so the log ends up in
# the project root's logs/ directory regardless of the working directory
# when the Flask app is launched (previously "logs/audit_log.json" was
# relative to cwd, which broke when app.py was run from webapp/).
AUDIT_LOG_PATH = Path(__file__).resolve().parent / "logs" / "audit_log.json"


def log_decision(record: dict) -> None:
    """Append a prediction record to the audit log (capped at 10 000 entries)."""
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(AUDIT_LOG_PATH) as f:
            logs = json.load(f)
        if not isinstance(logs, list):
            logs = []
    except Exception:
        logs = []

    # Stamp with write time if not already present
    entry = dict(record)
    entry.setdefault("audit_timestamp", datetime.now(timezone.utc).isoformat())

    logs.insert(0, entry)

    with open(AUDIT_LOG_PATH, "w") as f:
        json.dump(logs[:10_000], f, indent=2, default=str)

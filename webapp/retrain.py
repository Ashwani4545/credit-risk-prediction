# webapp/retrain.py
"""
Retrain the XGBoost model by invoking the training script as a subprocess.
Called by app.py when either the prediction count threshold or drift is triggered.
"""

import logging
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

# Project root is one level above webapp/
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)


def retrain_model() -> bool:
    """
    Run `python -m src.train_model` from the project root.
    Returns True on success, False on failure.
    """
    log.info("🔄 Retraining model …")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "src.train_model"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        log.info("✅ Model retrained successfully")
        if result.stdout:
            log.debug("stdout: %s", result.stdout[-500:])
        return True
    except subprocess.CalledProcessError as exc:
        log.error("❌ Retraining failed (exit %d): %s", exc.returncode, exc.stderr[-500:])
        return False
    except Exception as exc:
        log.error("❌ Retraining failed: %s", exc)
        return False

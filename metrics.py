
"""
metrics.py
~~~~~~~~~~

Minimal metrics recorder used to track micro-finetune steps.
This acts as a lightweight log instead of an external service.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict

from config import settings

_METRIC_PATH = settings.log_dir / "metrics.json"


def log_micro_update(model: str, delta: float) -> None:
    """Append a micro-finetune step to the metrics log."""  # MICRO-FINETUNE-ADD
    _METRIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    if _METRIC_PATH.exists():
        with open(_METRIC_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    else:
        data = {}
    entry = data.get(model, {"total_delta": 0.0, "steps": []})
    entry["total_delta"] += delta
    entry["steps"].append({"ts": datetime.utcnow().isoformat(), "delta": delta})
    data[model] = entry
    with open(_METRIC_PATH, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)

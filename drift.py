"""
drift.py
~~~~~~~~

Tracks behavioral drift between two local LLMs using
a sliding window of debate outcomes.

Metrics
-------
- agreement_rate           : % of debates where models agree
- guardian_override_rate   : % escalated to guardian that disagreed
- avg_response_length_delta: Δ in response length between models

Trigger
-------
- `check_drift(metrics)` emits warnings and returns True
  if drift exceeds defined heuristics.

Configuration pulled from `settings.drift_window_size`.

"""

import logging
from collections import deque
from typing import Deque, Dict

from config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Metrics Window
# ---------------------------------------------------------------------
class DriftWindow:
    """
    Sliding window to track last N debate metrics.
    """

    def __init__(self, size: int = settings.drift_window_size) -> None:
        self.size = size
        self._window: Deque[Dict[str, float]] = deque(maxlen=size)

    def update(self, metrics: Dict[str, float]) -> None:
        """
        Add a new debate's metrics to the sliding window.
        """
        self._window.append(metrics)

    def compute_summary(self) -> Dict[str, float]:
        """
        Compute average values over the current window.
        Returns a dict with keys:
            - agreement_rate
            - guardian_override_rate
            - avg_response_length_delta
        """
        if not self._window:
            return {"agreement_rate": 0.0, "guardian_override_rate": 0.0, "avg_response_length_delta": 0.0}

        keys = ["agreement_rate", "guardian_override_rate", "avg_response_length_delta"]
        summary = {k: 0.0 for k in keys}
        for m in self._window:
            for k in keys:
                summary[k] += m.get(k, 0.0)
        for k in keys:
            summary[k] /= len(self._window)
        return summary


# ---------------------------------------------------------------------
# Drift Detection Logic
# ---------------------------------------------------------------------
def check_drift(metrics: Dict[str, float]) -> bool:
    """
    Evaluate drift metrics and determine if anomaly is present.

    Parameters
    ----------
    metrics : dict
        Summary metrics as returned by `DriftWindow.compute_summary`.

    Returns
    -------
    bool
        True if drift is detected (e.g. models diverge too often).
    """
    logger.debug("Checking drift on metrics: %s", metrics)

    drift = False
    if metrics.get("agreement_rate", 1.0) < 0.4:
        drift = True
    if metrics.get("guardian_override_rate", 0.0) > 0.5:
        drift = True
    if metrics.get("avg_response_length_delta", 0.0) > 100:
        drift = True

    if drift:
        logger.warning("Potential drift detected: %s", metrics)
    return drift

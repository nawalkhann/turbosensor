"""Turning RUL estimates into maintenance decisions.

A point estimate is not what a planner acts on. Pulling an engine is decided on
a conservative bound: if the lower end of the predicted RUL interval falls inside
the planning horizon, the engine is scheduled for a shop visit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def maintenance_plan(latest: pd.DataFrame, horizon: int = 30, watch: int = 60) -> pd.DataFrame:
    """Classify each engine as ``replace``, ``watch`` or ``ok``.

    ``latest`` is the output of :meth:`RULModel.predict_latest`. The decision uses
    ``rul_lower`` when available, otherwise the point estimate.
    """
    out = latest.copy()
    basis = out["rul_lower"] if "rul_lower" in out else out["rul_pred"]
    out["decision_rul"] = basis
    out["action"] = np.select([basis <= horizon, basis <= watch], ["replace", "watch"], default="ok")
    return out.sort_values("decision_rul").reset_index(drop=True)


def alert_metrics(rul_true, decision_rul, horizon: int = 30) -> dict[str, float]:
    """How well a horizon alert catches engines that truly fail inside it."""
    truth = np.asarray(rul_true) <= horizon
    alert = np.asarray(decision_rul) <= horizon
    tp = int(np.sum(truth & alert))
    fp = int(np.sum(~truth & alert))
    fn = int(np.sum(truth & ~alert))
    return {
        "horizon": horizon,
        "engines_at_risk": int(truth.sum()),
        "alerts": int(alert.sum()),
        "caught": tp,
        "missed": fn,
        "false_alarms": fp,
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
    }

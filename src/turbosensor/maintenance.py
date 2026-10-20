"""Turning RUL estimates into maintenance decisions.

The planner asks one question per engine: will it fail inside the planning
horizon? Two mistakes are possible and they do not cost the same:

* a **missed failure** (the engine fails inside the horizon without an alert)
  means an unscheduled removal, priced at ``miss_cost`` cycles of engine life;
* a **false alarm** pulls an engine that still had life left. Its cost is the
  life thrown away, ``false_alarm_cost`` per cycle beyond the horizon, so an
  engine pulled 3 cycles early costs far less than one pulled 80 cycles early.

``fit_alert_threshold`` picks the RUL threshold that minimises total cost on
out-of-fold predictions, so the operating point is learned from training
engines only and the test set is never used to tune it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_HORIZONS = tuple(range(10, 65, 5))


def alert_cost(rul_true, rul_pred, threshold: float, horizon: int, miss_cost: float, false_alarm_cost: float) -> float:
    rul_true = np.asarray(rul_true, float)
    truth = rul_true <= horizon
    alert = np.asarray(rul_pred) <= threshold
    wasted = np.sum((rul_true - horizon)[~truth & alert])
    return float(miss_cost * np.sum(truth & ~alert) + false_alarm_cost * wasted)


def fit_alert_threshold(
    rul_true,
    rul_pred,
    horizon: int,
    miss_cost: float = 100.0,
    false_alarm_cost: float = 1.0,
    grid_step: float = 0.5,
) -> float:
    """Threshold on predicted RUL that minimises total alert cost.

    Ties go to the larger threshold, the safer side.
    """
    rul_true = np.asarray(rul_true, float)
    rul_pred = np.asarray(rul_pred, float)
    grid = np.arange(0, max(horizon * 3, 1) + grid_step, grid_step)
    truth = rul_true <= horizon
    at_risk = np.sort(rul_pred[truth])
    order = np.argsort(rul_pred[~truth])
    healthy = rul_pred[~truth][order]
    wasted = np.concatenate([[0.0], np.cumsum((rul_true[~truth] - horizon)[order])])
    missed = len(at_risk) - np.searchsorted(at_risk, grid, side="right")
    costs = miss_cost * missed + false_alarm_cost * wasted[np.searchsorted(healthy, grid, side="right")]
    best = np.flatnonzero(costs == costs.min())[-1]
    return float(grid[best])


def maintenance_plan(
    latest: pd.DataFrame,
    horizon: int = 30,
    watch: int = 60,
    thresholds: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Classify each engine as ``replace``, ``watch`` or ``ok``.

    ``latest`` is the output of :meth:`RULModel.predict_latest`. With a learned
    threshold for ``horizon`` (``RULModel.alert_thresholds_``) an engine is
    flagged ``replace`` when its predicted RUL is at or below that threshold.
    Without one, the rule falls back to the lower bound of the RUL interval.
    ``watch`` flags engines whose lower bound is inside the wider watch window.
    """
    out = latest.copy()
    lower = out["rul_lower"] if "rul_lower" in out else out["rul_pred"]
    if thresholds and horizon in thresholds:
        out["alert_threshold"] = thresholds[horizon]
        replace = out["rul_pred"] <= thresholds[horizon]
    else:
        replace = lower <= horizon
    out["action"] = np.select([replace, lower <= watch], ["replace", "watch"], default="ok")
    return out.sort_values("rul_pred").reset_index(drop=True)


def alert_metrics(rul_true, rul_basis, horizon: int = 30, threshold: float | None = None) -> dict[str, float]:
    """How well a horizon alert catches engines that truly fail inside it.

    An alert fires when ``rul_basis`` (a point estimate or lower bound) is at or
    below ``threshold``, which defaults to the horizon itself.
    """
    truth = np.asarray(rul_true) <= horizon
    alert = np.asarray(rul_basis) <= (horizon if threshold is None else threshold)
    early = np.asarray(rul_true, float)[~truth & alert] - horizon
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
        # A false alarm pulls an engine early; this is how early, in cycles.
        "false_alarm_mean_cycles_early": float(early.mean()) if len(early) else 0.0,
        "wasted_cycles": float(early.sum()),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
    }

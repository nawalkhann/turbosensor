"""Prognostics metrics, including the asymmetric PHM08 scoring function."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(y_true, y_pred) -> float:
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sqrt(np.mean(d**2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_pred, float) - np.asarray(y_true, float))))


def nasa_score(y_true, y_pred) -> float:
    """PHM08 challenge score (Saxena et al., 2008). Lower is better.

    Over-estimating RUL (predicting failure later than it happens) is penalised
    more steeply than under-estimating, because a late prediction means an engine
    keeps flying past the point it should have been pulled.
    """
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


def interval_coverage(y_true, lower, upper) -> float:
    y = np.asarray(y_true, float)
    return float(np.mean((y >= np.asarray(lower)) & (y <= np.asarray(upper))))


def regression_report(y_true, y_pred, lower=None, upper=None) -> dict[str, float]:
    report = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "nasa_score": nasa_score(y_true, y_pred),
        "late_rate": float(np.mean(np.asarray(y_pred) > np.asarray(y_true))),
        "n": int(len(y_true)),
    }
    if lower is not None and upper is not None:
        report["interval_coverage"] = interval_coverage(y_true, lower, upper)
        report["interval_width"] = float(np.mean(np.asarray(upper) - np.asarray(lower)))
    return report


def error_by_rul_bucket(y_true, y_pred, edges=(0, 25, 50, 75, 100, 125, np.inf)) -> pd.DataFrame:
    """RMSE broken down by how close the engine truly is to failure.

    Errors near end of life matter most operationally, and an aggregate RMSE
    can hide a model that is accurate only on healthy engines.
    """
    df = pd.DataFrame({"y": np.asarray(y_true, float), "p": np.asarray(y_pred, float)})
    df["bucket"] = pd.cut(df["y"], list(edges), right=False)
    return (
        df.groupby("bucket", observed=True)
        .apply(lambda g: pd.Series({"n": len(g), "rmse": rmse(g.y, g.p), "bias": float((g.p - g.y).mean())}))
        .reset_index()
    )

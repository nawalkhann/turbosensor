"""Causal time-series features per engine.

Every feature at cycle t uses only cycles <= t of the same unit, so a model
trained on these features can run online on a live engine without leakage
(``tests/test_features.py`` checks this property directly).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def _per_unit(grouped_result: pd.Series | pd.DataFrame):
    """Drop the group level pandas adds to ``groupby().rolling()`` results."""
    return grouped_result.reset_index(level=0, drop=True)


class RollingFeatures(BaseEstimator, TransformerMixin):
    """Rolling statistics, trend and baseline drift for each sensor.

    * ``mean{w}`` / ``std{w}``: level and noise over trailing windows.
    * ``slope{w}``: least-squares degradation rate over the longest window.
    * ``ewm``: smoothed current value.
    * ``drift``: smoothed value minus the engine's own early-life baseline, which
      removes the unit-to-unit manufacturing variation the dataset describes.
    """

    def __init__(
        self,
        sensors: Sequence[str] | None = None,
        windows: Sequence[int] = (5, 15, 30),
        ewm_span: int = 10,
        baseline_cycles: int = 15,
    ):
        self.sensors = sensors
        self.windows = windows
        self.ewm_span = ewm_span
        self.baseline_cycles = baseline_cycles

    def fit(self, X: pd.DataFrame, y=None):
        self.sensors_ = list(self.sensors) if self.sensors else [c for c in X.columns if c.startswith("s_")]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.sort_values(["unit", "cycle"])
        cols = self.sensors_
        S = X[cols]
        g = S.groupby(X["unit"], sort=False)
        parts = [X[["unit", "cycle"]], S]

        ewm = _per_unit(g.ewm(span=self.ewm_span).mean())
        parts.append(ewm.add_suffix("_ewm"))

        position = X.groupby("unit").cumcount()
        early = _per_unit(g.expanding().mean())
        # Freeze the baseline once the early-life window has passed.
        early[position.to_numpy() >= self.baseline_cycles] = np.nan
        early = early.groupby(X["unit"]).ffill()
        parts.append((ewm - early).add_suffix("_drift"))

        for w in self.windows:
            parts.append(_per_unit(g.rolling(w, min_periods=1).mean()).add_suffix(f"_mean{w}"))
            parts.append(_per_unit(g.rolling(w, min_periods=2).std()).fillna(0.0).add_suffix(f"_std{w}"))

        w = max(self.windows)
        t = X["cycle"].astype(float)
        T = pd.DataFrame({c: t for c in cols}, index=X.index)
        gt = T.groupby(X["unit"], sort=False)
        gst = (S * T).groupby(X["unit"], sort=False)
        gtt = (T * T).groupby(X["unit"], sort=False)
        m_s = _per_unit(g.rolling(w, min_periods=2).mean())
        m_t = _per_unit(gt.rolling(w, min_periods=2).mean())
        m_st = _per_unit(gst.rolling(w, min_periods=2).mean())
        m_tt = _per_unit(gtt.rolling(w, min_periods=2).mean())
        var_t = (m_tt - m_t**2).replace(0, np.nan)
        slope = ((m_st - m_s * m_t) / var_t).fillna(0.0)
        parts.append(slope.add_suffix(f"_slope{w}"))

        out = pd.concat(parts, axis=1)
        if "regime" in X:
            out["regime"] = X["regime"]
        return out.loc[X.index]


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Model inputs: everything except identifiers and targets."""
    return [c for c in frame.columns if c not in {"unit", "rul", "rul_true"}]

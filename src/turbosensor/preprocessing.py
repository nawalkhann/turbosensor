"""Operating-condition normalisation and sensor selection.

FD002 and FD004 are flown in six operating regimes (altitude, Mach, throttle).
Raw sensor values jump between regimes by far more than degradation moves them,
so each sensor is standardised against statistics of its own regime. On the
single-condition subsets this reduces to plain standardisation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans

from .data import SENSOR_COLS, SETTING_COLS


class ConditionNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, n_conditions: int | str = "auto", min_std: float = 1e-4, random_state: int = 0):
        self.n_conditions = n_conditions
        self.min_std = min_std
        self.random_state = random_state

    def _n_conditions(self, X: pd.DataFrame) -> int:
        if self.n_conditions != "auto":
            return int(self.n_conditions)
        # Settings are tightly clustered; rounding reveals how many regimes exist.
        distinct = X[SETTING_COLS].round(0).drop_duplicates()
        return 1 if len(distinct) <= 2 else 6

    def fit(self, X: pd.DataFrame, y=None):
        k = self._n_conditions(X)
        self.n_conditions_ = k
        if k > 1:
            self.kmeans_ = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
            regimes = self.kmeans_.fit_predict(X[SETTING_COLS].to_numpy())
        else:
            self.kmeans_ = None
            regimes = np.zeros(len(X), dtype=int)

        sensors = X[SENSOR_COLS]
        grouped = sensors.groupby(regimes)
        self.mean_ = grouped.mean()
        self.std_ = grouped.std(ddof=0)
        # Sensors that never move within any regime carry no degradation signal.
        informative = (self.std_ > self.min_std).all(axis=0)
        self.sensors_ = [c for c in SENSOR_COLS if informative[c]]
        return self

    def regimes(self, X: pd.DataFrame) -> np.ndarray:
        if self.kmeans_ is None:
            return np.zeros(len(X), dtype=int)
        return self.kmeans_.predict(X[SETTING_COLS].to_numpy())

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        regimes = self.regimes(X)
        mean = self.mean_.loc[regimes, self.sensors_].to_numpy()
        std = self.std_.loc[regimes, self.sensors_].to_numpy()
        out = X.drop(columns=SENSOR_COLS + SETTING_COLS).copy()
        out[self.sensors_] = (X[self.sensors_].to_numpy() - mean) / std
        out["regime"] = regimes
        return out

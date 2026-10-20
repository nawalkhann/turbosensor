"""End-to-end RUL estimator: normalisation, features, regressor and intervals."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import __version__
from .data import last_cycles
from .features import RollingFeatures, feature_columns
from .maintenance import DEFAULT_HORIZONS, fit_alert_threshold
from .preprocessing import ConditionNormalizer

logger = logging.getLogger(__name__)

DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "n_estimators": 600,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.6,
    "reg_lambda": 1.0,
    "verbose": -1,
}


class SeedEnsemble:
    """Average of identically configured models trained with different seeds.

    Row and column subsampling make each boosted model a slightly different
    estimate; averaging them removes that variance at no cost in bias.
    """

    def __init__(self, models: list):
        self.models = models

    def fit(self, X, y):
        for m in self.models:
            m.fit(X, y)
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)


def make_regressor(kind: str, params: dict[str, Any] | None = None, seed: int = 0, n_seeds: int = 1):
    params = dict(params or {})
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=params.get("alpha", 1.0)))
    if kind == "lgbm":
        from lightgbm import LGBMRegressor

        members = [LGBMRegressor(**{**DEFAULT_LGBM_PARAMS, **params, "random_state": seed + i}) for i in range(n_seeds)]
        return members[0] if n_seeds == 1 else SeedEnsemble(members)
    raise ValueError(f"Unknown model kind {kind!r}; expected 'ridge' or 'lgbm'")


class RULModel:
    """Predicts remaining useful life (in cycles) from raw C-MAPSS sensor history.

    ``fit`` takes labelled run-to-failure rows; ``predict`` takes any sensor history
    (one or many units) and returns a prediction for every row, using only that
    row's past. ``predict_latest`` returns one estimate per unit at its newest cycle,
    which is what a maintenance planner consumes.

    Prediction intervals come from out-of-fold residuals (split-conformal, binned
    by predicted RUL): the regressor is refit ``cv_folds`` times with whole engines
    held out, and the spread of its held-out errors in each RUL band sets the
    interval for new predictions in that band. This is calibrated by construction,
    unlike quantile boosting, which breaks down on the capped target because a
    large share of labels sit exactly at the cap. The same out-of-fold predictions
    give the unit-grouped cross-validation score in ``cv_report_``.
    """

    INTERVAL_BINS = (0, 15, 30, 50, 75, 100, np.inf)

    def __init__(
        self,
        kind: str = "lgbm",
        params: dict[str, Any] | None = None,
        windows: tuple[int, ...] = (5, 15, 30, 60),
        rul_cap: int = 125,
        interval: tuple[float, float] | None = (0.1, 0.9),
        cv_folds: int = 5,
        n_seeds: int = 1,
        miss_cost: float = 100.0,
        false_alarm_cost: float = 1.0,
        cnn: dict[str, Any] | None = None,
        seed: int = 0,
    ):
        self.kind = kind
        self.params = params or {}
        self.windows = tuple(windows)
        self.rul_cap = rul_cap
        self.interval = interval
        self.cv_folds = cv_folds
        self.n_seeds = n_seeds
        self.miss_cost = miss_cost
        self.false_alarm_cost = false_alarm_cost
        self.cnn = cnn
        self.seed = seed

    # -- training ---------------------------------------------------------
    def _features(self, raw: pd.DataFrame) -> pd.DataFrame:
        return self.features_.transform(self.normalizer_.transform(raw))

    def fit(self, train: pd.DataFrame) -> RULModel:
        if "rul" not in train:
            raise ValueError("training frame needs an 'rul' column; see data.add_rul")
        self.normalizer_ = ConditionNormalizer(random_state=self.seed).fit(train)
        self.features_ = RollingFeatures(sensors=self.normalizer_.sensors_, windows=self.windows).fit(train)
        F = self._features(train)
        y = train.loc[F.index, "rul"].to_numpy(dtype=float)
        self.feature_names_ = feature_columns(F)
        X = F[self.feature_names_]

        units = F["unit"].to_numpy()
        folds = list(GroupKFold(n_splits=self.cv_folds).split(X, groups=units)) if self.cv_folds > 1 else []

        self.regressor_ = make_regressor(self.kind, self.params, self.seed, self.n_seeds).fit(X, y)
        self.residual_quantiles_ = None
        self.cv_report_ = None
        self.alert_thresholds_ = {}
        self.cnn_ = None
        self.cnn_weight_ = 0.0
        oof = self._out_of_fold(X, y, folds) if folds else None

        if self.cnn is not None:
            from .sequence import CNNRegressor

            S = self._sequences(train, F.index)
            cnn_args = {"seed": self.seed, **self.cnn}
            if folds:
                oof_cnn = np.empty(len(y))
                for fit_idx, val_idx in folds:
                    oof_cnn[val_idx] = CNNRegressor(**cnn_args).fit(S[fit_idx], y[fit_idx]).predict(S[val_idx])
                oof_cnn = np.clip(oof_cnn, 0, self.rul_cap)
                self.cnn_weight_ = self._blend_weight(y, oof, oof_cnn)
                cnn_cv = float(np.sqrt(np.mean((oof_cnn - y) ** 2)))
                gbm_cv = float(np.sqrt(np.mean((oof - y) ** 2)))
                oof = (1 - self.cnn_weight_) * oof + self.cnn_weight_ * oof_cnn
            else:
                self.cnn_weight_ = 0.5
            if self.cnn_weight_ > 0:
                self.cnn_ = CNNRegressor(**cnn_args).fit(S, y)

        if oof is not None:
            resid = y - oof
            self.cv_report_ = {
                "cv_rmse": float(np.sqrt(np.mean(resid**2))),
                "cv_mae": float(np.mean(np.abs(resid))),
                "cv_folds": self.cv_folds,
            }
            if self.cnn is not None:
                self.cv_report_.update(cv_rmse_gbm=gbm_cv, cv_rmse_cnn=cnn_cv, cnn_weight=self.cnn_weight_)
            if self.interval:
                self.residual_quantiles_ = self._binned_quantiles(oof, resid)
            self.alert_thresholds_ = {
                h: fit_alert_threshold(y, oof, h, self.miss_cost, self.false_alarm_cost) for h in DEFAULT_HORIZONS
            }
        logger.info("Fitted %s on %d rows, %d features", self.kind, len(X), X.shape[1])
        return self

    def _sequences(self, raw: pd.DataFrame, order: pd.Index) -> np.ndarray:
        """CNN input windows for every row of ``raw``, in the order of ``order``."""
        from .sequence import sequence_inputs, windows

        Z = self.normalizer_.transform(raw)
        S, index = windows(sequence_inputs(Z, self.normalizer_.sensors_), self.cnn.get("window", 60))
        return S[index.get_indexer(order)]

    @staticmethod
    def _blend_weight(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        """Share of ``b`` in a two-model blend that minimises out-of-fold RMSE."""
        grid = np.round(np.arange(0, 1.0001, 0.05), 2)
        errors = [np.mean(((1 - w) * a + w * b - y) ** 2) for w in grid]
        return float(grid[int(np.argmin(errors))])

    def _out_of_fold(self, X: pd.DataFrame, y: np.ndarray, folds: list) -> np.ndarray:
        """Predictions for every row from a model that never saw that row's engine."""
        oof = np.empty(len(y))
        for fit_idx, val_idx in folds:
            reg = make_regressor(self.kind, self.params, self.seed, self.n_seeds).fit(X.iloc[fit_idx], y[fit_idx])
            oof[val_idx] = reg.predict(X.iloc[val_idx])
        return np.clip(oof, 0, self.rul_cap)

    def _binned_quantiles(self, pred: np.ndarray, resid: np.ndarray) -> pd.DataFrame:
        lo_q, hi_q = self.interval
        bins = pd.cut(pred, list(self.INTERVAL_BINS), right=False, labels=False)
        table = pd.DataFrame({"bin": bins, "r": resid}).groupby("bin")["r"].quantile([lo_q, hi_q]).unstack()
        table.columns = ["lo", "hi"]
        # Bands with no out-of-fold rows fall back to the global residual spread.
        overall = np.quantile(resid, [lo_q, hi_q])
        return table.reindex(range(len(self.INTERVAL_BINS) - 1)).fillna({"lo": overall[0], "hi": overall[1]})

    # -- inference --------------------------------------------------------
    def predict_frame(self, raw: pd.DataFrame) -> pd.DataFrame:
        F = self._features(raw)
        X = F[self.feature_names_]
        out = F[["unit", "cycle"]].copy()
        pred = np.clip(self.regressor_.predict(X), 0, self.rul_cap)
        if self.cnn_ is not None:
            cnn = np.clip(self.cnn_.predict(self._sequences(raw, F.index)), 0, self.rul_cap)
            pred = (1 - self.cnn_weight_) * pred + self.cnn_weight_ * cnn
        out["rul_pred"] = pred
        if self.residual_quantiles_ is not None:
            bins = pd.cut(out["rul_pred"], list(self.INTERVAL_BINS), right=False, labels=False).to_numpy()
            q = self.residual_quantiles_.loc[bins]
            out["rul_lower"] = np.clip(out["rul_pred"].to_numpy() + np.minimum(q["lo"].to_numpy(), 0), 0, None)
            out["rul_upper"] = out["rul_pred"].to_numpy() + np.maximum(q["hi"].to_numpy(), 0)
        return out.sort_index()

    def predict(self, raw: pd.DataFrame) -> np.ndarray:
        return self.predict_frame(raw)["rul_pred"].to_numpy()

    def predict_latest(self, raw: pd.DataFrame) -> pd.DataFrame:
        return last_cycles(self.predict_frame(raw))

    def feature_importance(self, top: int = 20) -> pd.Series:
        reg = self.regressor_
        members = reg.models if isinstance(reg, SeedEnsemble) else [reg]
        if hasattr(members[0], "booster_"):
            values = np.mean([m.booster_.feature_importance(importance_type="gain") for m in members], axis=0)
        else:
            values = np.abs(reg[-1].coef_)
        s = pd.Series(values, index=self.feature_names_, name="importance")
        return (s / s.sum()).sort_values(ascending=False).head(top)

    # -- persistence ------------------------------------------------------
    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        meta = {
            "turbosensor_version": __version__,
            "kind": self.kind,
            "rul_cap": self.rul_cap,
            "windows": list(self.windows),
            "interval": list(self.interval) if self.interval else None,
            "sensors": self.normalizer_.sensors_,
            "n_conditions": self.normalizer_.n_conditions_,
            "n_features": len(self.feature_names_),
            "n_seeds": self.n_seeds,
            "cnn": self.cnn,
            "cnn_weight": self.cnn_weight_,
            "alert_thresholds": self.alert_thresholds_,
            **(metadata or {}),
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=float))
        return path

    @staticmethod
    def load(path: str | Path) -> RULModel:
        model = joblib.load(path)
        if not isinstance(model, RULModel):
            raise TypeError(f"{path} does not contain a RULModel")
        return model

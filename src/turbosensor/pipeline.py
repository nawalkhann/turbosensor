"""Training and evaluation workflow driven by a YAML config."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .data import add_rul, load_subset
from .evaluation import error_by_rul_bucket, regression_report
from .maintenance import alert_metrics
from .model import RULModel

logger = logging.getLogger(__name__)


@dataclass
class Config:
    subset: str = "FD001"
    data_dir: str = "data/raw"
    output_dir: str = "artifacts"
    rul_cap: int = 125
    windows: tuple[int, ...] = (5, 15, 30, 60)
    models: dict[str, dict[str, Any]] = field(default_factory=lambda: {"ridge": {}, "lgbm": {}})
    cv_folds: int = 5
    maintenance_horizon: int = 30
    miss_cost: float = 100.0
    false_alarm_cost: float = 1.0
    n_seeds: int = 1
    cnn: dict[str, Any] | None = None  # e.g. {"window": 60, "epochs": 40}; blends a CNN into lgbm
    seed: int = 0

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides) -> Config:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        raw.update({k: v for k, v in overrides.items() if v is not None})
        if "windows" in raw:
            raw["windows"] = tuple(raw["windows"])
        return cls(**raw)


def _build(cfg: Config, kind: str) -> RULModel:
    return RULModel(
        kind=kind,
        params=cfg.models.get(kind) or {},
        windows=cfg.windows,
        rul_cap=cfg.rul_cap,
        cv_folds=cfg.cv_folds,
        n_seeds=cfg.n_seeds,
        miss_cost=cfg.miss_cost,
        false_alarm_cost=cfg.false_alarm_cost,
        cnn=cfg.cnn if kind == "lgbm" else None,
        seed=cfg.seed,
    )


def run(cfg: Config) -> dict[str, Any]:
    out_dir = Path(cfg.output_dir) / cfg.subset
    out_dir.mkdir(parents=True, exist_ok=True)
    split = load_subset(cfg.subset, cfg.data_dir)
    train = add_rul(split.train, cap=cfg.rul_cap)
    test_truth = split.test_rul

    results: dict[str, Any] = {"subset": cfg.subset, "config": asdict(cfg), "models": {}}
    for kind in cfg.models:
        t0 = time.perf_counter()
        logger.info("[%s] fitting %s with %d-fold engine-grouped CV", cfg.subset, kind, cfg.cv_folds)
        model = _build(cfg, kind).fit(train)
        latest = model.predict_latest(split.test)
        y_true = test_truth.loc[latest["unit"]].to_numpy()
        report = regression_report(y_true, latest["rul_pred"], latest.get("rul_lower"), latest.get("rul_upper"))
        # Literature protocol: truth capped like the training target, since the model
        # by design cannot tell 150 from 125 cycles left on an engine with no wear yet.
        capped = regression_report(
            np.minimum(y_true, cfg.rul_cap), latest["rul_pred"], latest.get("rul_lower"), latest.get("rul_upper")
        )
        report["capped"] = {k: v for k, v in capped.items() if k != "n"}
        h = cfg.maintenance_horizon
        threshold = model.alert_thresholds_.get(h)
        report["maintenance"] = alert_metrics(y_true, latest["rul_pred"], h, threshold)
        report["maintenance"]["threshold"] = threshold
        if "rul_lower" in latest:
            report["maintenance_lower_bound_rule"] = alert_metrics(y_true, latest["rul_lower"], h)
        report.update(model.cv_report_ or {})
        report["train_seconds"] = round(time.perf_counter() - t0, 1)
        results["models"][kind] = report

        model.save(out_dir / f"{kind}.joblib", metadata={"subset": cfg.subset, "test": report})
        preds = latest.assign(rul_true=y_true)
        preds.to_csv(out_dir / f"{kind}_test_predictions.csv", index=False)
        error_by_rul_bucket(y_true, latest["rul_pred"]).to_csv(out_dir / f"{kind}_error_by_bucket.csv", index=False)
        logger.info(
            "[%s] %s test RMSE %.2f (capped %.2f), NASA score %.0f, CV RMSE %.2f",
            cfg.subset,
            kind,
            report["rmse"],
            report["capped"]["rmse"],
            report["nasa_score"],
            report.get("cv_rmse", float("nan")),
        )

    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2, default=float))
    return results


def evaluate_saved(model_path: str | Path, subset: str, data_dir: str | Path) -> dict[str, float]:
    model = RULModel.load(model_path)
    split = load_subset(subset, data_dir)
    latest = model.predict_latest(split.test)
    y_true = split.test_rul.loc[latest["unit"]].to_numpy()
    return regression_report(y_true, latest["rul_pred"], latest.get("rul_lower"), latest.get("rul_upper"))


__all__ = ["Config", "run", "evaluate_saved"]

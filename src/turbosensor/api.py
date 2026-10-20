"""REST API for online RUL inference.

POST a unit's recent sensor history and get back its RUL estimate, prediction
interval and maintenance action. Install with ``pip install turbosensor[serve]``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .data import SENSOR_COLS, SETTING_COLS
from .maintenance import maintenance_plan
from .model import RULModel


class Reading(BaseModel):
    unit: int
    cycle: int
    settings: list[float] = Field(min_length=3, max_length=3)
    sensors: list[float] = Field(min_length=21, max_length=21, description="s_1 .. s_21 in dataset order")


class PredictRequest(BaseModel):
    readings: list[Reading] = Field(min_length=1)
    horizon: int = 30


class EnginePrediction(BaseModel):
    unit: int
    cycle: int
    rul_pred: float
    rul_lower: float | None = None
    rul_upper: float | None = None
    alert_threshold: float | None = None
    action: str


def _to_frame(readings: list[Reading]) -> pd.DataFrame:
    rows = [
        {
            "unit": r.unit,
            "cycle": r.cycle,
            **dict(zip(SETTING_COLS, r.settings, strict=True)),
            **dict(zip(SENSOR_COLS, r.sensors, strict=True)),
        }
        for r in readings
    ]
    return pd.DataFrame(rows).sort_values(["unit", "cycle"]).reset_index(drop=True)


def create_app(model: RULModel | str | Path) -> FastAPI:
    if not isinstance(model, RULModel):
        model = RULModel.load(model)
    app = FastAPI(title="TurboSensor RUL API", version=__version__)

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model.kind, "version": __version__}

    @app.post("/predict", response_model=list[EnginePrediction])
    def predict(req: PredictRequest):
        frame = _to_frame(req.readings)
        if frame.duplicated(["unit", "cycle"]).any():
            raise HTTPException(422, "duplicate (unit, cycle) readings")
        plan = maintenance_plan(model.predict_latest(frame), horizon=req.horizon, thresholds=model.alert_thresholds_)
        return plan.to_dict(orient="records")

    return app

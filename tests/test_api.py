import pytest
from conftest import make_fleet

from turbosensor.data import SENSOR_COLS, SETTING_COLS
from turbosensor.model import RULModel

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from turbosensor.api import create_app  # noqa: E402


def test_predict_endpoint(fleet):
    model = RULModel(kind="lgbm", params={"n_estimators": 30}).fit(fleet)
    client = TestClient(create_app(model))
    assert client.get("/health").json()["status"] == "ok"

    hist = make_fleet(n_units=2, seed=3, truncate=True)
    readings = [
        {
            "unit": int(r["unit"]),
            "cycle": int(r["cycle"]),
            "settings": [r[c] for c in SETTING_COLS],
            "sensors": [r[c] for c in SENSOR_COLS],
        }
        for r in hist.to_dict(orient="records")
    ]
    resp = client.post("/predict", json={"readings": readings, "horizon": 30})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert {b["unit"] for b in body} == {1, 2}
    assert all(b["action"] in {"replace", "watch", "ok"} for b in body)


def test_rejects_bad_sensor_count(fleet):
    model = RULModel(kind="ridge").fit(fleet)
    client = TestClient(create_app(model))
    bad = {"readings": [{"unit": 1, "cycle": 1, "settings": [0, 0, 100], "sensors": [1.0] * 5}]}
    assert client.post("/predict", json=bad).status_code == 422

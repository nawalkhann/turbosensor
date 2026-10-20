from pathlib import Path

import numpy as np
import pytest
from conftest import make_fleet

from turbosensor.data import add_rul, load_subset
from turbosensor.evaluation import rmse
from turbosensor.model import RULModel

DATA = Path(__file__).resolve().parents[1] / "data" / "raw"


@pytest.mark.parametrize("kind", ["ridge", "lgbm"])
def test_fit_predict_roundtrip(fleet, kind, tmp_path):
    model = RULModel(kind=kind, params={"n_estimators": 50} if kind == "lgbm" else {}).fit(fleet)
    test = make_fleet(n_units=6, seed=1, truncate=True)
    latest = model.predict_latest(test)
    assert len(latest) == 6
    assert latest["rul_pred"].between(0, model.rul_cap).all()
    if kind == "lgbm":
        assert (latest["rul_lower"] <= latest["rul_pred"]).all()
        assert (latest["rul_pred"] <= latest["rul_upper"]).all()

    path = model.save(tmp_path / "m.joblib")
    assert path.with_suffix(".json").exists()
    np.testing.assert_allclose(RULModel.load(path).predict(test), model.predict(test))


def test_beats_constant_baseline(fleet):
    model = RULModel(kind="lgbm", params={"n_estimators": 100}, interval=None).fit(fleet)
    holdout = make_fleet(n_units=6, seed=2)
    holdout = add_rul(holdout)
    pred = model.predict(holdout)
    assert rmse(holdout["rul"], pred) < 0.5 * rmse(holdout["rul"], np.full(len(pred), holdout["rul"].mean()))


@pytest.mark.data
@pytest.mark.skipif(not (DATA / "train_FD001.txt").exists(), reason="C-MAPSS not downloaded")
def test_fd001_shapes():
    split = load_subset("FD001", DATA)
    assert split.train["unit"].nunique() == 100
    assert len(split.train) == 20631
    assert len(split.test_rul) == split.test["unit"].nunique() == 100


def test_out_of_fold_intervals(fleet):
    model = RULModel(kind="lgbm", params={"n_estimators": 60}, cv_folds=3).fit(fleet)
    assert model.cv_report_["cv_folds"] == 3 and model.cv_report_["cv_rmse"] > 0
    frame = model.predict_frame(fleet)
    assert (frame["rul_lower"] <= frame["rul_pred"]).all() and (frame["rul_pred"] <= frame["rul_upper"]).all()
    assert (frame["rul_lower"] >= 0).all()


def test_cnn_blend(fleet):
    pytest.importorskip("torch")
    model = RULModel(
        kind="lgbm", params={"n_estimators": 40}, cv_folds=3, cnn={"window": 20, "epochs": 2, "threads": 1}
    ).fit(fleet)
    assert 0.0 <= model.cnn_weight_ <= 1.0
    assert {"cv_rmse_gbm", "cv_rmse_cnn", "cnn_weight"} <= set(model.cv_report_)
    latest = model.predict_latest(make_fleet(n_units=3, seed=4, truncate=True))
    assert len(latest) == 3 and latest["rul_pred"].between(0, model.rul_cap).all()

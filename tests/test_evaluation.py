import numpy as np
import pandas as pd
import pytest

from turbosensor.data import add_rul, last_cycles
from turbosensor.evaluation import nasa_score, regression_report, rmse
from turbosensor.maintenance import alert_metrics, maintenance_plan


def test_nasa_score_penalises_late_more_than_early():
    assert nasa_score([50], [50]) == 0
    assert nasa_score([50], [60]) > nasa_score([50], [40])
    assert nasa_score([50], [60]) == pytest.approx(np.e - 1)


def test_rmse():
    assert rmse([0, 0], [3, 4]) == pytest.approx(np.sqrt(12.5))
    assert regression_report([1, 2], [1, 2])["rmse"] == 0


def test_add_rul_run_to_failure_and_test():
    df = pd.DataFrame({"unit": [1, 1, 1, 2, 2], "cycle": [1, 2, 3, 1, 2]})
    out = add_rul(df, cap=1)
    assert out["rul_true"].tolist() == [2, 1, 0, 1, 0]
    assert out["rul"].tolist() == [1, 1, 0, 1, 0]
    out = add_rul(df, cap=None, final_rul=pd.Series({1: 10, 2: 5}))
    assert last_cycles(out)["rul"].tolist() == [10, 5]


def test_maintenance_plan_uses_lower_bound():
    latest = pd.DataFrame({"unit": [1, 2, 3], "cycle": [9, 9, 9], "rul_pred": [40, 80, 20], "rul_lower": [25, 70, 10]})
    plan = maintenance_plan(latest, horizon=30, watch=60).set_index("unit")
    assert plan.loc[[1, 2, 3], "action"].tolist() == ["replace", "ok", "replace"]
    m = alert_metrics([10, 100, 50], [25, 70, 10], horizon=30)
    assert (m["caught"], m["missed"], m["false_alarms"]) == (1, 0, 1)

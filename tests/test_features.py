import numpy as np

from turbosensor.features import RollingFeatures, feature_columns
from turbosensor.preprocessing import ConditionNormalizer


def test_normalizer_drops_constant_sensors(fleet):
    norm = ConditionNormalizer().fit(fleet)
    assert norm.n_conditions_ == 1
    assert "s_1" not in norm.sensors_ and "s_2" in norm.sensors_
    z = norm.transform(fleet)
    assert np.allclose(z[norm.sensors_].mean(), 0, atol=1e-8)


def test_features_are_causal(fleet):
    """Appending future cycles must not change features of past cycles."""
    norm = ConditionNormalizer().fit(fleet)
    z = norm.transform(fleet)
    feats = RollingFeatures(sensors=norm.sensors_).fit(z)
    full = feats.transform(z)
    past = z[z["cycle"] <= 50]
    partial = feats.transform(past)
    cols = feature_columns(partial)
    np.testing.assert_allclose(full.loc[partial.index, cols].to_numpy(), partial[cols].to_numpy())


def test_slope_tracks_trend(fleet):
    norm = ConditionNormalizer().fit(fleet)
    z = norm.transform(fleet)
    out = RollingFeatures(sensors=["s_3"], windows=(30,)).fit(z).transform(z)
    near_end = out.loc[fleet["rul"] < 10, "s_3_slope30"]
    early = out.loc[(fleet["cycle"] > 30) & (fleet["rul"] > 100), "s_3_slope30"]
    assert near_end.mean() > early.abs().mean() * 5

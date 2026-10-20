import numpy as np
import pandas as pd
import pytest

from turbosensor.data import COLUMNS, SENSOR_COLS, SETTING_COLS, add_rul


def make_fleet(n_units: int = 12, seed: int = 0, truncate: bool = False) -> pd.DataFrame:
    """Small synthetic fleet: a few sensors drift exponentially towards failure."""
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(1, n_units + 1):
        life = int(rng.integers(120, 220))
        length = int(rng.integers(40, life)) if truncate else life
        t = np.arange(1, length + 1)
        wear = np.exp((t - life) / 40.0)
        frame = pd.DataFrame({"unit": unit, "cycle": t})
        for c in SETTING_COLS:
            frame[c] = rng.normal(0, 0.001, length)
        for i, c in enumerate(SENSOR_COLS):
            if i % 4 == 0:  # constant sensor, should be dropped
                frame[c] = 100.0
            else:
                frame[c] = 500 + i + rng.normal(0, 0.5, length) + (i % 3) * 5 * wear
        rows.append(frame[COLUMNS])
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="session")
def fleet():
    return add_rul(make_fleet())

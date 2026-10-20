"""Loading and labelling the NASA C-MAPSS turbofan degradation dataset."""

from __future__ import annotations

import io
import logging
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATASET_URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
SUBSETS = ("FD001", "FD002", "FD003", "FD004")

ID_COLS = ["unit", "cycle"]
SETTING_COLS = [f"setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]
COLUMNS = ID_COLS + SETTING_COLS + SENSOR_COLS

# Human-readable sensor names from Saxena et al. (2008), Table 2.
SENSOR_DESCRIPTIONS = {
    "s_1": "Total temperature at fan inlet (T2)",
    "s_2": "Total temperature at LPC outlet (T24)",
    "s_3": "Total temperature at HPC outlet (T30)",
    "s_4": "Total temperature at LPT outlet (T50)",
    "s_5": "Pressure at fan inlet (P2)",
    "s_6": "Total pressure in bypass-duct (P15)",
    "s_7": "Total pressure at HPC outlet (P30)",
    "s_8": "Physical fan speed (Nf)",
    "s_9": "Physical core speed (Nc)",
    "s_10": "Engine pressure ratio (epr)",
    "s_11": "Static pressure at HPC outlet (Ps30)",
    "s_12": "Ratio of fuel flow to Ps30 (phi)",
    "s_13": "Corrected fan speed (NRf)",
    "s_14": "Corrected core speed (NRc)",
    "s_15": "Bypass ratio (BPR)",
    "s_16": "Burner fuel-air ratio (farB)",
    "s_17": "Bleed enthalpy (htBleed)",
    "s_18": "Demanded fan speed (Nf_dmd)",
    "s_19": "Demanded corrected fan speed (PCNfR_dmd)",
    "s_20": "HPT coolant bleed (W31)",
    "s_21": "LPT coolant bleed (W32)",
}


@dataclass
class CMAPSSSplit:
    """One C-MAPSS subset: run-to-failure training data and truncated test data."""

    subset: str
    train: pd.DataFrame
    test: pd.DataFrame
    test_rul: pd.Series  # true RUL at the last observed cycle, indexed by unit


def download(data_dir: str | Path, url: str = DATASET_URL) -> Path:
    """Download and extract the raw text files into ``data_dir``."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if all((data_dir / f"train_{s}.txt").exists() for s in SUBSETS):
        logger.info("C-MAPSS already present in %s", data_dir)
        return data_dir

    logger.info("Downloading C-MAPSS from %s", url)
    with urllib.request.urlopen(url, timeout=120) as resp:
        outer = zipfile.ZipFile(io.BytesIO(resp.read()))
    # The archive nests the data in an inner CMAPSSData.zip.
    inner_name = next(n for n in outer.namelist() if n.endswith("CMAPSSData.zip"))
    inner = zipfile.ZipFile(io.BytesIO(outer.read(inner_name)))
    for name in inner.namelist():
        if name.endswith(".txt"):
            (data_dir / Path(name).name).write_bytes(inner.read(name))
    logger.info("Extracted C-MAPSS to %s", data_dir)
    return data_dir


def read_table(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS, engine="python")
    df[ID_COLS] = df[ID_COLS].astype(int)
    return df


def load_subset(subset: str, data_dir: str | Path) -> CMAPSSSplit:
    subset = subset.upper()
    if subset not in SUBSETS:
        raise ValueError(f"Unknown subset {subset!r}; expected one of {SUBSETS}")
    data_dir = Path(data_dir)
    train = read_table(data_dir / f"train_{subset}.txt")
    test = read_table(data_dir / f"test_{subset}.txt")
    rul = pd.read_csv(data_dir / f"RUL_{subset}.txt", header=None).iloc[:, 0].to_numpy()
    test_rul = pd.Series(rul, index=np.sort(test["unit"].unique()), name="rul")
    return CMAPSSSplit(subset, train, test, test_rul)


def add_rul(df: pd.DataFrame, cap: int | None = 125, final_rul: pd.Series | None = None) -> pd.DataFrame:
    """Attach the remaining-useful-life target to every row.

    For run-to-failure data the last cycle has RUL 0. For truncated test data pass
    ``final_rul`` (RUL at the last observed cycle per unit). ``cap`` applies the
    standard piecewise-linear target: engines are treated as healthy (constant RUL)
    until degradation becomes observable, which avoids asking the model to
    distinguish 300 from 250 cycles left on an engine that shows no wear yet.
    """
    out = df.copy()
    last = out.groupby("unit")["cycle"].transform("max")
    rul = last - out["cycle"]
    if final_rul is not None:
        rul = rul + out["unit"].map(final_rul)
    out["rul_true"] = rul
    out["rul"] = rul.clip(upper=cap) if cap is not None else rul
    return out


def last_cycles(df: pd.DataFrame) -> pd.DataFrame:
    """Rows holding the most recent observation of each unit."""
    idx = df.groupby("unit")["cycle"].idxmax()
    return df.loc[idx].sort_values("unit").reset_index(drop=True)

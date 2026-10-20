"""Generate notebooks/01_exploration_and_results.ipynb (run with nbconvert --execute)."""

from pathlib import Path

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s))  # noqa: E731

md("""# TurboSensor: exploration and results

Remaining-useful-life (RUL) prediction on NASA's C-MAPSS turbofan simulation data.
This notebook walks through what the data looks like, why the pipeline is built the
way it is, and how the trained model performs, both as a regressor and as a
maintenance-alert system.

All modelling code lives in the `turbosensor` package; the notebook only calls it.""")

code("""from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from turbosensor.data import SENSOR_COLS, SENSOR_DESCRIPTIONS, SETTING_COLS, add_rul, load_subset
from turbosensor.evaluation import error_by_rul_bucket, regression_report
from turbosensor.maintenance import maintenance_plan
from turbosensor.model import RULModel
from turbosensor.preprocessing import ConditionNormalizer

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA, ART, FIG = ROOT / "data/raw", ROOT / "artifacts", ROOT / "reports/figures"
FIG.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GRAY, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.titlesize": 11,
    "axes.titlecolor": INK, "lines.linewidth": 1.6, "font.size": 9,
})""")

md(
    "## 1. The data\n\nEach engine starts healthy with unknown initial wear, develops a fault, and runs until failure (training) or is cut off some time before failure (test)."
)

code("""split = load_subset("FD001", DATA)
train = add_rul(split.train, cap=None)
life = train.groupby("unit")["cycle"].max()
print(f"{split.train.unit.nunique()} training engines, {len(split.train):,} cycles")
print(f"engine life: min {life.min()}, median {life.median():.0f}, max {life.max()} cycles")
print(f"{split.test.unit.nunique()} test engines; true RUL at cut-off ranges {split.test_rul.min()}-{split.test_rul.max()}")
split.train.head()""")

md(
    "### Which sensors carry signal?\n\nSeven of the 21 sensors are constant (or nearly) in FD001; the rest drift as the engine degrades. The pipeline drops the flat ones automatically."
)

code("""std = split.train[SENSOR_COLS].std()
corr = train[SENSOR_COLS + ["rul_true"]].corr()["rul_true"].drop("rul_true")
summary = pd.DataFrame({"description": pd.Series(SENSOR_DESCRIPTIONS), "std": std, "corr_with_rul": corr}).round(3)
summary.sort_values("corr_with_rul", key=abs, ascending=False)""")

code("""show = ["s_11", "s_4", "s_12", "s_7", "s_15", "s_21"]
fig, axes = plt.subplots(2, 3, figsize=(11, 5.2), sharex=True)
for ax, s in zip(axes.flat, show):
    for u in range(1, 21):
        g = train[train.unit == u]
        ax.plot(g.rul_true, g[s], color=BLUE, alpha=0.25, lw=0.8)
    ax.invert_xaxis()
    ax.set_title(f"{s}: {SENSOR_DESCRIPTIONS[s]}", loc="left")
for ax in axes[1]:
    ax.set_xlabel("cycles until failure")
fig.suptitle("Sensor trajectories of 20 engines, aligned at failure", x=0.01, ha="left", color=INK)
fig.tight_layout()
fig.savefig(FIG / "sensor_trajectories.png", bbox_inches="tight")""")

md("""Degradation is invisible for most of an engine's life and only bends the sensors in the last ~100-150 cycles. That is why the target is **capped at 125 cycles** (piecewise-linear RUL): asking a model to tell 250 from 200 cycles left on an engine that shows no wear is asking it to memorise noise.

### Operating conditions (FD002 / FD004)

The multi-condition subsets fly in six regimes. Raw sensor values jump between regimes by far more than degradation moves them, so each sensor is standardised against its own regime's statistics.""")

code("""fd2 = load_subset("FD002", DATA)
norm = ConditionNormalizer().fit(fd2.train)
regimes = norm.regimes(fd2.train)
z = norm.transform(fd2.train)
u1 = fd2.train.unit == 1
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
axes[0].plot(fd2.train.loc[u1, "cycle"], fd2.train.loc[u1, "s_11"], color=GRAY, lw=0.8)
axes[0].set_title("FD002 engine 1, raw Ps30: dominated by regime switches", loc="left")
axes[1].plot(z.loc[u1, "cycle"], z.loc[u1, "s_11"], color=BLUE, lw=0.8)
axes[1].plot(z.loc[u1, "cycle"], z.loc[u1, "s_11"].rolling(15, min_periods=1).mean(), color=ORANGE, lw=2, label="15-cycle mean")
axes[1].set_title("After per-regime normalisation: the degradation trend appears", loc="left")
axes[1].legend(frameon=False)
for ax in axes:
    ax.set_xlabel("cycle")
fig.tight_layout()
fig.savefig(FIG / "regime_normalisation.png", bbox_inches="tight")
print("cycles flown in each regime:", np.bincount(regimes))""")

md("""## 2. Results on the official test sets

Models were trained with `turbosensor train --config configs/fdXXX.yaml` (see README). Each engine in the test set is scored at its last observed cycle, against NASA's ground truth.

Two numbers are reported for each:
* **RMSE vs true RUL**: the strict number.
* **RMSE vs capped truth**: the protocol most papers use (truth capped at 125 like the target).

The NASA score is the asymmetric PHM08 metric that punishes late predictions harder than early ones.""")

code("""rows = []
for s in ["FD001", "FD002", "FD003", "FD004"]:
    m = json.loads((ART / s / "metrics.json").read_text())["models"]
    for kind, r in m.items():
        rows.append({"subset": s, "model": kind, "RMSE": r["rmse"], "RMSE (capped)": r["capped"]["rmse"],
                     "NASA score": r["nasa_score"], "NASA (capped)": r["capped"]["nasa_score"],
                     "CV RMSE": r["cv_rmse"], "80% interval coverage (capped)": r["capped"].get("interval_coverage"),
                     "missed @30": r["maintenance"]["missed"], "false alarms @30": r["maintenance"]["false_alarms"], "cycles early per false alarm": r["maintenance"]["false_alarm_mean_cycles_early"]})
results = pd.DataFrame(rows).set_index(["subset", "model"]).round(2)
results""")

code("""model = RULModel.load(ART / "FD001/lgbm.joblib")
latest = model.predict_latest(split.test)
latest["rul_true"] = split.test_rul.loc[latest.unit].to_numpy()
order = latest.sort_values("rul_true").reset_index(drop=True)

fig, ax = plt.subplots(figsize=(11, 3.8))
x = np.arange(len(order))
ax.fill_between(x, order.rul_lower, order.rul_upper, color=BLUE, alpha=0.15, lw=0, label="80% interval")
ax.plot(x, order.rul_true, color=INK, lw=1.6, label="true RUL")
ax.scatter(x, order.rul_pred, color=BLUE, s=14, zorder=3, label="predicted")
ax.axhline(125, color=GRAY, ls="--", lw=0.8)
ax.text(len(order) - 1, 127, "target cap (125)", ha="right", color=MUTED, fontsize=8)
ax.set_xlabel("test engine (sorted by true RUL)"); ax.set_ylabel("cycles")
ax.set_title("FD001 test set: LightGBM + CNN blend vs ground truth", loc="left")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "fd001_predictions.png", bbox_inches="tight")
regression_report(latest.rul_true, latest.rul_pred, latest.rul_lower, latest.rul_upper)""")

md(
    'Errors are small exactly where they matter, close to failure. The large errors sit on engines with more than 125 cycles left, which the model deliberately reports as "healthy, at least 125".'
)

code("""error_by_rul_bucket(latest.rul_true, latest.rul_pred).round(1)""")

code("""imp = model.feature_importance(15)[::-1]
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.barh(imp.index, imp.values, color=BLUE, height=0.6)
ax.set_xlabel("share of total gain")
ax.set_title("Top features (FD001, LightGBM member)", loc="left")
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(FIG / "fd001_feature_importance.png", bbox_inches="tight")""")

md("""## 3. From prediction to maintenance decision

The planner asks one question per engine: will it fail within the next 30 cycles? Two mistakes cost different amounts:

* **missed failure**: an unscheduled removal, priced at 100 cycles of engine life;
* **false alarm**: the engine is pulled while it still has life left; the cost is the cycles thrown away.

The alert threshold that minimises this cost is learned from **out-of-fold predictions on the training engines** (never the test set) and stored in the model.""")

code("""print("learned alert thresholds (horizon -> pull when predicted RUL <=):", {h: model.alert_thresholds_[h] for h in (20, 30, 40)})
plan = maintenance_plan(latest, horizon=30, watch=60, thresholds=model.alert_thresholds_)
print(plan.action.value_counts().to_string())
plan.head(10).round(1)""")

code("""truth = plan.rul_true <= 30
alert = plan.action == "replace"
early = plan.loc[alert & ~truth, "rul_true"] - 30
print(f"false alarms are pulled on average {early.mean():.1f} cycles before the 30-cycle horizon (max {early.max():.0f})")
pd.crosstab(truth.rename("fails within 30 cycles"), alert.rename("replace alert"))""")

md("""## Takeaways

* Regime normalisation plus causal rolling features over 5-60 cycles (level, noise, trend, drift from the engine's own baseline) a 3-seed ensemble, and a small 1D-CNN blended in at a weight learned on cross-validation let a gradient-boosted model reach strong accuracy without a GPU.
* Validation is grouped by engine, so no engine is ever in both training and validation folds.
* Out-of-fold residuals give a calibrated interval per engine, and a cost-based alert threshold learned on training engines turns predictions into replace/watch/ok decisions.
* The weakest region is engines with more than 125 cycles left: they are reported as healthy, which is the intended behaviour, but it is what drives raw RMSE on FD002/FD004, where over a fifth of test engines are in that range.""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
Path("notebooks").mkdir(exist_ok=True)
nbf.write(nb, "notebooks/01_exploration_and_results.ipynb")

# TurboSensor

**Predictive maintenance for jet engines: estimating how many flight cycles a turbofan has left before it fails.**

TurboSensor is an end-to-end remaining-useful-life (RUL) system built on NASA's
[C-MAPSS turbofan degradation dataset](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).
It takes the raw history of 21 engine sensors and 3 operating settings, and gives back
a calibrated RUL estimate for each engine and a maintenance action (`replace`, `watch`, `ok`).
It can run as a batch job, a CLI, or a REST service.

![FD001 predictions](reports/figures/fd001_predictions.png)

## Results

All numbers are on NASA's held-out test engines, scored at each engine's last observed cycle.
LightGBM is the production model; ridge regression on the same features is the baseline.

| Subset | Conditions / faults | Test engines | RMSE (capped truth) | RMSE (raw truth) | NASA score (capped) | Missed failures @30 cycles | False alarms @30 |
|---|---|---|---|---|---|---|---|
| FD001 | 1 / HPC | 100 | **11.22** | 12.48 | 182 | 0 of 25 | 2 |
| FD002 | 6 / HPC | 259 | **10.48** | 23.03 | 491 | 0 of 61 | 13 |
| FD003 | 1 / HPC + fan | 100 | **9.48** | 11.30 | 193 | 0 of 20 | 2 |
| FD004 | 6 / HPC + fan | 248 | **10.69** | 23.75 | 553 | 0 of 53 | 15 |

Ridge baseline, capped RMSE: 15.66 / 14.92 / 14.60 / 16.31. Full table: [`reports/metrics_summary.csv`](reports/metrics_summary.csv).

**How to read these numbers:**

- **Capped truth** is the protocol most published work uses. The true RUL is capped at 125 cycles, the same as the training target.
  **Raw truth** compares against NASA's RUL as given. The gap between the two comes from engines with more than 125
  cycles left. The model reports them as "healthy, at least 125", which is the intended behaviour. That's 22% of FD002's
  test engines and 27% of FD004's, which is why their raw RMSE is higher.
- **NASA score** is the asymmetric PHM08 metric. It penalises late predictions (the engine keeps flying past
  the point it should have been pulled) more steeply than early ones. Lower is better.
- **Missed failures / false alarms** come from the maintenance rule. An engine is flagged `replace` when the lower
  bound of its 80% RUL interval is within 30 cycles. On all four test sets, this rule caught every engine that
  failed within 30 cycles.
- Prediction intervals cover 77-78% of capped truths on FD002, FD003 and FD004, against an 80% target.
  On FD001's 100 test engines, coverage is 65%.

## Why it's built this way

| Design choice | Reason |
|---|---|
| **Piecewise-linear target, capped at 125** | Wear can't be seen for most of an engine's life (see the figure below). Asking a model to tell 250 from 200 cycles left on an engine with no wear means training it on noise. |
| **Per-regime sensor normalisation** | FD002/FD004 fly six operating regimes, which are found with k-means on the settings. Switching regimes moves the sensors far more than degradation does. |
| **Causal features only** | Rolling mean/std (5/15/30 cycles), 30-cycle trend slope, EWMA, and drift from the engine's own early-life baseline. A test checks that adding future cycles never changes past features, so the model can run on a live engine. |
| **Engine-grouped cross-validation** | No engine appears in both a training fold and a validation fold. A random row split would leak each engine's trajectory into validation. |
| **Split-conformal intervals** | Each prediction's interval comes from out-of-fold residuals, binned by predicted RUL. Quantile boosting fell apart on the capped target, because so many labels sit exactly at 125. |
| **Decisions act on the lower bound** | A planner pulls an engine because of the pessimistic estimate, not the average one. Uncertainty leads to earlier, safer action. |

![Sensor trajectories](reports/figures/sensor_trajectories.png)
![Regime normalisation](reports/figures/regime_normalisation.png)

## Quick start

```bash
git clone https://github.com/nawalkhann/turbosensor.git && cd turbosensor
pip install -e ".[dev,serve,notebook]"

turbosensor download                              # NASA C-MAPSS into data/raw (~12 MB)
turbosensor train --config configs/fd001.yaml     # CV + fit + test, artifacts in artifacts/FD001/
make train-all                                    # all four subsets (a few minutes on a laptop CPU)
```

**Batch scoring of a fleet**, with one row per engine sorted by urgency:

```bash
turbosensor predict --model artifacts/FD001/lgbm.joblib --input data/raw/test_FD001.txt --output plan.csv
```

```
 unit  cycle  rul_pred  rul_lower  rul_upper  decision_rul  action
   81    213       4.3        1.5        6.9           1.5 replace
   34    203       5.4        2.6        8.0           2.6 replace
   ...
```

**REST API**:

```bash
turbosensor serve --model artifacts/FD001/lgbm.joblib --port 8000
curl -s localhost:8000/health
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' -d '{
  "horizon": 30,
  "readings": [{"unit": 7, "cycle": 1, "settings": [0.0, 0.0, 100.0],
                "sensors": [518.67, 642.1, 1589.7, 1400.6, 14.62, 21.61, 554.36, 2388.06, 9046.19, 1.3,
                            47.47, 521.66, 2388.02, 8138.62, 8.4195, 0.03, 392, 2388, 100.0, 39.06, 23.419]}]
}'
```

**Docker**:

```bash
docker build -t turbosensor .
docker run -p 8000:8000 turbosensor
```

The image downloads the data and trains the FD001 model during the build, so the container serves right away.

## Project layout

```
src/turbosensor/
  data.py           download, parse, RUL labelling
  preprocessing.py  regime clustering, per-regime normalisation, flat-sensor removal
  features.py       causal rolling, trend, and baseline-drift features
  model.py          RULModel: pipeline, grouped CV, conformal intervals, save/load
  evaluation.py     RMSE, MAE, NASA PHM08 score, interval coverage, error by RUL band
  maintenance.py    RUL to replace/watch/ok decisions, alert precision/recall
  pipeline.py       config-driven train/evaluate run
  cli.py, api.py    command line and FastAPI service
configs/            one YAML per subset
notebooks/          exploration and results walkthrough (executed)
tests/              unit tests on synthetic fleets plus a real-data check
```

## Development

```bash
make test     # pytest
make lint     # ruff check + format check
make notebook # rebuild and execute the notebook (needs trained artifacts)
```

CI runs lint and tests on Python 3.10-3.12. Then it downloads the real data, trains FD001, and fails if
the capped test RMSE goes above 13.

## Data

The data comes from A. Saxena, K. Goebel, D. Simon, N. Eklund, *"Damage Propagation Modeling for Aircraft Engine
Run-to-Failure Simulation"*, PHM08, and is distributed by the NASA Ames Prognostics Center of Excellence.
The raw files aren't committed. `turbosensor download` fetches them from NASA's public data repository.

## License

MIT

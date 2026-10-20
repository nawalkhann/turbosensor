"""Command line interface: ``turbosensor download|train|evaluate|predict|serve``."""

from __future__ import annotations

import argparse
import json
import logging
import sys

import pandas as pd

from .data import download, read_table
from .maintenance import maintenance_plan
from .model import RULModel
from .pipeline import Config, evaluate_saved, run


def _cmd_download(args):
    download(args.data_dir)


def _cmd_train(args):
    cfg = Config.from_yaml(args.config, subset=args.subset, data_dir=args.data_dir, output_dir=args.output_dir)
    results = run(cfg)
    rows = {
        k: {
            "rmse": v["rmse"],
            "rmse_capped": v["capped"]["rmse"],
            "nasa_score": v["nasa_score"],
            "nasa_capped": v["capped"]["nasa_score"],
            "coverage_capped": v["capped"].get("interval_coverage"),
            "cv_rmse": v["cv_rmse"],
            "missed": v["maintenance"]["missed"],
            "false_alarms": v["maintenance"]["false_alarms"],
        }
        for k, v in results["models"].items()
    }
    print(pd.DataFrame(rows).T.round(2).to_string())


def _cmd_evaluate(args):
    print(json.dumps(evaluate_saved(args.model, args.subset, args.data_dir), indent=2))


def _cmd_predict(args):
    model = RULModel.load(args.model)
    history = pd.read_csv(args.input) if args.input.endswith(".csv") else read_table(args.input)
    plan = maintenance_plan(model.predict_latest(history), horizon=args.horizon, thresholds=model.alert_thresholds_)
    if args.output:
        plan.to_csv(args.output, index=False)
    print(plan.round(1).to_string(index=False))


def _cmd_serve(args):
    import uvicorn

    from .api import create_app

    uvicorn.run(create_app(args.model), host=args.host, port=args.port)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="turbosensor", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("download", help="fetch the NASA C-MAPSS dataset")
    d.add_argument("--data-dir", default="data/raw")
    d.set_defaults(func=_cmd_download)

    t = sub.add_parser("train", help="cross-validate, fit and test every model in a config")
    t.add_argument("--config", default="configs/fd001.yaml")
    t.add_argument("--subset")
    t.add_argument("--data-dir")
    t.add_argument("--output-dir")
    t.set_defaults(func=_cmd_train)

    e = sub.add_parser("evaluate", help="score a saved model on a subset's test set")
    e.add_argument("--model", required=True)
    e.add_argument("--subset", default="FD001")
    e.add_argument("--data-dir", default="data/raw")
    e.set_defaults(func=_cmd_evaluate)

    pr = sub.add_parser("predict", help="RUL and maintenance action for each engine in a history file")
    pr.add_argument("--model", required=True)
    pr.add_argument("--input", required=True, help="C-MAPSS .txt file or CSV with the same columns")
    pr.add_argument("--output")
    pr.add_argument("--horizon", type=int, default=30)
    pr.set_defaults(func=_cmd_predict)

    s = sub.add_parser("serve", help="run the REST API")
    s.add_argument("--model", required=True)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=_cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

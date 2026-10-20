"""Fail CI if a trained model regresses past a capped-RMSE threshold."""

import json
import sys
from pathlib import Path

path, model, threshold = Path(sys.argv[1]), sys.argv[2], float(sys.argv[3])
report = json.loads(path.read_text())["models"][model]
rmse = report["capped"]["rmse"]
print(f"{path.parent.name} {model}: capped RMSE {rmse:.2f} (threshold {threshold})")
sys.exit(0 if rmse <= threshold else 1)

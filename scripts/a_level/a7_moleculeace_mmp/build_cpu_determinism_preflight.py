import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "diagnostics" / "cpu_determinism_preflight" / "inputs"
TARGETS = ["CHEMBL244_Ki", "CHEMBL4792_Ki"]
SEEDS = [17, 43]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(ROOT / "mmp_pairs.csv").query("dataset in @TARGETS")
    controls = pd.read_csv(ROOT / "noncliff_matched_controls.csv").query("dataset in @TARGETS")
    predictions = pd.read_csv(ROOT / "predictions.csv").query("dataset in @TARGETS and seed in @SEEDS")
    metrics = pd.read_csv(ROOT / "model_metrics.csv").query("dataset in @TARGETS and seed in @SEEDS")
    pairs.to_csv(OUT / "mmp_pairs.csv", index=False)
    controls.to_csv(OUT / "noncliff_matched_controls.csv", index=False)
    predictions.to_csv(OUT / "predictions.csv", index=False)
    metrics.to_csv(OUT / "model_metrics.csv", index=False)
    shutil.copyfile(ROOT / "cross_target_overlap_audit.json", OUT / "cross_target_overlap_audit.json")
    audit = {
        "status": "PASS",
        "targets": TARGETS,
        "seeds": SEEDS,
        "model_cells": len(metrics),
        "cliff_pairs": len(pairs),
        "control_pairs": len(controls),
        "reason": "CHEMBL244_Ki has the largest selected panel and severe calibration clustering; CHEMBL4792_Ki contains the observed near-zero GPU sign instability and only three calibration components.",
    }
    (OUT.parent / "input_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

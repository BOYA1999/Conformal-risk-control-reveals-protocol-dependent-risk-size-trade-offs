import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def interval(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


parser = argparse.ArgumentParser()
parser.add_argument("--cells-csv", type=Path, required=True)
parser.add_argument("--out-json", type=Path, required=True)
parser.add_argument("--cells-dir", type=Path)
parser.add_argument("--bootstraps", type=int, default=2000)
args = parser.parse_args()
frame = pd.read_csv(args.cells_csv)
cells_dir = args.cells_dir or args.cells_csv.parent / "cells"
expected_methods = {"gradinput", "ig", "saliency", "atom_occlusion", "gnnexplainer"}
metric_columns = ["risk", "atom_fraction", "precision", "iou"]
if len(frame) != 330 or frame["cell_id"].nunique() != 66 or set(frame["method"]) != expected_methods:
    raise ValueError("Expected 66 complete cells and five methods")
if frame.duplicated(["cell_id", "method"]).any() or frame.groupby("method").size().ne(66).any():
    raise ValueError("Expected one row per cell-method pair")
if frame.groupby("cell_id")["method"].nunique().ne(5).any():
    raise ValueError("Each cell must contain all five methods")
if frame[["family", "task"]].drop_duplicates().shape[0] != 11:
    raise ValueError("Expected 11 task clusters")
if not np.isfinite(frame[metric_columns].to_numpy()).all():
    raise ValueError("All comparison metrics must be finite")

timings = {method: {"seconds": 0.0, "molecules": 0, "peak_vram_mib": 0.0} for method in expected_methods}
cell_paths = sorted(cells_dir.glob("*.json"))
if len(cell_paths) != 66:
    raise ValueError("Expected 66 cell JSON files for timing analysis")
for path in cell_paths:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for method in expected_methods:
        for split in ("calibration_timing", "test_timing"):
            timing = payload["methods"][method][split]
            timings[method]["seconds"] += float(timing["seconds"])
            timings[method]["molecules"] += int(timing["molecules"])
            timings[method]["peak_vram_mib"] = max(
                timings[method]["peak_vram_mib"], float(timing["peak_vram_mib"])
            )

methods = []
for method, group in frame.groupby("method", sort=True):
    task = group.groupby(["family", "task"], sort=True)
    methods.append(
        {
            "method": method,
            "cells": len(group),
            "macro_task_risk": float(task["risk"].mean().mean()),
            "macro_task_atom_fraction": float(task["atom_fraction"].mean().mean()),
            "macro_task_precision": float(task["precision"].mean().mean()),
            "macro_task_iou": float(task["iou"].mean().mean()),
            "risk_pass_cells": int(group["risk_pass"].sum()),
            "efficiency_pass_cells": int(group["efficiency_pass"].sum()),
            "joint_pass_cells": int((group["risk_pass"] & group["efficiency_pass"]).sum()),
            "runtime_seconds": timings[method]["seconds"],
            "seconds_per_molecule": timings[method]["seconds"] / timings[method]["molecules"],
            "peak_vram_mib": timings[method]["peak_vram_mib"],
        }
    )

task_means = frame.groupby(["family", "task", "method"], as_index=False)[metric_columns].mean()
family_means = task_means.groupby(["family", "method"], as_index=False)[metric_columns].mean()
family_labels = {"bxaic": "B-XAIC", "google": "Graph Attribution"}
family_task_counts = task_means.groupby("family")["task"].nunique()
family_cell_counts = frame.groupby("family")["cell_id"].nunique()
family_means.insert(1, "family_label", family_means["family"].map(family_labels))
family_means.insert(2, "tasks", family_means["family"].map(family_task_counts))
family_means.insert(3, "cells", family_means["family"].map(family_cell_counts))
family_csv = args.out_json.with_name("family_stratified_summary.csv")
family_means.to_csv(family_csv, index=False)

wide = frame.pivot(index=["cell_id", "family", "task"], columns="method")
clusters = list(dict.fromkeys(zip(wide.index.get_level_values("family"), wide.index.get_level_values("task"))))
rng = np.random.default_rng(20260901)
paired = []
for method in sorted(expected_methods - {"ig"}):
    entry = {"method": method, "reference": "ig", "differences": {}}
    for metric in metric_columns:
        differences = wide[(metric, method)] - wide[(metric, "ig")]
        cluster_means = differences.groupby(level=["family", "task"]).mean()
        samples = np.empty(args.bootstraps)
        for i in range(args.bootstraps):
            chosen = rng.integers(0, len(clusters), len(clusters))
            samples[i] = np.mean([cluster_means.loc[clusters[j]] for j in chosen])
        entry["differences"][metric] = {
            "mean": float(cluster_means.mean()),
            "task_cluster_bootstrap_95ci": interval(samples),
            "positive_cells": int((differences > 0).sum()),
            "negative_cells": int((differences < 0).sum()),
            "ties": int((differences == 0).sum()),
        }
    paired.append(entry)

payload = {
    "status": "PASS",
    "cells": 66,
    "tasks": 11,
    "methods": methods,
    "family_stratified": family_means.to_dict(orient="records"),
    "paired_vs_ig": paired,
    "bootstraps": args.bootstraps,
    "bootstrap_unit": "task cluster",
    "interpretation": "Matched reviewer-driven subset analysis; cell rows are paired, and uncertainty resamples the observed 11 task clusters from two benchmark families. Family summaries are macro means of task means and are descriptive.",
    "family_summary_csv": family_csv.name,
}
args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))

import csv
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "artifacts" / "analysis" / "phase_a_followup_20260809" / "random_model_control"
TRAINED = ROOT / "artifacts" / "experiment" / "gradient_grid_main" / "cells"
OUT = ROOT / "results" / "random_model_control"
TASKS = {
    "bxaic": ["B", "P", "X", "indole", "PAINS", "rings-count", "rings-max"],
    "google": ["benzene", "logic7", "logic8", "logic10"],
}


def finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return True


def mean(values):
    return sum(values) / len(values)


contract = json.loads((CONTROL / "run_contract.json").read_text(encoding="utf-8"))
manifest = json.loads((CONTROL / "manifest.json").read_text(encoding="utf-8"))
expected = {
    f"{family}__{task}__{model}__seed{seed}"
    for family, tasks in TASKS.items()
    for task in tasks
    for model in ["gin", "gcn"]
    for seed in [42, 123, 2026]
}
paths = sorted((CONTROL / "cells").glob("*.json"))
errors = []
if manifest.get("status") != "complete" or manifest.get("cell_count") != 66:
    errors.append("manifest")
if manifest.get("code_sha256") != contract["code_sha256"]:
    errors.append("manifest_code_hash")
if {path.stem for path in paths} != expected:
    errors.append("cell_set")

rows = []
for path in paths:
    control = json.loads(path.read_text(encoding="utf-8"))
    trained = json.loads((TRAINED / path.name).read_text(encoding="utf-8"))
    if control.get("status") != "complete" or control.get("code_sha256") != contract["code_sha256"]:
        errors.append(f"control:{path.name}")
    if trained.get("status") != "complete" or not finite(control) or not finite(trained):
        errors.append(f"paired_cell:{path.name}")
    for explainer in ["gradinput", "ig"]:
        trained_fixed = trained["explainers"][explainer]["metrics"]["fixed"]["0.20"]
        control_fixed = control["explainers"][explainer]["metrics"]["fixed"]["0.20"]
        difference = trained_fixed["iou"] - control_fixed["iou"]
        rows.append({
            "cell_id": path.stem,
            "family": trained["family"],
            "task": trained["task"],
            "model": trained["model"],
            "seed": trained["seed"],
            "explainer": explainer,
            "trained_test_auroc": trained["predictor"]["test"]["auroc"],
            "random_test_auroc": control["predictor"]["test"]["auroc"],
            "trained_fixed20_iou": trained_fixed["iou"],
            "random_fixed20_iou": control_fixed["iou"],
            "iou_difference": difference,
            "trained_wins": difference > 1e-12,
            "tie": abs(difference) <= 1e-12,
        })
if errors:
    raise SystemExit("VALIDATION_FAILED\n" + "\n".join(errors))

summary = []
for explainer in ["gradinput", "ig"]:
    group = [row for row in rows if row["explainer"] == explainer]
    summary.append({
        "explainer": explainer,
        "cells": len(group),
        "trained_wins": sum(row["trained_wins"] for row in group),
        "ties": sum(row["tie"] for row in group),
        "trained_macro_iou": mean([row["trained_fixed20_iou"] for row in group]),
        "random_macro_iou": mean([row["random_fixed20_iou"] for row in group]),
        "macro_iou_improvement": mean([row["iou_difference"] for row in group]),
        "trained_macro_auroc": mean([row["trained_test_auroc"] for row in group]),
        "random_macro_auroc": mean([row["random_test_auroc"] for row in group]),
        "trained_auroc_wins": sum(row["trained_test_auroc"] > row["random_test_auroc"] for row in group),
    })

groups = defaultdict(list)
for row in rows:
    groups[(row["family"], row["task"], row["explainer"])].append(row)
task_summary = []
for (family, task, explainer), group in sorted(groups.items()):
    task_summary.append({
        "family": family,
        "task": task,
        "explainer": explainer,
        "cells": len(group),
        "trained_wins": sum(row["trained_wins"] for row in group),
        "trained_mean_iou": mean([row["trained_fixed20_iou"] for row in group]),
        "random_mean_iou": mean([row["random_fixed20_iou"] for row in group]),
        "mean_iou_improvement": mean([row["iou_difference"] for row in group]),
    })

ig = next(item for item in summary if item["explainer"] == "ig")
gate = {
    "minimum_trained_wins": 44,
    "observed_trained_wins": ig["trained_wins"],
    "minimum_macro_iou_improvement": 0.05,
    "observed_macro_iou_improvement": ig["macro_iou_improvement"],
    "pass": ig["trained_wins"] >= 44 and ig["macro_iou_improvement"] >= 0.05,
}

OUT.mkdir(parents=True, exist_ok=True)
for filename, records in [("cells.csv", rows), ("tasks.csv", task_summary)]:
    with (OUT / filename).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
payload = {
    "validation": {"status": "PASS", "paired_cells": len(paths), "hashes_match": True, "finite": True},
    "runtime_seconds": manifest["seconds"],
    "summary": summary,
    "ig_primary_gate": gate,
    "task_summary": task_summary,
}
(OUT / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts" / "analysis" / "phase_a_followup_20260809" / "gnnexplainer_dev_gate_v2"
OUT = ROOT / "results" / "gnnexplainer_dev_gate_v2"
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


contract = json.loads((RUN / "run_contract.json").read_text(encoding="utf-8"))
manifest = json.loads((RUN / "manifest.json").read_text(encoding="utf-8"))
expected = {
    f"{family}__{task}__{model}__seed42"
    for family, tasks in TASKS.items()
    for task in tasks
    for model in ["gin", "gcn"]
}
paths = sorted((RUN / "cells").glob("*.json"))
errors = []
if manifest.get("status") != "complete" or manifest.get("cell_count") != 22:
    errors.append("manifest")
if manifest.get("script_sha256") != contract["script_sha256"]:
    errors.append("manifest_script_hash")
if {path.stem for path in paths} != expected:
    errors.append("cell_set")

cells = []
for path in paths:
    cell = json.loads(path.read_text(encoding="utf-8"))
    if cell.get("status") != "complete" or cell.get("surface") != "development_compute_gate":
        errors.append(f"status:{path.name}")
    if cell.get("script_sha256") != contract["script_sha256"]:
        errors.append(f"script_hash:{path.name}")
    if not finite(cell):
        errors.append(f"non_finite:{path.name}")
    cells.append(cell)
if errors:
    raise SystemExit("VALIDATION_FAILED\n" + "\n".join(errors))

rows = []
for cell in cells:
    for explainer in ["gnnexplainer", "ig"]:
        metrics = cell["explainers"][explainer]["metrics"]
        crc = metrics["alpha"]["0.10"]["crc"]
        rows.append({
            "cell_id": cell["cell_id"],
            "family": cell["family"],
            "task": cell["task"],
            "model": cell["model"],
            "explainer": explainer,
            "n_calibration": metrics["n_calibration_rationale"],
            "n_evaluation": metrics["n_evaluation_rationale"],
            "test_risk": crc["risk"],
            "mean_atom_fraction": crc["mean_atom_fraction"],
            "precision": crc["precision"],
            "iou": crc["iou"],
            "risk_pass": crc["risk"] <= 0.10,
            "efficiency_pass": crc["mean_atom_fraction"] < 0.80,
        })

gnn = [row for row in rows if row["explainer"] == "gnnexplainer"]
ig = [row for row in rows if row["explainer"] == "ig"]
gnn_fraction = mean([row["mean_atom_fraction"] for row in gnn])
ig_fraction = mean([row["mean_atom_fraction"] for row in ig])
gate = {
    "risk_pass_cells": sum(row["risk_pass"] for row in gnn),
    "efficiency_pass_cells": sum(row["efficiency_pass"] for row in gnn),
    "macro_mean_risk": mean([row["test_risk"] for row in gnn]),
    "macro_mean_atom_fraction": gnn_fraction,
    "ig_macro_mean_atom_fraction": ig_fraction,
    "macro_atom_fraction_reduction_vs_ig": ig_fraction - gnn_fraction,
}
gate["go_full_gnnexplainer"] = (
    gate["risk_pass_cells"] >= 15
    and gate["efficiency_pass_cells"] >= 15
    and gate["macro_atom_fraction_reduction_vs_ig"] >= 0.05
    and gate["macro_mean_risk"] <= 0.10
)

OUT.mkdir(parents=True, exist_ok=True)
with (OUT / "cells.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
summary = {
    "validation": {"status": "PASS", "observed_cells": len(cells), "script_hashes_match": True, "finite": True},
    "runtime_seconds": manifest["seconds"],
    "gate": gate,
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))

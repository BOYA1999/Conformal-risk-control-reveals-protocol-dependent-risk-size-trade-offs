import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts" / "analysis" / "phase_a_followup_20260809" / "molecule_level_ig_seed42"
OUT = ROOT / "results" / "molecule_level_ig_seed42"
TASKS = {
    "bxaic": ["B", "P", "X", "indole", "PAINS", "rings-count", "rings-max"],
    "google": ["benzene", "logic7", "logic8", "logic10"],
}


def records(frame):
    return json.loads(frame.to_json(orient="records"))


contract = json.loads((RUN / "run_contract.json").read_text(encoding="utf-8"))
manifest = json.loads((RUN / "manifest.json").read_text(encoding="utf-8"))
expected = {
    f"{family}__{task}__{model}__seed42"
    for family, tasks in TASKS.items()
    for task in tasks
    for model in ["gin", "gcn"]
}
cell_paths = sorted((RUN / "cells").glob("*.json"))
errors, frames = [], []
if manifest.get("status") != "complete" or manifest.get("cell_count") != 22:
    errors.append("manifest")
if manifest.get("code_sha256") != contract["code_sha256"]:
    errors.append("manifest_code_hash")
if {path.stem for path in cell_paths} != expected:
    errors.append("cell_set")
for path in cell_paths:
    cell = json.loads(path.read_text(encoding="utf-8"))
    table = ROOT / cell["table"]
    if cell.get("status") != "complete" or cell.get("code_sha256") != contract["code_sha256"]:
        errors.append(f"cell:{path.name}")
    frame = pd.read_csv(table)
    if len(frame) != cell["molecules"] or frame["cell_id"].nunique() != 1 or frame["cell_id"].iloc[0] != path.stem:
        errors.append(f"table:{path.name}")
    required = frame[["n_atoms", "n_rationale", "selected_fraction", "true_logit_drop_selected", "true_logit_drop_random", "fidelity_advantage"]]
    if not required.map(math.isfinite).all().all():
        errors.append(f"non_finite:{path.name}")
    frames.append(frame)
if errors:
    raise SystemExit("VALIDATION_FAILED\n" + "\n".join(errors))

data = pd.concat(frames, ignore_index=True)
nonnull = data[data["n_rationale"] > 0].copy()

cell_summary = data.groupby(["cell_id", "family", "task", "model"], as_index=False).agg(
    molecules=("source_index", "size"),
    rationale_molecules=("n_rationale", lambda values: int((values > 0).sum())),
    null_molecules=("n_rationale", lambda values: int((values == 0).sum())),
    accuracy=("prediction_correct", "mean"),
    mean_selected_fraction=("selected_fraction", "mean"),
    mean_selected_logit_drop=("true_logit_drop_selected", "mean"),
    mean_random_logit_drop=("true_logit_drop_random", "mean"),
    mean_fidelity_advantage=("fidelity_advantage", "mean"),
)
rationale_cell = nonnull.groupby("cell_id", as_index=False).agg(
    mean_miss_loss=("miss_loss", "mean"),
    mean_precision=("precision", "mean"),
    mean_iou=("iou", "mean"),
    rationale_mean_fidelity_advantage=("fidelity_advantage", "mean"),
)
cell_summary = cell_summary.merge(rationale_cell, on="cell_id", how="left")
cell_summary["fidelity_win"] = cell_summary["mean_fidelity_advantage"] > 0

task_summary = cell_summary.groupby(["family", "task"], as_index=False).agg(
    models=("model", "size"),
    mean_miss_loss=("mean_miss_loss", "mean"),
    mean_iou=("mean_iou", "mean"),
    mean_selected_fraction=("mean_selected_fraction", "mean"),
    mean_fidelity_advantage=("mean_fidelity_advantage", "mean"),
    fidelity_wins=("fidelity_win", "sum"),
)

rationale_strata = nonnull.groupby("rationale_group", as_index=False).agg(
    molecules=("source_index", "size"),
    mean_rationale_fraction=("rationale_fraction", "mean"),
    mean_miss_loss=("miss_loss", "mean"),
    mean_iou=("iou", "mean"),
    mean_selected_fraction=("selected_fraction", "mean"),
    mean_fidelity_advantage=("fidelity_advantage", "mean"),
)
atom_strata = nonnull.groupby("atom_size_group", as_index=False).agg(
    molecules=("source_index", "size"),
    mean_atoms=("n_atoms", "mean"),
    mean_miss_loss=("miss_loss", "mean"),
    mean_iou=("iou", "mean"),
    mean_selected_fraction=("selected_fraction", "mean"),
    mean_fidelity_advantage=("fidelity_advantage", "mean"),
)
null_summary = data[data["n_rationale"] == 0].groupby(["family", "task"], as_index=False).agg(
    molecules=("source_index", "size"),
    mean_false_highlight_fraction=("selected_fraction", "mean"),
    mean_fidelity_advantage=("fidelity_advantage", "mean"),
)

gate = {
    "minimum_cell_wins": 15,
    "observed_cell_wins": int(cell_summary["fidelity_win"].sum()),
    "mean_fidelity_advantage": float(cell_summary["mean_fidelity_advantage"].mean()),
}
gate["pass"] = gate["observed_cell_wins"] >= gate["minimum_cell_wins"]

OUT.mkdir(parents=True, exist_ok=True)
data.to_csv(OUT / "molecules.csv.gz", index=False, compression="gzip")
for filename, frame in [
    ("cells.csv", cell_summary),
    ("tasks.csv", task_summary),
    ("rationale_strata.csv", rationale_strata),
    ("atom_size_strata.csv", atom_strata),
    ("null_summary.csv", null_summary),
]:
    frame.to_csv(OUT / filename, index=False)
payload = {
    "validation": {"status": "PASS", "cells": len(cell_paths), "molecule_rows": len(data), "hashes_match": True, "finite": True},
    "runtime_seconds": manifest["seconds"],
    "fidelity_gate": gate,
    "task_summary": records(task_summary),
    "rationale_strata": records(rationale_strata),
    "atom_size_strata": records(atom_strata),
    "null_summary": records(null_summary),
}
(OUT / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))

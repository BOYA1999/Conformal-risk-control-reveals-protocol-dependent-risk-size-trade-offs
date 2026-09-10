import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts" / "experiment" / "gradient_grid_main"
OUT = ROOT / "results" / "gradient_grid_phase_a"
TASKS = {
    "bxaic": ["B", "P", "X", "indole", "PAINS", "rings-count", "rings-max"],
    "google": ["benzene", "logic7", "logic8", "logic10"],
}
MODELS = ["gin", "gcn"]
SEEDS = [42, 123, 2026]
EXPLAINERS = ["gradinput", "ig"]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def finite_numbers(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_numbers(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_numbers(v) for v in value)
    return True


def mean(values):
    return sum(values) / len(values)


contract = json.loads((RUN / "run_contract.json").read_text(encoding="utf-8"))
errors = []
for relative, expected_hash in contract["hashes"].items():
    path = ROOT / relative
    if not path.is_file() or sha256(path) != expected_hash.upper():
        errors.append(f"hash_mismatch:{relative}")

expected = {
    f"{family}__{task}__{model}__seed{seed}"
    for family, tasks in TASKS.items()
    for task in tasks
    for model in MODELS
    for seed in SEEDS
}
paths = sorted((RUN / "cells").glob("*.json"))
found = {path.stem for path in paths}
errors.extend(f"missing:{cell_id}" for cell_id in sorted(expected - found))
errors.extend(f"unexpected:{cell_id}" for cell_id in sorted(found - expected))
if len(paths) != contract["expected_cells"]:
    errors.append(f"cell_count:{len(paths)}")

cells = []
for path in paths:
    cell = json.loads(path.read_text(encoding="utf-8"))
    if cell.get("cell_id") != path.stem:
        errors.append(f"cell_id_mismatch:{path.name}")
    if cell.get("status") != "complete" or cell.get("surface") != "final":
        errors.append(f"not_final_complete:{path.name}")
    if cell.get("contract_sha256") != contract["hashes"]["artifacts/contract/experiment_contract.md"]:
        errors.append(f"contract_hash_mismatch:{path.name}")
    if cell.get("code_sha256") != contract["hashes"]["src/run_gradient_grid.py"]:
        errors.append(f"code_hash_mismatch:{path.name}")
    if not finite_numbers(cell):
        errors.append(f"non_finite:{path.name}")
    cells.append(cell)

if errors:
    raise SystemExit("VALIDATION_FAILED\n" + "\n".join(errors))

rows = []
for cell in cells:
    for explainer in EXPLAINERS:
        metrics = cell["explainers"][explainer]["metrics"]
        crc = metrics["alpha"]["0.10"]["crc"]
        random_crc = metrics["alpha"]["0.10"]["random_crc"]
        rows.append({
            "cell_id": cell["cell_id"],
            "family": cell["family"],
            "task": cell["task"],
            "model": cell["model"],
            "seed": cell["seed"],
            "explainer": explainer,
            "predictor_competence_pass": cell["predictor_competence_pass"],
            "test_auroc": cell["predictor"]["test"]["auroc"],
            "n_calibration_rationale": metrics["n_calibration_rationale"],
            "n_test_rationale": metrics["n_test_rationale"],
            "fraction_grid": crc["fraction"],
            "test_risk": crc["risk"],
            "mean_atom_fraction": crc["mean_atom_fraction"],
            "precision": crc["precision"],
            "iou": crc["iou"],
            "random_test_risk": random_crc["risk"],
            "random_mean_atom_fraction": random_crc["mean_atom_fraction"],
            "atom_fraction_saving_vs_random": random_crc["mean_atom_fraction"] - crc["mean_atom_fraction"],
            "fixed20_risk": metrics["fixed"]["0.20"]["risk"],
            "fixed50_risk": metrics["fixed"]["0.50"]["risk"],
            "risk_pass": crc["risk"] <= 0.10,
            "efficiency_pass": crc["mean_atom_fraction"] < 0.80,
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
        "mean_test_auroc": mean([r["test_auroc"] for r in group]),
        "mean_test_risk": mean([r["test_risk"] for r in group]),
        "mean_atom_fraction": mean([r["mean_atom_fraction"] for r in group]),
        "mean_random_test_risk": mean([r["random_test_risk"] for r in group]),
        "mean_random_atom_fraction": mean([r["random_mean_atom_fraction"] for r in group]),
        "mean_atom_fraction_saving_vs_random": mean([r["atom_fraction_saving_vs_random"] for r in group]),
        "mean_fixed20_risk": mean([r["fixed20_risk"] for r in group]),
        "mean_fixed50_risk": mean([r["fixed50_risk"] for r in group]),
        "risk_pass_rate": mean([r["risk_pass"] for r in group]),
        "efficiency_pass_rate": mean([r["efficiency_pass"] for r in group]),
    })

explainer_summary = []
for explainer in EXPLAINERS:
    group = [r for r in rows if r["explainer"] == explainer]
    task_group = [r for r in task_summary if r["explainer"] == explainer]
    explainer_summary.append({
        "explainer": explainer,
        "cells": len(group),
        "competent_predictor_rate": mean([r["predictor_competence_pass"] for r in group]),
        "risk_pass_rate": mean([r["risk_pass"] for r in group]),
        "efficiency_pass_rate": mean([r["efficiency_pass"] for r in group]),
        "macro_task_mean_risk": mean([r["mean_test_risk"] for r in task_group]),
        "macro_task_mean_atom_fraction": mean([r["mean_atom_fraction"] for r in task_group]),
        "macro_task_mean_random_atom_fraction": mean([r["mean_random_atom_fraction"] for r in task_group]),
        "macro_task_mean_atom_fraction_saving_vs_random": mean([r["mean_atom_fraction_saving_vs_random"] for r in task_group]),
        "practical_efficiency_gate": mean([r["efficiency_pass"] for r in group]) >= 2 / 3,
    })

alpha_rows = []
for cell in cells:
    for explainer in EXPLAINERS:
        metrics = cell["explainers"][explainer]["metrics"]
        for alpha in [0.05, 0.10, 0.20]:
            result = metrics["alpha"][f"{alpha:.2f}"]
            crc, naive, random_crc = result["crc"], result["naive_calibration"], result["random_crc"]
            alpha_rows.append({
                "cell_id": cell["cell_id"],
                "family": cell["family"],
                "task": cell["task"],
                "model": cell["model"],
                "seed": cell["seed"],
                "explainer": explainer,
                "alpha": alpha,
                "crc_fraction": crc["fraction"],
                "crc_test_risk": crc["risk"],
                "crc_mean_atom_fraction": crc["mean_atom_fraction"],
                "naive_fraction": naive["fraction"],
                "naive_test_risk": naive["risk"],
                "random_fraction": random_crc["mean_atom_fraction"],
                "crc_minus_naive_fraction": crc["fraction"] - naive["fraction"],
                "risk_pass": crc["risk"] <= alpha,
                "efficiency_pass": crc["mean_atom_fraction"] < 0.80,
            })

alpha_summary = []
for explainer in EXPLAINERS:
    for alpha in [0.05, 0.10, 0.20]:
        group = [row for row in alpha_rows if row["explainer"] == explainer and row["alpha"] == alpha]
        alpha_summary.append({
            "explainer": explainer,
            "alpha": alpha,
            "cells": len(group),
            "mean_test_risk": mean([row["crc_test_risk"] for row in group]),
            "mean_atom_fraction": mean([row["crc_mean_atom_fraction"] for row in group]),
            "risk_pass_rate": mean([row["risk_pass"] for row in group]),
            "efficiency_pass_rate": mean([row["efficiency_pass"] for row in group]),
            "mean_random_atom_fraction": mean([row["random_fraction"] for row in group]),
            "mean_crc_minus_naive_fraction": mean([row["crc_minus_naive_fraction"] for row in group]),
            "crc_equals_naive_rate": mean([abs(row["crc_minus_naive_fraction"]) < 1e-12 for row in group]),
        })

monotonic_violations = []
for cell_id in expected:
    for explainer in EXPLAINERS:
        group = sorted(
            [row for row in alpha_rows if row["cell_id"] == cell_id and row["explainer"] == explainer],
            key=lambda row: row["alpha"],
        )
        fractions = [row["crc_fraction"] for row in group]
        if not (fractions[0] >= fractions[1] >= fractions[2]):
            monotonic_violations.append(f"{cell_id}:{explainer}")

OUT.mkdir(parents=True, exist_ok=True)
for filename, records in [
    ("cell_alpha010.csv", rows),
    ("task_alpha010.csv", task_summary),
    ("cell_all_alpha.csv", alpha_rows),
    ("alpha_summary.csv", alpha_summary),
]:
    with (OUT / filename).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

summary = {
    "validation": {
        "status": "PASS",
        "expected_cells": len(expected),
        "observed_cells": len(cells),
        "hashes_match": True,
        "all_cells_final_complete": True,
        "all_required_numbers_finite": True,
    },
    "alpha": 0.10,
    "explainer_summary": explainer_summary,
    "task_summary": task_summary,
    "alpha_sensitivity": {
        "summary": alpha_summary,
        "fraction_monotonicity_violations": monotonic_violations,
    },
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary["validation"], indent=2))
print(json.dumps(explainer_summary, indent=2))

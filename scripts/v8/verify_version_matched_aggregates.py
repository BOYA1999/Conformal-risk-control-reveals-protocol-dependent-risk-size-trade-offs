import csv
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
failures = []


def rows(relative):
    with (ROOT / relative).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def payload(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        failures.append(message)


def close(value, expected, tolerance=1e-12):
    return math.isclose(float(value), expected, rel_tol=0.0, abs_tol=tolerance)


reference = rows("results/v8/pharmacophore_oracle/reference_return_summary.csv")
tie = rows("results/v8/pharmacophore_oracle/reference_return_tie_summary.csv")
capacity = rows("results/v8/pharmacophore_oracle/schedule_matched_capacity_oracle_summary.csv")
test_reference = next(row for row in reference if row["split"] == "test")
test_tie = next(row for row in tie if row["split"] == "test")
gradcam = next(row for row in capacity if row["method_schedule"] == "gradcam")
require(len(reference) == 2 and len(tie) == 2 and len(capacity) == 3, "pharmacophore row counts")
require(test_reference["molecules"] == "512" and test_reference["equality_count"] == "512", "exact reference-return identity")
require(close(test_reference["mean_reference_fraction"], 0.5822173758435784), "reference retained fraction")
require(close(test_tie["mean_tie_inflation"], 0.543763691111102), "tie inflation")
require(close(gradcam["actual_test_risk"], 0.08885152110450824), "Grad-CAM risk")
require(close(gradcam["actual_mean_retained_fraction"], 0.9206774728504318), "Grad-CAM retained fraction")
require(close(gradcam["schedule_oracle_mean_retained_fraction"], 0.5567140269957713), "schedule-matched oracle fraction")

rings_qa = payload("results/v8/rings_stratified/QA.json")
rings_validation = payload("results/v8/rings_stratified/validation.json")
rings = rows("results/v8/rings_stratified/manuscript_ready_stratified_table.csv")
require(rings_qa.get("status") == "PASS_WITH_RECOMPUTATION_DRIFT", "rings QA status")
require(rings_validation.get("status") == "PASS" and len(rings) == 12, "rings validation")
require({row["stratum"] for row in rings} == {"all_nonempty_reference", "positive_label", "negative_label"}, "rings strata")

provenance = rows("results/v8/version_provenance/analysis_version_result_provenance.csv")
provenance_qa = payload("results/v8/version_provenance/provenance_qa.json")
require(len(provenance) == 19, "provenance row count")
require(provenance_qa.get("status") == "PASS", "provenance QA")
require(all(not re.search(r"(?:^[A-Za-z]:[\\/]|^/)", row["canonical_output_file"]) for row in provenance), "repository-relative output paths")

for relative, marker in [
    ("results/v8/pharmacophore_oracle/DATA_LICENSE.md", "CC BY 4.0"),
    ("results/v8/rings_stratified/DATA_LICENSE.md", "CC BY-SA 4.0"),
]:
    path = ROOT / relative
    require(path.is_file() and marker in path.read_text(encoding="utf-8"), f"data-source notice: {relative}")

if failures:
    raise SystemExit("FAIL\n" + "\n".join(failures))
print("PASS: current aggregate and provenance checks")

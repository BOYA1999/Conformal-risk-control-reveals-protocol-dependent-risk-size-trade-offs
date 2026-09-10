import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


REPOSITORY = Path(__file__).resolve().parents[3]
ROOT = Path(os.environ.get("MOLXAI_PHARMACOPHORE_ROOT", "external/pharmacophore_benchmark")).resolve()
OUT = Path(os.environ.get("MOLXAI_OUTPUT_DIR", REPOSITORY / "results/v8/pharmacophore_oracle")).resolve()
OUT.mkdir(parents=True, exist_ok=True)
ATTRIBUTIONS = ROOT / "outputs" / "attributions.parquet"
ORIGINAL_SUMMARY = ROOT / "outputs" / "explanation_summary.csv"
ORIGINAL_SET_METRICS = ROOT / "outputs" / "test_set_metrics.parquet"
SOURCE_CODE = ROOT / "run_pharmacophore_benchmark.py"
RUN_CONTRACT = ROOT / "run_contract.json"
FRACTIONS = np.linspace(0.0, 1.0, 101)
ALPHA = 0.10
SALT = "molxai-crc-k4_2ar-v1"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def random_scores(identifier, n_atoms):
    values = []
    for atom_index in range(n_atoms):
        digest = hashlib.sha256(f"{identifier}|{atom_index}|{SALT}|random".encode()).digest()
        values.append(int.from_bytes(digest[:8], "big") / 2**64)
    return np.asarray(values, dtype=float)


def top_fraction(scores, fraction):
    scores = np.asarray(scores, dtype=float)
    n_atoms = len(scores)
    nominal_count = min(n_atoms, max(0, int(math.ceil(fraction * n_atoms))))
    selected = np.zeros(n_atoms, dtype=bool)
    if nominal_count:
        order = np.argsort(-scores, kind="stable")
        threshold = scores[order[nominal_count - 1]]
        selected = scores >= threshold
    return selected, nominal_count


def set_values(selected, truth):
    truth = np.asarray(truth, dtype=bool)
    intersection = int(np.logical_and(selected, truth).sum())
    selected_count = int(selected.sum())
    reference_count = int(truth.sum())
    union = int(np.logical_or(selected, truth).sum())
    return {
        "risk": 1.0 - intersection / reference_count,
        "retained_fraction": selected_count / len(truth),
        "precision": intersection / selected_count if selected_count else 0.0,
        "iou": intersection / union if union else 0.0,
        "selected_atoms": selected_count,
        "intersection_atoms": intersection,
    }


def score_for(row, method):
    if method == "gradcam":
        return np.asarray(row.gradcam, dtype=float)
    if method == "vanilla_gradient":
        return np.asarray(row.vanilla_gradient, dtype=float)
    if method == "random":
        return random_scores(str(row.ID), int(row.n_atoms))
    if method == "reference_return":
        return np.asarray(row.y_true, dtype=float)
    raise ValueError(method)


def actual_tables(frame, method):
    losses = np.zeros((len(frame), len(FRACTIONS)), dtype=float)
    counts = np.zeros((len(frame), len(FRACTIONS)), dtype=int)
    for row_index, row in enumerate(frame.itertuples(index=False)):
        truth = np.asarray(row.y_true, dtype=bool)
        scores = score_for(row, method)
        for fraction_index, fraction in enumerate(FRACTIONS):
            selected, _ = top_fraction(scores, float(fraction))
            losses[row_index, fraction_index] = set_values(selected, truth)["risk"]
            counts[row_index, fraction_index] = int(selected.sum())
    return losses, counts


def calibrate(losses):
    corrected = (len(losses) * losses.mean(axis=0) + 1.0) / (len(losses) + 1)
    feasible = np.flatnonzero(corrected <= ALPHA)
    index = int(feasible[0]) if len(feasible) else len(FRACTIONS) - 1
    return index, losses.mean(axis=0), corrected


def evaluate_actual(frame, method, fraction_index):
    rows = []
    fraction = float(FRACTIONS[fraction_index])
    for row in frame.itertuples(index=False):
        truth = np.asarray(row.y_true, dtype=bool)
        selected, nominal_count = top_fraction(score_for(row, method), fraction)
        values = set_values(selected, truth)
        rows.append({
            "split": str(row.split),
            "ID": str(row.ID),
            "method": method,
            "fraction_index": fraction_index,
            "nominal_fraction": fraction,
            "n_atoms": int(row.n_atoms),
            "reference_atoms": int(truth.sum()),
            "nominal_atoms": nominal_count,
            **values,
        })
    result = pd.DataFrame(rows)
    return result, {
        "test_risk": float(result.risk.mean()),
        "mean_retained_fraction": float(result.retained_fraction.mean()),
        "mean_precision": float(result.precision.mean()),
        "mean_iou": float(result.iou.mean()),
    }


def evaluate_schedule_oracle(frame, method, count_table, fraction_index):
    rows = []
    for row_index, row in enumerate(frame.itertuples(index=False)):
        n_atoms = int(row.n_atoms)
        reference_atoms = int(np.asarray(row.y_true, dtype=bool).sum())
        selected_atoms = int(count_table[row_index, fraction_index])
        recovered_atoms = min(selected_atoms, reference_atoms)
        union_atoms = selected_atoms + reference_atoms - recovered_atoms
        rows.append({
            "split": str(row.split),
            "ID": str(row.ID),
            "method_schedule": method,
            "fraction_index": fraction_index,
            "nominal_fraction": float(FRACTIONS[fraction_index]),
            "n_atoms": n_atoms,
            "reference_atoms": reference_atoms,
            "selected_atoms": selected_atoms,
            "recovered_atoms": recovered_atoms,
            "risk": 1.0 - recovered_atoms / reference_atoms,
            "retained_fraction": selected_atoms / n_atoms,
            "precision": recovered_atoms / selected_atoms if selected_atoms else 0.0,
            "iou": recovered_atoms / union_atoms if union_atoms else 0.0,
        })
    result = pd.DataFrame(rows)
    return result, {
        "test_risk": float(result.risk.mean()),
        "mean_retained_fraction": float(result.retained_fraction.mean()),
        "mean_precision": float(result.precision.mean()),
        "mean_iou": float(result.iou.mean()),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_parquet(ATTRIBUTIONS).sort_values(["split", "selection_rank"], kind="stable").reset_index(drop=True)
    calibration = data.loc[data.split == "val"].reset_index(drop=True)
    test = data.loc[data.split == "test"].reset_index(drop=True)
    original_summary = pd.read_csv(ORIGINAL_SUMMARY).set_index("method")
    original_set_metrics = pd.read_parquet(ORIGINAL_SET_METRICS)

    reference_losses_cal, reference_counts_cal = actual_tables(calibration, "reference_return")
    reference_losses_test, reference_counts_test = actual_tables(test, "reference_return")
    reference_index, reference_empirical, reference_corrected = calibrate(reference_losses_cal)
    reference_rows = []
    for split_name, frame, count_table in [
        ("val", calibration, reference_counts_cal),
        ("test", test, reference_counts_test),
    ]:
        for row_index, row in enumerate(frame.itertuples(index=False)):
            truth = np.asarray(row.y_true, dtype=bool)
            selected, nominal_count = top_fraction(truth.astype(float), float(FRACTIONS[reference_index]))
            reference_rows.append({
                "split": split_name,
                "selection_rank": int(row.selection_rank),
                "ID": str(row.ID),
                "n_atoms": int(row.n_atoms),
                "reference_atoms": int(truth.sum()),
                "reference_fraction": float(truth.mean()),
                "oracle_fraction_index": reference_index,
                "oracle_nominal_fraction": float(FRACTIONS[reference_index]),
                "oracle_nominal_atoms": nominal_count,
                "oracle_selected_atoms": int(selected.sum()),
                "oracle_tie_extra_atoms": int(selected.sum()) - nominal_count,
                "selected_minus_reference_atoms": int(selected.sum()) - int(truth.sum()),
                "selected_equals_reference_mask": bool(np.array_equal(selected, truth)),
                "oracle_risk": set_values(selected, truth)["risk"],
            })
    reference_audit = pd.DataFrame(reference_rows)
    reference_audit.to_csv(OUT / "reference_return_molecule_audit.csv", index=False)

    schedule_rows = []
    schedule_molecule_frames = []
    reconstruction_checks = {}
    for method in ["gradcam", "vanilla_gradient", "random"]:
        cal_losses, cal_counts = actual_tables(calibration, method)
        test_losses, test_counts = actual_tables(test, method)
        actual_index, actual_empirical, actual_corrected = calibrate(cal_losses)
        actual_rows, actual_metrics = evaluate_actual(test, method, actual_index)
        oracle_cal_losses = np.zeros_like(cal_losses)
        oracle_test_losses = np.zeros_like(test_losses)
        for row_index, row in enumerate(calibration.itertuples(index=False)):
            reference_atoms = int(np.asarray(row.y_true, dtype=bool).sum())
            oracle_cal_losses[row_index] = 1.0 - np.minimum(cal_counts[row_index], reference_atoms) / reference_atoms
        for row_index, row in enumerate(test.itertuples(index=False)):
            reference_atoms = int(np.asarray(row.y_true, dtype=bool).sum())
            oracle_test_losses[row_index] = 1.0 - np.minimum(test_counts[row_index], reference_atoms) / reference_atoms
        oracle_index, oracle_empirical, oracle_corrected = calibrate(oracle_cal_losses)
        oracle_rows, oracle_metrics = evaluate_schedule_oracle(test, method, test_counts, oracle_index)
        oracle_rows["actual_calibrated_fraction_index"] = actual_index
        oracle_rows["actual_calibrated_nominal_fraction"] = float(FRACTIONS[actual_index])
        schedule_molecule_frames.append(oracle_rows)
        schedule_rows.append({
            "method_schedule": method,
            "actual_fraction_index": actual_index,
            "actual_nominal_fraction": float(FRACTIONS[actual_index]),
            "actual_calibration_empirical_risk": float(actual_empirical[actual_index]),
            "actual_calibration_corrected_risk": float(actual_corrected[actual_index]),
            "actual_test_risk": actual_metrics["test_risk"],
            "actual_mean_retained_fraction": actual_metrics["mean_retained_fraction"],
            "schedule_oracle_fraction_index": oracle_index,
            "schedule_oracle_nominal_fraction": float(FRACTIONS[oracle_index]),
            "schedule_oracle_calibration_empirical_risk": float(oracle_empirical[oracle_index]),
            "schedule_oracle_calibration_corrected_risk": float(oracle_corrected[oracle_index]),
            "schedule_oracle_test_risk": oracle_metrics["test_risk"],
            "schedule_oracle_mean_retained_fraction": oracle_metrics["mean_retained_fraction"],
            "schedule_oracle_mean_precision": oracle_metrics["mean_precision"],
            "schedule_oracle_mean_iou": oracle_metrics["mean_iou"],
            "capacity_excess_mean_retained_fraction": actual_metrics["mean_retained_fraction"] - oracle_metrics["mean_retained_fraction"],
            "oracle_pointwise_dominates_actual_calibration": bool(np.all(oracle_cal_losses <= cal_losses + 1e-12)),
            "oracle_pointwise_dominates_actual_test": bool(np.all(oracle_test_losses <= test_losses + 1e-12)),
            "oracle_fraction_index_not_later_than_actual": bool(oracle_index <= actual_index),
        })
        source = original_summary.loc[method]
        reconstruction_checks[method] = {
            "fraction_index_exact": actual_index == int(source.fraction_index),
            "test_risk_abs_difference": abs(actual_metrics["test_risk"] - float(source.test_risk)),
            "retained_fraction_abs_difference": abs(actual_metrics["mean_retained_fraction"] - float(source.mean_retained_fraction)),
            "precision_abs_difference": abs(actual_metrics["mean_precision"] - float(source.mean_precision)),
            "iou_abs_difference": abs(actual_metrics["mean_iou"] - float(source.mean_iou)),
        }
    schedule_summary = pd.DataFrame(schedule_rows)
    random_retained = float(schedule_summary.loc[schedule_summary.method_schedule == "random", "actual_mean_retained_fraction"].iloc[0])
    reference_return_retained = float(original_summary.loc["oracle", "mean_retained_fraction"])
    schedule_summary["random_to_exact_reference_return_size_gap_fraction_recovered"] = (
        random_retained - schedule_summary.actual_mean_retained_fraction
    ) / (random_retained - reference_return_retained)
    schedule_summary["random_to_method_schedule_oracle_size_gap_fraction_recovered"] = (
        random_retained - schedule_summary.actual_mean_retained_fraction
    ) / (random_retained - schedule_summary.schedule_oracle_mean_retained_fraction)
    schedule_summary.to_csv(OUT / "schedule_matched_capacity_oracle_summary.csv", index=False)
    pd.concat(schedule_molecule_frames, ignore_index=True).to_csv(OUT / "schedule_matched_capacity_oracle_molecule_audit.csv", index=False)

    source_oracle = original_summary.loc["oracle"]
    original_test_oracle = original_set_metrics.loc[original_set_metrics.method == "oracle"].sort_values("ID")
    current_test_oracle = reference_audit.loc[reference_audit.split == "test"].sort_values("ID")
    source_count_match = np.array_equal(
        original_test_oracle.selected_atoms.to_numpy(dtype=int),
        current_test_oracle.oracle_selected_atoms.to_numpy(dtype=int),
    )
    split_summary = reference_audit.groupby("split", sort=True).agg(
        molecules=("ID", "size"),
        equality_count=("selected_equals_reference_mask", "sum"),
        mean_reference_fraction=("reference_fraction", "mean"),
        mean_oracle_selected_fraction=("oracle_selected_atoms", lambda s: float(np.mean(s.to_numpy() / reference_audit.loc[s.index, "n_atoms"].to_numpy()))),
        mean_tie_extra_atoms=("oracle_tie_extra_atoms", "mean"),
        min_reference_atoms=("reference_atoms", "min"),
        max_reference_atoms=("reference_atoms", "max"),
    ).reset_index()
    split_summary["equality_rate"] = split_summary.equality_count / split_summary.molecules
    split_summary.to_csv(OUT / "reference_return_summary.csv", index=False)

    tolerances_pass = all(
        item["fraction_index_exact"]
        and item["test_risk_abs_difference"] <= 1e-12
        and item["retained_fraction_abs_difference"] <= 1e-12
        and item["precision_abs_difference"] <= 1e-12
        and item["iou_abs_difference"] <= 1e-12
        for item in reconstruction_checks.values()
    )
    schedule_proofs_pass = bool(
        schedule_summary.oracle_pointwise_dominates_actual_calibration.all()
        and schedule_summary.oracle_pointwise_dominates_actual_test.all()
        and schedule_summary.oracle_fraction_index_not_later_than_actual.all()
        and (schedule_summary.capacity_excess_mean_retained_fraction >= -1e-12).all()
    )
    validation = {
        "status": "PASS" if all([
            len(calibration) == 512,
            len(test) == 512,
            bool(reference_audit.selected_equals_reference_mask.all()),
            reference_index == 1,
            source_count_match,
            tolerances_pass,
            schedule_proofs_pass,
        ]) else "FAIL",
        "input_sha256": {
            str(ATTRIBUTIONS): sha256(ATTRIBUTIONS),
            str(ORIGINAL_SUMMARY): sha256(ORIGINAL_SUMMARY),
            str(ORIGINAL_SET_METRICS): sha256(ORIGINAL_SET_METRICS),
            str(SOURCE_CODE): sha256(SOURCE_CODE),
            str(RUN_CONTRACT): sha256(RUN_CONTRACT),
        },
        "checks": {
            "calibration_rows_512": len(calibration) == 512,
            "test_rows_512": len(test) == 512,
            "binary_reference_scores_only": sorted(set(np.concatenate([np.asarray(x, dtype=int) for x in data.y_true]).tolist())) == [0, 1],
            "reference_return_calibrated_index_is_first_nonempty_grid_point": reference_index == 1,
            "selected_mask_equals_reference_mask_all_1024": bool(reference_audit.selected_equals_reference_mask.all()),
            "source_test_oracle_selected_counts_exact": source_count_match,
            "source_summary_reconstructed_within_1e_12": tolerances_pass,
            "schedule_matched_oracle_proofs_pass": schedule_proofs_pass,
        },
        "reference_return_oracle": {
            "score_definition": "binary y_true atom-reference vector: 1 for computed-reference atoms and 0 otherwise",
            "alpha": ALPHA,
            "fraction_grid": "0.00 to 1.00 in increments of 0.01",
            "tie_rule": "select all atoms whose score is at least the kth score threshold; stable sorting only locates the threshold",
            "calibrated_fraction_index": reference_index,
            "calibrated_nominal_fraction": float(FRACTIONS[reference_index]),
            "calibration_empirical_risk": float(reference_empirical[reference_index]),
            "calibration_corrected_risk": float(reference_corrected[reference_index]),
            "test_risk": float(source_oracle.test_risk),
            "test_mean_retained_fraction": float(source_oracle.mean_retained_fraction),
            "test_mean_iou": float(source_oracle.mean_iou),
            "test_mean_tie_inflation": float(source_oracle.mean_tie_inflation),
            "construction_verdict": "S equals Y because the first nonzero grid point requests at least one atom, a reference atom has the maximum binary score, and exact-tie inclusion returns every score-1 reference atom.",
        },
        "schedule_matched_capacity_oracle": {
            "definition": "For each method, molecule, and grid fraction, preserve that method's realized tie-inclusive selected atom count k; the capacity oracle recovers min(k, |Y|) reference atoms, then calibrates this method-specific oracle loss table independently.",
            "comparability_boundary": "This is a method-specific capacity benchmark on an observed count schedule. It is not the original reference-return oracle, not one shared comparator across methods, and not an independently observed explanation method.",
            "gradcam_random_to_method_schedule_oracle_size_gap_fraction_recovered": float(schedule_summary.loc[schedule_summary.method_schedule == "gradcam", "random_to_method_schedule_oracle_size_gap_fraction_recovered"].iloc[0]),
        },
        "source_reconstruction": reconstruction_checks,
    }
    write_json(OUT / "validation.json", validation)

    gradcam_schedule = schedule_summary.loc[schedule_summary.method_schedule == "gradcam"].iloc[0]
    report = f"""# Pharmacophore oracle evidence audit

## Decision

The reported `Reference-first oracle` is exactly the full computed reference returned by construction. It is not a tie-free CRC oracle that spends the 0.10 miss budget. Rename it **exact reference-return benchmark** (or **full-reference return**) and describe the legacy efficiency as a size-gap calculation anchored to that benchmark.

## Frozen definition and result

- Atom score: the binary released `y_true` reference vector (reference atoms = 1; all other atoms = 0).
- Fraction grid: 0.00--1.00 in increments of 0.01; corrected risk `(n * mean_loss + 1) / (n + 1)` at alpha = 0.10.
- Selection: `k = ceil(lambda * N)` followed by inclusion of every atom tied at the kth score threshold.
- The first feasible grid point was lambda = {FRACTIONS[reference_index]:.2f}. Because at least one score-1 atom enters and all score-1 atoms tie, the selector returned every reference atom.
- `|S| = |Y|` for {int(reference_audit.selected_equals_reference_mask.sum())}/{len(reference_audit)} molecules: 512/512 calibration and 512/512 test.
- Test risk = {float(source_oracle.test_risk):.12f}, mean retained fraction = {float(source_oracle.mean_retained_fraction):.12f}, IoU = {float(source_oracle.mean_iou):.12f}, and mean tie inflation = {float(source_oracle.mean_tie_inflation):.12f}.
- Therefore 0.582217 is simply the test mean computed-reference fraction, not evidence that calibration discovered a compact 58.2% threshold.

## Separately labelled schedule-matched capacity audit

To make the capacity comparison consistent with the schedule-matched definition used elsewhere in the project, a new **Grad-CAM-schedule-matched capacity oracle** preserves Grad-CAM's realized tie-inclusive atom-count schedule at every grid point and places reference atoms first within each count. It is calibrated independently on the same 512 validation molecules.

- Actual Grad-CAM: lambda = {gradcam_schedule.actual_nominal_fraction:.2f}, test risk = {gradcam_schedule.actual_test_risk:.6f}, retained fraction = {gradcam_schedule.actual_mean_retained_fraction:.6f}.
- Grad-CAM-schedule-matched capacity oracle: lambda = {gradcam_schedule.schedule_oracle_nominal_fraction:.2f}, test risk = {gradcam_schedule.schedule_oracle_test_risk:.6f}, retained fraction = {gradcam_schedule.schedule_oracle_mean_retained_fraction:.6f}.
- Schedule-matched oracle precision = {gradcam_schedule.schedule_oracle_mean_precision:.6f}; IoU = {gradcam_schedule.schedule_oracle_mean_iou:.6f}.
- Capacity excess = {gradcam_schedule.capacity_excess_mean_retained_fraction:.6f}.
- Using independently calibrated random as the numerator anchor, the exact-reference-return size-gap fraction recovered is {gradcam_schedule.random_to_exact_reference_return_size_gap_fraction_recovered:.15f}; if the denominator is instead the Grad-CAM-schedule-matched capacity oracle, the corresponding value is {gradcam_schedule.random_to_method_schedule_oracle_size_gap_fraction_recovered:.15f}. Both are negative because Grad-CAM retained more atoms than calibrated random.

This comparator is method-specific and uses the observed Grad-CAM cardinality schedule. It must not be pooled with or silently substituted for the exact reference-return benchmark. The adverse equal-size-random result is unchanged because it does not use either oracle denominator.

## Manuscript wording boundary

Use: `exact reference-return benchmark`, `full computed-reference return`, and, for the new comparison, `method-specific schedule-matched capacity oracle`.

Avoid treating the 0.582217 reference-return fraction as a CRC-optimized lower bound at alpha = 0.10. If the historical ratio is retained, call it the **fraction of the random-to-exact-reference-return size gap recovered** and state that a negative value means Grad-CAM retained more atoms than independently calibrated random; it is not a general efficiency parameter.

All statements remain limited to one K4_2ar task, one released checkpoint, one fixed 512/512 positive slice, and a computed operational atom reference. No experimental, biological, mechanistic, causal, family-wide, or population-level atom validation follows.

## Evidence files

- `reference_return_molecule_audit.csv`: all 1,024 molecule-level equality checks.
- `reference_return_summary.csv`: split-level counts and mean fractions.
- `schedule_matched_capacity_oracle_summary.csv`: actual and method-specific schedule-matched results.
- `schedule_matched_capacity_oracle_molecule_audit.csv`: molecule-level schedule-matched test results.
- `validation.json`: input hashes, source-result reconstruction, and proof checks.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")

    contract = {
        "slice_id": "V8-PHARM-ORACLE",
        "parent_claim": "The pharmacophore case separates molecular prediction from atom localization, while oracle-relative size quantities require an explicit oracle definition.",
        "question": "Does the reported rationale-first oracle return the full computed reference by construction, and what changes under a method-specific schedule-matched capacity oracle?",
        "fixed_conditions": {
            "task": "K4_2ar Close released GCN",
            "calibration_molecules": 512,
            "test_molecules": 512,
            "alpha": ALPHA,
            "fraction_grid": FRACTIONS.tolist(),
            "tie_policy": "include all exact ties",
            "no_retraining": True,
            "no_new_attribution": True,
        },
        "outputs": [
            "reference_return_molecule_audit.csv",
            "reference_return_summary.csv",
            "schedule_matched_capacity_oracle_summary.csv",
            "schedule_matched_capacity_oracle_molecule_audit.csv",
            "validation.json",
            "REPORT.md",
        ],
        "paper_role": "supporting_information_claim_boundary",
        "section_id": "constructed_pharmacophore_reference",
        "item_id": "V8-PHARM-ORACLE",
        "claim_links": ["pharmacophore_localization_boundary", "oracle_definition_boundary"],
        "target_display": "new SI oracle-definition audit table",
        "comparability": "Reference-return and schedule-matched capacity results are distinct estimands; only within-method actual-versus-schedule comparisons are direct.",
        "next_route": "write",
    }
    write_json(OUT / "analysis_contract.json", contract)

    manifest_paths = sorted(path for path in OUT.iterdir() if path.is_file() and path.name != "MANIFEST_SHA256.txt")
    manifest = "".join(f"{sha256(path)}  {path.name}\n" for path in manifest_paths)
    (OUT / "MANIFEST_SHA256.txt").write_text(manifest, encoding="utf-8")
    if validation["status"] != "PASS":
        raise RuntimeError("pharmacophore oracle audit failed validation")
    print(schedule_summary.to_string(index=False))
    print("PASS")


if __name__ == "__main__":
    main()

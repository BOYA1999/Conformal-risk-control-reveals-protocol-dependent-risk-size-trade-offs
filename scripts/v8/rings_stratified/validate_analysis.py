import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


REPOSITORY = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get("MOLXAI_OUTPUT_DIR", REPOSITORY / "results/v8/rings_stratified")).resolve()
PROJECT = Path(os.environ.get("MOLXAI_PROJECT_ROOT", "external/project")).resolve()
CACHE = Path(os.environ.get("MOLXAI_SCORE_CACHE", "external/score_cache/bxaic__rings-count__gin__seed42.npz")).resolve()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def close(a, b, tolerance=1e-12):
    return bool(np.isclose(float(a), float(b), rtol=0, atol=tolerance))


def rebuild_manifest():
    rows = []
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "output_manifest.csv":
            rows.append({"file": path.name, "bytes": path.stat().st_size, "sha256": digest(path)})
    with (OUT / "output_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("file", "bytes", "sha256"))
        writer.writeheader()
        writer.writerows(rows)


checks = {}
full_source = pd.read_csv(PROJECT / "results/molecule_level_ig_seed42/molecules.csv.gz")
full_source = full_source[(full_source.family == "bxaic") & (full_source.task == "rings-count") & full_source.n_rationale.gt(0)].copy()
full_rows = pd.read_csv(OUT / "full_test_observed_class_by_model.csv")
full_differences = []
for row in full_rows.itertuples(index=False):
    subset = full_source[full_source.cell_id == row.cell_id]
    if row.stratum == "positive_label":
        subset = subset[subset.label == 1]
    elif row.stratum == "negative_label":
        subset = subset[subset.label == 0]
    k = np.ceil(row.oracle_nominal_fraction * subset.n_atoms).astype(int)
    oracle_retained = k / subset.n_atoms
    oracle_risk = 1 - np.minimum(k, subset.n_rationale) / subset.n_rationale
    expected = {
        "n_test": len(subset),
        "risk": subset.miss_loss.mean(),
        "retained_fraction": subset.selected_fraction.mean(),
        "oracle_risk": oracle_risk.mean(),
        "oracle_retained_fraction": oracle_retained.mean(),
        "oracle_excess": (subset.selected_fraction - oracle_retained).mean(),
        "selected_minus_random": subset.fidelity_advantage.mean(),
        "positive_contrast_fraction": (subset.fidelity_advantage > 0).mean(),
    }
    full_differences.extend(abs(float(getattr(row, key)) - float(value)) for key, value in expected.items())
checks["full_rows_independently_recomputed"] = max(full_differences) < 1e-12
checks["full_counts_are_4933_1503_3430_per_cell"] = bool(
    full_rows[full_rows.stratum == "all_nonempty_reference"].n_test.eq(4933).all()
    and full_rows[full_rows.stratum == "positive_label"].n_test.eq(1503).all()
    and full_rows[full_rows.stratum == "negative_label"].n_test.eq(3430).all()
)

matched_molecules = pd.read_csv(OUT / "matched_target_molecule_metrics.csv")
matched_rows = pd.read_csv(OUT / "matched_target_class_strata.csv")
matched_differences = []
for row in matched_rows.itertuples(index=False):
    subset = matched_molecules[(matched_molecules.method == row.method) & (matched_molecules.target == row.target)]
    if row.stratum == "positive_label":
        subset = subset[subset.label == 1]
    elif row.stratum == "negative_label":
        subset = subset[subset.label == 0]
    expected = {
        "n_test": len(subset),
        "risk": subset.missed_reference_loss.mean(),
        "retained_fraction": subset.retained_fraction.mean(),
        "oracle_risk": subset.oracle_risk.mean(),
        "oracle_retained_fraction": subset.oracle_retained_fraction.mean(),
        "oracle_excess": subset.oracle_excess.mean(),
        "selected_minus_random": subset.selected_minus_random.mean(),
        "positive_contrast_fraction": (subset.selected_minus_random > 0).mean(),
    }
    matched_differences.extend(abs(float(getattr(row, key)) - float(value)) for key, value in expected.items())
checks["matched_rows_independently_recomputed"] = max(matched_differences) < 1e-12
checks["matched_each_method_target_has_same_100_sources"] = bool(
    matched_molecules.groupby(["method", "target"]).size().eq(100).all()
    and matched_molecules.groupby(["method", "target"]).source_index.nunique().eq(100).all()
    and matched_molecules.groupby(["method", "target"]).label.sum().eq(29).all()
)
checks["calibration_never_repeated_within_stratum"] = bool(
    full_rows.calibration_was_not_repeated_within_stratum.all()
    and matched_rows.calibration_was_not_repeated_within_stratum.all()
)
checks["matched_global_fraction_constant_across_strata"] = bool(
    matched_rows.groupby(["method", "target"]).nominal_fraction.nunique().eq(1).all()
    and matched_rows.groupby(["method", "target"]).oracle_nominal_fraction.nunique().eq(1).all()
)

reproduction = pd.read_csv(OUT / "a5_reproduction_check.csv")
checks["a5_nominal_fractions_exact"] = bool(reproduction.fraction_difference_vs_A5.abs().le(1e-12).all())
checks["a5_recorded_tie_variation_bounded"] = bool(
    reproduction[["risk_difference_vs_A5", "retained_difference_vs_A5"]].abs().to_numpy().max() <= 0.002
)
checks["core_outputs_finite"] = bool(np.isfinite(pd.concat([
    full_rows[["risk", "retained_fraction", "oracle_excess", "selected_minus_random"]],
    matched_rows[["risk", "retained_fraction", "oracle_excess", "selected_minus_random"]],
]).to_numpy()).all())

inputs = json.loads((OUT / "input_manifest.json").read_text(encoding="utf-8"))
checks["full_source_hash_matches"] = inputs["molecule_level_ig_seed42_molecules_sha256"] == digest(PROJECT / "results/molecule_level_ig_seed42/molecules.csv.gz")
checks["a5_cache_hash_matches"] = inputs["a5_score_cache_sha256"] == digest(CACHE)

result = {
    "status": "PASS" if all(checks.values()) else "FAIL",
    "checks": checks,
    "max_full_reconstruction_difference": max(full_differences),
    "max_matched_reconstruction_difference": max(matched_differences),
    "interpretation": "The full-test aggregation is an exact summary of frozen molecule-level outputs. The matched target audit uses one new same-run score calculation on the frozen A5 identities and checkpoint; all nominal fractions match A5, while small exact-tie set variation remains explicitly recorded.",
}
(OUT / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
rebuild_manifest()
print(json.dumps(result, indent=2))

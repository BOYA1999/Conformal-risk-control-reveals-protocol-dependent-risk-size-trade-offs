import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RUN1 = ROOT / "diagnostics" / "postrun_before_numerical_lineage_correction"
RUN3 = ROOT / "diagnostics" / "gpu_attribution_instability" / "run3_predictive_summary_gpu"
OUT = ROOT / "diagnostics" / "gpu_attribution_instability"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main():
    keys = ["dataset", "split", "pair_type", "pair_id", "seed", "method"]
    old = pd.read_csv(RUN1 / "pair_attributions.csv")
    new = pd.read_csv(RUN3 / "pair_attributions.csv")
    merged = old.merge(new, on=keys, suffixes=("_run1", "_run3"), validate="one_to_one")
    attribution_differences = []
    for row in merged.itertuples(index=False):
        a1 = np.fromstring(row.scores_a_run1, sep=";")
        a3 = np.fromstring(row.scores_a_run3, sep=";")
        b1 = np.fromstring(row.scores_b_run1, sep=";")
        b3 = np.fromstring(row.scores_b_run3, sep=";")
        scores1 = np.concatenate([a1, b1])
        scores3 = np.concatenate([a3, b3])
        changed_a = row.scores_a_run1 != row.scores_a_run3
        changed_b = row.scores_b_run1 != row.scores_b_run3
        rank_changed = not np.array_equal(np.argsort(-scores1, kind="stable"), np.argsort(-scores3, kind="stable"))
        if changed_a or changed_b or rank_changed:
            attribution_differences.append(
                {
                    **{key: getattr(row, key) for key in keys},
                    "scores_a_changed": changed_a,
                    "scores_b_changed": changed_b,
                    "full_rank_changed": rank_changed,
                    "max_absolute_score_difference": float(np.max(np.abs(scores1 - scores3))),
                }
            )
    attribution_frame = pd.DataFrame(attribution_differences)
    attribution_frame.to_csv(OUT / "gpu_run1_vs_run3_attribution_row_differences.csv", index=False)

    result_keys = ["dataset", "seed", "method", "proxy", "evaluation_split", "stratum", "policy", "aggregation"]
    metrics = ["risk", "retained_fraction", "precision", "iou", "nominal_fraction", "calibration_empirical_risk", "calibration_corrected_risk"]
    old_results = pd.read_csv(RUN1 / "crc_results.csv")
    new_results = pd.read_csv(RUN3 / "crc_results.csv")
    result_compare = old_results.merge(new_results, on=result_keys, suffixes=("_run1", "_run3"), validate="one_to_one")
    any_difference = np.zeros(len(result_compare), dtype=bool)
    for metric in metrics:
        any_difference |= ~np.isclose(result_compare[f"{metric}_run1"], result_compare[f"{metric}_run3"], atol=1e-12, rtol=0, equal_nan=True)
        result_compare[f"{metric}_absolute_difference"] = np.abs(result_compare[f"{metric}_run1"] - result_compare[f"{metric}_run3"])
    result_differences = result_compare.loc[any_difference]
    result_differences.to_csv(OUT / "gpu_run1_vs_run3_crc_differences.csv", index=False)

    audit = {
        "status": "GPU_ATTRIBUTION_NOT_FROZEN",
        "run1_full_artifacts": str(RUN1.relative_to(ROOT)).replace("\\", "/"),
        "run2_evidence": "diagnostics/gpu_attribution_instability/run2_lineage_corrected_log_only; detailed CSVs were overwritten before cross-run nondeterminism was recognized",
        "run3_full_artifacts": str(RUN3.relative_to(ROOT)).replace("\\", "/"),
        "pair_attribution_rows": len(merged),
        "scores_a_text_changed_rows": int((merged.scores_a_run1 != merged.scores_a_run3).sum()),
        "scores_b_text_changed_rows": int((merged.scores_b_run1 != merged.scores_b_run3).sum()),
        "full_rank_changed_rows": int(attribution_frame.full_rank_changed.sum()),
        "max_absolute_atom_score_difference": float(attribution_frame.max_absolute_score_difference.max()),
        "crc_rows": len(result_compare),
        "crc_rows_with_any_difference": len(result_differences),
        "maximum_absolute_result_differences": {
            metric: float(result_compare[f"{metric}_absolute_difference"].max(skipna=True)) for metric in metrics
        },
        "near_zero_sign_instability": {
            "dataset": "CHEMBL4792_Ki",
            "split": "test",
            "pair_id": 23,
            "seed": 17,
            "prediction_table_delta": 0.0,
            "gpu_checkpoint_delta_observed": 4.76837158203125e-7,
            "reobserved_in_run3": False,
        },
        "run1_pair_attributions_sha256": sha256(RUN1 / "pair_attributions.csv"),
        "run3_pair_attributions_sha256": sha256(RUN3 / "pair_attributions.csv"),
        "boundary": "GPU attribution outputs are failed-run diagnostic history only and cannot be selected or reported as the formal A7 attribution result.",
    }
    (OUT / "gpu_attribution_instability_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

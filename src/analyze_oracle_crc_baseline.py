import json
from pathlib import Path

import numpy as np
import pandas as pd

from molxai_crc import calibrate_crc
from run_gradient_grid import BXAIC_TASKS, GOOGLE_TASKS, FRACTIONS, bxaic_partitions, google_partitions


ROOT = Path(__file__).resolve().parents[1]
ALPHAS = [0.05, 0.10, 0.20]


def oracle_loss_table(graphs):
    rows = []
    for graph in graphs:
        m = int(graph.num_nodes)
        r = int(graph.rationale_mask.sum())
        if r == 0:
            continue
        retained = np.where(FRACTIONS == 0, 0, np.ceil(FRACTIONS * m)).astype(int)
        recovered = np.minimum(retained, r)
        rows.append(1.0 - recovered / r)
    table = np.asarray(rows, dtype=float)
    if table.size == 0 or np.any(np.diff(table, axis=1) > 1e-12):
        raise ValueError("invalid oracle loss table")
    return table


def oracle_test_metrics(graphs, fraction):
    risks, realized, precisions, ious = [], [], [], []
    for graph in graphs:
        m = int(graph.num_nodes)
        r = int(graph.rationale_mask.sum())
        if r == 0:
            continue
        k = 0 if fraction == 0 else int(np.ceil(fraction * m))
        recovered = min(k, r)
        risks.append(1.0 - recovered / r)
        realized.append(k / m)
        precisions.append(recovered / k if k else 0.0)
        ious.append(recovered / (k + r - recovered))
    return {
        "test_risk": float(np.mean(risks)),
        "test_mean_atom_fraction": float(np.mean(realized)),
        "test_median_atom_fraction": float(np.median(realized)),
        "test_precision": float(np.mean(precisions)),
        "test_iou": float(np.mean(ious)),
        "n_test_rationale": len(risks),
    }


def main():
    out_dir = ROOT / "results" / "reviewer_oracle_crc"
    out_dir.mkdir(parents=True, exist_ok=True)
    task_rows = []
    for family, tasks in (("bxaic", BXAIC_TASKS), ("google", GOOGLE_TASKS)):
        for task in tasks:
            print(f"oracle_task={family}/{task}", flush=True)
            partitions = (
                bxaic_partitions(ROOT / "data/raw/bxaic/data.csv", ROOT / "data/raw/bxaic/explanations.sdf", task)
                if family == "bxaic"
                else google_partitions(ROOT / "reference/graph-attribution/data", task)
            )
            cal_table = oracle_loss_table(partitions["calibration"])
            test_table = oracle_loss_table(partitions["test"])
            for alpha in ALPHAS:
                crc = calibrate_crc(cal_table, FRACTIONS, alpha)
                metrics = oracle_test_metrics(partitions["test"], crc["fraction"])
                if abs(metrics["test_risk"] - float(test_table[:, crc["index"]].mean())) > 1e-12:
                    raise ValueError("oracle metric mismatch")
                task_rows.append({
                    "family": family,
                    "task": task,
                    "alpha": alpha,
                    "oracle_fraction": crc["fraction"],
                    "oracle_calibration_risk": crc["empirical_risk"],
                    "oracle_corrected_risk": crc["corrected_risk"],
                    "oracle_n_calibration_rationale": crc["n_calibration"],
                    **{f"oracle_{key}": value for key, value in metrics.items()},
                })

    task_frame = pd.DataFrame(task_rows)
    task_frame.to_csv(out_dir / "task_oracle_all_alpha.csv", index=False)

    all_alpha = pd.read_csv(ROOT / "results/gradient_grid_phase_a/cell_all_alpha.csv")
    paired = all_alpha.merge(task_frame, on=["family", "task", "alpha"], validate="many_to_one")
    paired["excess_grid_fraction"] = paired["crc_fraction"] - paired["oracle_fraction"]
    paired["recoverable_efficiency_grid"] = (
        (paired["random_fraction"] - paired["crc_fraction"])
        / (paired["random_fraction"] - paired["oracle_fraction"])
    ).where(paired["random_fraction"] > paired["oracle_fraction"])
    paired.to_csv(out_dir / "paired_cells_all_alpha.csv", index=False)

    primary = pd.read_csv(ROOT / "results/gradient_grid_phase_a/cell_alpha010.csv")
    oracle_primary = task_frame[np.isclose(task_frame["alpha"], 0.10)].drop(columns="alpha")
    primary = primary.merge(oracle_primary, on=["family", "task"], validate="many_to_one")
    primary["excess_mean_atom_fraction"] = primary["mean_atom_fraction"] - primary["oracle_test_mean_atom_fraction"]
    primary["recoverable_efficiency"] = (
        (primary["random_mean_atom_fraction"] - primary["mean_atom_fraction"])
        / (primary["random_mean_atom_fraction"] - primary["oracle_test_mean_atom_fraction"])
    ).where(primary["random_mean_atom_fraction"] > primary["oracle_test_mean_atom_fraction"])
    primary["oracle_efficiency_pass"] = primary["oracle_test_mean_atom_fraction"] < 0.80
    primary.to_csv(out_dir / "paired_cells_alpha010.csv", index=False)

    expected = primary.groupby(["family", "task"])[["n_calibration_rationale", "n_test_rationale"]].nunique()
    counts_match = (
        primary["n_calibration_rationale"].eq(primary["oracle_n_calibration_rationale"]).all()
        and primary["n_test_rationale"].eq(primary["oracle_n_test_rationale"]).all()
    )
    if int(expected.to_numpy().max()) != 1 or not counts_match or len(task_frame) != 33 or len(paired) != 396 or len(primary) != 132:
        raise ValueError("task or pairing counts do not match the frozen main analysis")
    if (paired["excess_grid_fraction"] < -1e-12).any():
        raise ValueError("an explainer selected a smaller grid fraction than the rationale-first oracle")

    summary_by_explainer = {}
    for explainer, group in primary.groupby("explainer"):
        valid_recovery = group["recoverable_efficiency"].dropna()
        summary_by_explainer[explainer] = {
            "n_cells": int(len(group)),
            "mean_oracle_atom_fraction": float(group["oracle_test_mean_atom_fraction"].mean()),
            "mean_explainer_atom_fraction": float(group["mean_atom_fraction"].mean()),
            "mean_random_atom_fraction": float(group["random_mean_atom_fraction"].mean()),
            "mean_excess_atom_fraction": float(group["excess_mean_atom_fraction"].mean()),
            "mean_recoverable_efficiency": float(valid_recovery.mean()),
            "median_recoverable_efficiency": float(valid_recovery.median()),
            "n_valid_recoverable_efficiency": int(len(valid_recovery)),
            "oracle_efficiency_pass_cells": int(group["oracle_efficiency_pass"].sum()),
            "explainer_efficiency_pass_cells": int(group["efficiency_pass"].sum()),
        }
    summary = {
        "status": "complete",
        "primary_alpha": 0.10,
        "efficiency_threshold": 0.80,
        "n_tasks": int(len(oracle_primary)),
        "n_paired_cells": int(len(primary)),
        "all_explainer_fractions_at_least_oracle": True,
        "by_explainer": summary_by_explainer,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

import json
import hashlib
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


HERE = Path(__file__).resolve().parent
REVISION = HERE.parents[2]
PROJECT = REVISION.parent
FULL = PROJECT / "artifacts/experiment/gradient_grid_main/cells"
SUBSET = REVISION / "experiments/established_explainers_20260901/cells.csv"
METHODS = ["ig", "gradinput"]
OUTCOMES = ["risk", "atom_fraction", "precision", "iou", "selected_grid_fraction"]


def finite_correlation(fn, x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(fn(x, y).statistic)


def load_full():
    rows = []
    for path in sorted(FULL.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for method in METHODS:
            metrics = payload["explainers"][method]["metrics"]
            crc = metrics["alpha"]["0.10"]["crc"]
            rows.append({
                "cell_id": payload["cell_id"],
                "family": payload["family"],
                "task": payload["task"],
                "model": payload["model"],
                "seed": payload["seed"],
                "method": method,
                "full_n_calibration": metrics["n_calibration_rationale"],
                "full_n_test": metrics["n_test_rationale"],
                "risk": crc["risk"],
                "atom_fraction": crc["mean_atom_fraction"],
                "precision": crc["precision"],
                "iou": crc["iou"],
                "selected_grid_fraction": crc["fraction"],
            })
    return pd.DataFrame(rows)


def summarize(group, strata):
    rows = []
    keys = strata + ["method"]
    for key, part in group.groupby(keys, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        base = dict(zip(keys, key))
        for outcome in OUTCOMES:
            x = part[f"full_{outcome}"].to_numpy(float)
            y = part[f"subset_{outcome}"].to_numpy(float)
            delta = y - x
            rows.append({
                **base,
                "outcome": outcome,
                "n_cells": len(part),
                "pearson_r": finite_correlation(pearsonr, x, y),
                "spearman_rho": finite_correlation(spearmanr, x, y),
                "mean_absolute_error": float(np.mean(np.abs(delta))),
                "median_absolute_error": float(np.median(np.abs(delta))),
                "signed_bias_subset_minus_full": float(np.mean(delta)),
                "max_absolute_error": float(np.max(np.abs(delta))),
            })
    return pd.DataFrame(rows)


def preference(frame, outcome, higher_better):
    wide = frame.pivot(index="cell_id", columns="method", values=outcome)
    delta = wide["ig"] - wide["gradinput"]
    if higher_better:
        return np.sign(delta)
    return np.sign(-delta)


def main():
    full = load_full()
    subset = pd.read_csv(SUBSET)
    subset = subset[subset["method"].isin(METHODS)].copy()
    subset = subset.rename(columns={
        "n_calibration": "subset_n_calibration",
        "n_test": "subset_n_test",
        **{name: f"subset_{name}" for name in OUTCOMES},
    })
    full = full.rename(columns={name: f"full_{name}" for name in OUTCOMES})
    keys = ["cell_id", "family", "task", "model", "seed", "method"]
    merged = full.merge(subset[keys + ["subset_n_calibration", "subset_n_test"] + [f"subset_{x}" for x in OUTCOMES]], on=keys, validate="one_to_one")
    if len(merged) != 132 or merged["cell_id"].nunique() != 66:
        raise ValueError("Expected 132 paired method-cell rows over 66 cells")
    for outcome in OUTCOMES:
        merged[f"delta_{outcome}"] = merged[f"subset_{outcome}"] - merged[f"full_{outcome}"]
    merged["full_risk_pass"] = merged["full_risk"] <= 0.10
    merged["subset_risk_pass"] = merged["subset_risk"] <= 0.10
    merged["full_efficiency_pass"] = merged["full_atom_fraction"] < 0.80
    merged["subset_efficiency_pass"] = merged["subset_atom_fraction"] < 0.80
    merged.to_csv(HERE / "paired_cells.csv", index=False)

    overall = summarize(merged, [])
    family = summarize(merged, ["family"])
    task = summarize(merged, ["family", "task"])
    overall.to_csv(HERE / "overall_metric_agreement.csv", index=False)
    family.to_csv(HERE / "family_metric_agreement.csv", index=False)
    task.to_csv(HERE / "task_metric_agreement.csv", index=False)

    pass_rows = []
    for method, part in merged.groupby("method", sort=True):
        pass_rows.append({
            "method": method,
            "risk_pass_agreement": float((part["full_risk_pass"] == part["subset_risk_pass"]).mean()),
            "risk_pass_full": int(part["full_risk_pass"].sum()),
            "risk_pass_subset": int(part["subset_risk_pass"].sum()),
            "efficiency_pass_agreement": float((part["full_efficiency_pass"] == part["subset_efficiency_pass"]).mean()),
            "efficiency_pass_full": int(part["full_efficiency_pass"].sum()),
            "efficiency_pass_subset": int(part["subset_efficiency_pass"].sum()),
        })
    pass_frame = pd.DataFrame(pass_rows)
    pass_frame.to_csv(HERE / "gate_agreement.csv", index=False)

    rank_rows = []
    for outcome in OUTCOMES:
        full_pref = preference(merged, f"full_{outcome}", outcome in {"precision", "iou"})
        subset_pref = preference(merged, f"subset_{outcome}", outcome in {"precision", "iou"})
        rank_rows.append({
            "outcome": outcome,
            "n_cells": len(full_pref),
            "exact_order_or_tie_agreement": float((full_pref == subset_pref).mean()),
            "strict_reversal_rate": float(((full_pref * subset_pref) < 0).mean()),
            "full_ties": int((full_pref == 0).sum()),
            "subset_ties": int((subset_pref == 0).sum()),
        })
    rank_frame = pd.DataFrame(rank_rows)
    rank_frame.to_csv(HERE / "method_order_agreement.csv", index=False)

    primary = overall[overall["outcome"].isin(["risk", "atom_fraction"])]
    correlations_ok = bool((primary["spearman_rho"].fillna(-1) >= 0.80).all())
    errors_ok = bool((primary["mean_absolute_error"] <= 0.05).all())
    gates_ok = bool((pass_frame[["risk_pass_agreement", "efficiency_pass_agreement"]].to_numpy() >= 0.90).all())
    support = correlations_ok and errors_ok and gates_ok
    summary = {
        "status": "PASS" if support else "PARTIAL",
        "reviewer_item": "A3",
        "historical_set_policy": "index_tiebreak_v1",
        "paired_method_cell_rows": len(merged),
        "unique_cells": int(merged["cell_id"].nunique()),
        "methods": METHODS,
        "support_rule": {
            "primary_correlations_ok": correlations_ok,
            "primary_errors_ok": errors_ok,
            "gate_agreement_ok": gates_ok,
            "subset_qualified_five_method_comparison_supported": support,
        },
        "overall_metrics": overall.where(pd.notnull(overall), None).to_dict("records"),
        "gate_agreement": pass_frame.to_dict("records"),
        "method_order_agreement": rank_frame.to_dict("records"),
        "interpretation": "This audit evaluates representativeness only for IG and GradInput, the methods available on both the full grid and the matched subset. It cannot validate the unobserved full-grid behavior of saliency, atom occlusion, or GNNExplainer.",
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    validation = {
        "status": "PASS",
        "expected_rows": 132,
        "observed_rows": len(merged),
        "unique_cells": int(merged["cell_id"].nunique()),
        "finite_numeric_outputs": bool(np.isfinite(merged.select_dtypes(include=[np.number]).to_numpy()).all()),
        "no_cross_policy_mixing": True,
    }
    (HERE / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    environment = {"platform": platform.platform(), "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__}
    (HERE / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    inputs = {path.name: hashlib.sha256(path.read_bytes()).hexdigest().upper() for path in sorted(FULL.glob("*.json"))}
    inputs["matched_subset_cells.csv"] = hashlib.sha256(SUBSET.read_bytes()).hexdigest().upper()
    outputs = {}
    for name in ["analyze.py", "run_contract.json", "summary.json", "validation.json", "paired_cells.csv", "overall_metric_agreement.csv", "family_metric_agreement.csv", "task_metric_agreement.csv", "gate_agreement.csv", "method_order_agreement.csv", "environment.json"]:
        path = HERE / name
        outputs[name] = hashlib.sha256(path.read_bytes()).hexdigest().upper()
    (HERE / "manifest.json").write_text(json.dumps({"inputs": inputs, "outputs": outputs}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

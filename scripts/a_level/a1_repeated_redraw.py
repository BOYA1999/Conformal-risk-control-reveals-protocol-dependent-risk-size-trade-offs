import csv
import gzip
import hashlib
import json
import platform
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
REVISION = HERE.parents[2]
PROJECT = REVISION.parent
CACHE = Path(r"<tie-inclusive-score-cache>")
FULL = PROJECT / "artifacts/experiment/gradient_grid_main/cells"
METHODS = ["gradinput", "ig", "saliency", "atom_occlusion", "gnnexplainer"]
FRACTIONS = np.round(np.linspace(0, 1, 101), 2)
CALIBRATION_FRACTIONS = [0.125, 0.25, 0.5]
ALPHA = 0.10
REPEATS = 500
BASE_SEED = 2026090400


def unpack(values, offsets):
    return [values[a:b] for a, b in zip(offsets[:-1], offsets[1:])]


def tables(score_rows, rationale_rows):
    losses = np.empty((len(score_rows), len(FRACTIONS)))
    sizes = np.empty_like(losses)
    for i, (scores, truth_bits) in enumerate(zip(score_rows, rationale_rows)):
        truth = np.flatnonzero(truth_bits)
        order = np.argsort(-scores, kind="stable")
        ordered = scores[order]
        recovered = np.cumsum(np.isin(order, truth))
        nominal = np.where(FRACTIONS == 0, 0, np.ceil(FRACTIONS * len(scores))).astype(int)
        counts = np.zeros(len(FRACTIONS), dtype=int)
        positive = nominal > 0
        thresholds = ordered[nominal[positive] - 1]
        counts[positive] = np.searchsorted(-ordered, -thresholds, side="right")
        found = np.zeros(len(FRACTIONS), dtype=int)
        found[positive] = recovered[counts[positive] - 1]
        losses[i] = 1 - found / len(truth)
        sizes[i] = counts / len(scores)
    return losses, sizes


def select(calibration_losses, corrected):
    empirical = calibration_losses.mean(axis=0)
    criterion = (len(calibration_losses) * empirical + 1) / (len(calibration_losses) + 1) if corrected else empirical
    feasible = np.flatnonzero(criterion <= ALPHA)
    return int(feasible[0]) if len(feasible) else len(FRACTIONS) - 1, not len(feasible)


def wilson(successes, total):
    z = 1.959963984540054
    p = successes / total
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return float(center - half), float(center + half)


def covariates():
    rows = []
    for path in FULL.glob("*.json"):
        d = json.loads(path.read_text(encoding="utf-8"))
        m = d["explainers"]["ig"]["metrics"]
        nonnull = m["n_calibration_rationale"] + m["n_test_rationale"]
        total = nonnull + m["n_calibration_null"] + m["n_test_null"]
        rows.append({
            "cell_id": d["cell_id"], "family": d["family"], "task": d["task"],
            "rationale_prevalence": nonnull / total,
            "test_auroc": d["predictor"]["test"]["auroc"],
        })
    return pd.DataFrame(rows)


def main():
    cov = covariates()
    metadata = cov.set_index("cell_id").to_dict("index")
    aggregates = defaultdict(list)
    output = HERE / "per_repeat_results.csv.gz"
    fields = [
        "cell_id", "family", "task", "method", "repeat", "split_seed", "pool_n",
        "calibration_fraction", "n_calibration", "n_test", "selector", "test_risk",
        "risk_above_alpha", "nominal_fraction", "realized_atom_fraction", "fallback",
    ]
    with gzip.open(output, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for path in sorted(CACHE.glob("*.npz")):
            cell_id = path.stem
            meta = metadata[cell_id]
            z = np.load(path, allow_pickle=False)
            cal_offsets = z["calibration__offsets"]
            test_offsets = z["test__offsets"]
            offsets = np.concatenate([cal_offsets, test_offsets[1:] + cal_offsets[-1]])
            rationales = unpack(
                np.concatenate([z["calibration__rationale_values"], z["test__rationale_values"]]), offsets
            )
            method_tables = {}
            for method in METHODS:
                score_values = np.concatenate([
                    z[f"{method}__calibration__score_values"], z[f"{method}__test__score_values"]
                ])
                method_tables[method] = tables(unpack(score_values, offsets), rationales)
            pool_n = len(rationales)
            cell_offset = int(hashlib.sha256(cell_id.encode()).hexdigest()[:8], 16) % 1000000
            for repeat in range(REPEATS):
                split_seed = BASE_SEED + cell_offset + repeat
                order = np.random.default_rng(split_seed).permutation(pool_n)
                for cal_fraction in CALIBRATION_FRACTIONS:
                    n_cal = max(10, int(round(pool_n * cal_fraction)))
                    cal_idx, test_idx = order[:n_cal], order[n_cal:]
                    for method in METHODS:
                        losses, sizes = method_tables[method]
                        for selector, corrected in (("corrected", True), ("naive", False)):
                            index, fallback = select(losses[cal_idx], corrected)
                            row = {
                                "cell_id": cell_id, "family": meta["family"], "task": meta["task"],
                                "method": method, "repeat": repeat, "split_seed": split_seed,
                                "pool_n": pool_n, "calibration_fraction": cal_fraction,
                                "n_calibration": n_cal, "n_test": len(test_idx), "selector": selector,
                                "test_risk": float(losses[test_idx, index].mean()),
                                "risk_above_alpha": bool(losses[test_idx, index].mean() > ALPHA),
                                "nominal_fraction": float(FRACTIONS[index]),
                                "realized_atom_fraction": float(sizes[test_idx, index].mean()),
                                "fallback": fallback,
                            }
                            writer.writerow(row)
                            key = (cell_id, meta["family"], meta["task"], method, cal_fraction, n_cal, selector)
                            aggregates[key].append((row["test_risk"], row["risk_above_alpha"], row["nominal_fraction"], row["realized_atom_fraction"], fallback))

    summaries = []
    for key, values in aggregates.items():
        cell_id, family, task, method, cal_fraction, n_cal, selector = key
        array = np.asarray(values, dtype=float)
        violations = int(array[:, 1].sum())
        low, high = wilson(violations, len(array))
        summaries.append({
            "cell_id": cell_id, "family": family, "task": task, "method": method,
            "calibration_fraction": cal_fraction, "n_calibration": n_cal, "selector": selector,
            "repeats": len(array), "mean_test_risk": array[:, 0].mean(),
            "sd_test_risk": array[:, 0].std(ddof=1), "violation_count": violations,
            "violation_frequency": violations / len(array), "violation_wilson_low": low,
            "violation_wilson_high": high, "mean_nominal_fraction": array[:, 2].mean(),
            "mean_realized_atom_fraction": array[:, 3].mean(), "fallback_count": int(array[:, 4].sum()),
        })
    cell = pd.DataFrame(summaries)
    cell = cell.merge(cov, on=["cell_id", "family", "task"], validate="many_to_one")
    cell.to_csv(HERE / "cell_method_size_summary.csv", index=False)

    paired = cell.pivot(index=["cell_id", "family", "task", "method", "calibration_fraction", "n_calibration"], columns="selector", values=["mean_test_risk", "violation_frequency", "mean_nominal_fraction", "mean_realized_atom_fraction"])
    paired.columns = [f"{metric}_{selector}" for metric, selector in paired.columns]
    paired = paired.reset_index()
    for metric in ["mean_test_risk", "violation_frequency", "mean_nominal_fraction", "mean_realized_atom_fraction"]:
        paired[f"corrected_minus_naive_{metric}"] = paired[f"{metric}_corrected"] - paired[f"{metric}_naive"]
    paired.to_csv(HERE / "paired_corrected_vs_naive.csv", index=False)

    macro = cell.groupby(["calibration_fraction", "selector", "method"])[["mean_test_risk", "violation_frequency", "mean_nominal_fraction", "mean_realized_atom_fraction"]].mean().reset_index()
    macro.to_csv(HERE / "macro_summary.csv", index=False)
    task = cell.groupby(["family", "task", "calibration_fraction", "selector"])[["mean_test_risk", "violation_frequency", "mean_realized_atom_fraction", "rationale_prevalence", "test_auroc"]].mean().reset_index()
    association_rows = []
    for (cal_fraction, selector), part in task.groupby(["calibration_fraction", "selector"]):
        for covariate in ["rationale_prevalence", "test_auroc"]:
            for outcome in ["mean_test_risk", "violation_frequency", "mean_realized_atom_fraction"]:
                result = spearmanr(part[covariate], part[outcome])
                association_rows.append({
                    "calibration_fraction": cal_fraction, "selector": selector,
                    "covariate": covariate, "outcome": outcome, "n_tasks": len(part),
                    "spearman_rho": float(result.statistic), "descriptive_p_value": float(result.pvalue),
                })
    pd.DataFrame(association_rows).to_csv(HERE / "task_covariate_associations.csv", index=False)
    task.to_csv(HERE / "task_macro_summary.csv", index=False)

    corrected = cell[cell["selector"] == "corrected"]
    naive = cell[cell["selector"] == "naive"]
    summary = {
        "status": "PASS",
        "reviewer_item": "A1/A7-statistical",
        "set_policy": "include_all_exact_ties_v2",
        "cells": int(cell["cell_id"].nunique()), "methods": METHODS,
        "repeats_per_cell_size": REPEATS,
        "per_repeat_rows": int(sum(len(v) for v in aggregates.values())),
        "calibration_fractions": CALIBRATION_FRACTIONS,
        "corrected_macro": corrected.groupby("calibration_fraction")[["mean_test_risk", "violation_frequency", "mean_realized_atom_fraction"]].mean().reset_index().to_dict("records"),
        "naive_macro": naive.groupby("calibration_fraction")[["mean_test_risk", "violation_frequency", "mean_realized_atom_fraction"]].mean().reset_index().to_dict("records"),
        "paired_corrected_minus_naive": paired.groupby("calibration_fraction")[[c for c in paired if c.startswith("corrected_minus_naive")]].mean().reset_index().to_dict("records"),
        "fallback_count": int(cell["fallback_count"].sum()),
        "interpretation": "Finite-pool repeated mutually exclusive redraw diagnostic. The controlled estimand is expected nonempty-rationale loss, not per-repeat test-set violation frequency; no new-population guarantee is inferred.",
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    validation = {
        "status": "PASS", "cache_files": len(list(CACHE.glob("*.npz"))),
        "expected_cell_method_size_selector_rows": 66 * 5 * 3 * 2,
        "observed_cell_method_size_selector_rows": len(cell),
        "expected_per_repeat_rows": 66 * 5 * 3 * 2 * REPEATS,
        "observed_per_repeat_rows": summary["per_repeat_rows"],
        "finite_summary_values": bool(np.isfinite(cell.select_dtypes(include=[np.number]).to_numpy()).all()),
        "all_test_complements_nonempty": bool((cell["n_calibration"] < cell["cell_id"].map({p.stem: (len(np.load(p, allow_pickle=False)["calibration__offsets"]) - 1) + (len(np.load(p, allow_pickle=False)["test__offsets"]) - 1) for p in CACHE.glob("*.npz")})).all()),
    }
    (HERE / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    environment = {
        "timestamp_local": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "platform": platform.platform(), "python": sys.version,
        "numpy": np.__version__, "pandas": pd.__version__,
    }
    (HERE / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    inputs = {}
    for path in sorted(CACHE.glob("*.npz")):
        inputs[path.name] = hashlib.sha256(path.read_bytes()).hexdigest().upper()
    outputs = {}
    for name in ["run.py", "run_contract.json", "summary.json", "validation.json", "cell_method_size_summary.csv", "paired_corrected_vs_naive.csv", "macro_summary.csv", "task_macro_summary.csv", "task_covariate_associations.csv", "per_repeat_results.csv.gz", "environment.json"]:
        path = HERE / name
        outputs[name] = hashlib.sha256(path.read_bytes()).hexdigest().upper()
    (HERE / "manifest.json").write_text(json.dumps({"inputs": inputs, "outputs": outputs}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

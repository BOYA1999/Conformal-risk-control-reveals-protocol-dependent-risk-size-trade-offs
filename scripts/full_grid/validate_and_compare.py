import os
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
PACKAGE = CODE.parents[1]
WORK = Path(os.environ.get('MOLXAI_WORK_ROOT', '../molxai-work')).resolve()
HERE = WORK / 'full_grid'
SCIENCE = WORK
CONTRACT = PACKAGE / 'contracts/full_grid/run_contract.json'
ORIGINAL = WORK / 'artifacts/experiment/gradient_grid_main/cells'
SUBSET = WORK / 'experiments/established_subset/cells'
GRID = np.linspace(0, 1, 101)
OUTCOMES = ["risk", "mean_atom_fraction", "precision", "iou"]
METHODS = ["atom_occlusion", "saliency"]
POLICIES = ["index_tiebreak_v1", "include_all_exact_ties_v2"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def split_cache(z, split, method):
    boundaries = z[f"{split}__offsets"]
    scores = [z[f"{split}__{method}"][a:b] for a, b in zip(boundaries[:-1], boundaries[1:])]
    masks = [z[f"{split}__rationale"][a:b].astype(bool) for a, b in zip(boundaries[:-1], boundaries[1:])]
    return scores, masks


def loss_matrix(scores, masks, inclusive):
    table = []
    for score, mask in zip(scores, masks):
        order = np.lexsort((np.arange(len(score)), -score.astype(float)))
        truth_ranks = np.flatnonzero(mask[order])
        counts = np.ceil(GRID * len(score)).astype(int)
        if inclusive:
            ordered = score[order]
            thresholds = ordered[counts[1:] - 1]
            counts[1:] = (ordered[:, None] >= thresholds[None, :]).sum(axis=0)
        hits = np.searchsorted(truth_ranks, counts, side="left")
        table.append(1.0 - hits / len(truth_ranks))
    return np.asarray(table)


def direct_metrics(scores, masks, fraction, inclusive):
    values = {key: [] for key in OUTCOMES + ["tie_inflation"]}
    for score, truth in zip(scores, masks):
        n = len(score)
        k = int(np.ceil(fraction * n))
        if k == 0:
            chosen = np.zeros(n, dtype=bool)
        elif inclusive:
            cutoff = np.partition(score, n - k)[n - k]
            chosen = score >= cutoff
        else:
            indices = np.lexsort((np.arange(n), -score.astype(float)))[:k]
            chosen = np.zeros(n, dtype=bool)
            chosen[indices] = True
        hits = np.count_nonzero(chosen & truth)
        selected, actual = np.count_nonzero(chosen), np.count_nonzero(truth)
        values["risk"].append(1 - hits / actual)
        values["mean_atom_fraction"].append(selected / n)
        values["precision"].append(hits / selected if selected else 0)
        values["iou"].append(hits / (selected + actual - hits))
        values["tie_inflation"].append((selected - k) / n)
    return {**{k: float(np.mean(v)) for k, v in values.items()}, "median_atom_fraction": float(np.median(values["mean_atom_fraction"]))}


def replay(cal_scores, cal_masks, test_scores, test_masks, policy, alphas):
    inclusive = policy == POLICIES[1]
    losses = loss_matrix(cal_scores, cal_masks, inclusive)
    assert np.isfinite(losses).all() and (np.diff(losses, axis=1) <= 1e-12).all()
    n = len(cal_scores)
    corrected = (losses.sum(axis=0) + 1) / (n + 1)
    result = {}
    for alpha in alphas:
        found = np.flatnonzero(corrected <= alpha)
        assert len(found)
        selected = int(found[0])
        result[f"{alpha:.2f}"] = {"fraction": float(GRID[selected]), "index": selected, "empirical_risk": float(losses[:, selected].mean()), "corrected_risk": float(corrected[selected]), "n_calibration": n, "alpha": alpha, **direct_metrics(test_scores, test_masks, GRID[selected], inclusive)}
    return result


def paired_interval(frame, metric, left, right, name):
    merged = left[["cell_id", "family", "task", metric]].merge(right[["cell_id", metric]], on="cell_id", validate="one_to_one", suffixes=("_left", "_right"))
    merged["delta"] = merged[f"{metric}_left"] - merged[f"{metric}_right"]
    task_delta = merged.groupby(["family", "task"])["delta"].mean().to_numpy()
    rng = np.random.default_rng(20260912)
    samples = task_delta[rng.integers(0, len(task_delta), (2000, len(task_delta)))].mean(1)
    row = {"comparison_left_minus_right": name, "metric": metric, "cells": len(merged), "observed_tasks": len(task_delta), "mean_difference": float(task_delta.mean()), "q025": float(np.quantile(samples, .025)), "q975": float(np.quantile(samples, .975)), "cell_min": float(merged.delta.min()), "cell_max": float(merged.delta.max()), "cell_positive": int((merged.delta > 0).sum()), "cell_negative": int((merged.delta < 0).sum()), "cell_tie": int((merged.delta == 0).sum())}
    frame.append(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    paths = sorted((HERE / "cells").glob("*.json"))
    if not args.partial:
        assert len(paths) == 66
    checks, full_rows, subset_rows, fixed_rows, fresh_subset_rows, tie_rows = [], [], [], [], [], []
    maximum_delta = 0.0
    for path in paths:
        data = json.loads(path.read_text())
        cell_id = data["cell_id"]
        assert data["status"] == "complete" and cell_id == path.stem
        ident = {key: data[key] for key in ["cell_id", "family", "task", "model", "seed"]}
        assert data["contract_sha256"] == sha(CONTRACT)
        assert data["runner_sha256"] == sha(CODE / "run_full_grid.py")
        cache_path = HERE / "scores" / f"{cell_id}.npz"
        assert sha(cache_path) == data["score_cache_sha256"]
        historical = json.loads((ORIGINAL / path.name).read_text())
        subset = json.loads((SUBSET / path.name).read_text())
        random_fraction = historical["explainers"]["ig"]["metrics"]["alpha"]["0.10"]["random_crc"]["mean_atom_fraction"]
        assert abs(random_fraction - historical["explainers"]["gradinput"]["metrics"]["alpha"]["0.10"]["random_crc"]["mean_atom_fraction"]) < 1e-12
        assert data["checkpoint_sha256"] == subset["checkpoint_sha256"]
        with np.load(cache_path, allow_pickle=False) as z:
            assert not set(z["calibration__all_source_ids"]) & set(z["test__all_source_ids"])
            for split in ["calibration", "test"]:
                ids, full_ids = z[f"{split}__source_ids"], z[f"{split}__all_source_ids"]
                offsets = z[f"{split}__offsets"]
                assert len(full_ids) == data["counts"][split]["all_count"]
                assert len(ids) == data["counts"][split]["eligible_count"] == historical["explainers"]["ig"]["metrics"][f"n_{split}_rationale"]
                assert len(set(ids)) == len(ids) and set(ids) <= set(full_ids)
                assert len(set(full_ids)) == len(full_ids)
                assert hashlib.sha256(np.asarray(full_ids, dtype="<i8").tobytes()).hexdigest().upper() == data["counts"][split]["all_id_sha256"]
                assert len(offsets) == len(ids) + 1 and (np.diff(offsets) > 0).all() and offsets[0] == 0
                assert offsets[-1] == len(z[f"{split}__rationale"])
                assert np.isin(z[f"{split}__rationale"], [0, 1]).all()
                assert len(z[f"{split}__labels"]) == len(ids) and np.isin(z[f"{split}__labels"], [0, 1]).all()
                assert all(len(z[f"{split}__{method}"]) == offsets[-1] for method in METHODS)
            for method in METHODS:
                cal_scores, cal_masks = split_cache(z, "calibration", method)
                test_scores, test_masks = split_cache(z, "test", method)
                for values, masks in [(cal_scores, cal_masks), (test_scores, test_masks)]:
                    assert all(np.isfinite(v).all() and len(v) == len(m) and m.any() for v, m in zip(values, masks))
                for policy in POLICIES:
                    observed = replay(cal_scores, cal_masks, test_scores, test_masks, policy, [.05, .1, .2])
                    for alpha, metrics in observed.items():
                        reported = data["methods"][method][policy]["alpha"][alpha]["crc"]
                        delta = max(abs(float(value) - float(reported[key])) for key, value in metrics.items())
                        maximum_delta = max(maximum_delta, delta)
                        assert delta < 1e-10, (cell_id, method, policy, alpha, delta)
                    metrics = observed["0.10"]
                    (full_rows if policy == POLICIES[0] else tie_rows).append({**ident, "method": method, "origin": "fresh_full_partition", **metrics, **({"random_atom_fraction": random_fraction, "saving_vs_random": random_fraction - metrics["mean_atom_fraction"]} if policy == POLICIES[0] else {})})
                for fraction in [.2, .5]:
                    actual = direct_metrics(test_scores, test_masks, fraction, False)
                    reported = data["methods"][method][POLICIES[0]]["fixed"][f"{fraction:.2f}"]
                    assert all(abs(actual[k] - reported[k]) < 1e-10 for k in actual)
                    fixed_rows.append({**ident, "method": method, "nominal_fraction": fraction, **actual})
                fresh = []
                for split, scores, masks in [("calibration", cal_scores, cal_masks), ("test", test_scores, test_masks)]:
                    by_id = {int(value): index for index, value in enumerate(z[f"{split}__source_ids"])}
                    selected = [by_id[value] for value in subset["sample"][f"{split}_source_indices"]]
                    fresh.extend([[scores[i] for i in selected], [masks[i] for i in selected]])
                fresh_metrics = replay(*fresh, POLICIES[0], [.1])["0.10"]
                fresh_subset_rows.append({**ident, "method": method, **fresh_metrics})
                subset_rows.append({**ident, "method": method, **subset["methods"][method]["metrics"]["alpha"]["0.10"]["crc"]})
        for method in ["ig", "gradinput"]:
            metrics = historical["explainers"][method]["metrics"]["alpha"]["0.10"]["crc"]
            full_rows.append({**ident, "method": method, "origin": "historical_full_partition", **metrics, "random_atom_fraction": random_fraction, "saving_vs_random": random_fraction - metrics["mean_atom_fraction"]})
            subset_rows.append({**ident, "method": method, **subset["methods"][method]["metrics"]["alpha"]["0.10"]["crc"]})
            for fraction in [.2, .5]:
                fixed_rows.append({**ident, "method": method, "nominal_fraction": fraction, **historical["explainers"][method]["metrics"]["fixed"][f"{fraction:.2f}"]})
        checks.append({"cell_id": cell_id, "status": "PASS", "n_calibration": len(cal_scores), "n_test": len(test_scores)})
        print(f"validated {len(checks)}/{len(paths)} {cell_id}", flush=True)
    prefix = "partial_" if args.partial else ""
    full, subset, fresh, ties, fixed = map(pd.DataFrame, [full_rows, subset_rows, fresh_subset_rows, tie_rows, fixed_rows])
    for name, frame in [("comparison_cells", full), ("historical_subset_cells", subset), ("same_run_subset_cells", fresh), ("tie_sensitivity_cells", ties), ("fixed_budget_cells", fixed)]:
        frame.to_csv(HERE / f"{prefix}{name}.csv", index=False)
    summary, family = [], []
    for method, group in full.groupby("method"):
        task = group.groupby(["family", "task"])[OUTCOMES + ["random_atom_fraction", "saving_vs_random"]].mean()
        summary.append({"method": method, "cells": len(group), **{f"macro_task_{key}": float(value) for key, value in task.mean().items()}, "risk_pass_cells": int((group.risk <= .1).sum()), "efficiency_pass_cells": int((group.mean_atom_fraction < .8).sum()), "joint_pass_cells": int(((group.risk <= .1) & (group.mean_atom_fraction < .8)).sum())})
        for label, part in task.groupby(level="family"):
            family.append({"method": method, "family": label, **{f"macro_task_{key}": float(value) for key, value in part.mean().items()}})
    pd.DataFrame(summary).to_csv(HERE / f"{prefix}method_summary.csv", index=False)
    pd.DataFrame(family).to_csv(HERE / f"{prefix}family_summary.csv", index=False)
    paired = []
    for method in METHODS:
        for metric in OUTCOMES:
            base = full[full.method == method]
            paired_interval(paired, metric, base, subset[subset.method == method], f"full_{method}_minus_historical_subset")
            paired_interval(paired, metric, base, fresh[fresh.method == method], f"full_{method}_minus_same_run_subset")
            paired_interval(paired, metric, fresh[fresh.method == method], subset[subset.method == method], f"same_run_subset_{method}_minus_historical_subset")
            paired_interval(paired, metric, ties[ties.method == method], base, f"tie_inclusive_minus_index_{method}")
    for left, right in [("atom_occlusion", "ig"), ("saliency", "ig"), ("atom_occlusion", "saliency")]:
        for metric in OUTCOMES:
            paired_interval(paired, metric, full[full.method == left], full[full.method == right], f"full_{left}_minus_full_{right}")
    pd.DataFrame(paired).to_csv(HERE / f"{prefix}paired_comparisons.csv", index=False)
    reversals = []
    for left, right in itertools.combinations(["ig", "gradinput"] + METHODS, 2):
        for metric in ["risk", "mean_atom_fraction", "iou"]:
            pf = full.pivot(index="cell_id", columns="method", values=metric)
            ps = subset.pivot(index="cell_id", columns="method", values=metric)
            first, second = pf[left] - pf[right], ps[left] - ps[right]
            reversals.append({"left": left, "right": right, "metric": metric, "comparison": "fresh_full_added_methods_and_historical_full_gradients_vs_historical_subset", "cells": len(first), "strict_order_reversals": int(((first * second) < 0).sum()), "full_ties": int((first == 0).sum()), "subset_ties": int((second == 0).sum())})
    pd.DataFrame(reversals).to_csv(HERE / f"{prefix}order_reversals.csv", index=False)
    protocol_rows = []
    left, right = METHODS
    for metric in ["risk", "mean_atom_fraction", "iou"]:
        first = full.pivot(index="cell_id", columns="method", values=metric)
        second = ties.pivot(index="cell_id", columns="method", values=metric)
        a, b = first[left] - first[right], second[left] - second[right]
        protocol_rows.append({"comparison": "index_vs_exact_ties", "left": left, "right": right, "metric": metric, "cells": len(a), "non_tied_cells": int(((a != 0) & (b != 0)).sum()), "strict_reversals": int((a * b < 0).sum())})
    for fraction in [.2, .5]:
        first = full.pivot(index="cell_id", columns="method", values="mean_atom_fraction")
        second = fixed[fixed.nominal_fraction == fraction].pivot(index="cell_id", columns="method", values="iou")
        a, b = first[left] - first[right], second[left] - second[right]
        protocol_rows.append({"comparison": f"calibrated_size_vs_fixed_{fraction:.2f}_iou", "left": left, "right": right, "metric": "smaller_size_vs_larger_iou", "cells": len(a), "non_tied_cells": int(((a != 0) & (b != 0)).sum()), "strict_reversals": int((a * b > 0).sum())})
    pd.DataFrame(protocol_rows).to_csv(HERE / f"{prefix}policy_order_comparisons.csv", index=False)
    result = {"status": "PARTIAL_PASS" if args.partial else "PASS", "cells": len(checks), "maximum_metric_replay_difference": maximum_delta, "eligible_calibration_occurrences": sum(c["n_calibration"] for c in checks), "eligible_test_occurrences": sum(c["n_test"] for c in checks), "checks": checks, "method_summary": summary, "comparison_boundary": "New methods use fresh attribution with original checkpoints. Original gradient results are historical P01; same-run subset versus full uses the identical fresh cached rankings. Historical-subset replay drift is retained separately. Bootstrap intervals describe the 11 observed task clusters from two dependent benchmark families."}
    if not args.partial:
        assert result["eligible_calibration_occurrences"] == 82866 and result["eligible_test_occurrences"] == 81672
    write_json(HERE / f"{prefix}validation.json", result)


if __name__ == "__main__":
    main()

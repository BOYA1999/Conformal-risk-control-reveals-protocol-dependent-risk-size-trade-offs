import argparse
import hashlib
import json
import math
import platform
import sys
import time
import warnings
from pathlib import Path

import torch
import torch_geometric
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import rankdata
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool


HERE = Path(__file__).resolve().parent
REVISION = HERE.parents[2]
PROJECT = REVISION.parent
sys.path.insert(0, str(PROJECT / "src"))
from run_gradient_grid import BXAIC_TASKS, GOOGLE_TASKS, GraphClassifier, bxaic_partitions, google_partitions


DEFAULT_CACHE = PROJECT.parents[1] / "<reviewer-working-tree>/tie_inclusive/established_explainers_tie_inclusive/score_cache"
CHECKPOINTS = PROJECT / "artifacts/experiment/gradient_grid_main/checkpoints"
CONTRACT = HERE / "run_contract.json"
FRACTIONS = np.round(np.linspace(0, 1, 101), 2)
METHODS = ["gradinput", "ig"]
TARGETS = ["true", "positive", "predicted"]
ALPHA = 0.10
HASH_CACHE = {}


def sha256(path):
    path = Path(path)
    if path not in HASH_CACHE:
        HASH_CACHE[path] = hashlib.sha256(path.read_bytes()).hexdigest().upper()
    return HASH_CACHE[path]


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_ready(payload), indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def top_set(scores, fraction):
    scores = np.asarray(scores, dtype=float)
    if fraction == 0 or not len(scores):
        return set()
    k = int(np.ceil(fraction * len(scores)))
    threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
    return set(np.flatnonzero(scores >= threshold))


def loss_table(scores, truths):
    table = np.empty((len(scores), len(FRACTIONS)))
    for i, (row, truth) in enumerate(zip(scores, truths)):
        row = np.asarray(row)
        truth = np.asarray(sorted(truth), dtype=int)
        order = np.argsort(-row, kind="stable")
        ordered = row[order]
        recovered = np.cumsum(np.isin(order, truth))
        nominal = np.where(FRACTIONS == 0, 0, np.ceil(FRACTIONS * len(row))).astype(int)
        counts = np.zeros(len(FRACTIONS), dtype=int)
        positive = nominal > 0
        thresholds = ordered[nominal[positive] - 1]
        counts[positive] = np.searchsorted(-ordered, -thresholds, side="right")
        found = np.zeros(len(FRACTIONS), dtype=int)
        found[positive] = recovered[counts[positive] - 1]
        table[i] = 1 - found / len(truth)
    return table


def calibrate(scores, truths):
    if len(scores) < 10:
        return len(FRACTIONS) - 1, False, "deterministic_full_set_fallback"
    losses = loss_table(scores, truths)
    n = len(losses)
    corrected = (n * losses.mean(0) + 1) / (n + 1)
    feasible = np.flatnonzero(corrected <= ALPHA)
    if not len(feasible):
        return len(FRACTIONS) - 1, False, "deterministic_full_set_fallback"
    return int(feasible[0]), True, "crc"


def set_metrics(scores, truths, fraction, indices=None):
    indices = range(len(scores)) if indices is None else indices
    rows = []
    for i in indices:
        selected = top_set(scores[i], fraction)
        truth = truths[i]
        overlap = len(selected & truth)
        rows.append((1 - overlap / len(truth), len(selected) / len(scores[i]), overlap / len(selected) if selected else 0, overlap / len(selected | truth)))
    if not rows:
        return {"n": 0, "risk": None, "retained": None, "precision": None, "iou": None}
    values = np.asarray(rows)
    return {"n": len(rows), "risk": float(values[:, 0].mean()), "retained": float(values[:, 1].mean()), "precision": float(values[:, 2].mean()), "iou": float(values[:, 3].mean())}


def targets_for(model, batch, mode):
    if mode == "true":
        return batch.y
    if mode == "positive":
        return torch.ones(len(batch.y), dtype=torch.long, device=batch.x.device)
    with torch.no_grad():
        return model(batch.x, batch.edge_index, batch.batch).argmax(1)


def attribution_scores(model, graphs, device, method, target_mode, batch_size):
    rows, predictions, labels, source_indices = [], [], [], []
    cursor = 0
    model.eval()
    for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
        batch = batch.to(device)
        frozen_targets = targets_for(model, batch, target_mode)
        with torch.no_grad():
            predictions.extend(model(batch.x, batch.edge_index, batch.batch).argmax(1).cpu().tolist())
        if method == "gradinput":
            x = batch.x.detach().clone().requires_grad_(True)
            logits = model(x, batch.edge_index, batch.batch)
            objective = logits[torch.arange(len(batch.y), device=device), frozen_targets].sum()
            scores = (torch.autograd.grad(objective, x)[0] * x).sum(-1)
        else:
            total = torch.zeros_like(batch.x)
            for step in range(1, 21):
                x = (batch.x * step / 20).detach().requires_grad_(True)
                logits = model(x, batch.edge_index, batch.batch)
                objective = logits[torch.arange(len(batch.y), device=device), frozen_targets].sum()
                total += torch.autograd.grad(objective, x)[0]
            scores = (batch.x * total / 20).sum(-1)
        ptr = batch.ptr.cpu().tolist()
        values = scores.detach().cpu().numpy()
        rows.extend(values[a:b] for a, b in zip(ptr[:-1], ptr[1:]))
        labels.extend(batch.y.cpu().tolist())
        count = len(batch.y)
        source_indices.extend(int(graph.source_index) for graph in graphs[cursor:cursor + count])
        cursor += count
    return rows, np.asarray(predictions), np.asarray(labels), np.asarray(source_indices)


@torch.no_grad()
def embeddings(model, graphs, device, batch_size=512):
    rows = []
    model.eval()
    for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
        batch = batch.to(device)
        node = model.encoder(batch.x, batch.edge_index)
        rows.append(global_add_pool(node, batch.batch).cpu().numpy())
    return np.concatenate(rows)


def unpack(values, offsets):
    return [values[a:b] for a, b in zip(offsets[:-1], offsets[1:])]


def cached_true(z, method, split):
    return unpack(z[f"{method}__{split}__score_values"], z[f"{split}__offsets"])


def graph_truths(graphs):
    return [set(torch.nonzero(g.rationale_mask, as_tuple=False).flatten().tolist()) for g in graphs]


def task_input_hashes(family, task):
    if family == "bxaic":
        paths = {
            "bxaic_data_csv": PROJECT / "data/raw/bxaic/data.csv",
            "bxaic_explanations_sdf": PROJECT / "data/raw/bxaic/explanations.sdf",
        }
    else:
        folder = PROJECT / "reference/graph-attribution/data" / task
        paths = {
            "smiles_csv": folder / f"{task}_smiles.csv",
            "split_npz": folder / f"{task}_traintest_indices.npz",
            "graph_npz": folder / "x_true.npz",
            "label_npz": folder / "y_true.npz",
            "rationale_npz": folder / "true_raw_attribution_datadicts.npz",
        }
    return {name: sha256(path) for name, path in paths.items()}


def cell_provenance(cache_dir, cell_id, checkpoint, family, task, run_mode):
    cache_path = cache_dir / f"{cell_id}.npz"
    upstream_path = cache_dir.parent / "cells" / f"{cell_id}.json"
    upstream = json.loads(upstream_path.read_text(encoding="utf-8"))
    cache_hash, checkpoint_hash = sha256(cache_path), sha256(checkpoint)
    if upstream["score_cache"]["sha256"].upper() != cache_hash:
        raise ValueError(f"{cell_id}: score-cache SHA-256 disagrees with upstream cell record")
    if upstream["checkpoint_sha256"].upper() != checkpoint_hash:
        raise ValueError(f"{cell_id}: checkpoint SHA-256 disagrees with upstream cell record")
    return {
        "run_mode": run_mode,
        "script_sha256": sha256(__file__),
        "contract_sha256": sha256(CONTRACT),
        "run_gradient_grid_sha256": sha256(PROJECT / "src/run_gradient_grid.py"),
        "run_bxaic_probe_sha256": sha256(PROJECT / "src/run_bxaic_probe.py"),
        "audit_graph_attribution_sha256": sha256(PROJECT / "src/audit_graph_attribution.py"),
        "upstream_cell_sha256": sha256(upstream_path),
        "score_cache_sha256": cache_hash,
        "checkpoint_sha256": checkpoint_hash,
        "task_input_sha256": task_input_hashes(family, task),
        "alpha": ALPHA,
        "fraction_grid": FRACTIONS.tolist(),
        "methods": METHODS,
        "targets": TARGETS,
        "runtime": {
            "python": platform.python_version(), "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__, "sklearn": sklearn.__version__,
            "numpy": np.__version__, "pandas": pd.__version__,
        },
    }


def validate_selected_cache(z, selected, split):
    source_indices = z[f"{split}__source_indices"].astype(int)
    if source_indices.tolist() != [int(graph.source_index) for graph in selected]:
        raise ValueError(f"{split}: cache source order disagrees with selected graphs")
    offsets = z[f"{split}__offsets"].astype(int)
    rationale_values = z[f"{split}__rationale_values"].astype(bool)
    if len(offsets) != len(selected) + 1 or offsets[0] != 0 or offsets[-1] != len(rationale_values):
        raise ValueError(f"{split}: invalid cache offsets")
    for i, graph in enumerate(selected):
        start, stop = offsets[i:i + 2]
        expected = graph.rationale_mask.detach().cpu().numpy().astype(bool)
        if stop - start != graph.num_nodes or not np.array_equal(rationale_values[start:stop], expected):
            raise ValueError(f"{split}/{int(graph.source_index)}: cached atom/rationale alignment failed")
    for method in METHODS:
        values = z[f"{method}__{split}__score_values"]
        if len(values) != len(rationale_values) or not np.isfinite(values).all():
            raise ValueError(f"{split}/{method}: cached scores are misaligned or nonfinite")


def score_consistency(cached, recomputed):
    differences = np.concatenate([
        np.abs(np.asarray(left, dtype=float) - np.asarray(right, dtype=float))
        for left, right in zip(cached, recomputed)
    ])
    return {
        "max_abs_difference": float(differences.max(initial=0.0)),
        "mean_abs_difference": float(differences.mean()) if len(differences) else 0.0,
    }


def reusable_cell(result_path, molecule_path, provenance, expected_test_rows):
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        return (
            payload.get("status") == "complete"
            and payload.get("provenance") == provenance
            and molecule_path.exists()
            and len(pd.read_csv(molecule_path, usecols=["source_index"])) == expected_test_rows
        )
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        return False


def paired_agreement(a, b, truths, predictions, labels, fraction_a, fraction_b, equality_expected):
    rhos, fixed_j, calibrated_j, max_diffs = [], [], [], []
    strata = {"correct": [], "error": []}
    for i, (left, right, truth) in enumerate(zip(a, b, truths)):
        ra, rb = rankdata(left), rankdata(right)
        if np.std(ra) > 0 and np.std(rb) > 0:
            rhos.append(float(np.corrcoef(ra, rb)[0, 1]))
        sa, sb = top_set(left, 0.20), top_set(right, 0.20)
        fixed_j.append(len(sa & sb) / len(sa | sb) if sa | sb else 1.0)
        sa, sb = top_set(left, fraction_a), top_set(right, fraction_b)
        calibrated_j.append(len(sa & sb) / len(sa | sb) if sa | sb else 1.0)
        loss = 1 - len(sb & truth) / len(truth)
        strata["correct" if predictions[i] == labels[i] else "error"].append((loss, len(sb) / len(right)))
        if equality_expected and predictions[i] == labels[i]:
            max_diffs.append(float(np.max(np.abs(np.asarray(left) - np.asarray(right)))))
    def summarize(values):
        if not values:
            return {"n": 0, "risk": None, "retained": None}
        x = np.asarray(values)
        return {"n": len(x), "risk": float(x[:, 0].mean()), "retained": float(x[:, 1].mean())}
    return {
        "score_spearman_mean": float(np.mean(rhos)) if rhos else None,
        "score_spearman_finite_n": len(rhos),
        "fixed20_set_jaccard_mean": float(np.mean(fixed_j)),
        "own_calibrated_set_jaccard_mean": float(np.mean(calibrated_j)),
        "correct_subset_max_abs_score_difference": max(max_diffs) if max_diffs else None,
        "right_target_correct_prediction_stratum": summarize(strata["correct"]),
        "right_target_error_prediction_stratum": summarize(strata["error"]),
    }


def evaluate_gate(model, partitions, all_scores, device):
    fit_y = np.asarray([bool(g.rationale_mask.any()) for g in partitions["fit"]], dtype=int)
    cal_y = np.asarray([bool(g.rationale_mask.any()) for g in partitions["calibration"]], dtype=int)
    test_y = np.asarray([bool(g.rationale_mask.any()) for g in partitions["test"]], dtype=int)
    fit_labels = np.asarray([int(g.y) for g in partitions["fit"]])
    cal_labels = np.asarray([int(g.y) for g in partitions["calibration"]])
    test_labels = np.asarray([int(g.y) for g in partitions["test"]])
    gate = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=20260904),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        gate.fit(embeddings(model, partitions["fit"], device), fit_y)
    convergence_warnings = sum(issubclass(item.category, ConvergenceWarning) for item in caught)
    cal_prob = gate.predict_proba(embeddings(model, partitions["calibration"], device))[:, 1]
    test_prob = gate.predict_proba(embeddings(model, partitions["test"], device))[:, 1]
    cal_gate, test_gate = cal_prob >= 0.5, test_prob >= 0.5
    cal_truths, test_truths = graph_truths(partitions["calibration"]), graph_truths(partitions["test"])
    eligible = np.flatnonzero(cal_gate & cal_y.astype(bool))
    index, certified, selection_status = calibrate([all_scores["calibration"][i] for i in eligible], [cal_truths[i] for i in eligible])
    fraction = float(FRACTIONS[index])
    molecule_rows, selective_losses, overall_nonnull_losses, retained = [], [], [], []
    null_sets, nonnull_abstain = 0, 0
    for i, graph in enumerate(partitions["test"]):
        truth = test_truths[i]
        selected = top_set(all_scores["test"][i], fraction) if test_gate[i] else set()
        size = len(selected) / graph.num_nodes
        retained.append(size)
        loss = None
        if truth:
            loss = 1 - len(selected & truth) / len(truth) if test_gate[i] else 1.0
            overall_nonnull_losses.append(loss)
            if test_gate[i]:
                selective_losses.append(loss)
            else:
                nonnull_abstain += 1
        elif selected:
            null_sets += 1
        molecule_rows.append({
            "source_index": int(graph.source_index), "label": int(graph.y), "nonnull": bool(truth),
            "gate_probability": test_prob[i], "gate_positive": bool(test_gate[i]),
            "selected_atoms": len(selected), "atom_count": graph.num_nodes, "retained_fraction": size,
            "missed_rationale_loss": loss,
        })
    return {
        "fit_n": len(fit_y), "fit_nonnull_prevalence": float(fit_y.mean()),
        "calibration_n": len(cal_y), "calibration_nonnull_prevalence": float(cal_y.mean()),
        "test_n": len(test_y), "test_nonnull_prevalence": float(test_y.mean()),
        "gate_test_auroc": float(roc_auc_score(test_y, test_prob)),
        "gate_test_auprc": float(average_precision_score(test_y, test_prob)),
        "gate_test_accuracy": float(accuracy_score(test_y, test_gate)),
        "gate_test_balanced_accuracy": float(balanced_accuracy_score(test_y, test_gate)),
        "gate_convergence_warning_count": convergence_warnings,
        "gate_converged": convergence_warnings == 0,
        "fit_nonnull_label_agreement": float((fit_y == fit_labels).mean()),
        "calibration_nonnull_label_agreement": float((cal_y == cal_labels).mean()),
        "test_nonnull_label_agreement": float((test_y == test_labels).mean()),
        "gate_threshold": 0.5, "eligible_calibration_n": len(eligible),
        "selection_status": selection_status, "crc_certified": certified,
        "nominal_fraction": fraction,
        "null_false_positive_set_rate": null_sets / int((test_y == 0).sum()),
        "nonnull_abstention_rate": nonnull_abstain / int(test_y.sum()),
        "selective_nonnull_risk": float(np.mean(selective_losses)) if selective_losses else None,
        "overall_nonnull_risk_abstention_as_one": float(np.mean(overall_nonnull_losses)),
        "overall_retained_fraction": float(np.mean(retained)),
        "overall_abstention_rate": float((~test_gate).mean()),
    }, molecule_rows


def run_cell(family, task, model_kind, seed, partitions, device, out_root, cache_dir, run_mode):
    cell_id = f"{family}__{task}__{model_kind}__seed{seed}"
    result_path = out_root / "cells" / f"{cell_id}.json"
    molecule_path = out_root / "gate_test_molecules" / f"{cell_id}.csv.gz"
    checkpoint = CHECKPOINTS / f"{cell_id}.pt"
    provenance = cell_provenance(cache_dir, cell_id, checkpoint, family, task, run_mode)
    if reusable_cell(result_path, molecule_path, provenance, len(partitions["test"])):
        print(f"skip_complete={cell_id}", flush=True)
        return
    model = GraphClassifier(model_kind, partitions["fit"][0].x.shape[1]).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["state_dict"])
    model.eval()
    selected, cached = {}, {method: {} for method in METHODS}
    with np.load(cache_dir / f"{cell_id}.npz", allow_pickle=False) as z:
        if z["schema_version"].tolist() != [1]:
            raise ValueError(f"{cell_id}: unsupported score-cache schema")
        for split in ["calibration", "test"]:
            graph_map = {int(g.source_index): g for g in partitions[split]}
            selected[split] = [graph_map[int(i)] for i in z[f"{split}__source_indices"]]
            validate_selected_cache(z, selected[split], split)
            for method in METHODS:
                cached[method][split] = [np.asarray(row).copy() for row in cached_true(z, method, split)]
    truths = {split: graph_truths(selected[split]) for split in selected}

    scores = {method: {target: {} for target in TARGETS} for method in METHODS}
    predictions, labels = {}, {}
    all_predicted_grad = {}
    cache_consistency = {method: {} for method in METHODS}
    for split in ["calibration", "test"]:
        all_rows, _, _, all_sources = attribution_scores(model, partitions[split], device, "gradinput", "predicted", 256)
        expected_sources = np.asarray([int(graph.source_index) for graph in partitions[split]])
        if not np.array_equal(all_sources, expected_sources):
            raise ValueError(f"{cell_id}/{split}: full-split attribution order changed")
        if not all(np.isfinite(row).all() and len(row) == graph.num_nodes for row, graph in zip(all_rows, partitions[split])):
            raise ValueError(f"{cell_id}/{split}: full-split attribution is nonfinite or misaligned")
        all_predicted_grad[split] = all_rows
        for method in METHODS:
            batch_size = 64 if method == "gradinput" else 32
            for target in TARGETS:
                rows, pred, lab, sources = attribution_scores(model, selected[split], device, method, target, batch_size)
                if not np.array_equal(sources, np.asarray([int(graph.source_index) for graph in selected[split]])):
                    raise ValueError(f"{cell_id}/{split}/{method}/{target}: selected attribution order changed")
                if not all(np.isfinite(row).all() and len(row) == graph.num_nodes for row, graph in zip(rows, selected[split])):
                    raise ValueError(f"{cell_id}/{split}/{method}/{target}: nonfinite or misaligned attribution")
                scores[method][target][split] = rows
                if target == "predicted":
                    if split in predictions and (not np.array_equal(predictions[split], pred) or not np.array_equal(labels[split], lab)):
                        raise ValueError(f"{cell_id}/{split}: prediction labels changed across attribution methods")
                    predictions[split], labels[split] = pred, lab
            cache_consistency[method][split] = score_consistency(cached[method][split], scores[method]["true"][split])

    target_results = {}
    agreements = {}
    for method in METHODS:
        target_results[method] = {}
        fractions = {}
        for target in TARGETS:
            index, certified, status = calibrate(scores[method][target]["calibration"], truths["calibration"])
            fraction = float(FRACTIONS[index])
            fractions[target] = fraction
            target_results[method][target] = {
                "calibration_n": len(truths["calibration"]), "test_n": len(truths["test"]),
                "selection_status": status, "crc_certified": certified, "nominal_fraction": fraction,
                **set_metrics(scores[method][target]["test"], truths["test"], fraction),
            }
        agreements[method] = {}
        for target in ["predicted", "positive"]:
            agreements[method][f"true_vs_{target}"] = paired_agreement(
                scores[method]["true"]["test"], scores[method][target]["test"], truths["test"],
                predictions["test"], labels["test"], fractions["true"], fractions[target],
                equality_expected=target == "predicted",
            )

    gate_result, molecule_rows = evaluate_gate(model, partitions, all_predicted_grad, device)
    pd.DataFrame(molecule_rows).to_csv(molecule_path, index=False, compression="gzip")
    payload = {
        "status": "complete", "reviewer_item": "A5", "cell_id": cell_id, "family": family,
        "task": task, "model": model_kind, "seed": seed, "set_policy": "include_all_exact_ties_v2",
        "provenance": provenance, "cache_true_score_consistency": cache_consistency,
        "target_audit": target_results, "target_agreement": agreements, "two_stage": gate_result,
    }
    write_json(result_path, payload)
    print(f"cell_complete={cell_id}", flush=True)


def aggregate(out_root):
    target_rows, gate_rows, agreement_rows, cache_rows = [], [], [], []
    for path in sorted((out_root / "cells").glob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        base = {k: d[k] for k in ["cell_id", "family", "task", "model", "seed"]}
        for method, targets in d["target_audit"].items():
            for target, metrics in targets.items():
                target_rows.append({**base, "method": method, "target": target, **metrics})
        for method, pairs in d["target_agreement"].items():
            for pair, metrics in pairs.items():
                agreement_rows.append({
                    **base, "method": method, "comparison": pair,
                    "score_spearman_mean": metrics["score_spearman_mean"],
                    "score_spearman_finite_n": metrics["score_spearman_finite_n"],
                    "fixed20_set_jaccard_mean": metrics["fixed20_set_jaccard_mean"],
                    "own_calibrated_set_jaccard_mean": metrics["own_calibrated_set_jaccard_mean"],
                    "correct_subset_max_abs_score_difference": metrics["correct_subset_max_abs_score_difference"],
                    **{f"correct_{k}": v for k, v in metrics["right_target_correct_prediction_stratum"].items()},
                    **{f"error_{k}": v for k, v in metrics["right_target_error_prediction_stratum"].items()},
                })
        for method, splits in d["cache_true_score_consistency"].items():
            for split, metrics in splits.items():
                cache_rows.append({**base, "method": method, "split": split, **metrics})
        gate_rows.append({**base, **d["two_stage"]})
    target, gate = pd.DataFrame(target_rows), pd.DataFrame(gate_rows)
    agreement, cache = pd.DataFrame(agreement_rows), pd.DataFrame(cache_rows)
    target.to_csv(out_root / "target_cells.csv", index=False)
    gate.to_csv(out_root / "two_stage_cells.csv", index=False)
    agreement.to_csv(out_root / "target_agreement_cells.csv", index=False)
    cache.to_csv(out_root / "cache_true_score_consistency.csv", index=False)
    expected_ids = {
        f"{family}__{task}__{model}__seed{seed}"
        for family, tasks in (("bxaic", BXAIC_TASKS), ("google", GOOGLE_TASKS))
        for task in tasks for model in ("gin", "gcn") for seed in (42, 123, 2026)
    }
    observed_ids = set(gate.get("cell_id", []))
    structural_pass = (
        observed_ids == expected_ids and len(target) == 396 and len(agreement) == 264
        and target.groupby("cell_id").size().eq(6).all()
        and agreement.groupby("cell_id").size().eq(4).all()
        and len(list((out_root / "gate_test_molecules").glob("*.csv.gz"))) == len(expected_ids)
        and gate["gate_converged"].all()
    )
    finite_targets = bool(np.isfinite(target[["risk", "retained", "precision", "iou"]].to_numpy()).all())
    cache_max_difference = float(cache["max_abs_difference"].max())
    summary = {
        "status": "PASS" if structural_pass and finite_targets else "PARTIAL", "reviewer_item": "A5",
        "cells": len(gate), "target_rows": len(target),
        "target_macro": target.groupby(["method", "target"])[["risk", "retained", "precision", "iou"]].mean().reset_index().to_dict("records"),
        "agreement_macro": agreement.groupby(["method", "comparison"])[["score_spearman_mean", "fixed20_set_jaccard_mean", "own_calibrated_set_jaccard_mean"]].mean().reset_index().to_dict("records"),
        "two_stage_macro": gate[["gate_test_auroc", "gate_test_auprc", "gate_test_balanced_accuracy", "null_false_positive_set_rate", "nonnull_abstention_rate", "selective_nonnull_risk", "overall_nonnull_risk_abstention_as_one", "overall_retained_fraction", "overall_abstention_rate", "test_nonnull_label_agreement"]].mean().to_dict(),
        "two_stage_by_family_task": gate.groupby(["family", "task"])[["gate_test_auroc", "gate_test_auprc", "gate_test_balanced_accuracy", "null_false_positive_set_rate", "nonnull_abstention_rate", "selective_nonnull_risk", "overall_nonnull_risk_abstention_as_one", "overall_retained_fraction", "overall_abstention_rate", "test_nonnull_label_agreement"]].mean().reset_index().to_dict("records"),
        "fallback_cells": int((~gate["crc_certified"]).sum()),
        "gate_nonconverged_cells": int((~gate["gate_converged"]).sum()),
        "undefined_selective_risk_cells": int(gate["selective_nonnull_risk"].isna().sum()),
        "perfect_test_nonnull_label_alignment_cells": int(np.isclose(gate["test_nonnull_label_agreement"], 1.0).sum()),
        "cache_true_score_max_abs_difference": cache_max_difference,
        "interpretation": "Predicted-class rankings are deployment-available; true-class rankings remain audit-only. Mask presence is identical to the class label in most studied tasks, so the supervised gate is not an independent detector of universal chemical relevance; the nondegenerate task is reported separately.",
    }
    write_json(out_root / "summary.json", summary)
    write_json(out_root / "validation.json", {
        "status": "PASS" if structural_pass and finite_targets else "PARTIAL",
        "expected_cells": 66, "observed_cells": len(gate), "expected_target_rows": 396,
        "observed_target_rows": len(target), "expected_agreement_rows": 264, "observed_agreement_rows": len(agreement),
        "exact_expected_cell_ids": observed_ids == expected_ids,
        "finite_core_outputs": finite_targets,
        "cache_true_score_max_abs_difference": cache_max_difference,
        "cache_consistency_within_1e-5": cache_max_difference <= 1e-5,
        "gate_nonconverged_cells": int((~gate["gate_converged"]).sum()),
        "per_molecule_gate_files": len(list((out_root / "gate_test_molecules").glob("*.csv.gz"))),
    })
    print(json.dumps(json_ready(summary), indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()
    out_root = HERE / "smoke" if args.smoke else HERE
    (out_root / "cells").mkdir(parents=True, exist_ok=True)
    (out_root / "gate_test_molecules").mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    run_mode = "smoke" if args.smoke else "full"
    jobs = [("bxaic", task) for task in BXAIC_TASKS] + [("google", task) for task in GOOGLE_TASKS]
    if args.smoke:
        jobs = jobs[:1]
    for family, task in jobs:
        partitions = bxaic_partitions(PROJECT / "data/raw/bxaic/data.csv", PROJECT / "data/raw/bxaic/explanations.sdf", task) if family == "bxaic" else google_partitions(PROJECT / "reference/graph-attribution/data", task)
        for model_kind in (["gin"] if args.smoke else ["gin", "gcn"]):
            for seed in ([42] if args.smoke else [42, 123, 2026]):
                run_cell(family, task, model_kind, seed, partitions, device, out_root, args.cache_dir, run_mode)
    if not args.smoke:
        aggregate(out_root)
    write_json(out_root / "environment.json", {
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S %z"), "platform": platform.platform(),
        "python": sys.version, "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "device": args.device, "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "numpy": np.__version__, "pandas": pd.__version__, "run_mode": run_mode,
        "script_sha256": sha256(__file__), "contract_sha256": sha256(CONTRACT),
    })


if __name__ == "__main__":
    main()

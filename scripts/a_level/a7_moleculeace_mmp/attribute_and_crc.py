import argparse
import json
import math
import os
from pathlib import Path

for thread_variable in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[thread_variable] = "1"

import torch
import torch_geometric
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from torch_geometric.data import Batch

from gine_model import GINERegressor, graph_from_smiles


FRACTIONS = np.linspace(0.0, 1.0, 101)
DIRECTION_NUMERICAL_TIE_ATOL = 1e-6
SCORE_SERIALIZATION_SIGNIFICANT_DIGITS = 10
PROXIES = {
    "main_variable_plus_attachment_symmetry_closed": "main_mask",
    "variable_only_symmetry_closed": "variable_mask",
    "one_hop_expanded_symmetry_closed": "expanded_mask",
}


def parse_indices(value):
    if pd.isna(value) or value == "":
        return set()
    return {int(item) for item in str(value).split(";")}


def scores_text(values):
    return ";".join(f"{float(value):.{SCORE_SERIALIZATION_SIGNIFICANT_DIGITS}g}" for value in values)


def parse_scores(value):
    return np.asarray([float(item) for item in str(value).split(";")], dtype=float)


def predict_original(model, graph, device, y_mean, y_std, x=None):
    x = graph.x if x is None else x
    batch = torch.zeros(x.shape[0], dtype=torch.long, device=device)
    return model(x, graph.edge_index, graph.edge_attr, batch) * y_std + y_mean


def molecule_attributions(model, smiles, device, y_mean, y_std, ig_steps):
    graph = graph_from_smiles(smiles).to(device)
    model.eval()
    original_x = graph.x.detach()
    baseline_x = torch.zeros_like(original_x)
    with torch.no_grad():
        prediction = float(predict_original(model, graph, device, y_mean, y_std).item())
        baseline_prediction = float(predict_original(model, graph, device, y_mean, y_std, baseline_x).item())
    ig_batch = Batch.from_data_list([graph.clone() for _ in range(ig_steps)]).to(device)
    scales = torch.linspace(1.0 / ig_steps, 1.0, ig_steps, device=device).repeat_interleave(graph.num_nodes).unsqueeze(1)
    interpolated = (scales * original_x.repeat(ig_steps, 1)).detach().requires_grad_(True)
    ig_outputs = model(interpolated, ig_batch.edge_index, ig_batch.edge_attr, ig_batch.batch) * y_std + y_mean
    gradients = torch.autograd.grad(ig_outputs.sum(), interpolated)[0]
    gradient_sum = gradients.view(ig_steps, graph.num_nodes, -1).sum(dim=0).detach()
    signed_ig = ((original_x - baseline_x) * gradient_sum / ig_steps).sum(dim=1).detach().cpu().numpy()
    occluded_predictions = []
    for start in range(0, graph.num_nodes, 32):
        data_list = []
        for atom_index in range(start, min(start + 32, graph.num_nodes)):
            occluded = graph.clone()
            occluded.x = original_x.clone()
            occluded.x[atom_index] = 0.0
            data_list.append(occluded)
        batch = Batch.from_data_list(data_list).to(device)
        with torch.no_grad():
            values = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch) * y_std + y_mean
        occluded_predictions.extend(values.cpu().tolist())
    signed_occlusion = prediction - np.asarray(occluded_predictions, dtype=float)
    return {
        "prediction": prediction,
        "baseline_prediction": baseline_prediction,
        "integrated_gradients": np.abs(signed_ig),
        "atom_occlusion": np.abs(signed_occlusion),
        "ig_completeness_error": float(abs(signed_ig.sum() - (prediction - baseline_prediction))),
    }


def top_fraction_set(scores, fraction):
    scores = np.asarray(scores, dtype=float)
    if fraction == 0 or not len(scores):
        return set()
    count = int(np.ceil(fraction * len(scores)))
    threshold = np.partition(scores, len(scores) - count)[len(scores) - count]
    return set(np.flatnonzero(scores >= threshold).tolist())


def loss_table(score_rows, rationale_rows):
    table = np.empty((len(score_rows), len(FRACTIONS)), dtype=float)
    for row_index, (scores, rationale) in enumerate(zip(score_rows, rationale_rows)):
        scores = np.asarray(scores, dtype=float)
        rationale = np.asarray(sorted(rationale), dtype=int)
        order = np.argsort(-scores)
        ordered = scores[order]
        rationale_mask = np.zeros(len(scores), dtype=int)
        rationale_mask[rationale] = 1
        recovered_by_rank = np.cumsum(rationale_mask[order])
        nominal = np.where(FRACTIONS == 0, 0, np.ceil(FRACTIONS * len(scores))).astype(int)
        recovered = np.zeros(len(FRACTIONS), dtype=int)
        positive = nominal > 0
        thresholds = ordered[nominal[positive] - 1]
        tie_counts = np.searchsorted(-ordered, -thresholds, side="right")
        recovered[positive] = recovered_by_rank[tie_counts - 1]
        table[row_index] = 1.0 - recovered / len(rationale)
    if np.any(np.diff(table, axis=1) > 1e-12):
        raise RuntimeError("loss is not monotone")
    return table


def calibrate(losses, alpha=0.1):
    n = losses.shape[0]
    corrected = (n * losses.mean(axis=0) + 1.0) / (n + 1)
    feasible = np.flatnonzero(corrected <= alpha)
    if not len(feasible):
        raise RuntimeError("no feasible CRC fraction")
    index = int(feasible[0])
    return index, float(losses[:, index].mean()), float(corrected[index])


def rationale_for_row(row, proxy_column):
    a = parse_indices(row[f"{proxy_column}_a"])
    b = {index + int(row["n_atoms_a"]) for index in parse_indices(row[f"{proxy_column}_b"])}
    rationale = a | b
    if not rationale:
        raise RuntimeError("empty transformation-site proxy")
    return rationale


def scores_for_row(row):
    return np.concatenate([parse_scores(row["scores_a"]), parse_scores(row["scores_b"])])


def deterministic_metrics(rows, proxy_column, fraction):
    values = []
    for row in rows.itertuples(index=False):
        scores = scores_for_row(row._asdict())
        rationale = rationale_for_row(row._asdict(), proxy_column)
        selected = top_fraction_set(scores, fraction)
        overlap = len(selected & rationale)
        union = len(selected | rationale)
        values.append(
            (
                1.0 - overlap / len(rationale),
                len(selected) / len(scores),
                overlap / len(selected) if selected else 0.0,
                overlap / union if union else 1.0,
            )
        )
    return np.asarray(values, dtype=float)


def oracle_metrics(rows, proxy_column):
    values = []
    for row in rows.itertuples(index=False):
        data = row._asdict()
        rationale = rationale_for_row(data, proxy_column)
        total = len(scores_for_row(data))
        values.append((0.0, len(rationale) / total, 1.0, 1.0))
    return np.asarray(values, dtype=float)


def random_metrics(rows, proxy_column, fraction):
    values = []
    for row in rows.itertuples(index=False):
        data = row._asdict()
        total = len(scores_for_row(data))
        rationale_size = len(rationale_for_row(data, proxy_column))
        selected_size = int(np.ceil(fraction * total))
        expected_iou = 0.0
        for overlap in range(max(0, selected_size + rationale_size - total), min(selected_size, rationale_size) + 1):
            expected_iou += hypergeom.pmf(overlap, total, rationale_size, selected_size) * overlap / (
                selected_size + rationale_size - overlap
            )
        values.append(
            (
                1.0 - selected_size / total,
                selected_size / total,
                rationale_size / total if selected_size else 0.0,
                expected_iou,
            )
        )
    return np.asarray(values, dtype=float)


def component_mean(values, components):
    frame = pd.DataFrame(values, columns=["risk", "retained_fraction", "precision", "iou"])
    frame["component_id"] = np.asarray(components, dtype=str)
    return frame.groupby("component_id")[["risk", "retained_fraction", "precision", "iou"]].mean().to_numpy(float)


def aggregate_result_rows(target, seed, method, proxy, evaluation_split, stratum, policy, nominal_fraction, values, subset, cal_info, calibration_unit):
    components = subset.component_id.astype(str).to_numpy()
    rows = []
    for aggregation, aggregated in [
        ("pair_weighted", values),
        ("component_weighted", component_mean(values, components)),
    ]:
        rows.append(
            {
                "dataset": target,
                "seed": seed,
                "method": method,
                "proxy": proxy,
                "evaluation_split": evaluation_split,
                "stratum": stratum,
                "policy": policy,
                "aggregation": aggregation,
                "n_pairs": len(values),
                "n_components": len(set(components)),
                "nominal_fraction": nominal_fraction,
                "risk": float(aggregated[:, 0].mean()),
                "retained_fraction": float(aggregated[:, 1].mean()),
                "precision": float(aggregated[:, 2].mean()),
                "iou": float(aggregated[:, 3].mean()),
                "calibration_unit": calibration_unit,
                "calibration_empirical_risk": cal_info[0],
                "calibration_corrected_risk": cal_info[1],
            }
        )
    return rows


def component_losses(frame, losses):
    components = frame.component_id.astype(str).to_numpy()
    return np.vstack([losses[components == component].mean(axis=0) for component in sorted(set(components))])


def bootstrap_macro(results, replicates=5000):
    rows = []
    metrics = ["risk", "retained_fraction", "precision", "iou"]
    selected = results[(results.evaluation_split == "test") & (results.stratum == "all")]
    rng = np.random.default_rng(20260904)
    group_columns = ["method", "proxy", "policy", "aggregation"]
    for keys, frame in selected.groupby(group_columns):
        by_target = frame.groupby("dataset")[metrics].mean()
        target_values = by_target.to_numpy(float)
        samples = target_values[rng.integers(0, len(target_values), size=(replicates, len(target_values)))].mean(axis=1)
        for metric_index, metric in enumerate(metrics):
            rows.append(
                {
                    **dict(zip(group_columns, keys)),
                    "metric": metric,
                    "target_macro_mean": float(target_values[:, metric_index].mean()),
                    "bootstrap_ci_low": float(np.quantile(samples[:, metric_index], 0.025)),
                    "bootstrap_ci_high": float(np.quantile(samples[:, metric_index], 0.975)),
                    "n_targets": len(target_values),
                }
            )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--controls", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ig-steps", type=int, default=32)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    pairs = pd.read_csv(args.pairs)
    pairs["pair_type"] = "sar_cliff_proxy"
    controls = pd.read_csv(args.controls)
    combined_pairs = pd.concat([pairs, controls], ignore_index=True, sort=False)
    predictions = pd.read_csv(args.predictions)
    device = torch.device(args.device)
    if device.type == "cpu":
        if args.cpu_threads != 1:
            raise RuntimeError("formal CPU attribution contract requires --cpu-threads 1")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
    torch.manual_seed(20260905)
    attribution_rows = []
    direction_instability_rows = []
    checkpoint_reproduction_max_error = 0.0
    for (target, seed), prediction_frame in predictions.groupby(["dataset", "seed"]):
        checkpoint_path = Path(args.model_dir) / target / f"seed_{int(seed)}.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = GINERegressor(**checkpoint["architecture"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        target_pairs = combined_pairs[combined_pairs.dataset == target]
        prediction_map = dict(zip(prediction_frame.canonical_smiles, prediction_frame.prediction))
        cache = {}
        for pair_index, row in enumerate(target_pairs.itertuples(index=False), start=1):
            for smiles in [row.smiles_a, row.smiles_b]:
                if smiles not in cache:
                    cache[smiles] = molecule_attributions(
                        model,
                        smiles,
                        device,
                        checkpoint["y_mean"],
                        checkpoint["y_std"],
                        args.ig_steps,
                    )
                    checkpoint_reproduction_max_error = max(
                        checkpoint_reproduction_max_error,
                        abs(cache[smiles]["prediction"] - prediction_map[smiles]),
                    )
            a, b = cache[row.smiles_a], cache[row.smiles_b]
            prediction_a = float(prediction_map[row.smiles_a])
            prediction_b = float(prediction_map[row.smiles_b])
            predicted_delta = prediction_a - prediction_b
            checkpoint_delta = a["prediction"] - b["prediction"]
            if np.sign(predicted_delta) != np.sign(checkpoint_delta):
                maximum_absolute_delta = max(abs(predicted_delta), abs(checkpoint_delta))
                direction_instability_rows.append(
                    {
                        "dataset": target,
                        "split": row.split,
                        "pair_type": row.pair_type,
                        "pair_id": int(row.pair_id),
                        "seed": int(seed),
                        "prediction_table_delta": predicted_delta,
                        "checkpoint_attribution_forward_delta": checkpoint_delta,
                        "maximum_absolute_delta": maximum_absolute_delta,
                        "numerical_tie_atol": DIRECTION_NUMERICAL_TIE_ATOL,
                    }
                )
                if maximum_absolute_delta > DIRECTION_NUMERICAL_TIE_ATOL:
                    raise RuntimeError("prediction direction changed outside the frozen numerical-tie tolerance")
            direction_correct = bool(np.sign(predicted_delta) == np.sign(row.measured_delta_a_minus_b))
            base = row._asdict()
            for method in ["integrated_gradients", "atom_occlusion"]:
                attribution_rows.append(
                    {
                        **base,
                        "seed": int(seed),
                        "method": method,
                        "prediction_a": prediction_a,
                        "prediction_b": prediction_b,
                        "predicted_delta_a_minus_b": predicted_delta,
                        "direction_correct": direction_correct,
                        "scores_a": scores_text(a[method]),
                        "scores_b": scores_text(b[method]),
                        "ig_completeness_error_a": a["ig_completeness_error"] if method == "integrated_gradients" else np.nan,
                        "ig_completeness_error_b": b["ig_completeness_error"] if method == "integrated_gradients" else np.nan,
                    }
                )
            if pair_index % 25 == 0:
                print(f"{target} seed={seed}: attributed {pair_index}/{len(target_pairs)} pairs", flush=True)
    attributions = pd.DataFrame(attribution_rows)
    attributions.to_csv(output_dir / "pair_attributions.csv", index=False)
    direction_instabilities = pd.DataFrame(
        direction_instability_rows,
        columns=[
            "dataset",
            "split",
            "pair_type",
            "pair_id",
            "seed",
            "prediction_table_delta",
            "checkpoint_attribution_forward_delta",
            "maximum_absolute_delta",
            "numerical_tie_atol",
        ],
    )
    direction_instabilities.to_csv(output_dir / "near_zero_direction_instability.csv", index=False)
    result_rows = []
    feasibility_rows = []
    for (target, seed, method), frame in attributions.groupby(["dataset", "seed", "method"]):
        calibration = frame[(frame.pair_type == "sar_cliff_proxy") & (frame.split == "calibration")]
        test = frame[(frame.pair_type == "sar_cliff_proxy") & (frame.split == "test")]
        control = frame[frame.pair_type == "matched_noncliff_control"]
        for proxy, proxy_column in PROXIES.items():
            calibration_losses = loss_table(
                [scores_for_row(row._asdict()) for row in calibration.itertuples(index=False)],
                [rationale_for_row(row._asdict(), proxy_column) for row in calibration.itertuples(index=False)],
            )
            pair_index, pair_empirical, pair_corrected = calibrate(calibration_losses)
            pair_fraction = float(FRACTIONS[pair_index])
            grouped_losses = component_losses(calibration, calibration_losses)
            component_sizes = calibration.groupby("component_id").size()
            component_feasible = len(grouped_losses) >= 9
            if component_feasible:
                component_index, component_empirical, component_corrected = calibrate(grouped_losses)
                component_fraction = float(FRACTIONS[component_index])
            else:
                component_empirical = component_corrected = component_fraction = np.nan
            feasibility_rows.append(
                {
                    "dataset": target,
                    "seed": int(seed),
                    "method": method,
                    "proxy": proxy,
                    "n_calibration_pairs": len(calibration),
                    "n_calibration_components": len(grouped_losses),
                    "max_pairs_per_calibration_component": int(component_sizes.max()),
                    "minimum_components_alpha_0_10": 9,
                    "component_crc_numerically_feasible": component_feasible,
                    "component_crc_fraction": component_fraction,
                    "component_calibration_empirical_risk": component_empirical,
                    "component_calibration_corrected_risk": component_corrected,
                    "pair_crc_descriptive_fraction": pair_fraction,
                    "pair_calibration_empirical_risk": pair_empirical,
                    "pair_calibration_corrected_risk": pair_corrected,
                    "fallback_when_infeasible": "pair_crc_descriptive_finite_pool_only" if not component_feasible else "not_needed",
                    "guarantee": "none; clustered retrospective stress test",
                }
            )
            policies = [
                ("pair_crc_descriptive", pair_fraction, deterministic_metrics, (pair_empirical, pair_corrected), "pairs"),
                ("fixed_20_percent", 0.2, deterministic_metrics, (np.nan, np.nan), "none"),
                ("fixed_50_percent", 0.5, deterministic_metrics, (np.nan, np.nan), "none"),
                ("oracle_proxy", pair_fraction, oracle_metrics, (np.nan, np.nan), "none"),
                ("random_same_nominal_size", pair_fraction, random_metrics, (np.nan, np.nan), "none"),
            ]
            if component_feasible:
                policies.append(
                    (
                        "component_crc_cluster_stress_test",
                        component_fraction,
                        deterministic_metrics,
                        (component_empirical, component_corrected),
                        "components",
                    )
                )
            for policy, fraction, evaluator, cal_info, calibration_unit in policies:
                for evaluation_split, subset in [("calibration", calibration), ("test", test)]:
                    strata = [("all", subset)]
                    if evaluation_split == "test":
                        strata.extend(
                            [
                                ("correct_direction", subset[subset.direction_correct]),
                                ("direction_error", subset[~subset.direction_correct]),
                            ]
                        )
                    for stratum, stratum_frame in strata:
                        if not len(stratum_frame):
                            continue
                        if evaluator is deterministic_metrics:
                            values = evaluator(stratum_frame, proxy_column, fraction)
                        elif evaluator is random_metrics:
                            values = evaluator(stratum_frame, proxy_column, fraction)
                        else:
                            values = evaluator(stratum_frame, proxy_column)
                        result_rows.extend(
                            aggregate_result_rows(
                                target,
                                int(seed),
                                method,
                                proxy,
                                evaluation_split,
                                stratum,
                                policy,
                                fraction,
                                values,
                                stratum_frame,
                                cal_info,
                                calibration_unit,
                            )
                        )
                if policy in {"pair_crc_descriptive", "component_crc_cluster_stress_test"} and len(control):
                    values = deterministic_metrics(control, proxy_column, fraction)
                    result_rows.extend(
                        aggregate_result_rows(
                            target,
                            int(seed),
                            method,
                            proxy,
                            "matched_noncliff_control",
                            "descriptive_only",
                            policy,
                            fraction,
                            values,
                            control,
                            cal_info,
                            calibration_unit,
                        )
                    )
    results = pd.DataFrame(result_rows)
    results.to_csv(output_dir / "crc_results.csv", index=False)
    feasibility = pd.DataFrame(feasibility_rows)
    feasibility.to_csv(output_dir / "component_crc_feasibility.csv", index=False)
    macro = bootstrap_macro(results)
    macro.to_csv(output_dir / "crc_macro_summary.csv", index=False)
    primary = macro[
        (macro.proxy == "main_variable_plus_attachment_symmetry_closed")
        & macro.policy.isin(["pair_crc_descriptive", "component_crc_cluster_stress_test"])
    ].to_dict("records")
    feasible_targets = sorted(feasibility.loc[feasibility.component_crc_numerically_feasible, "dataset"].unique())
    infeasible_targets = sorted(feasibility.loc[~feasibility.component_crc_numerically_feasible, "dataset"].unique())
    ig_rows = attributions[attributions.method == "integrated_gradients"]
    ig_errors = np.concatenate(
        [ig_rows.ig_completeness_error_a.to_numpy(float), ig_rows.ig_completeness_error_b.to_numpy(float)]
    )
    overlap_path = Path(args.pairs).resolve().parent / "cross_target_overlap_audit.json"
    overlap = json.loads(overlap_path.read_text(encoding="utf-8"))
    model_metrics = pd.read_csv(Path(args.predictions).resolve().parent / "model_metrics.csv")
    metric_columns = ["test_rmse", "test_spearman", "test_pair_direction_accuracy"]
    target_performance = model_metrics.groupby("dataset")[metric_columns].mean()
    target_extremes = {}
    for metric in metric_columns:
        values = target_performance[metric]
        target_extremes[metric] = {
            "lowest": {"dataset": values.idxmin(), "value": float(values.min())},
            "highest": {"dataset": values.idxmax(), "value": float(values.max())},
        }
    seed_counts = model_metrics.groupby("dataset").seed.nunique()
    seeds_per_target = int(seed_counts.iloc[0]) if seed_counts.nunique() == 1 else sorted(seed_counts.unique().tolist())
    predictive_performance = {
        "aggregation": f"equal-target macro mean after averaging {seeds_per_target} seeds within each of {len(target_performance)} targets",
        "n_targets": len(target_performance),
        "seeds_per_target": seeds_per_target,
        "target_macro_mean": {metric: float(target_performance[metric].mean()) for metric in metric_columns},
        "target_extremes": target_extremes,
        "boundary": "Weak and heterogeneous activity-cliff prediction limits the interpretation of downstream explanation results; RMSE is lower-is-better, whereas Spearman and pair-direction accuracy are higher-is-better.",
    }
    gpu_instability = json.loads(
        (Path(__file__).resolve().parent / "diagnostics" / "gpu_attribution_instability" / "gpu_attribution_instability_audit.json").read_text(encoding="utf-8")
    )
    summary = {
        "status": "DESCRIPTIVE_CLUSTERED_STRESS_TEST",
        "estimand": "predicted potency difference f(A)-f(B)",
        "score": "absolute atom contribution on the disjoint union",
        "masking_baseline": "zero continuous atom features while preserving the molecular edge graph",
        "atom_occlusion_definition": "zero one atom's continuous features while preserving all molecular edges",
        "risk_definition": "mean fraction of experimental SAR-linked transformation-site proxy atoms missed (one minus proxy recall), not the probability of any miss",
        "tie_policy": "include all atoms tied at the nominal fraction boundary",
        "alpha": 0.1,
        "ig_steps": args.ig_steps,
        "checkpoint_reproduction_max_absolute_error": checkpoint_reproduction_max_error,
        "direction_strata_source": "predictions.csv serialized deployment predictions",
        "near_zero_direction_instability_pairs": len(direction_instabilities),
        "near_zero_direction_instability_atol": DIRECTION_NUMERICAL_TIE_ATOL,
        "near_zero_direction_instability_boundary": "Checkpoint attribution forwards are reproduction diagnostics only; direction strata use serialized predictions, and sign changes within the fixed tolerance are disclosed as numerically unstable rather than robustly correct.",
        "historical_gpu_direction_instability_pairs": 1,
        "gpu_attribution_reproducibility_boundary": gpu_instability["boundary"],
        "attribution_rows": len(attributions),
        "crc_result_rows": len(results),
        "calibration_component_range": [int(feasibility.n_calibration_components.min()), int(feasibility.n_calibration_components.max())],
        "component_crc_numerically_feasible_targets": feasible_targets,
        "component_crc_infeasible_targets": infeasible_targets,
        "ig_completeness_absolute_error": {
            "n_molecule_attributions": len(ig_errors),
            "median": float(np.median(ig_errors)),
            "p95": float(np.quantile(ig_errors, 0.95)),
            "max": float(np.max(ig_errors)),
            "boundary": "numerical path-integration diagnostic; not a fidelity or mechanistic-validity guarantee"
        },
        "cross_target_overlap": overlap,
        "predictive_performance": predictive_performance,
        "primary_target_macro": primary,
        "proxy_boundary": "Experimental SAR-linked transformation-site proxy; not causal, complete, mechanistic, or expert ground truth.",
        "control_boundary": "Matched noncliff controls are descriptive only because two target-specific balance checks exceeded |SMD| 0.25.",
        "cluster_boundary": "Pairs are clustered within MMP/scaffold components. Pair-weighted selection is a descriptive finite-pool stress test; component-level numerical feasibility does not establish exchangeability, distribution-free risk control, or generalization.",
        "bootstrap_boundary": "Target-resampling intervals are descriptive for the fixed, overlapping 11-target panel; they are not population confidence intervals or independent-target generalization evidence."
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    environment = {
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "device": args.device,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "ig_steps": args.ig_steps,
        "cpu_threads": torch.get_num_threads() if device.type == "cpu" else None,
        "cpu_interop_threads": torch.get_num_interop_threads() if device.type == "cpu" else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "default_dtype": str(torch.get_default_dtype()),
        "score_serialization_significant_digits": SCORE_SERIALIZATION_SIGNIFICANT_DIGITS,
        "ranking": "descending serialized absolute atom scores",
        "tie_policy": "include every atom exactly tied at the serialized-score boundary",
        "thread_environment": {name: os.environ[name] for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]},
    }
    (output_dir / "attribution_environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

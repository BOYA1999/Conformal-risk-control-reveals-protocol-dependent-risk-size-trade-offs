import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import torch
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader


REPOSITORY = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get("MOLXAI_OUTPUT_DIR", REPOSITORY / "results/v8/rings_stratified")).resolve()
PROJECT = Path(os.environ.get("MOLXAI_PROJECT_ROOT", "external/project")).resolve()
ANALYSIS = Path(os.environ.get("MOLXAI_ANALYSIS_ROOT", PROJECT / "analysis_workspace")).resolve()
A5_ROOT = ANALYSIS / "experiments/jcim_a_level_20260904/a5_target_null"
CACHE = Path(os.environ.get("MOLXAI_SCORE_CACHE", "external/score_cache/bxaic__rings-count__gin__seed42.npz")).resolve()
OUT.mkdir(parents=True, exist_ok=True)
CELL_ID = "bxaic__rings-count__gin__seed42"
METHODS = ("gradinput", "ig")
TARGETS = ("true", "positive")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def load_a5():
    spec = importlib.util.spec_from_file_location("a5_frozen", A5_ROOT / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def counts_at_grid(scores, fractions):
    ordered = np.sort(np.asarray(scores, dtype=float))[::-1]
    nominal = np.where(fractions == 0, 0, np.ceil(fractions * len(ordered))).astype(int)
    counts = np.zeros(len(fractions), dtype=int)
    positive = nominal > 0
    counts[positive] = np.searchsorted(-ordered, -ordered[nominal[positive] - 1], side="right")
    return counts


def calibrate_losses(losses, alpha=0.10):
    n = len(losses)
    corrected = (n * losses.mean(0) + 1) / (n + 1)
    feasible = np.flatnonzero(corrected <= alpha)
    if not len(feasible):
        return len(corrected) - 1, False
    return int(feasible[0]), True


def oracle_schedule(cal_scores, cal_truths, test_scores, test_truths, fractions):
    cal_counts = np.stack([counts_at_grid(row, fractions) for row in cal_scores])
    test_counts = np.stack([counts_at_grid(row, fractions) for row in test_scores])
    cal_sizes = np.asarray([len(row) for row in cal_truths])[:, None]
    cal_losses = 1 - np.minimum(cal_counts, cal_sizes) / cal_sizes
    index, certified = calibrate_losses(cal_losses)
    return index, certified, test_counts


def target_values(model, graphs, mode, device):
    values = []
    with torch.no_grad():
        for batch in DataLoader(graphs, batch_size=32, shuffle=False):
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            targets = batch.y if mode == "true" else torch.ones(len(batch.y), dtype=torch.long, device=device)
            values.extend(logits[torch.arange(len(batch.y), device=device), targets].cpu().tolist())
    return np.asarray(values, dtype=float)


def perturbation_rows(a5, model, graphs, scores, fraction, method, target, device):
    rows = []
    for graph, score in zip(graphs, scores):
        selected = sorted(a5.top_set(score, fraction))
        source = int(graph.source_index)
        seed = int.from_bytes(hashlib.sha256(f"{CELL_ID}:{source}:{len(selected)}:20260908".encode()).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        random_sets = [rng.choice(graph.num_nodes, len(selected), replace=False) for _ in range(20)]
        copies = [graph.clone()]
        for indices in [selected] + random_sets:
            clone = graph.clone()
            clone.x[torch.as_tensor(np.asarray(indices), dtype=torch.long)] = 0
            copies.append(clone)
        values = target_values(model, copies, target, device)
        drops = values[0] - values[1:]
        rows.append({
            "cell_id": CELL_ID,
            "method": "ig20" if method == "ig" else method,
            "target": "observed_class" if target == "true" else "positive_class",
            "source_index": source,
            "label": int(graph.y),
            "nominal_fraction": fraction,
            "selected_count": len(selected),
            "random_draws": 20,
            "selected_target_decrease": float(drops[0]),
            "random_mean_target_decrease": float(drops[1:].mean()),
            "selected_minus_random": float(drops[0] - drops[1:].mean()),
        })
    return rows


def target_audit(a5):
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to reproduce the frozen A5 attribution path")
    parts = a5.bxaic_partitions(PROJECT / "data/raw/bxaic/data.csv", PROJECT / "data/raw/bxaic/explanations.sdf", "rings-count")
    with np.load(CACHE, allow_pickle=False) as cache:
        selected = {}
        for split in ("calibration", "test"):
            graph_map = {int(graph.source_index): graph for graph in parts[split]}
            selected[split] = [graph_map[int(index)] for index in cache[f"{split}__source_indices"]]
            a5.validate_selected_cache(cache, selected[split], split)
    checkpoint = a5.CHECKPOINTS / f"{CELL_ID}.pt"
    model = a5.GraphClassifier("gin", parts["fit"][0].x.shape[1]).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["state_dict"])
    model.eval()
    truths = {split: a5.graph_truths(selected[split]) for split in selected}
    labels = np.asarray([int(graph.y) for graph in selected["test"]])
    scores = {method: {target: {} for target in TARGETS} for method in METHODS}
    source_checks = []
    for method in METHODS:
        batch_size = 64 if method == "gradinput" else 32
        for target in TARGETS:
            for split in ("calibration", "test"):
                rows, _, _, sources = a5.attribution_scores(model, selected[split], device, method, target, batch_size)
                expected = np.asarray([int(graph.source_index) for graph in selected[split]])
                source_checks.append(bool(np.array_equal(sources, expected)))
                scores[method][target][split] = rows

    strata_rows = []
    molecule_rows = []
    reproduction = []
    recorded = json.loads((A5_ROOT / "cells" / f"{CELL_ID}.json").read_text(encoding="utf-8"))
    for method in METHODS:
        for target in TARGETS:
            index, certified, status = a5.calibrate(scores[method][target]["calibration"], truths["calibration"])
            fraction = float(a5.FRACTIONS[index])
            oracle_index, oracle_certified, test_counts = oracle_schedule(
                scores[method][target]["calibration"], truths["calibration"],
                scores[method][target]["test"], truths["test"], a5.FRACTIONS,
            )
            perturb = perturbation_rows(a5, model, selected["test"], scores[method][target]["test"], fraction, method, target, device)
            perturb_by_source = {row["source_index"]: row for row in perturb}
            for i, (graph, score, truth) in enumerate(zip(selected["test"], scores[method][target]["test"], truths["test"])):
                chosen = a5.top_set(score, fraction)
                overlap = len(chosen & truth)
                oracle_count = int(test_counts[i, oracle_index])
                p = perturb_by_source[int(graph.source_index)]
                molecule_rows.append({
                    **p,
                    "n_atoms": int(graph.num_nodes),
                    "n_reference": len(truth),
                    "missed_reference_loss": 1 - overlap / len(truth),
                    "retained_fraction": len(chosen) / graph.num_nodes,
                    "oracle_nominal_fraction": float(a5.FRACTIONS[oracle_index]),
                    "oracle_retained_fraction": oracle_count / graph.num_nodes,
                    "oracle_risk": 1 - min(oracle_count, len(truth)) / len(truth),
                    "oracle_excess": (len(chosen) - oracle_count) / graph.num_nodes,
                })
            current = pd.DataFrame([row for row in molecule_rows if row["method"] == ("ig20" if method == "ig" else method) and row["target"] == ("observed_class" if target == "true" else "positive_class")])
            for stratum, mask in (
                ("all_nonempty_reference", np.ones(len(current), dtype=bool)),
                ("positive_label", current["label"].to_numpy() == 1),
                ("negative_label", current["label"].to_numpy() == 0),
            ):
                subset = current.loc[mask]
                strata_rows.append({
                    "surface": "V8_fresh_attribution_matched_A5_exact_tie",
                    "cell_id": CELL_ID,
                    "method": "ig20" if method == "ig" else method,
                    "target": "observed_class" if target == "true" else "positive_class",
                    "stratum": stratum,
                    "n_test": len(subset),
                    "calibration_n_all_labels": len(truths["calibration"]),
                    "calibration_was_not_repeated_within_stratum": True,
                    "selection_status": status,
                    "crc_certified_on_all_label_calibration_pool": certified,
                    "nominal_fraction": fraction,
                    "risk": subset["missed_reference_loss"].mean(),
                    "retained_fraction": subset["retained_fraction"].mean(),
                    "oracle_status": "crc" if oracle_certified else "deterministic_full_set_fallback",
                    "oracle_nominal_fraction": float(a5.FRACTIONS[oracle_index]),
                    "oracle_risk": subset["oracle_risk"].mean(),
                    "oracle_retained_fraction": subset["oracle_retained_fraction"].mean(),
                    "oracle_excess": subset["oracle_excess"].mean(),
                    "replacement": "zero_features",
                    "random_draws_per_molecule": 20,
                    "selected_minus_random": subset["selected_minus_random"].mean(),
                    "selected_minus_random_sd": subset["selected_minus_random"].std(ddof=1),
                    "positive_contrast_fraction": (subset["selected_minus_random"] > 0).mean(),
                })
            expected = recorded["target_audit"][method][target]
            overall = strata_rows[-3]
            reproduction.append({
                "method": "ig20" if method == "ig" else method,
                "target": "observed_class" if target == "true" else "positive_class",
                "fraction_difference_vs_A5": fraction - expected["nominal_fraction"],
                "risk_difference_vs_A5": overall["risk"] - expected["risk"],
                "retained_difference_vs_A5": overall["retained_fraction"] - expected["retained"],
            })
    del model
    torch.cuda.empty_cache()
    return pd.DataFrame(strata_rows), pd.DataFrame(molecule_rows), pd.DataFrame(reproduction), all(source_checks), sha256(checkpoint)


def full_test_audit():
    source_path = PROJECT / "results/molecule_level_ig_seed42/molecules.csv.gz"
    frame = pd.read_csv(source_path)
    frame = frame[(frame["family"] == "bxaic") & (frame["task"] == "rings-count") & frame["n_rationale"].gt(0)].copy()
    oracle_path = PROJECT / "results/reviewer_oracle_crc/task_oracle_all_alpha.csv"
    oracle = pd.read_csv(oracle_path)
    oracle = oracle[(oracle["family"] == "bxaic") & (oracle["task"] == "rings-count") & np.isclose(oracle["alpha"], 0.10)]
    if len(oracle) != 1:
        raise ValueError("expected one rings-count alpha=0.10 oracle row")
    oracle_fraction = float(oracle.iloc[0]["oracle_fraction"])
    oracle_count = np.ceil(oracle_fraction * frame["n_atoms"]).astype(int)
    frame["oracle_retained_fraction"] = oracle_count / frame["n_atoms"]
    frame["oracle_risk"] = 1 - np.minimum(oracle_count, frame["n_rationale"]) / frame["n_rationale"]
    frame["oracle_excess"] = frame["selected_fraction"] - frame["oracle_retained_fraction"]
    rows = []
    for cell_id, cell in frame.groupby("cell_id", sort=True):
        for stratum, subset in (
            ("all_nonempty_reference", cell),
            ("positive_label", cell[cell["label"] == 1]),
            ("negative_label", cell[cell["label"] == 0]),
        ):
            rows.append({
                "surface": "full_test_historical_index_tiebreak",
                "cell_id": cell_id,
                "model": subset["model"].iloc[0],
                "seed": 42,
                "method": "ig20",
                "target": "observed_class",
                "stratum": stratum,
                "n_test": len(subset),
                "calibration_n_all_labels": 4935,
                "calibration_was_not_repeated_within_stratum": True,
                "nominal_fraction": subset["crc_fraction"].iloc[0],
                "risk": subset["miss_loss"].mean(),
                "retained_fraction": subset["selected_fraction"].mean(),
                "oracle_nominal_fraction": oracle_fraction,
                "oracle_risk": subset["oracle_risk"].mean(),
                "oracle_retained_fraction": subset["oracle_retained_fraction"].mean(),
                "oracle_excess": subset["oracle_excess"].mean(),
                "replacement": "zero_features",
                "random_draws_per_molecule": 1,
                "selected_minus_random": subset["fidelity_advantage"].mean(),
                "selected_minus_random_sd": subset["fidelity_advantage"].std(ddof=1),
                "positive_contrast_fraction": (subset["fidelity_advantage"] > 0).mean(),
            })
    cells = pd.DataFrame(rows)
    macro = cells.groupby("stratum", as_index=False).agg(
        cell_count=("cell_id", "size"),
        n_test_per_cell=("n_test", "first"),
        risk=("risk", "mean"),
        retained_fraction=("retained_fraction", "mean"),
        oracle_risk=("oracle_risk", "mean"),
        oracle_retained_fraction=("oracle_retained_fraction", "mean"),
        oracle_excess=("oracle_excess", "mean"),
        selected_minus_random=("selected_minus_random", "mean"),
        positive_contrast_fraction=("positive_contrast_fraction", "mean"),
    )
    macro.insert(0, "surface", "full_test_historical_index_tiebreak")
    macro["aggregation"] = "equal mean across two seed-42 model cells; no molecule pooling for effect summaries"
    return cells, macro, sha256(source_path), sha256(oracle_path)


def contrasts(target_rows, full_macro):
    rows = []
    for (method, target), group in target_rows.groupby(["method", "target"]):
        indexed = group.set_index("stratum")
        for metric in ("risk", "retained_fraction", "oracle_excess", "selected_minus_random"):
            rows.append({
                "surface": "V8_fresh_attribution_matched_A5_exact_tie",
                "method": method,
                "target": target,
                "contrast": "positive_label_minus_negative_label",
                "metric": metric,
                "difference": indexed.loc["positive_label", metric] - indexed.loc["negative_label", metric],
            })
    for (method, stratum), group in target_rows.groupby(["method", "stratum"]):
        indexed = group.set_index("target")
        for metric in ("risk", "retained_fraction", "oracle_excess", "selected_minus_random"):
            rows.append({
                "surface": "V8_fresh_attribution_matched_A5_exact_tie",
                "method": method,
                "target": "positive_class_minus_observed_class",
                "contrast": stratum,
                "metric": metric,
                "difference": indexed.loc["positive_class", metric] - indexed.loc["observed_class", metric],
            })
    indexed = full_macro.set_index("stratum")
    for metric in ("risk", "retained_fraction", "oracle_excess", "selected_minus_random"):
        rows.append({
            "surface": "full_test_historical_index_tiebreak",
            "method": "ig20",
            "target": "observed_class",
            "contrast": "positive_label_minus_negative_label",
            "metric": metric,
            "difference": indexed.loc["positive_label", metric] - indexed.loc["negative_label", metric],
        })
    return pd.DataFrame(rows)


def write_manifest():
    rows = []
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "output_manifest.csv":
            rows.append({"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    with (OUT / "output_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("file", "bytes", "sha256"))
        writer.writeheader()
        writer.writerows(rows)


def main():
    a5 = load_a5()
    full_cells, full_macro, molecule_hash, oracle_hash = full_test_audit()
    target_rows, molecule_rows, reproduction, sources_ok, checkpoint_hash = target_audit(a5)
    contrast_rows = contrasts(target_rows, full_macro)

    manuscript_rows = pd.concat([
        full_cells[["surface", "cell_id", "model", "method", "target", "stratum", "n_test", "nominal_fraction", "risk", "retained_fraction", "oracle_excess", "random_draws_per_molecule", "selected_minus_random"]],
        target_rows[target_rows["method"] == "ig20"].assign(model="gin")[["surface", "cell_id", "model", "method", "target", "stratum", "n_test", "nominal_fraction", "risk", "retained_fraction", "oracle_excess", "random_draws_per_molecule", "selected_minus_random"]],
    ], ignore_index=True)

    full_cells.to_csv(OUT / "full_test_observed_class_by_model.csv", index=False)
    full_macro.to_csv(OUT / "full_test_observed_class_macro.csv", index=False)
    target_rows.to_csv(OUT / "matched_target_class_strata.csv", index=False)
    molecule_rows.to_csv(OUT / "matched_target_molecule_metrics.csv", index=False)
    contrast_rows.to_csv(OUT / "class_and_target_contrasts.csv", index=False)
    reproduction.to_csv(OUT / "a5_reproduction_check.csv", index=False)
    manuscript_rows.to_csv(OUT / "manuscript_ready_stratified_table.csv", index=False)

    counts = full_cells.pivot(index="cell_id", columns="stratum", values="n_test")
    full_counts_ok = bool(
        counts["all_nonempty_reference"].eq(4933).all()
        and counts["positive_label"].eq(1503).all()
        and counts["negative_label"].eq(3430).all()
    )
    reproduction_max = float(reproduction[["fraction_difference_vs_A5", "risk_difference_vs_A5", "retained_difference_vs_A5"]].abs().to_numpy().max())
    fractions_exact = bool(reproduction["fraction_difference_vs_A5"].abs().le(1e-12).all())
    replay_within_recorded_scale = reproduction_max <= 0.002
    core_pass = bool(full_counts_ok and sources_ok and fractions_exact and replay_within_recorded_scale and target_rows["oracle_excess"].ge(-1e-12).all() and full_cells["oracle_excess"].ge(-1e-12).all())
    qa = {
        "status": "PASS_WITH_RECOMPUTATION_DRIFT" if core_pass and reproduction_max > 1e-12 else "PASS" if core_pass else "FAIL",
        "full_test_counts_match_4933_1503_3430": full_counts_ok,
        "matched_source_order_verified": sources_ok,
        "matched_test_count": int(target_rows[target_rows["stratum"] == "all_nonempty_reference"]["n_test"].iloc[0]),
        "matched_positive_count": int(target_rows[target_rows["stratum"] == "positive_label"]["n_test"].iloc[0]),
        "matched_negative_count": int(target_rows[target_rows["stratum"] == "negative_label"]["n_test"].iloc[0]),
        "a5_unstratified_max_abs_reproduction_difference": reproduction_max,
        "a5_nominal_fractions_reproduced_exactly": fractions_exact,
        "a5_risk_and_retained_differences_within_0.002": replay_within_recorded_scale,
        "a5_replay_interpretation": f"The same four nominal fractions were recovered. Tie-inclusive test set metrics differed by at most {reproduction_max:.6f}, consistent with A5's recorded non-bitwise exact-tie replay limitation; the new class rows are bound to this same-run calculation rather than substituted with historical aggregate values.",
        "matched_surface_is_independent_v8_fresh_attribution_rerun": True,
        "matched_surface_is_not_a_new_independent_molecule_sample": True,
        "no_within_stratum_recalibration": True,
        "all_oracle_excess_nonnegative": bool(target_rows["oracle_excess"].ge(-1e-12).all() and full_cells["oracle_excess"].ge(-1e-12).all()),
        "all_reported_core_metrics_finite": bool(np.isfinite(pd.concat([
            full_cells[["risk", "retained_fraction", "oracle_excess", "selected_minus_random"]],
            target_rows[["risk", "retained_fraction", "oracle_excess", "selected_minus_random"]],
        ]).to_numpy()).all()),
    }
    write_json(OUT / "QA.json", qa)
    write_json(OUT / "environment.json", {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    })
    write_json(OUT / "input_manifest.json", {
        "molecule_level_ig_seed42_molecules_sha256": molecule_hash,
        "task_oracle_all_alpha_sha256": oracle_hash,
        "a5_run_script_sha256": sha256(A5_ROOT / "run.py"),
        "a5_cell_record_sha256": sha256(A5_ROOT / "cells" / f"{CELL_ID}.json"),
        "a5_score_cache_sha256": sha256(CACHE),
        "checkpoint_sha256": checkpoint_hash,
        "bxaic_data_csv_sha256": sha256(PROJECT / "data/raw/bxaic/data.csv"),
        "bxaic_explanations_sdf_sha256": sha256(PROJECT / "data/raw/bxaic/explanations.sdf"),
    })

    f = full_macro.set_index("stratum")
    g = full_cells.set_index(["model", "stratum"])
    t = target_rows.set_index(["method", "target", "stratum"])
    report = f"""# Rings-count class-stratified result

Status: **{qa['status']}**. The two analysis surfaces below are deliberately separate because their molecule sets, tie policies, random comparators and oracle schedules differ.

## Full official test surface

The historical observed-class IG20 analysis contains 4,933 nonempty-reference test molecules per seed-42 model cell: 1,503 positive-label and 3,430 negative-label molecules. Calibration remained fixed on all 4,935 nonempty-reference calibration molecules; no class-stratum threshold was selected. Equal means across GIN and GCN give:

| Stratum | n per cell | Risk | Retained | Oracle excess | Selected minus one frozen random |
|---|---:|---:|---:|---:|---:|
| Positive | 1,503 | {f.loc['positive_label','risk']:.6f} | {f.loc['positive_label','retained_fraction']:.6f} | {f.loc['positive_label','oracle_excess']:.6f} | {f.loc['positive_label','selected_minus_random']:.6f} |
| Negative | 3,430 | {f.loc['negative_label','risk']:.6f} | {f.loc['negative_label','retained_fraction']:.6f} | {f.loc['negative_label','oracle_excess']:.6f} | {f.loc['negative_label','selected_minus_random']:.6f} |

The GIN/GCN-specific rows remain in `full_test_observed_class_by_model.csv`; the aggregate does not hide the GIN adverse fidelity cell.

In the GIN cell that supplied the repeated fidelity counterexample, the observed-class risk was {g.loc[('gin','positive_label'),'risk']:.6f} in positives versus {g.loc[('gin','negative_label'),'risk']:.6f} in negatives, and selected-minus-random was {g.loc[('gin','positive_label'),'selected_minus_random']:.6f} versus {g.loc[('gin','negative_label'),'selected_minus_random']:.6f}. The GCN contrast remained positive in both strata. Thus the adverse GIN operating result is concentrated in the negative-label stratum rather than reproduced uniformly across classes.

## Matched target-audit surface

The independent V8 fresh-attribution rerun uses the frozen A5 rings-count GIN seed-42 identities and checkpoint; it is not a new independent molecule sample. It contains {qa['matched_positive_count']} positives and {qa['matched_negative_count']} negatives among the same 100 test molecules. For each method and target, calibration used all 100 calibration molecules once, after which the fraction was frozen for both strata. Zero-feature replacement used 20 deterministic equal-size random masks per molecule.

| Method / target | Stratum | n | Risk | Retained | Oracle excess | Selected minus random |
|---|---|---:|---:|---:|---:|---:|
| IG20 / observed | Positive | {int(t.loc[('ig20','observed_class','positive_label'),'n_test'])} | {t.loc[('ig20','observed_class','positive_label'),'risk']:.6f} | {t.loc[('ig20','observed_class','positive_label'),'retained_fraction']:.6f} | {t.loc[('ig20','observed_class','positive_label'),'oracle_excess']:.6f} | {t.loc[('ig20','observed_class','positive_label'),'selected_minus_random']:.6f} |
| IG20 / observed | Negative | {int(t.loc[('ig20','observed_class','negative_label'),'n_test'])} | {t.loc[('ig20','observed_class','negative_label'),'risk']:.6f} | {t.loc[('ig20','observed_class','negative_label'),'retained_fraction']:.6f} | {t.loc[('ig20','observed_class','negative_label'),'oracle_excess']:.6f} | {t.loc[('ig20','observed_class','negative_label'),'selected_minus_random']:.6f} |
| IG20 / positive | Positive | {int(t.loc[('ig20','positive_class','positive_label'),'n_test'])} | {t.loc[('ig20','positive_class','positive_label'),'risk']:.6f} | {t.loc[('ig20','positive_class','positive_label'),'retained_fraction']:.6f} | {t.loc[('ig20','positive_class','positive_label'),'oracle_excess']:.6f} | {t.loc[('ig20','positive_class','positive_label'),'selected_minus_random']:.6f} |
| IG20 / positive | Negative | {int(t.loc[('ig20','positive_class','negative_label'),'n_test'])} | {t.loc[('ig20','positive_class','negative_label'),'risk']:.6f} | {t.loc[('ig20','positive_class','negative_label'),'retained_fraction']:.6f} | {t.loc[('ig20','positive_class','negative_label'),'oracle_excess']:.6f} | {t.loc[('ig20','positive_class','negative_label'),'selected_minus_random']:.6f} |

The matched IG20 decomposition gives the same direction for the observed-class fidelity contrast: {t.loc[('ig20','observed_class','positive_label'),'selected_minus_random']:.6f} in positives and {t.loc[('ig20','observed_class','negative_label'),'selected_minus_random']:.6f} in negatives. Retargeting to the positive-class logit changed the globally calibrated fraction from {t.loc[('ig20','observed_class','all_nonempty_reference'),'nominal_fraction']:.2f} to {t.loc[('ig20','positive_class','all_nonempty_reference'),'nominal_fraction']:.2f}; it did not provide a class-independent rescue. For positive-label molecules the two target logits are definitionally identical, so their changed operating results arise from the globally recalibrated fraction, not from a different within-molecule attribution target.

## Claim boundary

The result is a descriptive conditional audit of one benchmark task, not a class-conditional CRC guarantee. The positive-class and observed-class rankings control different logits on negative molecules; the supplied ring mask is a formal benchmark reference and is not evidence that all ring atoms support the negative prediction. Feature-zeroing is an off-support computational intervention rather than a chemical deletion. No population, cross-family, causal or medicinal-chemistry claim is supported.

The A5 same-run reproduction recovered all four nominal fractions exactly. Tie-inclusive risk/retained values differed from the historical A5 record by at most {reproduction_max:.6f}, which is retained as the previously documented non-bitwise exact-tie replay limitation rather than silently overwritten.
"""
    (OUT / "evidence_update.md").write_text(report, encoding="utf-8")
    write_manifest()
    print(json.dumps({"status": qa["status"], "full": full_macro.to_dict("records"), "matched": target_rows.to_dict("records")}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

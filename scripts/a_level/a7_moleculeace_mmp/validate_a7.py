import ast
import hashlib
import json
import math
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.stats import hypergeom, spearmanr


ROOT = Path(__file__).resolve().parent
TARGETS = [
    "CHEMBL204_Ki",
    "CHEMBL214_Ki",
    "CHEMBL228_Ki",
    "CHEMBL233_Ki",
    "CHEMBL234_Ki",
    "CHEMBL235_EC50",
    "CHEMBL236_Ki",
    "CHEMBL237_Ki",
    "CHEMBL244_Ki",
    "CHEMBL264_Ki",
    "CHEMBL4792_Ki",
]
EXPECTED_COUNTS = {
    "CHEMBL204_Ki": (59, 40),
    "CHEMBL214_Ki": (47, 31),
    "CHEMBL228_Ki": (51, 34),
    "CHEMBL233_Ki": (75, 52),
    "CHEMBL234_Ki": (64, 35),
    "CHEMBL235_EC50": (30, 29),
    "CHEMBL236_Ki": (33, 32),
    "CHEMBL237_Ki": (37, 43),
    "CHEMBL244_Ki": (81, 109),
    "CHEMBL264_Ki": (47, 27),
    "CHEMBL4792_Ki": (35, 44),
}
MANIFEST = Path(r"<split-semantics-directory>\split_manifest.csv")
EXPECTED_CONTRACT_HASH = "C466C755EB910745DC8E5748E9F4E85AE8EBB4925732C484C548D0D9E0FBAB4E"
MMP_BOND_PATTERN = Chem.MolFromSmarts("[#6+0;!$(*=,#[!#6])]!@!=!#[*]")
FRACTIONS = np.linspace(0.0, 1.0, 101)
PROXIES = {
    "main_variable_plus_attachment_symmetry_closed": "main_mask",
    "variable_only_symmetry_closed": "variable_mask",
    "one_hop_expanded_symmetry_closed": "expanded_mask",
}


checks = []


def add(name, result, detail):
    checks.append({"name": name, "result": result, "detail": str(detail)})


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def exact_dev_groups(frame, seed):
    sizes = frame.groupby("group_id").size().to_dict()
    ordered = sorted(sizes, key=lambda value: hashlib.sha256(f"{seed}|{value}".encode()).hexdigest())
    reachable = {0: ()}
    for group in ordered:
        additions = {}
        for count, selected in list(reachable.items()):
            proposed = count + sizes[group]
            if proposed not in reachable and proposed not in additions:
                additions[proposed] = selected + (group,)
        reachable.update(additions)
    closest = min((count for count in reachable if 0 < count < len(frame)), key=lambda count: (abs(count - 0.2 * len(frame)), count))
    return set(reachable[closest]), sizes


def row_scores(row):
    return np.concatenate(
        [
            np.asarray([float(value) for value in str(row.scores_a).split(";")]),
            np.asarray([float(value) for value in str(row.scores_b).split(";")]),
        ]
    )


def row_proxy(row, column):
    left = indices(getattr(row, f"{column}_a"))
    right = {index + int(row.n_atoms_a) for index in indices(getattr(row, f"{column}_b"))}
    return left | right


def selected_atoms(scores, fraction):
    if fraction == 0:
        return set()
    count = int(np.ceil(fraction * len(scores)))
    threshold = np.partition(scores, len(scores) - count)[len(scores) - count]
    return set(np.flatnonzero(scores >= threshold))


def independent_loss_table(frame, proxy_column):
    output = np.empty((len(frame), len(FRACTIONS)), dtype=float)
    for row_index, row in enumerate(frame.itertuples(index=False)):
        scores = row_scores(row)
        proxy = row_proxy(row, proxy_column)
        for fraction_index, fraction in enumerate(FRACTIONS):
            output[row_index, fraction_index] = 1.0 - len(selected_atoms(scores, fraction) & proxy) / len(proxy)
    return output


def independent_calibrate(losses):
    corrected = (len(losses) * losses.mean(axis=0) + 1.0) / (len(losses) + 1)
    feasible = np.flatnonzero(corrected <= 0.1)
    if not len(feasible):
        return None
    index = int(feasible[0])
    return float(FRACTIONS[index]), float(losses[:, index].mean()), float(corrected[index])


def independent_metrics(frame, proxy_column, fraction, evaluator):
    values = []
    for row in frame.itertuples(index=False):
        scores = row_scores(row)
        proxy = row_proxy(row, proxy_column)
        if evaluator == "oracle":
            values.append((0.0, len(proxy) / len(scores), 1.0, 1.0))
            continue
        count = int(np.ceil(fraction * len(scores)))
        if evaluator == "random":
            expected_iou = sum(
                hypergeom.pmf(overlap, len(scores), len(proxy), count) * overlap / (count + len(proxy) - overlap)
                for overlap in range(max(0, count + len(proxy) - len(scores)), min(count, len(proxy)) + 1)
            )
            values.append((1.0 - count / len(scores), count / len(scores), len(proxy) / len(scores) if count else 0.0, expected_iou))
            continue
        selected = selected_atoms(scores, fraction)
        overlap = len(selected & proxy)
        values.append(
            (
                1.0 - overlap / len(proxy),
                len(selected) / len(scores),
                overlap / len(selected) if selected else 0.0,
                overlap / len(selected | proxy),
            )
        )
    return np.asarray(values, dtype=float)


def independently_recompute_crc(attributions):
    result_rows = []
    feasibility_rows = []
    for (target, seed, method), frame in attributions.groupby(["dataset", "seed", "method"]):
        calibration = frame[(frame.pair_type == "sar_cliff_proxy") & (frame.split == "calibration")]
        test = frame[(frame.pair_type == "sar_cliff_proxy") & (frame.split == "test")]
        control = frame[frame.pair_type == "matched_noncliff_control"]
        for proxy, column in PROXIES.items():
            losses = independent_loss_table(calibration, column)
            pair_selection = independent_calibrate(losses)
            components = calibration.component_id.astype(str).to_numpy()
            grouped_losses = np.vstack([losses[components == component].mean(axis=0) for component in sorted(set(components))])
            component_selection = independent_calibrate(grouped_losses) if len(grouped_losses) >= 9 else None
            feasibility_rows.append(
                {
                    "dataset": target,
                    "seed": int(seed),
                    "method": method,
                    "proxy": proxy,
                    "n_calibration_components": len(grouped_losses),
                    "pair_crc_descriptive_fraction": pair_selection[0],
                    "pair_calibration_empirical_risk": pair_selection[1],
                    "pair_calibration_corrected_risk": pair_selection[2],
                    "component_crc_fraction": component_selection[0] if component_selection else np.nan,
                    "component_calibration_empirical_risk": component_selection[1] if component_selection else np.nan,
                    "component_calibration_corrected_risk": component_selection[2] if component_selection else np.nan,
                }
            )
            policies = [
                ("pair_crc_descriptive", pair_selection[0], "deterministic", pair_selection[1:], "pairs"),
                ("fixed_20_percent", 0.2, "deterministic", (np.nan, np.nan), "none"),
                ("fixed_50_percent", 0.5, "deterministic", (np.nan, np.nan), "none"),
                ("oracle_proxy", pair_selection[0], "oracle", (np.nan, np.nan), "none"),
                ("random_same_nominal_size", pair_selection[0], "random", (np.nan, np.nan), "none"),
            ]
            if component_selection:
                policies.append(("component_crc_cluster_stress_test", component_selection[0], "deterministic", component_selection[1:], "components"))
            for policy, fraction, evaluator, cal_info, calibration_unit in policies:
                for evaluation_split, subset in [("calibration", calibration), ("test", test)]:
                    strata = [("all", subset)]
                    if evaluation_split == "test":
                        strata += [("correct_direction", subset[subset.direction_correct]), ("direction_error", subset[~subset.direction_correct])]
                    for stratum, stratum_frame in strata:
                        if not len(stratum_frame):
                            continue
                        values = independent_metrics(stratum_frame, column, fraction, evaluator)
                        component_ids = stratum_frame.component_id.astype(str).to_numpy()
                        component_values = np.vstack([values[component_ids == component].mean(axis=0) for component in sorted(set(component_ids))])
                        for aggregation, aggregated in [("pair_weighted", values), ("component_weighted", component_values)]:
                            result_rows.append(
                                {
                                    "dataset": target,
                                    "seed": int(seed),
                                    "method": method,
                                    "proxy": proxy,
                                    "evaluation_split": evaluation_split,
                                    "stratum": stratum,
                                    "policy": policy,
                                    "aggregation": aggregation,
                                    "n_pairs": len(values),
                                    "n_components": len(set(component_ids)),
                                    "nominal_fraction": fraction,
                                    "risk": float(aggregated[:, 0].mean()),
                                    "retained_fraction": float(aggregated[:, 1].mean()),
                                    "precision": float(aggregated[:, 2].mean()),
                                    "iou": float(aggregated[:, 3].mean()),
                                    "calibration_unit": calibration_unit,
                                    "calibration_empirical_risk": cal_info[0],
                                    "calibration_corrected_risk": cal_info[1],
                                }
                            )
                if policy in {"pair_crc_descriptive", "component_crc_cluster_stress_test"} and len(control):
                    values = independent_metrics(control, column, fraction, evaluator)
                    component_ids = control.component_id.astype(str).to_numpy()
                    component_values = np.vstack([values[component_ids == component].mean(axis=0) for component in sorted(set(component_ids))])
                    for aggregation, aggregated in [("pair_weighted", values), ("component_weighted", component_values)]:
                        result_rows.append(
                            {
                                "dataset": target,
                                "seed": int(seed),
                                "method": method,
                                "proxy": proxy,
                                "evaluation_split": "matched_noncliff_control",
                                "stratum": "descriptive_only",
                                "policy": policy,
                                "aggregation": aggregation,
                                "n_pairs": len(values),
                                "n_components": len(set(component_ids)),
                                "nominal_fraction": fraction,
                                "risk": float(aggregated[:, 0].mean()),
                                "retained_fraction": float(aggregated[:, 1].mean()),
                                "precision": float(aggregated[:, 2].mean()),
                                "iou": float(aggregated[:, 3].mean()),
                                "calibration_unit": calibration_unit,
                                "calibration_empirical_risk": cal_info[0],
                                "calibration_corrected_risk": cal_info[1],
                            }
                        )
    return pd.DataFrame(result_rows), pd.DataFrame(feasibility_rows)


def independently_recompute_macro(results):
    rows = []
    metrics = ["risk", "retained_fraction", "precision", "iou"]
    selected = results[(results.evaluation_split == "test") & (results.stratum == "all")]
    rng = np.random.default_rng(20260904)
    keys = ["method", "proxy", "policy", "aggregation"]
    for values, frame in selected.groupby(keys):
        target_values = frame.groupby("dataset")[metrics].mean().to_numpy(float)
        samples = target_values[rng.integers(0, len(target_values), size=(5000, len(target_values)))].mean(axis=1)
        for index, metric in enumerate(metrics):
            rows.append(
                {
                    **dict(zip(keys, values)),
                    "metric": metric,
                    "target_macro_mean": float(target_values[:, index].mean()),
                    "bootstrap_ci_low": float(np.quantile(samples[:, index], 0.025)),
                    "bootstrap_ci_high": float(np.quantile(samples[:, index], 0.975)),
                    "n_targets": len(target_values),
                }
            )
    return pd.DataFrame(rows)


def indices(value):
    if pd.isna(value) or value == "":
        return set()
    return {int(item) for item in str(value).split(";")}


def symmetry_orbits(mol):
    matches = mol.GetSubstructMatches(mol, uniquify=False, useChirality=True, maxMatches=100000)
    orbits = [{index} for index in range(mol.GetNumAtoms())]
    for match in matches:
        for source, target in enumerate(match):
            orbits[source].add(target)
    return orbits, len(matches) == 100000


def closure(raw, orbits):
    return set().union(*(orbits[index] for index in raw)) if raw else set()


def validate_masks(frame, label):
    failures = Counter()
    molecule_cache = {}
    orbit_cache = {}
    eligible_cache = {}
    for row in frame.itertuples(index=False):
        for suffix in ["a", "b"]:
            data = row._asdict()
            smiles = data[f"smiles_{suffix}"]
            if smiles not in molecule_cache:
                molecule_cache[smiles] = Chem.MolFromSmiles(smiles)
            mol = molecule_cache[smiles]
            if mol is None:
                failures["invalid_smiles"] += 1
                continue
            if smiles not in eligible_cache:
                eligible_cache[smiles] = {
                    mol.GetBondBetweenAtoms(match[0], match[1]).GetIdx()
                    for match in mol.GetSubstructMatches(MMP_BOND_PATTERN, uniquify=True)
                }
            n_atoms = int(data[f"n_atoms_{suffix}"])
            failures["atom_count"] += int(mol.GetNumAtoms() != n_atoms)
            variable_raw = indices(data[f"variable_mask_raw_{suffix}"])
            main_raw = indices(data[f"main_mask_raw_{suffix}"])
            variable = indices(data[f"variable_mask_{suffix}"])
            main = indices(data[f"main_mask_{suffix}"])
            expanded = indices(data[f"expanded_mask_{suffix}"])
            attachment = int(data[f"core_attachment_{suffix}"])
            cut = indices(data[f"cut_bond_{suffix}"])
            all_masks = [variable_raw, main_raw, variable, main, expanded, cut, {attachment}]
            failures["out_of_bounds"] += int(any(any(index < 0 or index >= n_atoms for index in mask) for mask in all_masks))
            failures["empty_variable"] += int(not variable_raw)
            failures["main_raw_definition"] += int(main_raw != variable_raw | {attachment})
            failures["attachment_in_variable"] += int(attachment in variable_raw)
            failures["fragment_size"] += int(len(variable_raw) != int(data[f"variable_heavy_atoms_{suffix}"]))
            failures["core_plus_variable_size"] += int(
                int(data["core_heavy_atoms"]) + int(data[f"variable_heavy_atoms_{suffix}"]) != n_atoms
            )
            failures["cut_size"] += int(len(cut) != 2)
            if len(cut) == 2 and all(0 <= index < n_atoms for index in cut):
                left, right = sorted(cut)
                bond = mol.GetBondBetweenAtoms(left, right)
                failures["cut_missing"] += int(bond is None)
                if bond is not None:
                    failures["cut_not_single_nonring"] += int(
                        bond.GetBondType() != Chem.BondType.SINGLE or bond.IsInRing()
                    )
                    failures["cut_not_default_rdmmpa_smarts"] += int(bond.GetIdx() not in eligible_cache[smiles])
                failures["attachment_not_cut_endpoint"] += int(attachment not in cut)
                other = (cut - {attachment})
                failures["variable_not_cut_endpoint"] += int(len(other) != 1 or not other <= variable_raw)
            if smiles not in orbit_cache:
                orbit_cache[smiles] = symmetry_orbits(mol)
            orbits, truncated = orbit_cache[smiles]
            failures["automorphism_truncated"] += int(truncated)
            expanded_raw = set(main_raw)
            for index in main_raw:
                expanded_raw.update(atom.GetIdx() for atom in mol.GetAtomWithIdx(index).GetNeighbors())
            failures["variable_closure"] += int(variable != closure(variable_raw, orbits))
            failures["main_closure"] += int(main != closure(main_raw, orbits))
            failures["expanded_closure"] += int(expanded != closure(expanded_raw, orbits))
            failures["mask_nesting"] += int(not variable <= main <= expanded)
    total = sum(failures.values())
    add(f"{label}: atom mapping and three proxy masks", "PASS" if total == 0 else "FAIL", dict(failures))
    return total == 0


def validate_pretrain():
    contract_hash = sha256(ROOT / "run_contract.json")
    add("Frozen run contract hash", "PASS" if contract_hash == EXPECTED_CONTRACT_HASH else "FAIL", contract_hash)
    ast_failures = []
    ast_hashes = {}
    for name in ["gine_model.py", "train_gine.py", "attribute_and_crc.py"]:
        path = ROOT / name
        try:
            ast.parse(path.read_text(encoding="utf-8"))
            ast_hashes[name] = sha256(path)
        except SyntaxError:
            ast_failures.append(name)
    add("Model and attribution AST", "PASS" if not ast_failures else "FAIL", {"hashes": ast_hashes, "failures": ast_failures})
    runtime = json.loads((ROOT / "contract_amendment_02_runtime_import_order.json").read_text(encoding="utf-8"))
    runtime_ok = runtime["runtime_import_smoke"]["status"] == "PASS" and runtime["cpu_end_to_end_smoke"]["status"] == "PASS_NOT_ANALYTIC_EVIDENCE"
    add("Runtime import amendment", "PASS" if runtime_ok else "FAIL", f"formal_cells={runtime['formal_run_contract_unchanged']['cells']}; code hashes superseded by cluster amendment")
    statistical_amendment = json.loads((ROOT / "contract_amendment_03_statistical_and_eligibility_boundaries.json").read_text(encoding="utf-8"))
    postrun_amendment = json.loads((ROOT / "contract_amendment_04_numerical_lineage.json").read_text(encoding="utf-8"))
    summary_amendment = json.loads((ROOT / "contract_amendment_05_predictive_summary.json").read_text(encoding="utf-8"))
    deterministic_amendment = json.loads((ROOT / "contract_amendment_06_cpu_deterministic_attribution.json").read_text(encoding="utf-8"))
    superseded = {"attribute_and_crc.py", "validate_a7.py"}
    retained_hashes_ok = all(sha256(ROOT / name) == expected for name, expected in statistical_amendment["code_sha256"].items() if name not in superseded)
    amendment_ok = statistical_amendment["status"] == "FROZEN_PRE_FORMAL_RUN" and retained_hashes_ok and statistical_amendment["formal_training"]["cells"] == 33 and statistical_amendment["statistical_claim"] == "descriptive finite-pool clustered stress test only"
    add("Pre-formal statistical and eligibility amendment", "PASS" if amendment_ok else "FAIL", f"retained_hashes_match={retained_hashes_ok}, claim={statistical_amendment['statistical_claim']}")
    transition_ok = postrun_amendment["status"] == "FROZEN_POST_RUN_NUMERICAL_LINEAGE_CORRECTION"
    for name in superseded:
        transition_ok = transition_ok and postrun_amendment["prior_code_sha256"][name] == statistical_amendment["code_sha256"][name]
    transition_ok = transition_ok and postrun_amendment["direction_strata_authority"] == "predictions.csv" and postrun_amendment["numerical_tie_atol"] == 1e-6
    add("Post-run numerical-lineage amendment", "PASS" if transition_ok else "FAIL", f"hash_transition={transition_ok}, direction_source={postrun_amendment['direction_strata_authority']}")
    summary_transition_ok = summary_amendment["status"] == "FROZEN_POST_RUN_PREDICTIVE_SUMMARY_ADDITION"
    for name in superseded:
        summary_transition_ok = summary_transition_ok and summary_amendment["prior_code_sha256"][name] == postrun_amendment["current_code_sha256"][name]
    add("Post-run predictive-summary amendment", "PASS" if summary_transition_ok else "FAIL", f"hash_transition={summary_transition_ok}, model metrics unchanged")
    deterministic_transition_ok = deterministic_amendment["status"] == "FROZEN_POST_RUN_CPU_DETERMINISTIC_ATTRIBUTION"
    for name in superseded:
        deterministic_transition_ok = deterministic_transition_ok and deterministic_amendment["prior_code_sha256"][name] == summary_amendment["current_code_sha256"][name]
        deterministic_transition_ok = deterministic_transition_ok and sha256(ROOT / name) == deterministic_amendment["current_code_sha256"][name]
    for name, expected in deterministic_amendment["supporting_code_sha256"].items():
        deterministic_transition_ok = deterministic_transition_ok and sha256(ROOT / name) == expected
    deterministic_transition_ok = deterministic_transition_ok and deterministic_amendment["formal_attribution"]["device"] == "cpu" and deterministic_amendment["formal_attribution"]["threads"] == 1
    add("Post-run CPU-deterministic attribution amendment", "PASS" if deterministic_transition_ok else "FAIL", f"hash_transition={deterministic_transition_ok}, device=cpu, threads=1")
    pairs = pd.read_csv(ROOT / "mmp_pairs.csv")
    candidates = pd.read_csv(ROOT / "mmp_pair_candidates.csv")
    counts = pd.read_csv(ROOT / "mmp_pair_counts.csv")
    exclusions = pd.read_csv(ROOT / "mmp_mapping_exclusions.csv")
    audit = json.loads((ROOT / "mmp_mapping_audit.json").read_text(encoding="utf-8"))
    amendment = json.loads((ROOT / "contract_amendment_01_pairs.json").read_text(encoding="utf-8"))
    actual = {
        target: (
            len(pairs[(pairs.dataset == target) & (pairs.split == "calibration")]),
            len(pairs[(pairs.dataset == target) & (pairs.split == "test")]),
        )
        for target in TARGETS
    }
    add("MMP frozen target/split counts", "PASS" if actual == EXPECTED_COUNTS else "FAIL", actual)
    add("MMP total and split counts", "PASS" if (len(pairs), sum(pairs.split == "calibration"), sum(pairs.split == "test")) == (1035, 559, 476) else "FAIL", f"total={len(pairs)}, calibration={sum(pairs.split == 'calibration')}, test={sum(pairs.split == 'test')}")
    add("MMP candidate count", "PASS" if len(candidates) == 2853 else "FAIL", len(candidates))
    add("MMP every target/split >=20", "PASS" if min(value for pair in actual.values() for value in pair) >= 20 else "FAIL", min(value for pair in actual.values() for value in pair))
    activity_ok = np.allclose(pairs.measured_abs_delta, np.abs(pairs.activity_a - pairs.activity_b), atol=1e-12) and (pairs.measured_abs_delta > 1.0).all()
    add("MMP cliff activity definition", "PASS" if activity_ok else "FAIL", "|measured delta| > 1.0 and internally consistent")
    candidate_keys = set(zip(candidates.dataset, candidates.split, candidates.candidate_pair_id))
    pair_keys = set(zip(pairs.dataset, pairs.split, pairs.candidate_pair_id))
    add("Selected pairs trace to candidates", "PASS" if pair_keys <= candidate_keys else "FAIL", f"missing={len(pair_keys - candidate_keys)}")
    reuse = 0
    overlap = 0
    for target in TARGETS:
        for split in ["calibration", "test"]:
            group = pairs[(pairs.dataset == target) & (pairs.split == split)]
            molecules = list(group.smiles_a) + list(group.smiles_b)
            reuse += len(molecules) - len(set(molecules))
        cal = pairs[(pairs.dataset == target) & (pairs.split == "calibration")]
        test = pairs[(pairs.dataset == target) & (pairs.split == "test")]
        overlap += len((set(cal.smiles_a) | set(cal.smiles_b)) & (set(test.smiles_a) | set(test.smiles_b)))
    add("MMP within-split molecule disjointness", "PASS" if reuse == 0 else "FAIL", f"reuse={reuse}")
    add("Calibration/test molecule disjointness", "PASS" if overlap == 0 else "FAIL", f"overlap={overlap}")
    manifest_groups = pd.read_csv(MANIFEST, usecols=["dataset", "environment", "split", "canonical_smiles", "group_id"])
    manifest_groups = manifest_groups[(manifest_groups.environment == "mmp_series") & manifest_groups.dataset.isin(TARGETS)]
    group_lookup = {
        (row.dataset, row.split, row.canonical_smiles): str(row.group_id)
        for row in manifest_groups.itertuples(index=False)
    }
    component_failures = 0
    for row in pairs.itertuples(index=False):
        expected_a = group_lookup.get((row.dataset, row.split, row.smiles_a))
        expected_b = group_lookup.get((row.dataset, row.split, row.smiles_b))
        component_failures += int(expected_a is None or expected_b is None or expected_a != expected_b or str(row.component_id) != expected_a)
    add("Pair component_id lineage", "PASS" if component_failures == 0 else "FAIL", f"failures={component_failures}")
    observed_sizes = pairs.groupby(["dataset", "split", "component_id"]).size().rename("n_pairs").reset_index()
    frozen_sizes = pd.read_csv(ROOT / "mmp_component_cluster_sizes.csv")
    size_compare = observed_sizes.merge(frozen_sizes, on=["dataset", "split", "component_id"], how="outer", suffixes=("_observed", "_frozen"))
    size_ok = not size_compare.isna().any().any() and (size_compare.n_pairs_observed == size_compare.n_pairs_frozen).all()
    diagnostics = pd.read_csv(ROOT / "mmp_component_diagnostics.csv")
    independently = []
    for (target, split), subset in observed_sizes.groupby(["dataset", "split"]):
        values = subset.n_pairs.to_numpy(float)
        independently.append(
            {
                "dataset": target,
                "split": split,
                "n_pairs": int(values.sum()),
                "n_components": len(values),
                "max_pairs_per_component": int(values.max()),
            }
        )
    independently = pd.DataFrame(independently)
    diag_compare = independently.merge(diagnostics, on=["dataset", "split"], suffixes=("_observed", "_frozen"))
    diag_ok = (diag_compare.n_pairs_observed == diag_compare.n_pairs_frozen).all() and (diag_compare.n_components_observed == diag_compare.n_components_frozen).all() and (diag_compare.max_pairs_per_component_observed == diag_compare.max_pairs_per_component_frozen).all()
    cal_diag = diagnostics[diagnostics.split == "calibration"]
    cluster_boundary_ok = size_ok and diag_ok and (cal_diag.n_components.min(), cal_diag.n_components.max()) == (3, 15) and int(cal_diag.group_crc_numerically_feasible_alpha_0_10.sum()) == 7 and audit["pair_exchangeability_supported"] is False
    extremes = {row.dataset: (int(row.n_components), int(row.max_pairs_per_component)) for row in cal_diag.itertuples(index=False)}
    cluster_boundary_ok = cluster_boundary_ok and extremes["CHEMBL4792_Ki"] == (3, 29) and extremes["CHEMBL244_Ki"] == (5, 65) and extremes["CHEMBL234_Ki"] == (7, 57)
    extreme_detail = {target: extremes[target] for target in ["CHEMBL4792_Ki", "CHEMBL244_Ki", "CHEMBL234_Ki"]}
    add("Independent component clustering audit", "PASS" if cluster_boundary_ok else "FAIL", f"calibration components=3-15; group CRC numerically feasible=7/11; extremes={extreme_detail}")
    add("Mapping exclusions fully counted", "PASS" if len(exclusions) == audit["mapping_exclusion_count"] == amendment["mapping_exclusions"] == 9 else "FAIL", f"csv={len(exclusions)}, audit={audit['mapping_exclusion_count']}, amendment={amendment['mapping_exclusions']}")
    fragmentation = pd.read_csv(ROOT / "fragmentation_limit_exclusions.csv")
    no_eligible = pd.read_csv(ROOT / "no_eligible_bond_exclusions.csv")
    fragmentation_failures = 0
    for smiles, subset in fragmentation.groupby("canonical_smiles"):
        mol = Chem.MolFromSmiles(smiles)
        eligible = {
            mol.GetBondBetweenAtoms(match[0], match[1]).GetIdx()
            for match in mol.GetSubstructMatches(MMP_BOND_PATTERN, uniquify=True)
        }
        fragmentation_failures += int(len(eligible) <= 20 or set(subset.eligible_mmp_bonds) != {len(eligible)} or set(subset.n_atoms) != {mol.GetNumAtoms()})
        expected_occurrences = manifest_groups[
            (manifest_groups.canonical_smiles == smiles) & manifest_groups.split.isin(["calibration", "test"])
        ][["dataset", "split", "group_id"]].drop_duplicates()
        observed_occurrences = subset[["dataset", "split", "group_id"]].drop_duplicates()
        fragmentation_failures += int(
            set(map(tuple, expected_occurrences.astype(str).to_numpy())) != set(map(tuple, observed_occurrences.astype(str).to_numpy()))
        )
    fragmentation_selected = (set(fragmentation.canonical_smiles) & (set(pairs.smiles_a) | set(pairs.smiles_b)))
    fragmentation_ok = len(fragmentation) == 297 and fragmentation.canonical_smiles.nunique() == 245 and not fragmentation_selected and fragmentation_failures == 0 and (fragmentation.rdmmpa_max_cut_bonds == 20).all()
    fragmentation_ok = fragmentation_ok and audit["fragmentation_limit_unique_molecules"] == 245 and audit["fragmentation_limit_target_split_occurrences"] == 297 and audit["fragmentation_limit_selected_molecule_overlap"] == 0
    add("RDKit default maxCutBonds=20 structural exclusions", "PASS" if fragmentation_ok else "FAIL", f"unique=245, occurrences=297, selected_overlap={len(fragmentation_selected)}, failures={fragmentation_failures}")
    add("No-default-SMARTS eligible bond exclusions", "PASS" if len(no_eligible) == audit["no_eligible_bond_unique_molecules"] == 3 else "FAIL", f"unique={len(no_eligible)}")
    add("rdMMPA alignment audit", "PASS" if audit["status"] == "PASS" and audit["all_retained_molecule_core_sets_match_rdmmpa"] else "FAIL", f"status={audit['status']}")
    add("Automorphism enumeration audit", "PASS" if audit["audit"]["automorphism_truncated_molecules"] == 0 else "FAIL", audit["audit"]["automorphism_truncated_molecules"])
    validate_masks(pairs, "MMP cliffs")

    controls = pd.read_csv(ROOT / "noncliff_matched_controls.csv")
    control_counts = pd.read_csv(ROOT / "noncliff_control_counts.csv")
    balance = pd.read_csv(ROOT / "noncliff_control_balance.csv")
    control_audit = json.loads((ROOT / "noncliff_control_audit.json").read_text(encoding="utf-8"))
    control_ok = (controls.split == "test").all() and (controls.pair_type == "matched_noncliff_control").all() and (controls.measured_abs_delta <= 0.5 + 1e-12).all() and np.allclose(controls.measured_abs_delta, np.abs(controls.activity_a - controls.activity_b), atol=1e-12)
    add("Noncliff control definition", "PASS" if control_ok else "FAIL", f"n={len(controls)}")
    control_reuse = 0
    cliff_overlap = 0
    matched_id_failures = 0
    for target in TARGETS:
        group = controls[controls.dataset == target]
        molecules = list(group.smiles_a) + list(group.smiles_b)
        control_reuse += len(molecules) - len(set(molecules))
        cliff_group = pairs[pairs.dataset == target]
        cliff_molecules = set(cliff_group.smiles_a) | set(cliff_group.smiles_b)
        cliff_overlap += len(set(molecules) & cliff_molecules)
        test_ids = set(pairs[(pairs.dataset == target) & (pairs.split == "test")].pair_id)
        matched_id_failures += sum(int(value not in test_ids) for value in group.matched_cliff_pair_id)
        matched_id_failures += int(group.matched_cliff_pair_id.duplicated().any())
    add("Control molecule disjointness", "PASS" if control_reuse == 0 else "FAIL", f"reuse={control_reuse}")
    add("Control/cliff molecule exclusion", "PASS" if cliff_overlap == 0 else "FAIL", f"overlap={cliff_overlap}")
    add("Control-to-test-cliff traceability", "PASS" if matched_id_failures == 0 else "FAIL", f"failures={matched_id_failures}")
    add("Control target coverage", "PASS" if len(controls) == 467 and (control_counts.n_matched_controls >= 20).all() and set(control_counts.dataset) == set(TARGETS) else "FAIL", f"n={len(controls)}, minimum={control_counts.n_matched_controls.min()}")
    balance_failures = []
    feature_values = {
        "core_heavy_atoms": lambda frame: frame.core_heavy_atoms.to_numpy(float),
        "variable_total": lambda frame: (frame.variable_heavy_atoms_a + frame.variable_heavy_atoms_b).to_numpy(float),
        "atom_total": lambda frame: (frame.n_atoms_a + frame.n_atoms_b).to_numpy(float),
        "atom_difference": lambda frame: np.abs(frame.n_atoms_a - frame.n_atoms_b).to_numpy(float),
    }
    for row in balance.itertuples(index=False):
        control_group = controls[controls.dataset == row.dataset]
        matched_ids = set(control_group.matched_cliff_pair_id)
        cliff_group = pairs[(pairs.dataset == row.dataset) & (pairs.split == "test") & pairs.pair_id.isin(matched_ids)]
        x = feature_values[row.feature](control_group)
        y = feature_values[row.feature](cliff_group)
        pooled = math.sqrt(max(0.0, (x.var(ddof=1) + y.var(ddof=1)) / 2))
        smd = (x.mean() - y.mean()) / pooled if pooled else 0.0
        if not np.isclose(smd, row.standardized_mean_difference, atol=1e-12):
            balance_failures.append((row.dataset, row.feature))
    passing = int(control_counts.balance_gate_0_25.sum())
    failed_targets = sorted(control_counts.loc[~control_counts.balance_gate_0_25, "dataset"])
    balance_ok = not balance_failures and passing == 9 and control_audit["status"] == "DESCRIPTIVE_ONLY"
    add("Descriptive control balance, including negative result", "PASS" if balance_ok else "FAIL", f"passing=9/11, failed={failed_targets}, recomputation_failures={balance_failures}")
    validate_masks(controls, "Noncliff controls")
    control_component_failures = 0
    for row in controls.itertuples(index=False):
        expected_a = group_lookup.get((row.dataset, row.split, row.smiles_a))
        expected_b = group_lookup.get((row.dataset, row.split, row.smiles_b))
        control_component_failures += int(expected_a is None or expected_b is None or expected_a != expected_b or str(row.component_id) != expected_a)
    add("Control component_id lineage", "PASS" if control_component_failures == 0 else "FAIL", f"failures={control_component_failures}")
    preflight = pd.read_csv(ROOT / "training_split_preflight.csv")
    preflight_summary = json.loads((ROOT / "training_split_preflight.json").read_text(encoding="utf-8"))
    preflight_failures = 0
    for row in preflight.itertuples(index=False):
        train_pool = manifest_groups[(manifest_groups.dataset == row.dataset) & (manifest_groups.split == "train")]
        dev_groups, sizes = exact_dev_groups(train_pool, int(row.seed))
        dev = train_pool.group_id.isin(dev_groups)
        train_groups = set(train_pool.loc[~dev, "group_id"])
        expected = (
            len(train_pool),
            int((~dev).sum()),
            int(dev.sum()),
            len(sizes),
            len(train_groups),
            len(dev_groups),
            max(sizes.values()),
        )
        observed = (
            int(row.n_training_pool),
            int(row.n_model_train),
            int(row.n_development),
            int(row.n_all_components),
            int(row.n_model_train_components),
            int(row.n_development_components),
            int(row.largest_component_molecules),
        )
        preflight_failures += int(expected != observed or int(row.group_overlap) != 0 or not np.isclose(row.development_fraction, dev.mean()))
    exception_targets = sorted(preflight.loc[preflight.structural_exception, "dataset"].unique())
    preflight_ok = preflight_failures == 0 and len(preflight) == 33 and exception_targets == ["CHEMBL214_Ki", "CHEMBL234_Ki"] and preflight_summary["status"] == "PASS_WITH_STRUCTURAL_EXCEPTIONS"
    add("Independent train/development component preflight", "PASS" if preflight_ok else "FAIL", f"cells={len(preflight)}, failures={preflight_failures}, structural exceptions={exception_targets}")
    eligibility = pd.read_csv(ROOT / "target_panel_eligibility.csv")
    eligibility_audit = json.loads((ROOT / "target_panel_eligibility_audit.json").read_text(encoding="utf-8"))
    excluded_expected = {
        ("CHEMBL2147_Ki", "calibration"): 93,
        ("CHEMBL2147_Ki", "test"): 1,
        ("CHEMBL219_Ki", "calibration"): 4,
        ("CHEMBL219_Ki", "test"): 5,
        ("CHEMBL239_EC50", "calibration"): 3,
        ("CHEMBL239_EC50", "test"): 8,
        ("CHEMBL287_Ki", "calibration"): 12,
        ("CHEMBL287_Ki", "test"): 10,
    }
    eligibility_lookup = {(row.dataset, row.split): int(row.n_disjoint_pairs) for row in eligibility.itertuples(index=False)}
    eligible_targets = sorted(eligibility.loc[eligibility.eligible_both_splits, "dataset"].unique())
    eligibility_ok = len(eligibility) == 30 and eligibility_audit["status"] == "PASS" and eligible_targets == sorted(TARGETS)
    eligibility_ok = eligibility_ok and all(eligibility_lookup[key] == value for key, value in excluded_expected.items())
    for row in counts.itertuples(index=False):
        eligibility_ok = eligibility_ok and eligibility_lookup[(row.dataset, row.split)] == int(row.n_pairs)
    add("Frozen 11-of-15 target panel eligibility", "PASS" if eligibility_ok else "FAIL", f"selected=11; excluded counts={excluded_expected}")
    overlap_table = pd.read_csv(ROOT / "cross_target_overlap.csv")
    overlap_audit = json.loads((ROOT / "cross_target_overlap_audit.json").read_text(encoding="utf-8"))
    selected_sets = {target: set(pairs.loc[pairs.dataset == target, "smiles_a"]) | set(pairs.loc[pairs.dataset == target, "smiles_b"]) for target in TARGETS}
    manifest_sets = {target: set(manifest_groups.loc[manifest_groups.dataset == target, "canonical_smiles"]) for target in TARGETS}
    overlap_rows = []
    for scope, sets in [("selected_mmp_panel", selected_sets), ("full_mmp_series_manifest", manifest_sets)]:
        overlap_rows.extend(
            {"scope": scope, "target_a": a, "target_b": b, "shared_molecules": len(sets[a] & sets[b])}
            for a, b in combinations(TARGETS, 2)
        )
    independent_overlap = pd.DataFrame(overlap_rows)
    overlap_compare = overlap_table.merge(independent_overlap, on=["scope", "target_a", "target_b"], how="outer", suffixes=("_saved", "_independent"))
    overlap_ok = len(overlap_compare) == 110 and not overlap_compare.isna().any().any() and (overlap_compare.shared_molecules_saved == overlap_compare.shared_molecules_independent).all()
    overlap_ok = overlap_ok and overlap_audit["selected_panel_nonzero_target_pairs"] == 6 and overlap_audit["selected_panel_pairwise_shared_molecule_sum"] == 97 and overlap_audit["full_manifest_nonzero_target_pairs"] == 27 and overlap_audit["full_manifest_pairwise_shared_molecule_sum"] == 7983
    add("Independent cross-target molecule-overlap audit", "PASS" if overlap_ok else "FAIL", "selected: 6 target-pairs/97 pairwise shared; full manifest: 27/7983")
    provenance = json.loads((ROOT / "provenance.json").read_text(encoding="utf-8"))
    provenance_ok = provenance["status"] == "PASS" and provenance["code_source"]["commit"] == "7e6de0bd2968c56589c580f2a397f01c531ede26" and provenance["code_source"]["license"] == "MIT" and provenance["data_source"]["license"] == "CC BY-SA 3.0" and len(provenance["data_source"]["benchmark_files"]) == 11
    provenance_ok = provenance_ok and provenance["data_source"]["license_url"] == "https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/general-questions" and provenance["data_source"]["publication_doi"] == "10.1021/acs.jcim.2c01073"
    provenance_ok = provenance_ok and "official licensing FAQ" in provenance["data_source"]["evidence_basis"] and "not covered" in provenance["data_source"]["license_scope"]
    add("Source, license, and input-hash provenance", "PASS" if provenance_ok else "FAIL", "MoleculeACE code MIT; ChEMBL v29-derived data CC BY-SA 3.0")


def validate_training():
    names = ["predictions.csv", "model_metrics.csv", "train_development_splits.csv", "training_membership.csv", "environment.json"]
    present = [name for name in names if (ROOT / name).exists()]
    if not present:
        add("GINE training outputs", "PENDING", "awaiting serialized GPU run")
        return False
    if len(present) != len(names):
        add("GINE training outputs", "FAIL", f"partial files={present}")
        return False
    environment = json.loads((ROOT / "environment.json").read_text(encoding="utf-8"))
    environment_ok = environment["targets"] == TARGETS and environment["seeds"] == [17, 29, 43] and environment["epochs"] == 120 and environment["patience"] == 15 and environment["batch_size"] == 96
    add("Formal training hyperparameter contract", "PASS" if environment_ok else "FAIL", f"targets={len(environment['targets'])}, seeds={environment['seeds']}, epochs={environment['epochs']}, patience={environment['patience']}, batch={environment['batch_size']}")
    metrics = pd.read_csv(ROOT / "model_metrics.csv")
    predictions = pd.read_csv(ROOT / "predictions.csv")
    splits = pd.read_csv(ROOT / "train_development_splits.csv")
    membership = pd.read_csv(ROOT / "training_membership.csv")
    expected_cells = {(target, seed) for target in TARGETS for seed in environment["seeds"]}
    metric_cells = set(zip(metrics.dataset, metrics.seed))
    split_cells = set(zip(splits.dataset, splits.seed))
    cell_ok = metric_cells == split_cells == expected_cells and len(metrics) == len(splits) == 33 and len(metrics.drop_duplicates(["dataset", "seed"])) == 33 and len(splits.drop_duplicates(["dataset", "seed"])) == 33 and environment["seeds"] == [17, 29, 43]
    add("GINE target/seed cell completeness", "PASS" if cell_ok else "FAIL", f"observed={len(metric_cells)}, expected={len(expected_cells)}, seeds={environment['seeds']}")
    preflight = pd.read_csv(ROOT / "training_split_preflight.csv")
    split_compare = splits.merge(preflight, on=["dataset", "seed"], suffixes=("_actual", "_frozen"))
    split_ok = len(split_compare) == 33 and (split_compare.group_overlap_actual == 0).all()
    split_ok = split_ok and (split_compare.n_model_train_actual == split_compare.n_model_train_frozen).all() and (split_compare.n_development_actual == split_compare.n_development_frozen).all()
    split_ok = split_ok and (split_compare.n_train_groups == split_compare.n_model_train_components).all() and (split_compare.n_development_groups == split_compare.n_development_components).all()
    split_ok = split_ok and np.allclose(split_compare.development_fraction_of_training_pool, split_compare.development_fraction)
    add("Frozen train-only component-disjoint development split", "PASS" if split_ok else "FAIL", "33 cells; CHEMBL214 10.33%, CHEMBL234 3.65%; no component splitting")
    manifest_train = pd.read_csv(MANIFEST, usecols=["canonical_smiles", "dataset", "environment", "split", "group_id"])
    manifest_train = manifest_train[(manifest_train.environment == "mmp_series") & (manifest_train.split == "train") & manifest_train.dataset.isin(TARGETS)]
    membership_failures = 0
    for target, seed in expected_cells:
        source = manifest_train[manifest_train.dataset == target]
        dev_groups, _ = exact_dev_groups(source, seed)
        observed = membership[(membership.dataset == target) & (membership.seed == seed)]
        observed_map = dict(zip(observed.canonical_smiles, observed.role))
        source_groups = dict(zip(source.canonical_smiles, source.group_id.astype(str)))
        observed_groups = dict(zip(observed.canonical_smiles, observed.group_id.astype(str)))
        membership_failures += int(len(observed) != len(source) or observed.canonical_smiles.nunique() != len(source))
        membership_failures += int(source_groups != observed_groups)
        for row in source.itertuples(index=False):
            expected_role = "development" if row.group_id in dev_groups else "model_train"
            membership_failures += int(observed_map.get(row.canonical_smiles) != expected_role)
        role_by_group = observed.groupby("group_id").role.nunique()
        membership_failures += int((role_by_group > 1).any())
    add("Independent training membership audit", "PASS" if membership_failures == 0 else "FAIL", f"failures={membership_failures}")
    numeric = ["development_rmse", "calibration_rmse", "test_rmse", "test_pair_direction_accuracy"]
    metric_ok = np.isfinite(metrics[numeric].to_numpy(float)).all() and metrics.test_pair_direction_accuracy.between(0, 1).all() and metrics.test_spearman.between(-1, 1).all()
    add("Predictive metric schema and bounds", "PASS" if metric_ok else "FAIL", f"rows={len(metrics)}")
    manifest = pd.read_csv(MANIFEST, usecols=["canonical_smiles", "activity", "split", "dataset", "environment"])
    manifest = manifest[(manifest.environment == "mmp_series") & manifest.dataset.isin(TARGETS)]
    pairs = pd.read_csv(ROOT / "mmp_pairs.csv")
    prediction_failures = 0
    for target, seed in expected_cells:
        observed = predictions[(predictions.dataset == target) & (predictions.seed == seed)]
        expected = manifest[manifest.dataset == target]
        expected_map = expected.set_index("canonical_smiles")[["activity", "split"]].sort_index()
        observed_map = observed.set_index("canonical_smiles")[["activity", "split", "prediction"]].sort_index()
        prediction_failures += int(len(observed) != len(expected) or observed.canonical_smiles.nunique() != len(expected))
        prediction_failures += int(set(observed_map.index) != set(expected_map.index))
        if set(observed_map.index) == set(expected_map.index):
            prediction_failures += int(not np.allclose(observed_map.activity, expected_map.activity, atol=1e-12) or not observed_map.split.equals(expected_map.split))
            prediction_failures += int(not np.isfinite(observed_map.prediction).all())
            metric_row = metrics[(metrics.dataset == target) & (metrics.seed == seed)].iloc[0]
            membership_cell = membership[(membership.dataset == target) & (membership.seed == seed)]
            dev_smiles = set(membership_cell.loc[membership_cell.role == "development", "canonical_smiles"])
            prediction_series = observed_map.prediction
            for split_name, smiles_set in [
                ("development", dev_smiles),
                ("calibration", set(expected.loc[expected.split == "calibration", "canonical_smiles"])),
                ("test", set(expected.loc[expected.split == "test", "canonical_smiles"])),
            ]:
                ordered = sorted(smiles_set)
                y_true = expected_map.loc[ordered, "activity"].to_numpy(float)
                y_pred = prediction_series.loc[ordered].to_numpy(float)
                rmse = float(np.sqrt(np.mean(np.square(y_true - y_pred))))
                rho = float(spearmanr(y_true, y_pred).statistic)
                prediction_failures += int(not np.isclose(rmse, metric_row[f"{split_name}_rmse"], atol=1e-10, rtol=1e-10))
                prediction_failures += int(not np.isclose(rho, metric_row[f"{split_name}_spearman"], atol=1e-10, rtol=1e-10))
            test_pairs = pairs[(pairs.dataset == target) & (pairs.split == "test")]
            directions = [
                np.sign(prediction_series[row.smiles_a] - prediction_series[row.smiles_b]) == np.sign(row.measured_delta_a_minus_b)
                for row in test_pairs.itertuples(index=False)
            ]
            prediction_failures += int(not np.isclose(np.mean(directions), metric_row.test_pair_direction_accuracy, atol=1e-12))
            prediction_failures += int(len(directions) != metric_row.n_test_pairs)
    add("Prediction exact keys and independently recomputed metrics", "PASS" if prediction_failures == 0 else "FAIL", f"failures={prediction_failures}")
    checkpoints_missing = sum(not (ROOT / "models" / target / f"seed_{seed}.pt").exists() for target, seed in expected_cells)
    add("Checkpoint coverage", "PASS" if checkpoints_missing == 0 else "FAIL", f"missing={checkpoints_missing}")
    deterministic_amendment = json.loads((ROOT / "contract_amendment_06_cpu_deterministic_attribution.json").read_text(encoding="utf-8"))
    frozen_inputs = deterministic_amendment["frozen_predictor_artifacts"]
    predictor_hashes_ok = all(sha256(ROOT / name) == expected for name, expected in frozen_inputs["file_sha256"].items())
    checkpoint_hashes = pd.read_csv(ROOT / "model_checkpoint_hashes.csv")
    checkpoint_hashes_ok = len(checkpoint_hashes) == 33 and checkpoint_hashes.relative_path.nunique() == 33
    for row in checkpoint_hashes.itertuples(index=False):
        path = ROOT / row.relative_path
        checkpoint_hashes_ok = checkpoint_hashes_ok and path.exists() and sha256(path) == row.sha256 and path.stat().st_size == int(row.bytes)
    checkpoint_hashes_ok = checkpoint_hashes_ok and sha256(ROOT / "model_checkpoint_hashes.csv") == frozen_inputs["checkpoint_hashes_csv_sha256"]
    add("Frozen predictor and checkpoint hashes before CPU attribution", "PASS" if predictor_hashes_ok and checkpoint_hashes_ok else "FAIL", f"predictor_files={len(frozen_inputs['file_sha256'])}, checkpoints={len(checkpoint_hashes)}")
    return True


def validate_attributions(training_complete):
    names = ["pair_attributions.csv", "crc_results.csv", "crc_macro_summary.csv", "component_crc_feasibility.csv", "near_zero_direction_instability.csv", "summary.json", "attribution_environment.json", "formal_cpu_reproducibility_audit.json"]
    present = [name for name in names if (ROOT / name).exists()]
    if not present:
        add("Attribution and CRC outputs", "PENDING", "awaiting completed GINE checkpoints")
        return False
    if len(present) != len(names) or not training_complete:
        add("Attribution and CRC outputs", "FAIL", f"partial files={present}, training_complete={training_complete}")
        return False
    attributions = pd.read_csv(ROOT / "pair_attributions.csv")
    results = pd.read_csv(ROOT / "crc_results.csv")
    macro = pd.read_csv(ROOT / "crc_macro_summary.csv")
    feasibility = pd.read_csv(ROOT / "component_crc_feasibility.csv")
    summary = json.loads((ROOT / "summary.json").read_text(encoding="utf-8"))
    direction_instabilities = pd.read_csv(ROOT / "near_zero_direction_instability.csv")
    environment = json.loads((ROOT / "environment.json").read_text(encoding="utf-8"))
    attribution_environment = json.loads((ROOT / "attribution_environment.json").read_text(encoding="utf-8"))
    ig_contract_ok = summary["ig_steps"] == 32 and attribution_environment["ig_steps"] == 32
    add("Formal integrated-gradients step contract", "PASS" if ig_contract_ok else "FAIL", f"summary={summary['ig_steps']}, environment={attribution_environment['ig_steps']}")
    thread_values = attribution_environment["thread_environment"]
    cpu_contract_ok = attribution_environment["device"] == "cpu" and attribution_environment["device_name"] == "CPU" and attribution_environment["cpu_threads"] == 1 and attribution_environment["cpu_interop_threads"] == 1
    cpu_contract_ok = cpu_contract_ok and attribution_environment["deterministic_algorithms"] and attribution_environment["default_dtype"] == "torch.float32" and attribution_environment["score_serialization_significant_digits"] == 10
    cpu_contract_ok = cpu_contract_ok and set(thread_values.values()) == {"1"} and attribution_environment["ranking"] == "descending serialized absolute atom scores" and "include every atom exactly tied" in attribution_environment["tie_policy"]
    add("Formal CPU deterministic numerical contract", "PASS" if cpu_contract_ok else "FAIL", f"threads={attribution_environment['cpu_threads']}/{attribution_environment['cpu_interop_threads']}, dtype={attribution_environment['default_dtype']}, score digits={attribution_environment['score_serialization_significant_digits']}")
    expected_rows = (1035 + 467) * len(environment["seeds"]) * 2
    pairs = pd.read_csv(ROOT / "mmp_pairs.csv")
    pairs["pair_type"] = "sar_cliff_proxy"
    controls = pd.read_csv(ROOT / "noncliff_matched_controls.csv")
    combined = pd.concat([pairs, controls], ignore_index=True, sort=False)
    base_keys = set(zip(combined.dataset, combined.split, combined.pair_type, combined.pair_id))
    expected_keys = {
        (*key, seed, method)
        for key in base_keys
        for seed in environment["seeds"]
        for method in ["integrated_gradients", "atom_occlusion"]
    }
    observed_key_columns = ["dataset", "split", "pair_type", "pair_id", "seed", "method"]
    observed_keys = set(map(tuple, attributions[observed_key_columns].itertuples(index=False, name=None)))
    attribution_complete = len(attributions) == expected_rows and len(attributions.drop_duplicates(observed_key_columns)) == expected_rows and observed_keys == expected_keys and attributions.component_id.notna().all()
    add("Attribution exact pair x seed x method coverage", "PASS" if attribution_complete else "FAIL", f"observed={len(attributions)}, expected={expected_rows}, missing={len(expected_keys - observed_keys)}, extra={len(observed_keys - expected_keys)}")
    predictions = pd.read_csv(ROOT / "predictions.csv")
    prediction_lookup = {(row.dataset, int(row.seed), row.canonical_smiles): float(row.prediction) for row in predictions.itertuples(index=False)}
    pair_lookup = {(row.dataset, row.split, row.pair_type, int(row.pair_id)): row for row in combined.itertuples(index=False)}
    attribution_lineage_failures = 0
    for row in attributions.itertuples(index=False):
        source = pair_lookup[(row.dataset, row.split, row.pair_type, int(row.pair_id))]
        prediction_a = prediction_lookup[(row.dataset, int(row.seed), row.smiles_a)]
        prediction_b = prediction_lookup[(row.dataset, int(row.seed), row.smiles_b)]
        delta = prediction_a - prediction_b
        direction = np.sign(delta) == np.sign(source.measured_delta_a_minus_b)
        attribution_lineage_failures += int(row.smiles_a != source.smiles_a or row.smiles_b != source.smiles_b or str(row.component_id) != str(source.component_id))
        attribution_lineage_failures += int(not np.isclose(row.prediction_a, prediction_a, atol=1e-6) or not np.isclose(row.prediction_b, prediction_b, atol=1e-6) or not np.isclose(row.predicted_delta_a_minus_b, delta, atol=1e-6))
        attribution_lineage_failures += int(bool(row.direction_correct) != bool(direction))
    add("Attribution prediction-delta and direction lineage", "PASS" if attribution_lineage_failures == 0 else "FAIL", f"failures={attribution_lineage_failures}")
    instability_ok = len(direction_instabilities) == summary["near_zero_direction_instability_pairs"]
    if len(direction_instabilities):
        instability_ok = instability_ok and (direction_instabilities.maximum_absolute_delta <= 1e-6).all() and (direction_instabilities.numerical_tie_atol == 1e-6).all()
        instability_ok = instability_ok and (np.sign(direction_instabilities.prediction_table_delta) != np.sign(direction_instabilities.checkpoint_attribution_forward_delta)).all()
    gpu_instability = json.loads((ROOT / "diagnostics" / "gpu_attribution_instability" / "gpu_attribution_instability_audit.json").read_text(encoding="utf-8"))
    gpu_instability_ok = gpu_instability["status"] == "GPU_ATTRIBUTION_NOT_FROZEN" and gpu_instability["scores_a_text_changed_rows"] == 7319 and gpu_instability["scores_b_text_changed_rows"] == 7259 and gpu_instability["full_rank_changed_rows"] == 1428
    gpu_instability_ok = gpu_instability_ok and gpu_instability["crc_rows_with_any_difference"] == 1264 and np.isclose(gpu_instability["maximum_absolute_result_differences"]["risk"], 0.0565284561237219)
    add("Near-zero direction and failed-GPU reproducibility disclosure", "PASS" if instability_ok and gpu_instability_ok else "FAIL", f"formal CPU sign-instability rows={len(direction_instabilities)}, historical GPU pair=1, GPU rank changes=1428")
    score_failures = 0
    for row in attributions.itertuples(index=False):
        scores_a = np.asarray([float(value) for value in str(row.scores_a).split(";")])
        scores_b = np.asarray([float(value) for value in str(row.scores_b).split(";")])
        score_failures += int(len(scores_a) != row.n_atoms_a or len(scores_b) != row.n_atoms_b)
        score_failures += int(not np.isfinite(scores_a).all() or not np.isfinite(scores_b).all() or (scores_a < 0).any() or (scores_b < 0).any())
    add("Per-atom attribution score schema", "PASS" if score_failures == 0 else "FAIL", f"failures={score_failures}")
    ig = attributions[attributions.method == "integrated_gradients"]
    ig_errors = np.concatenate([ig.ig_completeness_error_a.to_numpy(float), ig.ig_completeness_error_b.to_numpy(float)])
    disclosed = summary["ig_completeness_absolute_error"]
    ig_ok = np.isfinite(ig_errors).all() and disclosed["n_molecule_attributions"] == len(ig_errors) and np.isclose(disclosed["median"], np.median(ig_errors)) and np.isclose(disclosed["p95"], np.quantile(ig_errors, 0.95)) and np.isclose(disclosed["max"], np.max(ig_errors))
    add("IG completeness-error disclosure", "PASS" if ig_ok else "FAIL", f"n={len(ig_errors)}, median={np.median(ig_errors):.6g}, p95={np.quantile(ig_errors, 0.95):.6g}, max={np.max(ig_errors):.6g}")
    bounds = ["risk", "retained_fraction", "precision", "iou"]
    crc_ok = np.isfinite(results[bounds].to_numpy(float)).all() and all(results[column].between(0, 1).all() for column in bounds)
    calibrated = results[results.policy.isin(["pair_crc_descriptive", "component_crc_cluster_stress_test"])]
    crc_ok = crc_ok and (calibrated.calibration_corrected_risk <= 0.1 + 1e-12).all()
    required_policies = {"pair_crc_descriptive", "component_crc_cluster_stress_test", "fixed_20_percent", "fixed_50_percent", "oracle_proxy", "random_same_nominal_size"}
    crc_ok = crc_ok and set(results.policy) == required_policies and set(results.method) == {"integrated_gradients", "atom_occlusion"} and set(results.aggregation) == {"pair_weighted", "component_weighted"}
    crc_ok = crc_ok and (results.n_components >= 1).all()
    add("CRC policies, alpha gate, and metric bounds", "PASS" if crc_ok else "FAIL", f"rows={len(results)}, policies={sorted(set(results.policy))}")
    recomputed_results, recomputed_feasibility = independently_recompute_crc(attributions)
    result_keys = ["dataset", "seed", "method", "proxy", "evaluation_split", "stratum", "policy", "aggregation"]
    result_compare = results.merge(recomputed_results, on=result_keys, how="outer", suffixes=("_saved", "_independent"), indicator=True)
    crc_recompute_ok = len(result_compare) == len(results) == len(recomputed_results) and (result_compare._merge == "both").all()
    for column in ["n_pairs", "n_components", "nominal_fraction", "risk", "retained_fraction", "precision", "iou", "calibration_empirical_risk", "calibration_corrected_risk"]:
        crc_recompute_ok = crc_recompute_ok and np.allclose(result_compare[f"{column}_saved"], result_compare[f"{column}_independent"], atol=1e-10, rtol=1e-10, equal_nan=True)
    crc_recompute_ok = crc_recompute_ok and (result_compare.calibration_unit_saved == result_compare.calibration_unit_independent).all()
    add("Independent CRC table and aggregate recomputation", "PASS" if crc_recompute_ok else "FAIL", f"saved={len(results)}, independently_recomputed={len(recomputed_results)}")
    diagnostic_components = dict(
        zip(
            pd.read_csv(ROOT / "mmp_component_diagnostics.csv").query("split == 'calibration'").dataset,
            pd.read_csv(ROOT / "mmp_component_diagnostics.csv").query("split == 'calibration'").n_components,
        )
    )
    feasibility_failures = 0
    for row in feasibility.itertuples(index=False):
        expected_components = int(diagnostic_components[row.dataset])
        expected_feasible = expected_components >= 9
        feasibility_failures += int(int(row.n_calibration_components) != expected_components or bool(row.component_crc_numerically_feasible) != expected_feasible)
        if expected_feasible:
            feasibility_failures += int(not np.isfinite(row.component_calibration_corrected_risk) or row.component_calibration_corrected_risk > 0.1 + 1e-12)
        else:
            feasibility_failures += int(row.fallback_when_infeasible != "pair_crc_descriptive_finite_pool_only")
    feasible_targets = sorted(feasibility.loc[feasibility.component_crc_numerically_feasible, "dataset"].unique())
    infeasible_targets = sorted(feasibility.loc[~feasibility.component_crc_numerically_feasible, "dataset"].unique())
    feasibility_ok = len(feasibility) == 198 and feasibility_failures == 0 and len(feasible_targets) == 7 and infeasible_targets == ["CHEMBL234_Ki", "CHEMBL236_Ki", "CHEMBL244_Ki", "CHEMBL4792_Ki"]
    feasibility_keys = ["dataset", "seed", "method", "proxy"]
    feasibility_compare = feasibility.merge(recomputed_feasibility, on=feasibility_keys, how="outer", suffixes=("_saved", "_independent"), indicator=True)
    feasibility_recompute_ok = len(feasibility_compare) == len(feasibility) == len(recomputed_feasibility) and (feasibility_compare._merge == "both").all()
    for column in [
        "n_calibration_components",
        "pair_crc_descriptive_fraction",
        "pair_calibration_empirical_risk",
        "pair_calibration_corrected_risk",
        "component_crc_fraction",
        "component_calibration_empirical_risk",
        "component_calibration_corrected_risk",
    ]:
        feasibility_recompute_ok = feasibility_recompute_ok and np.allclose(feasibility_compare[f"{column}_saved"], feasibility_compare[f"{column}_independent"], atol=1e-10, rtol=1e-10, equal_nan=True)
    feasibility_ok = feasibility_ok and feasibility_recompute_ok
    add("Group-level corrected CRC feasibility", "PASS" if feasibility_ok else "FAIL", f"rows={len(feasibility)}, feasible=7/11, infeasible={infeasible_targets}, failures={feasibility_failures}")
    independent_macro = independently_recompute_macro(recomputed_results)
    macro_keys = ["method", "proxy", "policy", "aggregation", "metric"]
    macro_compare = macro.merge(independent_macro, on=macro_keys, how="outer", suffixes=("_saved", "_independent"), indicator=True)
    macro_ok = len(macro_compare) == len(macro) == len(independent_macro) and (macro_compare._merge == "both").all() and set(macro.n_targets) == {7, 11}
    for column in ["target_macro_mean", "bootstrap_ci_low", "bootstrap_ci_high", "n_targets"]:
        macro_ok = macro_ok and np.allclose(macro_compare[f"{column}_saved"], macro_compare[f"{column}_independent"], atol=1e-10, rtol=1e-10)
    macro_ok = macro_ok and macro.bootstrap_ci_low.le(macro.target_macro_mean).all() and macro.target_macro_mean.le(macro.bootstrap_ci_high).all()
    add("Target-macro bootstrap summaries", "PASS" if macro_ok else "FAIL", f"rows={len(macro)}")
    metrics = pd.read_csv(ROOT / "model_metrics.csv")
    metric_columns = ["test_rmse", "test_spearman", "test_pair_direction_accuracy"]
    target_performance = metrics.groupby("dataset")[metric_columns].mean()
    predictive = summary["predictive_performance"]
    predictive_ok = predictive["n_targets"] == 11 and predictive["seeds_per_target"] == 3 and set(target_performance.index) == set(TARGETS)
    for metric in metric_columns:
        values = target_performance[metric]
        reported = predictive["target_macro_mean"][metric]
        extremes = predictive["target_extremes"][metric]
        predictive_ok = predictive_ok and np.isclose(reported, values.mean(), atol=1e-12, rtol=1e-12)
        predictive_ok = predictive_ok and extremes["lowest"]["dataset"] == values.idxmin() and np.isclose(extremes["lowest"]["value"], values.min(), atol=1e-12, rtol=1e-12)
        predictive_ok = predictive_ok and extremes["highest"]["dataset"] == values.idxmax() and np.isclose(extremes["highest"]["value"], values.max(), atol=1e-12, rtol=1e-12)
    predictive_ok = predictive_ok and "limits the interpretation" in predictive["boundary"] and "heterogeneous" in predictive["boundary"]
    add("Predictive target-macro and adverse-extreme summary", "PASS" if predictive_ok else "FAIL", f"RMSE={target_performance.test_rmse.mean():.6f}, Spearman={target_performance.test_spearman.mean():.6f}, pair-direction={target_performance.test_pair_direction_accuracy.mean():.6f}")
    reproducibility = json.loads((ROOT / "formal_cpu_reproducibility_audit.json").read_text(encoding="utf-8"))
    current_hashes = {name: sha256(ROOT / name) for name in reproducibility["run_a_sha256"]}
    reproducibility_ok = reproducibility["status"] == "PASS_EXACT" and all(reproducibility["file_hashes_exact"].values()) and reproducibility["attribution_keys_and_serialized_scores_exact"] and reproducibility["crc_table_exact"]
    expected_run_a = (ROOT / "diagnostics" / "formal_cpu_reproducibility" / "run1").resolve()
    reproducibility_ok = reproducibility_ok and Path(reproducibility["run_a"]).resolve() == expected_run_a and current_hashes == reproducibility["run_a_sha256"] and reproducibility["attribution_rows"] == 9012 and reproducibility["crc_rows"] == 9576
    add("Independent full CPU attribution reproducibility", "PASS" if reproducibility_ok else "FAIL", f"files={reproducibility['files']}, byte-identical={all(reproducibility['file_hashes_exact'].values())}")
    overlap = summary["cross_target_overlap"]
    summary_ok = summary["status"] == "DESCRIPTIVE_CLUSTERED_STRESS_TEST" and summary["estimand"] == "predicted potency difference f(A)-f(B)" and summary["tie_policy"] == "include all atoms tied at the nominal fraction boundary" and summary["alpha"] == 0.1 and summary["checkpoint_reproduction_max_absolute_error"] <= 1e-4 and "not causal" in summary["proxy_boundary"] and "descriptive only" in summary["control_boundary"] and "does not establish" in summary["cluster_boundary"] and "distribution-free risk control" in summary["cluster_boundary"]
    summary_ok = summary_ok and "mean fraction" in summary["risk_definition"] and "not the probability" in summary["risk_definition"] and "preserving all molecular edges" in summary["atom_occlusion_definition"]
    summary_ok = summary_ok and summary["direction_strata_source"] == "predictions.csv serialized deployment predictions" and summary["near_zero_direction_instability_atol"] == 1e-6 and "numerically unstable" in summary["near_zero_direction_instability_boundary"]
    summary_ok = summary_ok and summary["historical_gpu_direction_instability_pairs"] == 1 and "failed-run diagnostic history" in summary["gpu_attribution_reproducibility_boundary"]
    summary_ok = summary_ok and overlap["selected_panel_nonzero_target_pairs"] == 6 and overlap["selected_panel_pairwise_shared_molecule_sum"] == 97 and overlap["full_manifest_nonzero_target_pairs"] == 27 and overlap["full_manifest_pairwise_shared_molecule_sum"] == 7983 and "not population confidence intervals" in summary["bootstrap_boundary"]
    add("Estimand, tie policy, reproduction, and claim boundaries", "PASS" if summary_ok else "FAIL", f"checkpoint max error={summary['checkpoint_reproduction_max_absolute_error']}")
    return True


def main():
    validate_pretrain()
    training_complete = validate_training()
    attribution_complete = validate_attributions(training_complete)
    failures = [check for check in checks if check["result"] == "FAIL"]
    pending = [check for check in checks if check["result"] == "PENDING"]
    status = "FAIL" if failures else "FULL_PASS" if training_complete and attribution_complete and not pending else "PRETRAIN_PASS"
    report = {
        "status": status,
        "checks_passed": sum(check["result"] == "PASS" for check in checks),
        "checks_failed": len(failures),
        "checks_pending": len(pending),
        "checks": checks,
        "claim_boundary": "Experimental SAR-linked transformation-site proxy; not causal, complete, mechanistic, expert, or ground truth.",
        "statistical_boundary": "Clustered pairs and overlapping targets support only a descriptive finite-pool stress test; no distribution-free pair-level risk control, population confidence interval, or generalization claim.",
    }
    (ROOT / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# A7 validation", "", f"Overall status: **{status}**", "", "| Check | Result | Detail |", "|---|---:|---|"]
    for check in checks:
        detail = check["detail"].replace("|", "/").replace("\n", " ")
        lines.append(f"| {check['name']} | {check['result']} | {detail} |")
    lines += ["", "## Claim boundaries", "", report["claim_boundary"], "", report["statistical_boundary"], ""]
    (ROOT / "validation.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ["status", "checks_passed", "checks_failed", "checks_pending"]}, indent=2))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

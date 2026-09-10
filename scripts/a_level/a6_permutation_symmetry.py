import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import torch
import torch_geometric
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from scipy.stats import rankdata
from torch_geometric.loader import DataLoader


RDLogger.DisableLog("rdApp.*")
HERE = Path(__file__).resolve().parent
REVISION = HERE.parents[2]
PROJECT = REVISION.parent
sys.path.insert(0, str(PROJECT / "src"))
from run_gradient_grid import BXAIC_TASKS, GOOGLE_TASKS, GraphClassifier, bxaic_partitions, google_partitions


DEFAULT_TIE_ROOT = PROJECT.parents[1] / "<reviewer-working-tree>/tie_inclusive"
DEFAULT_CACHE = DEFAULT_TIE_ROOT / "established_explainers_tie_inclusive/score_cache"
DEFAULT_SELECTOR_AUDIT = DEFAULT_TIE_ROOT / "permutation_invariance_audit.json"
CHECKPOINTS = PROJECT / "artifacts/experiment/gradient_grid_main/checkpoints"
CONTRACT = HERE / "run_contract.json"
FRACTIONS = np.round(np.linspace(0, 1, 101), 2)
POLICIES = ["index_tiebreak_v1", "include_all_exact_ties_v2", "rdkit_symmetry_orbit_closure_v3"]
ALPHA = 0.10
HASH_CACHE = {}
GOOGLE_ATOM_TYPES = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "Na", "Ca", "I", "B", "H", "*"]
BXAIC_ATOM_TYPES = ["C", "N", "O", "F", "Cl", "Br", "P", "S", "B", "I", "Unk"]


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


def unpack(values, offsets):
    return [values[a:b] for a, b in zip(offsets[:-1], offsets[1:])]


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


def cell_provenance(cache_dir, cell_id, checkpoint, family, task, run_mode, n_molecules, repetitions):
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
        "policies": POLICIES,
        "test_molecules": n_molecules,
        "permutations_per_molecule": repetitions,
        "runtime": {
            "python": platform.python_version(), "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__, "numpy": np.__version__,
            "pandas": pd.__version__, "rdkit": Chem.rdBase.rdkitVersion,
        },
    }


def validate_cache(z, calibration_graphs, test_graphs):
    if z["schema_version"].tolist() != [1]:
        raise ValueError("unsupported score-cache schema")
    for split, graphs, prefix_only in (
        ("calibration", calibration_graphs, False),
        ("test", test_graphs, True),
    ):
        sources = z[f"{split}__source_indices"].astype(int)
        expected = [int(graph.source_index) for graph in graphs]
        observed = sources[:len(graphs)].tolist() if prefix_only else sources.tolist()
        if observed != expected:
            raise ValueError(f"{split}: cache source order disagrees with graphs")
        offsets = z[f"{split}__offsets"].astype(int)
        rationale_values = z[f"{split}__rationale_values"].astype(bool)
        if len(offsets) != len(sources) + 1 or offsets[0] != 0 or offsets[-1] != len(rationale_values):
            raise ValueError(f"{split}: invalid cache offsets")
        for i, graph in enumerate(graphs):
            start, stop = offsets[i:i + 2]
            expected_mask = graph.rationale_mask.detach().cpu().numpy().astype(bool)
            if stop - start != graph.num_nodes or not np.array_equal(rationale_values[start:stop], expected_mask):
                raise ValueError(f"{split}/{int(graph.source_index)}: cached atom/rationale alignment failed")
        values = z[f"ig__{split}__score_values"]
        if len(values) != len(rationale_values) or not np.isfinite(values).all():
            raise ValueError(f"{split}: cached IG scores are misaligned or nonfinite")


def score_consistency(cached, recomputed):
    differences = np.concatenate([
        np.abs(np.asarray(left, dtype=float) - np.asarray(right, dtype=float))
        for left, right in zip(cached, recomputed)
    ])
    return {
        "max_abs_difference": float(differences.max(initial=0.0)),
        "mean_abs_difference": float(differences.mean()) if len(differences) else 0.0,
    }


def reusable_cell(result_path, score_path, set_path, calibration_path, provenance, expected_scores, expected_sets):
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        return (
            payload.get("status") == "complete"
            and payload.get("provenance") == provenance
            and len(pd.read_csv(score_path, usecols=["cell_id"])) == expected_scores
            and len(pd.read_csv(set_path, usecols=["cell_id"])) == expected_sets
            and len(pd.read_csv(calibration_path, usecols=["cell_id"])) == len(POLICIES)
        )
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        return False


def selected_set(scores, fraction, policy, orbit_ids):
    scores = np.asarray(scores)
    if fraction == 0:
        selected = set()
    else:
        k = int(np.ceil(fraction * len(scores)))
        if policy == "index_tiebreak_v1":
            selected = set(np.lexsort((np.arange(len(scores)), -scores))[:k].tolist())
        else:
            threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
            selected = set(np.flatnonzero(scores >= threshold).tolist())
    if policy == "rdkit_symmetry_orbit_closure_v3" and selected:
        chosen_orbits = {orbit_ids[i] for i in selected}
        selected = {i for i, orbit in enumerate(orbit_ids) if orbit in chosen_orbits}
    return selected


def calibrate(score_rows, truths, orbit_rows, policy):
    losses = np.empty((len(score_rows), len(FRACTIONS)))
    for i, (scores, truth, orbits) in enumerate(zip(score_rows, truths, orbit_rows)):
        for j, fraction in enumerate(FRACTIONS):
            selected = selected_set(scores, fraction, policy, orbits)
            losses[i, j] = 1 - len(selected & truth) / len(truth)
    corrected = (len(losses) * losses.mean(0) + 1) / (len(losses) + 1)
    feasible = np.flatnonzero(corrected <= ALPHA)
    if not len(feasible):
        return len(FRACTIONS) - 1, False
    return int(feasible[0]), True


def orbit_split(selected, orbit_ids):
    for orbit in set(orbit_ids):
        members = {i for i, value in enumerate(orbit_ids) if value == orbit}
        if selected & members and not members <= selected:
            return True
    return False


def molecule_map(family, task, graphs):
    wanted = {int(g.source_index): g for g in graphs}
    if family == "bxaic":
        frame = pd.read_csv(PROJECT / "data/raw/bxaic/data.csv", usecols=["smiles"])
    else:
        frame = pd.read_csv(PROJECT / f"reference/graph-attribution/data/{task}/{task}_smiles.csv", usecols=["smiles"])
    output = {}
    for index, graph in wanted.items():
        mol = Chem.MolFromSmiles(str(frame.at[index, "smiles"]))
        if mol is None or mol.GetNumAtoms() != graph.num_nodes:
            raise ValueError(f"{family}/{task}/{index}: SMILES atom mapping failed")
        atom_types = BXAIC_ATOM_TYPES if family == "bxaic" else GOOGLE_ATOM_TYPES
        unknown = atom_types[-1]
        expected_symbols = [atom.GetSymbol() if atom.GetSymbol() in atom_types else unknown for atom in mol.GetAtoms()]
        features = graph.x.detach().cpu().numpy()
        if features.ndim != 2 or features.shape[1] != len(atom_types):
            raise ValueError(f"{family}/{task}/{index}: unexpected atom-feature shape")
        observed_symbols = [atom_types[i] for i in features.argmax(axis=1)]
        if observed_symbols != expected_symbols:
            raise ValueError(f"{family}/{task}/{index}: SMILES atom-feature identity mapping failed")
        graph_edges = {tuple(sorted((int(a), int(b)))) for a, b in graph.edge_index.T.tolist()}
        mol_edges = {tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))) for bond in mol.GetBonds()}
        if graph_edges != mol_edges:
            raise ValueError(f"{family}/{task}/{index}: SMILES edge mapping failed")
        output[index] = mol
    return output


def symmetry_orbits(mol):
    return list(Chem.CanonicalRankAtoms(mol, breakTies=False, includeChirality=True))


def permuted_graph(graph, permutation):
    permutation = torch.as_tensor(permutation, dtype=torch.long)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(len(permutation))
    result = graph.clone()
    result.x = graph.x[permutation]
    result.edge_index = inverse[graph.edge_index]
    result.rationale_mask = graph.rationale_mask[permutation]
    return result


def ig_scores(model, graphs, device, batch_size=32):
    rows, logits = [], []
    model.eval()
    for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
        batch = batch.to(device)
        with torch.no_grad():
            logits.append(model(batch.x, batch.edge_index, batch.batch).cpu().numpy())
        total = torch.zeros_like(batch.x)
        for step in range(1, 21):
            x = (batch.x * step / 20).detach().requires_grad_(True)
            output = model(x, batch.edge_index, batch.batch)
            objective = output[torch.arange(len(batch.y), device=device), batch.y].sum()
            total += torch.autograd.grad(objective, x)[0]
        scores = (batch.x * total / 20).sum(-1).detach().cpu().numpy()
        ptr = batch.ptr.cpu().tolist()
        rows.extend(scores[a:b] for a, b in zip(ptr[:-1], ptr[1:]))
    return rows, np.concatenate(logits)


def one_cell(family, task, model_kind, partitions, device, n_molecules, repetitions, out_root, cache_dir, run_mode):
    cell_id = f"{family}__{task}__{model_kind}__seed42"
    result_path = out_root / "cells" / f"{cell_id}.json"
    score_path = out_root / "rows" / f"{cell_id}_scores.csv.gz"
    set_path = out_root / "rows" / f"{cell_id}_sets.csv.gz"
    calibration_path = out_root / "rows" / f"{cell_id}_calibration.csv"
    checkpoint = CHECKPOINTS / f"{cell_id}.pt"
    provenance = cell_provenance(cache_dir, cell_id, checkpoint, family, task, run_mode, n_molecules, repetitions)
    expected_scores, expected_sets = n_molecules * repetitions, n_molecules * repetitions * len(POLICIES)
    if reusable_cell(result_path, score_path, set_path, calibration_path, provenance, expected_scores, expected_sets):
        print(f"skip_complete={cell_id}", flush=True)
        return
    with np.load(cache_dir / f"{cell_id}.npz", allow_pickle=False) as z:
        source_cal = z["calibration__source_indices"].astype(int)
        source_test = z["test__source_indices"].astype(int)[:n_molecules]
        if len(source_test) != n_molecules:
            raise ValueError(f"{cell_id}: expected {n_molecules} test molecules, found {len(source_test)}")
        maps = {split: {int(g.source_index): g for g in partitions[split]} for split in ["calibration", "test"]}
        cal_graphs = [maps["calibration"][i] for i in source_cal]
        test_graphs = [maps["test"][i] for i in source_test]
        validate_cache(z, cal_graphs, test_graphs)
        cal_scores = [np.asarray(row).copy() for row in unpack(z["ig__calibration__score_values"], z["calibration__offsets"])]
        all_test_scores = unpack(z["ig__test__score_values"], z["test__offsets"])
        cached_baseline_scores = [np.asarray(row).copy() for row in all_test_scores[:len(test_graphs)]]
    mols = molecule_map(family, task, cal_graphs + test_graphs)
    cal_orbits = [symmetry_orbits(mols[int(g.source_index)]) for g in cal_graphs]
    test_orbits = [symmetry_orbits(mols[int(g.source_index)]) for g in test_graphs]
    cal_truths = [set(torch.nonzero(g.rationale_mask, as_tuple=False).flatten().tolist()) for g in cal_graphs]
    test_truths = [set(torch.nonzero(g.rationale_mask, as_tuple=False).flatten().tolist()) for g in test_graphs]
    calibration_rows, calibrated = [], {}
    for policy in POLICIES:
        index, certified = calibrate(cal_scores, cal_truths, cal_orbits, policy)
        calibrated[policy] = float(FRACTIONS[index])
        calibration_rows.append({"cell_id": cell_id, "policy": policy, "nominal_fraction": FRACTIONS[index], "crc_certified": certified, "n_calibration": len(cal_graphs)})

    model = GraphClassifier(model_kind, partitions["fit"][0].x.shape[1]).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["state_dict"])
    model.eval()
    baseline_scores, baseline_logits = ig_scores(model, test_graphs, device)
    if not all(np.isfinite(row).all() and len(row) == graph.num_nodes for row, graph in zip(baseline_scores, test_graphs)):
        raise ValueError(f"{cell_id}: nonfinite or misaligned baseline IG scores")
    cache_consistency = score_consistency(cached_baseline_scores, baseline_scores)
    permuted, metadata = [], []
    for molecule_index, graph in enumerate(test_graphs):
        source_index = int(graph.source_index)
        seen = set()
        for repetition in range(repetitions):
            attempt = 0
            while True:
                seed_text = f"{cell_id}|{source_index}|{repetition}|{attempt}"
                seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16)
                permutation = np.random.default_rng(seed).permutation(graph.num_nodes)
                key = tuple(permutation.tolist())
                if key != tuple(range(graph.num_nodes)) and key not in seen:
                    seen.add(key)
                    break
                attempt += 1
            permuted.append(permuted_graph(graph, permutation))
            metadata.append((molecule_index, source_index, repetition, permutation))
    permuted_scores, permuted_logits = ig_scores(model, permuted, device)
    if len(permuted_scores) != len(metadata) or not np.isfinite(permuted_logits).all():
        raise ValueError(f"{cell_id}: incomplete or nonfinite permuted outputs")
    if not all(np.isfinite(row).all() and len(row) == graph.num_nodes for row, graph in zip(permuted_scores, permuted)):
        raise ValueError(f"{cell_id}: nonfinite or misaligned permuted IG scores")
    score_rows, set_rows = [], []
    for scores_new, logits_new, meta in zip(permuted_scores, permuted_logits, metadata):
        molecule_index, source_index, repetition, permutation = meta
        mapped = np.empty_like(scores_new)
        mapped[permutation] = scores_new
        original = np.asarray(baseline_scores[molecule_index])
        ranks_a, ranks_b = rankdata(original), rankdata(mapped)
        variable_a, variable_b = bool(np.std(ranks_a)), bool(np.std(ranks_b))
        rho = float(np.corrcoef(ranks_a, ranks_b)[0, 1]) if variable_a and variable_b else None
        constant_status = "both_variable" if variable_a and variable_b else "both_constant" if not variable_a and not variable_b else "one_constant"
        original_logits = baseline_logits[molecule_index]
        score_rows.append({
            "cell_id": cell_id, "family": family, "task": task, "model": model_kind,
            "source_index": source_index, "repetition": repetition,
            "score_spearman": rho, "max_abs_score_drift": float(np.max(np.abs(original - mapped))),
            "max_abs_logit_drift": float(np.max(np.abs(original_logits - logits_new))),
            "prediction_flip": int(np.argmax(original_logits) != np.argmax(logits_new)), "score_variability": constant_status,
        })
        truth = test_truths[molecule_index]
        original_orbits = test_orbits[molecule_index]
        permuted_orbits = [original_orbits[i] for i in permutation]
        for policy in POLICIES:
            fraction = calibrated[policy]
            nominal_count = 0 if fraction == 0 else int(np.ceil(fraction * len(original)))
            reference = selected_set(original, fraction, policy, original_orbits)
            chosen_new = selected_set(scores_new, fraction, policy, permuted_orbits)
            chosen = {int(permutation[i]) for i in chosen_new}
            original_tie = selected_set(original, fraction, "include_all_exact_ties_v2", original_orbits)
            permuted_tie_new = selected_set(scores_new, fraction, "include_all_exact_ties_v2", permuted_orbits)
            permuted_tie = {int(permutation[i]) for i in permuted_tie_new}
            original_orbit_inflation = len(reference) - len(original_tie) if policy == "rdkit_symmetry_orbit_closure_v3" else 0
            permuted_orbit_inflation = len(chosen) - len(permuted_tie) if policy == "rdkit_symmetry_orbit_closure_v3" else 0
            set_rows.append({
                "cell_id": cell_id, "family": family, "task": task, "model": model_kind,
                "source_index": source_index, "repetition": repetition, "policy": policy,
                "nominal_fraction": fraction,
                "mapped_set_jaccard": len(reference & chosen) / len(reference | chosen) if reference | chosen else 1.0,
                "missed_rationale_loss": 1 - len(chosen & truth) / len(truth),
                "retained_fraction": len(chosen) / len(original), "nominal_atom_count": nominal_count,
                "selected_atom_count": len(chosen),
                "original_exact_tie_expansion_atoms": len(original_tie) - nominal_count,
                "permuted_exact_tie_expansion_atoms": len(permuted_tie) - nominal_count,
                "original_exact_tie_expanded": len(original_tie) > nominal_count,
                "permuted_exact_tie_expanded": len(permuted_tie) > nominal_count,
                "orbit_split": orbit_split(chosen, original_orbits),
                "original_orbit_closure_inflation_atoms": original_orbit_inflation,
                "size_inflation_vs_tie_atoms": permuted_orbit_inflation,
                "orbit_closure_inflation_fraction": permuted_orbit_inflation / len(original),
            })
    pd.DataFrame(score_rows).to_csv(score_path, index=False, compression="gzip")
    pd.DataFrame(set_rows).to_csv(set_path, index=False, compression="gzip")
    pd.DataFrame(calibration_rows).to_csv(calibration_path, index=False)
    write_json(result_path, {
        "status": "complete", "cell_id": cell_id, "family": family, "task": task, "model": model_kind,
        "seed": 42, "test_molecules": len(test_graphs), "permutations_per_molecule": repetitions,
        "calibrated_fractions": calibrated, "cache_true_score_consistency": cache_consistency,
        "permutation_policy": "unique deterministic nonidentity node permutations",
        "provenance": provenance,
    })
    print(f"cell_complete={cell_id}", flush=True)


def aggregate(out_root, selector_audit, n_molecules=25, repetitions=20):
    score = pd.concat([pd.read_csv(p) for p in sorted((out_root / "rows").glob("*_scores.csv.gz"))], ignore_index=True)
    sets = pd.concat([pd.read_csv(p) for p in sorted((out_root / "rows").glob("*_sets.csv.gz"))], ignore_index=True)
    calibration = pd.concat([pd.read_csv(p) for p in sorted((out_root / "rows").glob("*_calibration.csv"))], ignore_index=True)
    score.to_csv(out_root / "permutation_score_results.csv.gz", index=False, compression="gzip")
    sets.to_csv(out_root / "permutation_set_results.csv.gz", index=False, compression="gzip")
    calibration.to_csv(out_root / "policy_calibration.csv", index=False)
    policy_summary = sets.groupby("policy")[[
        "mapped_set_jaccard", "missed_rationale_loss", "retained_fraction", "orbit_split",
        "original_exact_tie_expanded", "permuted_exact_tie_expanded",
        "original_exact_tie_expansion_atoms", "permuted_exact_tie_expansion_atoms",
        "size_inflation_vs_tie_atoms", "orbit_closure_inflation_fraction",
    ]].mean().reset_index()
    policy_summary.to_csv(out_root / "policy_summary.csv", index=False)
    expected_ids = {
        f"{family}__{task}__{model}__seed42"
        for family, tasks in (("bxaic", BXAIC_TASKS), ("google", GOOGLE_TASKS))
        for task in tasks for model in ("gin", "gcn")
    }
    observed_ids = set(score["cell_id"])
    expected_scores = len(expected_ids) * n_molecules * repetitions
    expected_sets = expected_scores * len(POLICIES)
    per_molecule_counts = score.groupby(["cell_id", "source_index"]).size()
    structural_pass = (
        observed_ids == expected_ids and len(score) == expected_scores and len(sets) == expected_sets
        and len(calibration) == len(expected_ids) * len(POLICIES)
        and per_molecule_counts.eq(repetitions).all()
        and score.groupby("cell_id")["source_index"].nunique().eq(n_molecules).all()
        and sets.groupby(["cell_id", "source_index", "policy"]).size().eq(repetitions).all()
    )
    finite_core = bool(np.isfinite(sets[["mapped_set_jaccard", "missed_rationale_loss", "retained_fraction", "size_inflation_vs_tie_atoms"]].to_numpy()).all())
    cell_payloads = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((out_root / "cells").glob("*.json"))]
    cache_max_difference = max(payload["cache_true_score_consistency"]["max_abs_difference"] for payload in cell_payloads)
    summary = {
        "status": "PASS" if structural_pass and finite_core else "PARTIAL", "reviewer_item": "A6",
        "cells": int(score["cell_id"].nunique()), "molecule_permutation_rows": len(score), "policy_rows": len(sets),
        "mean_score_spearman": float(score["score_spearman"].mean()),
        "finite_score_spearman_rows": int(score["score_spearman"].notna().sum()),
        "undefined_score_spearman_rows": int(score["score_spearman"].isna().sum()),
        "max_abs_score_drift": float(score["max_abs_score_drift"].max()),
        "max_abs_logit_drift": float(score["max_abs_logit_drift"].max()),
        "prediction_flips": int(score["prediction_flip"].sum()),
        "cache_true_score_max_abs_difference": float(cache_max_difference),
        "policy_summary": policy_summary.to_dict("records"),
        "selector_only_audit_sha256": sha256(selector_audit),
        "selector_only_audit": json.loads(selector_audit.read_text(encoding="utf-8")),
        "interpretation": "This full graph re-attribution audit is distinct from the selector-only cached-score permutation check. RDKit orbit closure is reported as a graph-symmetry sensitivity policy.",
    }
    write_json(out_root / "summary.json", summary)
    write_json(out_root / "validation.json", {
        "status": "PASS" if structural_pass and finite_core else "PARTIAL",
        "expected_score_rows": expected_scores, "observed_score_rows": len(score),
        "expected_policy_rows": expected_sets, "observed_policy_rows": len(sets),
        "exact_expected_cell_ids": observed_ids == expected_ids,
        "all_policy_calibrations_present": len(calibration) == len(expected_ids) * len(POLICIES),
        "per_molecule_repetitions_complete": bool(per_molecule_counts.eq(repetitions).all()),
        "sampled_molecules_per_cell_complete": bool(score.groupby("cell_id")["source_index"].nunique().eq(n_molecules).all()),
        "finite_core_outputs": finite_core,
        "cache_true_score_max_abs_difference": float(cache_max_difference),
        "cache_consistency_within_1e-5": cache_max_difference <= 1e-5,
    })
    print(json.dumps(json_ready(summary), indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--selector-audit", type=Path, default=DEFAULT_SELECTOR_AUDIT)
    args = parser.parse_args()
    out_root = HERE / "smoke" if args.smoke else HERE
    (out_root / "cells").mkdir(parents=True, exist_ok=True)
    (out_root / "rows").mkdir(parents=True, exist_ok=True)
    jobs = [("bxaic", task) for task in BXAIC_TASKS] + [("google", task) for task in GOOGLE_TASKS]
    if args.smoke:
        jobs = jobs[:1]
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    run_mode = "smoke" if args.smoke else "full"
    n_molecules, repetitions = (3, 2) if args.smoke else (25, 20)
    for family, task in jobs:
        partitions = bxaic_partitions(PROJECT / "data/raw/bxaic/data.csv", PROJECT / "data/raw/bxaic/explanations.sdf", task) if family == "bxaic" else google_partitions(PROJECT / "reference/graph-attribution/data", task)
        for model_kind in (["gin"] if args.smoke else ["gin", "gcn"]):
            one_cell(family, task, model_kind, partitions, device, n_molecules, repetitions, out_root, args.cache_dir, run_mode)
    if not args.smoke:
        aggregate(out_root, args.selector_audit, n_molecules, repetitions)
    write_json(out_root / "environment.json", {
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S %z"), "platform": platform.platform(),
        "python": sys.version, "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "device": args.device, "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "rdkit": Chem.rdBase.rdkitVersion, "run_mode": run_mode,
        "script_sha256": sha256(__file__), "contract_sha256": sha256(CONTRACT),
    })


if __name__ == "__main__":
    main()

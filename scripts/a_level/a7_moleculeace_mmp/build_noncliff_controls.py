import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from build_mmp_pairs import TARGETS, build_pairs, cut_records, fingerprints


FEATURES = ["core_heavy_atoms", "variable_total", "atom_total", "atom_difference"]


def pair_features(rows):
    return np.asarray(
        [
            [
                row["core_heavy_atoms"],
                row["variable_heavy_atoms_a"] + row["variable_heavy_atoms_b"],
                row["n_atoms_a"] + row["n_atoms_b"],
                abs(row["n_atoms_a"] - row["n_atoms_b"]),
            ]
            for row in rows
        ],
        dtype=float,
    )


def disjoint_control_pool(candidates, cliffs):
    candidate_features = pair_features(candidates)
    cliff_features = pair_features(cliffs)
    pooled = np.vstack([candidate_features, cliff_features])
    scale = pooled.std(axis=0, ddof=0)
    scale[scale == 0] = 1.0
    distances = cdist(candidate_features / scale, cliff_features / scale)
    nearest = distances.min(axis=1)
    graph = nx.Graph()
    for index, row in enumerate(candidates):
        tie = int(hashlib.sha256(f"{row['smiles_a']}|{row['smiles_b']}".encode()).hexdigest()[:8], 16) % 1000
        weight = -int(round(nearest[index] * 1_000_000)) + tie
        graph.add_edge(row["smiles_a"], row["smiles_b"], row_index=index, weight=weight)
    matching = nx.algorithms.matching.max_weight_matching(graph, maxcardinality=True, weight="weight")
    indices = sorted(graph.edges[a, b]["row_index"] for a, b in matching)
    return [dict(candidates[index]) for index in indices]


def match_controls(pool, cliffs):
    control_features = pair_features(pool)
    cliff_features = pair_features(cliffs)
    pooled = np.vstack([control_features, cliff_features])
    scale = pooled.std(axis=0, ddof=0)
    scale[scale == 0] = 1.0
    costs = cdist(cliff_features / scale, control_features / scale)
    cliff_indices, control_indices = linear_sum_assignment(costs)
    matched_controls = []
    matched_cliffs = []
    for cliff_index, control_index in zip(cliff_indices, control_indices):
        row = dict(pool[control_index])
        row["matched_cliff_pair_id"] = int(cliffs[cliff_index]["pair_id"])
        row["structural_match_distance"] = float(costs[cliff_index, control_index])
        matched_controls.append(row)
        matched_cliffs.append(cliffs[cliff_index])
    return matched_controls, matched_cliffs


def balance_rows(target, controls, cliffs):
    control_features = pair_features(controls)
    cliff_features = pair_features(cliffs)
    rows = []
    for index, feature in enumerate(FEATURES):
        pooled_sd = math_sqrt((control_features[:, index].var(ddof=1) + cliff_features[:, index].var(ddof=1)) / 2)
        smd = (control_features[:, index].mean() - cliff_features[:, index].mean()) / pooled_sd if pooled_sd else 0.0
        rows.append(
            {
                "dataset": target,
                "feature": feature,
                "control_mean": float(control_features[:, index].mean()),
                "cliff_mean": float(cliff_features[:, index].mean()),
                "standardized_mean_difference": float(smd),
                "absolute_smd": abs(float(smd)),
            }
        )
    return rows


def math_sqrt(value):
    return float(np.sqrt(max(0.0, value)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cliff-pairs", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    usecols = ["canonical_smiles", "activity", "exp_nm", "split", "group_id", "dataset", "environment"]
    frame = pd.read_csv(args.manifest, usecols=usecols)
    frame = frame[(frame.environment == "mmp_series") & (frame.split == "test") & frame.dataset.isin(TARGETS)].copy()
    cliffs = pd.read_csv(args.cliff_pairs)
    cliffs = cliffs[cliffs.split == "test"].copy()
    audit = Counter()
    record_cache = {}
    atom_count_cache = {}
    fingerprint_cache = {}
    for position, smiles in enumerate(sorted(frame.canonical_smiles.unique()), start=1):
        records, atom_count, _ = cut_records(smiles, audit)
        record_cache[smiles] = records
        atom_count_cache[smiles] = atom_count
        fingerprint_cache[smiles] = fingerprints(smiles)
        if position % 500 == 0:
            print(f"control fragmentation {position}/{frame.canonical_smiles.nunique()}", flush=True)
    output_rows = []
    count_rows = []
    all_balance = []
    for target in TARGETS:
        target_frame = frame[frame.dataset == target].reset_index(drop=True)
        target_cliffs_frame = cliffs[cliffs.dataset == target].reset_index(drop=True)
        target_cliffs = target_cliffs_frame.to_dict("records")
        forbidden = set(target_cliffs_frame.smiles_a) | set(target_cliffs_frame.smiles_b)
        local_audit = Counter()
        candidates = build_pairs(
            target_frame, record_cache, atom_count_cache, fingerprint_cache, local_audit, pair_mode="noncliff"
        )
        candidates = [row for row in candidates if row["smiles_a"] not in forbidden and row["smiles_b"] not in forbidden]
        pool = disjoint_control_pool(candidates, target_cliffs)
        controls, matched_cliffs = match_controls(pool, target_cliffs)
        for control_id, row in enumerate(controls):
            row.update({"dataset": target, "split": "test", "pair_id": control_id, "pair_type": "matched_noncliff_control"})
        output_rows.extend(controls)
        balance = balance_rows(target, controls, matched_cliffs)
        all_balance.extend(balance)
        max_smd = max(row["absolute_smd"] for row in balance)
        count_rows.append(
            {
                "dataset": target,
                "n_test_cliff_pairs": len(target_cliffs),
                "n_noncliff_candidates_after_cliff_molecule_exclusion": len(candidates),
                "n_disjoint_control_pool": len(pool),
                "n_matched_controls": len(controls),
                "max_absolute_smd": max_smd,
                "minimum_20_controls": len(controls) >= 20,
                "balance_gate_0_25": max_smd <= 0.25,
            }
        )
        print(f"{target}: {len(controls)} controls, max |SMD|={max_smd:.3f}", flush=True)
    controls = pd.DataFrame(output_rows)
    controls = controls[["dataset", "split", "pair_id", "pair_type"] + [column for column in controls if column not in {"dataset", "split", "pair_id", "pair_type"}]]
    counts = pd.DataFrame(count_rows)
    balance = pd.DataFrame(all_balance)
    controls.to_csv(output_dir / "noncliff_matched_controls.csv", index=False)
    counts.to_csv(output_dir / "noncliff_control_counts.csv", index=False)
    balance.to_csv(output_dir / "noncliff_control_balance.csv", index=False)
    molecules = list(controls.smiles_a) + list(controls.smiles_b)
    summary = {
        "status": "PASS" if counts.minimum_20_controls.all() and counts.balance_gate_0_25.all() else "DESCRIPTIVE_ONLY",
        "definition": "Exact-single-cut, MoleculeACE-similar MMPs with measured absolute pActivity difference <= 0.5.",
        "cliff_molecule_overlap": 0,
        "within_control_molecule_reuse": len(molecules) - len(set((row.dataset, molecule) for row in controls.itertuples() for molecule in [row.smiles_a, row.smiles_b])),
        "n_controls": int(len(controls)),
        "targets_with_at_least_20": int(counts.minimum_20_controls.sum()),
        "targets_passing_balance": int(counts.balance_gate_0_25.sum()),
        "max_absolute_smd": float(balance.absolute_smd.max()),
        "audit": dict(audit),
        "interpretation": "Matched noncliff controls are descriptive specificity checks and do not provide causal rationale ground truth."
    }
    (output_dir / "noncliff_control_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

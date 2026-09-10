import argparse
import hashlib
import json
import platform
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import networkx as nx
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger, rdBase
from rdkit.Chem import AllChem
from rdkit.Chem.rdMMPA import FragmentMol
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol, MakeScaffoldGeneric


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
EXPECTED = {
    "CHEMBL204_Ki": (59, 40),
    "CHEMBL214_Ki": (47, 31),
    "CHEMBL228_Ki": (52, 34),
    "CHEMBL233_Ki": (75, 52),
    "CHEMBL234_Ki": (64, 36),
    "CHEMBL235_EC50": (30, 29),
    "CHEMBL236_Ki": (34, 32),
    "CHEMBL237_Ki": (41, 44),
    "CHEMBL244_Ki": (81, 110),
    "CHEMBL264_Ki": (47, 28),
    "CHEMBL4792_Ki": (35, 43),
}

RDLogger.DisableLog("rdApp.warning")
MMP_BOND_PATTERN = Chem.MolFromSmarts("[#6+0;!$(*=,#[!#6])]!@!=!#[*]")


def levenshtein_distance(a, b):
    if len(a) > len(b):
        a, b = b, a
    if not a:
        return len(b)
    masks = defaultdict(int)
    for index, character in enumerate(a):
        masks[character] |= 1 << index
    width_mask = (1 << len(a)) - 1
    high_bit = 1 << (len(a) - 1)
    positive = width_mask
    negative = 0
    score = len(a)
    for character in b:
        equal = masks[character]
        vertical = equal | negative
        horizontal = (((equal & positive) + positive) ^ positive) | equal
        positive_step = negative | ~(horizontal | positive)
        negative_step = positive & horizontal
        if positive_step & high_bit:
            score += 1
        elif negative_step & high_bit:
            score -= 1
        positive_step = ((positive_step << 1) | 1) & width_mask
        negative_step = (negative_step << 1) & width_mask
        positive = (negative_step | ~(vertical | positive_step)) & width_mask
        negative = positive_step & vertical
    return score


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def index_text(values):
    return ";".join(str(int(value)) for value in sorted(values))


def symmetry_orbits(mol):
    matches = mol.GetSubstructMatches(mol, uniquify=False, useChirality=True, maxMatches=100000)
    orbits = [{i} for i in range(mol.GetNumAtoms())]
    for match in matches:
        for query_index, target_index in enumerate(match):
            orbits[query_index].add(target_index)
    return orbits, len(matches), len(matches) == 100000


def closure(indices, orbits):
    output = set()
    for index in indices:
        output.update(orbits[index])
    return output


def clean_fragment_smiles(fragment):
    for atom in fragment.GetAtoms():
        atom.SetAtomMapNum(1 if atom.GetAtomicNum() == 0 else 0)
        atom.SetIsotope(0)
    return Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=True)


def reference_core_keys(mol):
    keys = set()
    for _, fragments in FragmentMol(mol, maxCuts=1, resultsAsMols=False):
        parts = fragments.split(".")
        parsed = [(part, Chem.MolFromSmiles(part)) for part in parts]
        valid = [(part, item.GetNumHeavyAtoms()) for part, item in parsed if item is not None]
        if valid:
            core = max(valid, key=lambda item: (item[1], item[0]))[0]
            keys.add(Chem.MolToSmiles(Chem.MolFromSmiles(core), canonical=True, isomericSmiles=True))
    return keys


def fingerprints(smiles):
    mol = Chem.MolFromSmiles(smiles)
    morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    try:
        scaffold_mol = MakeScaffoldGeneric(mol)
    except Exception:
        scaffold_mol = GetScaffoldForMol(mol)
    scaffold = AllChem.GetMorganFingerprintAsBitVect(scaffold_mol, 2, nBits=1024)
    return morgan, scaffold


def is_frozen_similar_pair(row_a, row_b, fingerprint_cache):
    morgan_a, scaffold_a = fingerprint_cache[row_a.canonical_smiles]
    morgan_b, scaffold_b = fingerprint_cache[row_b.canonical_smiles]
    morgan_similarity = DataStructs.TanimotoSimilarity(morgan_a, morgan_b)
    scaffold_similarity = DataStructs.TanimotoSimilarity(scaffold_a, scaffold_b)
    smiles_similarity = 1.0 - levenshtein_distance(row_a.canonical_smiles, row_b.canonical_smiles) / max(
        len(row_a.canonical_smiles), len(row_b.canonical_smiles)
    )
    similar = morgan_similarity >= 0.9 or scaffold_similarity >= 0.9 or smiles_similarity >= 0.9
    return similar


def cut_records(smiles, audit):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        audit["invalid_smiles"] += 1
        return [], None, {"invalid_smiles": True, "eligible_bonds": 0, "fragmentation_limit_exclusion": False, "mapping_mismatch": False}
    for atom in mol.GetAtoms():
        atom.SetIntProp("_orig_idx", atom.GetIdx())
    orbits, automorphisms, truncated = symmetry_orbits(mol)
    audit["automorphism_match_total"] += automorphisms
    audit["automorphism_truncated_molecules"] += int(truncated)
    records = []
    eligible_bonds = sorted(
        {
            mol.GetBondBetweenAtoms(match[0], match[1]).GetIdx()
            for match in mol.GetSubstructMatches(MMP_BOND_PATTERN, uniquify=True)
        }
    )
    for bond_index in eligible_bonds:
        bond = mol.GetBondWithIdx(bond_index)
        fragmented = Chem.FragmentOnBonds(
            mol,
            [bond.GetIdx()],
            addDummies=True,
            dummyLabels=[(0, 0)],
        )
        fragments = Chem.GetMolFrags(fragmented, asMols=True, sanitizeFrags=True)
        if len(fragments) != 2:
            audit["nonbinary_fragmentation"] += 1
            continue
        parts = []
        for fragment in fragments:
            originals = sorted(
                atom.GetIntProp("_orig_idx")
                for atom in fragment.GetAtoms()
                if atom.HasProp("_orig_idx")
            )
            dummy_neighbors = []
            for atom in fragment.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    dummy_neighbors.extend(
                        neighbor.GetIntProp("_orig_idx")
                        for neighbor in atom.GetNeighbors()
                        if neighbor.HasProp("_orig_idx")
                    )
            parts.append(
                {
                    "n_heavy": len(originals),
                    "smiles": clean_fragment_smiles(fragment),
                    "indices": originals,
                    "attachment": sorted(set(dummy_neighbors)),
                }
            )
        core_index = max(range(2), key=lambda index: (parts[index]["n_heavy"], parts[index]["smiles"]))
        variable_index = 1 - core_index
        core = parts[core_index]
        variable = parts[variable_index]
        audit["equal_size_cut_lexically_resolved"] += int(parts[0]["n_heavy"] == parts[1]["n_heavy"])
        if len(core["attachment"]) != 1 or len(variable["attachment"]) != 1:
            audit["attachment_mapping_failure"] += 1
            continue
        variable_raw = set(variable["indices"])
        main_raw = variable_raw | set(core["attachment"])
        expanded_raw = set(main_raw)
        for index in list(main_raw):
            expanded_raw.update(neighbor.GetIdx() for neighbor in mol.GetAtomWithIdx(index).GetNeighbors())
        records.append(
            {
                "core_smiles": core["smiles"],
                "variable_smiles": variable["smiles"],
                "core_heavy_atoms": core["n_heavy"],
                "variable_heavy_atoms": variable["n_heavy"],
                "core_attachment_atom": core["attachment"][0],
                "variable_atoms_raw": tuple(sorted(variable_raw)),
                "main_atoms_raw": tuple(sorted(main_raw)),
                "variable_atoms": tuple(sorted(closure(variable_raw, orbits))),
                "main_atoms": tuple(sorted(closure(main_raw, orbits))),
                "expanded_atoms": tuple(sorted(closure(expanded_raw, orbits))),
                "cut_bond": tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))),
            }
        )
    audit["molecules_without_eligible_cut"] += int(not eligible_bonds)
    audit["molecules_over_rdmmpa_max_cut_bonds"] += int(len(eligible_bonds) > 20)
    audit["eligible_cut_bonds"] += len(eligible_bonds)
    unique = {}
    for record in records:
        key = (
            record["core_smiles"],
            record["variable_smiles"],
            record["variable_atoms_raw"],
            record["main_atoms_raw"],
        )
        unique[key] = record
    audit["duplicate_cut_records_removed"] += len(records) - len(unique)
    records = list(unique.values())
    observed_keys = {record["core_smiles"] for record in records}
    expected_keys = reference_core_keys(mol)
    audit["rdmmpa_reference_core_keys"] += len(expected_keys)
    audit["raw_mapped_core_keys"] += len(observed_keys)
    audit["rdmmpa_reference_only_core_keys"] += len(expected_keys - observed_keys)
    audit["rdmmpa_filtered_extra_core_keys"] += len(observed_keys - expected_keys)
    records = [record for record in records if record["core_smiles"] in expected_keys]
    final_keys = {record["core_smiles"] for record in records}
    audit["mapped_core_keys"] += len(final_keys)
    mismatch = final_keys != expected_keys
    audit["rdmmpa_core_mismatch_molecules"] += int(mismatch)
    audit["rdmmpa_core_mismatch_molecules_excluded"] += int(mismatch)
    if mismatch:
        records = []
    fragmentation_limit_exclusion = len(eligible_bonds) > 20 and not expected_keys
    audit["rdmmpa_fragmentation_limit_exclusions"] += int(fragmentation_limit_exclusion)
    return records, mol.GetNumAtoms(), {
        "invalid_smiles": False,
        "eligible_bonds": len(eligible_bonds),
        "fragmentation_limit_exclusion": fragmentation_limit_exclusion,
        "mapping_mismatch": mismatch,
    }


def build_pairs(frame, record_cache, atom_count_cache, fingerprint_cache, audit, pair_mode="cliff"):
    core_index = defaultdict(list)
    for row_index, row in frame.iterrows():
        for record in record_cache[row.canonical_smiles]:
            core_index[record["core_smiles"]].append((row_index, record))
    candidates = defaultdict(list)
    for core_smiles, items in core_index.items():
        for (index_a, record_a), (index_b, record_b) in combinations(items, 2):
            if index_a == index_b:
                continue
            if record_a["variable_smiles"] == record_b["variable_smiles"]:
                audit["candidate_same_variable"] += 1
                continue
            activity_delta = abs(float(frame.at[index_a, "activity"]) - float(frame.at[index_b, "activity"]))
            activity_pass = activity_delta > 1.0 if pair_mode == "cliff" else activity_delta <= 0.5
            if not activity_pass:
                audit[f"candidate_not_{pair_mode}_activity"] += 1
                continue
            if not is_frozen_similar_pair(frame.loc[index_a], frame.loc[index_b], fingerprint_cache):
                audit["candidate_not_similar"] += 1
                continue
            key = tuple(sorted((index_a, index_b)))
            candidates[key].append((core_smiles, record_a, record_b))
    rows = []
    for (index_a, index_b), matches in sorted(candidates.items()):
        max_core = max(match[1]["core_heavy_atoms"] for match in matches)
        best = [match for match in matches if match[1]["core_heavy_atoms"] == max_core]
        signatures = {
            (
                match[0],
                match[1]["main_atoms_raw"],
                match[2]["main_atoms_raw"],
                match[1]["variable_atoms_raw"],
                match[2]["variable_atoms_raw"],
            )
            for match in best
        }
        if len(signatures) != 1:
            audit["ambiguous_pair_mapping_excluded"] += 1
            continue
        core_smiles, record_a, record_b = sorted(
            best,
            key=lambda match: (
                match[0],
                match[1]["variable_smiles"],
                match[2]["variable_smiles"],
            ),
        )[0]
        row_a = frame.loc[index_a]
        row_b = frame.loc[index_b]
        if str(row_a.group_id) != str(row_b.group_id):
            audit["candidate_cross_component_excluded"] += 1
            continue
        rows.append(
            {
                "component_id": str(row_a.group_id),
                "smiles_a": row_a.canonical_smiles,
                "smiles_b": row_b.canonical_smiles,
                "activity_a": float(row_a.activity),
                "activity_b": float(row_b.activity),
                "exp_nm_a": float(row_a.exp_nm),
                "exp_nm_b": float(row_b.exp_nm),
                "measured_delta_a_minus_b": float(row_a.activity - row_b.activity),
                "measured_abs_delta": abs(float(row_a.activity - row_b.activity)),
                "core_smiles": core_smiles,
                "variable_smiles_a": record_a["variable_smiles"],
                "variable_smiles_b": record_b["variable_smiles"],
                "core_heavy_atoms": record_a["core_heavy_atoms"],
                "variable_heavy_atoms_a": record_a["variable_heavy_atoms"],
                "variable_heavy_atoms_b": record_b["variable_heavy_atoms"],
                "n_atoms_a": atom_count_cache[row_a.canonical_smiles],
                "n_atoms_b": atom_count_cache[row_b.canonical_smiles],
                "core_attachment_a": record_a["core_attachment_atom"],
                "core_attachment_b": record_b["core_attachment_atom"],
                "variable_mask_a": index_text(record_a["variable_atoms"]),
                "variable_mask_b": index_text(record_b["variable_atoms"]),
                "main_mask_a": index_text(record_a["main_atoms"]),
                "main_mask_b": index_text(record_b["main_atoms"]),
                "expanded_mask_a": index_text(record_a["expanded_atoms"]),
                "expanded_mask_b": index_text(record_b["expanded_atoms"]),
                "variable_mask_raw_a": index_text(record_a["variable_atoms_raw"]),
                "variable_mask_raw_b": index_text(record_b["variable_atoms_raw"]),
                "main_mask_raw_a": index_text(record_a["main_atoms_raw"]),
                "main_mask_raw_b": index_text(record_b["main_atoms_raw"]),
                "cut_bond_a": index_text(record_a["cut_bond"]),
                "cut_bond_b": index_text(record_b["cut_bond"]),
            }
        )
    audit["mapped_candidate_pairs"] += len(rows)
    return rows


def disjoint_pairs(rows):
    graph = nx.Graph()
    for index, row in enumerate(rows):
        tie_break = int(hashlib.sha256(f"{row['smiles_a']}|{row['smiles_b']}".encode()).hexdigest()[:8], 16) % 1000
        weight = int(row["core_heavy_atoms"]) * 1_000_000
        weight -= int(row["variable_heavy_atoms_a"] + row["variable_heavy_atoms_b"]) * 1_000
        weight += tie_break
        graph.add_edge(row["smiles_a"], row["smiles_b"], row_index=index, weight=weight)
    matching = nx.algorithms.matching.max_weight_matching(graph, maxcardinality=True, weight="weight")
    selected_indices = sorted(graph.edges[a, b]["row_index"] for a, b in matching)
    selected = [dict(rows[index]) for index in selected_indices]
    selected.sort(key=lambda row: (row["smiles_a"], row["smiles_b"]))
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    manifest = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_hash = "C71E0E2624AFFE872E368131517FA1A565B81520FED25F87CA4C64B53F83174F"
    observed_hash = sha256(manifest)
    if observed_hash != expected_hash:
        raise RuntimeError(f"Manifest hash mismatch: {observed_hash}")
    usecols = [
        "canonical_smiles",
        "activity",
        "exp_nm",
        "split",
        "group_id",
        "dataset",
        "environment",
        "official_split",
        "scaffold",
    ]
    frame = pd.read_csv(manifest, usecols=usecols)
    frame = frame[(frame.environment == "mmp_series") & frame.dataset.isin(TARGETS)].copy()
    crc_frame = frame[frame.split.isin(["calibration", "test"])].copy()
    audit = Counter()
    record_cache = {}
    atom_count_cache = {}
    fingerprint_cache = {}
    mapping_exclusions = []
    fragmentation_limit_exclusions = []
    no_eligible_bond_exclusions = []
    for position, smiles in enumerate(sorted(crc_frame.canonical_smiles.unique()), start=1):
        records, atom_count, metadata = cut_records(smiles, audit)
        occurrences = crc_frame.loc[crc_frame.canonical_smiles == smiles, ["dataset", "split", "group_id"]].drop_duplicates()
        if metadata["mapping_mismatch"]:
            mapping_exclusions.append({"canonical_smiles": smiles, "reason": "rdMMPA core-set mapping mismatch"})
        if metadata["fragmentation_limit_exclusion"]:
            for occurrence in occurrences.itertuples(index=False):
                fragmentation_limit_exclusions.append(
                    {
                        "dataset": occurrence.dataset,
                        "split": occurrence.split,
                        "group_id": occurrence.group_id,
                        "canonical_smiles": smiles,
                        "n_atoms": atom_count,
                        "eligible_mmp_bonds": metadata["eligible_bonds"],
                        "rdmmpa_max_cut_bonds": 20,
                        "reason": "RDKit rdMMPA default maxCutBonds=20 returns no reference fragmentation",
                    }
                )
        if metadata["eligible_bonds"] == 0:
            no_eligible_bond_exclusions.append(
                {"canonical_smiles": smiles, "n_atoms": atom_count, "reason": "no bond matches RDKit rdMMPA default cut SMARTS"}
            )
        record_cache[smiles] = records
        atom_count_cache[smiles] = atom_count
        fingerprint_cache[smiles] = fingerprints(smiles)
        if position % 500 == 0:
            print(f"fragmented {position}/{crc_frame.canonical_smiles.nunique()}", flush=True)
    candidate_rows = []
    pair_rows = []
    count_rows = []
    for target in TARGETS:
        for split in ["calibration", "test"]:
            subset = crc_frame[(crc_frame.dataset == target) & (crc_frame.split == split)].reset_index(drop=True)
            local_audit = Counter()
            candidates = build_pairs(subset, record_cache, atom_count_cache, fingerprint_cache, local_audit)
            for candidate_id, row in enumerate(candidates):
                row.update({"dataset": target, "split": split, "candidate_pair_id": candidate_id})
            rows = disjoint_pairs(candidates)
            for pair_id, row in enumerate(rows):
                row.update({"dataset": target, "split": split, "pair_id": pair_id})
            candidate_rows.extend(candidates)
            pair_rows.extend(rows)
            expected = EXPECTED[target][0 if split == "calibration" else 1]
            count_rows.append(
                {
                    "dataset": target,
                    "split": split,
                    "n_molecules": len(subset),
                    "n_candidate_pairs": len(candidates),
                    "n_pairs": len(rows),
                    "expected_scout_pairs": expected,
                    "matches_scout_count": len(rows) == expected,
                    "minimum_20_pass": len(rows) >= 20,
                    **dict(local_audit),
                }
            )
            audit.update(local_audit)
            audit["selected_disjoint_pairs"] += len(rows)
            print(f"{target} {split}: {len(rows)} disjoint from {len(candidates)} candidates (scout {expected})", flush=True)
    candidates = pd.DataFrame(candidate_rows)
    pairs = pd.DataFrame(pair_rows)
    counts = pd.DataFrame(count_rows).fillna(0)
    pairs = pairs[
        ["dataset", "split", "pair_id"]
        + [column for column in pairs.columns if column not in {"dataset", "split", "pair_id"}]
    ]
    pairs.to_csv(output_dir / "mmp_pairs.csv", index=False)
    candidates.to_csv(output_dir / "mmp_pair_candidates.csv", index=False)
    pd.DataFrame(mapping_exclusions, columns=["canonical_smiles", "reason"]).to_csv(
        output_dir / "mmp_mapping_exclusions.csv", index=False
    )
    pd.DataFrame(fragmentation_limit_exclusions).to_csv(output_dir / "fragmentation_limit_exclusions.csv", index=False)
    pd.DataFrame(no_eligible_bond_exclusions).drop_duplicates().to_csv(
        output_dir / "no_eligible_bond_exclusions.csv", index=False
    )
    counts.to_csv(output_dir / "mmp_pair_counts.csv", index=False)
    cluster_sizes = (
        pairs.groupby(["dataset", "split", "component_id"], as_index=False)
        .size()
        .rename(columns={"size": "n_pairs"})
    )
    cluster_rows = []
    for (target, split), subset in cluster_sizes.groupby(["dataset", "split"]):
        sizes = subset.n_pairs.to_numpy(float)
        cluster_rows.append(
            {
                "dataset": target,
                "split": split,
                "n_pairs": int(sizes.sum()),
                "n_components": len(sizes),
                "max_pairs_per_component": int(sizes.max()),
                "median_pairs_per_component": float(np.median(sizes)),
                "kish_effective_components": float(sizes.sum() ** 2 / np.square(sizes).sum()),
                "group_crc_minimum_components_alpha_0_10": 9,
                "group_crc_numerically_feasible_alpha_0_10": bool(split == "calibration" and len(sizes) >= 9),
            }
        )
    cluster_diagnostics = pd.DataFrame(cluster_rows)
    cluster_sizes.to_csv(output_dir / "mmp_component_cluster_sizes.csv", index=False)
    cluster_diagnostics.to_csv(output_dir / "mmp_component_diagnostics.csv", index=False)
    overlap_by_target = {}
    for target in TARGETS:
        target_pairs = pairs[pairs.dataset == target]
        cal_molecules = set(target_pairs.loc[target_pairs.split == "calibration", "smiles_a"]) | set(
            target_pairs.loc[target_pairs.split == "calibration", "smiles_b"]
        )
        test_molecules = set(target_pairs.loc[target_pairs.split == "test", "smiles_a"]) | set(
            target_pairs.loc[target_pairs.split == "test", "smiles_b"]
        )
        overlap_by_target[target] = len(cal_molecules & test_molecules)
    overlap = sum(overlap_by_target.values())
    within_split_reuse = 0
    for (_, _), subset in pairs.groupby(["dataset", "split"]):
        molecules = list(subset.smiles_a) + list(subset.smiles_b)
        within_split_reuse += len(molecules) - len(set(molecules))
    selected_molecules = set(pairs.smiles_a) | set(pairs.smiles_b)
    fragmentation_molecules = {row["canonical_smiles"] for row in fragmentation_limit_exclusions}
    fragmentation_selected_overlap = len(selected_molecules & fragmentation_molecules)
    summary = {
        "status": "PASS"
        if counts.minimum_20_pass.all()
        and not overlap
        and not within_split_reuse
        and not fragmentation_selected_overlap
        and audit["automorphism_truncated_molecules"] == 0
        else "FAIL",
        "manifest_sha256": observed_hash,
        "environment": "mmp_series",
        "targets": len(TARGETS),
        "pairs": int(len(pairs)),
        "calibration_pairs": int((pairs.split == "calibration").sum()),
        "test_pairs": int((pairs.split == "test").sum()),
        "calibration_test_molecule_overlap": overlap,
        "calibration_test_molecule_overlap_by_target": overlap_by_target,
        "within_target_split_molecule_reuse": within_split_reuse,
        "mapping_exclusion_count": len(mapping_exclusions),
        "fragmentation_limit_unique_molecules": len(fragmentation_molecules),
        "fragmentation_limit_target_split_occurrences": len(fragmentation_limit_exclusions),
        "fragmentation_limit_selected_molecule_overlap": fragmentation_selected_overlap,
        "no_eligible_bond_unique_molecules": len(no_eligible_bond_exclusions),
        "all_retained_molecule_core_sets_match_rdmmpa": True,
        "all_target_split_counts_at_least_20": bool(counts.minimum_20_pass.all()),
        "all_counts_match_scout": bool(counts.matches_scout_count.all()),
        "pair_exchangeability_supported": False,
        "calibration_component_range": [
            int(cluster_diagnostics.loc[cluster_diagnostics.split == "calibration", "n_components"].min()),
            int(cluster_diagnostics.loc[cluster_diagnostics.split == "calibration", "n_components"].max()),
        ],
        "targets_group_crc_numerically_feasible_alpha_0_10": int(
            cluster_diagnostics.loc[
                cluster_diagnostics.split == "calibration", "group_crc_numerically_feasible_alpha_0_10"
            ].sum()
        ),
        "statistical_boundary": "Pairs cluster within MMP/scaffold components. Pair-weighted CRC is descriptive and finite-pool conditional; no distribution-free pair-level risk-control or generalization claim is supported.",
        "fragmentation_boundary": "The main line preserves RDKit rdMMPA defaults: cut SMARTS [#6+0;!$(*=,#[!#6])]!@!=!#[*] and maxCutBonds=20. Molecules above that eligible-bond cap receive no reference fragmentation and are structurally excluded.",
        "audit": dict(audit),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "networkx": nx.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
    }
    (output_dir / "mmp_mapping_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

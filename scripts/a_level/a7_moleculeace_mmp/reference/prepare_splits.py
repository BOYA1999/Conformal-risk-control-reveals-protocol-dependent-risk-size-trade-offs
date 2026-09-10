import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdMMPA import FragmentMol
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol

from prepare_data import cliff_pairs


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[b] = a


def scaffold_key(smiles):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = GetScaffoldForMol(mol)
    if scaffold.GetNumAtoms() == 0:
        return f"ACYCLIC::{smiles}"
    return Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=True)


def mmp_cores(smiles):
    mol = Chem.MolFromSmiles(smiles)
    keys = set()
    for _, fragments in FragmentMol(mol, maxCuts=1, resultsAsMols=False):
        parts = fragments.split(".")
        if len(parts) < 2:
            continue
        part_mols = [Chem.MolFromSmiles(part) for part in parts]
        valid = [(part, item.GetNumHeavyAtoms()) for part, item in zip(parts, part_mols) if item is not None]
        if valid:
            core = max(valid, key=lambda item: (item[1], item[0]))[0]
            keys.add(Chem.MolToSmiles(Chem.MolFromSmiles(core), canonical=True, isomericSmiles=True))
    return keys


def stable_order(values, seed):
    return sorted(values, key=lambda value: hashlib.sha256(f"{seed}|{value}".encode()).hexdigest())


def assign_groups(group_by_index, fractions, seed):
    members = defaultdict(list)
    for index, group in enumerate(group_by_index):
        members[group].append(index)
    labels = list(fractions)
    targets = {label: fractions[label] * len(group_by_index) for label in labels}
    counts = {label: 0 for label in labels}
    assignment = {}
    for group in stable_order(members, seed):
        label = max(labels, key=lambda item: (targets[item] - counts[item], -labels.index(item)))
        assignment[group] = label
        counts[label] += len(members[group])
    return [assignment[group] for group in group_by_index]


def original_split(frame, seed, calibration_fraction):
    labels = frame["official_split"].tolist()
    train_indices = [i for i, label in enumerate(labels) if label == "train"]
    train_scaffolds = [scaffold_key(frame.iloc[i]["canonical_smiles"]) for i in train_indices]
    assigned = assign_groups(
        train_scaffolds,
        {"train": 1.0 - calibration_fraction, "calibration": calibration_fraction},
        seed,
    )
    output = ["test" if label == "test" else None for label in labels]
    for index, label in zip(train_indices, assigned):
        output[index] = label
    return output, [f"official::{value}" for value in frame["canonical_smiles"]]


def strict_group_split(frame, fractions, seed):
    n = len(frame)
    union_find = UnionFind(n)
    core_members = defaultdict(list)
    scaffold_members = defaultdict(list)
    scaffolds = []
    for i, smiles in enumerate(frame["canonical_smiles"]):
        scaffold = scaffold_key(smiles)
        scaffolds.append(scaffold)
        scaffold_members[scaffold].append(i)
        for core in mmp_cores(smiles):
            core_members[core].append(i)
        if (i + 1) % 250 == 0:
            print(f"fragmented {i + 1}/{n}", flush=True)
    for collection in (core_members, scaffold_members):
        for members in collection.values():
            if len(members) > 1:
                anchor = members[0]
                for member in members[1:]:
                    union_find.union(anchor, member)
    roots = [union_find.find(i) for i in range(n)]
    split = assign_groups(roots, fractions, seed)
    return split, [f"component::{root}" for root in roots], scaffolds


def split_qc(dataset, environment, frame):
    test = frame[frame["split"] == "test"].reset_index(drop=True)
    pairs = cliff_pairs(test)
    subsets = {label: set(frame.loc[frame["split"] == label, "canonical_smiles"]) for label in frame["split"].unique()}
    labels = sorted(subsets)
    overlap = sum(len(subsets[labels[i]] & subsets[labels[j]]) for i in range(len(labels)) for j in range(i + 1, len(labels)))
    row = {
        "dataset": dataset,
        "environment": environment,
        "n_train": int((frame["split"] == "train").sum()),
        "n_calibration": int((frame["split"] == "calibration").sum()),
        "n_test": int((frame["split"] == "test").sum()),
        "test_cliff_pairs": int(len(pairs)),
        "canonical_cross_split_overlap": int(overlap),
        "largest_group": int(frame.groupby("group_id").size().max()),
    }
    pair_rows = []
    for pair_id, (i, j) in enumerate(pairs):
        pair_rows.append(
            {
                "dataset": dataset,
                "environment": environment,
                "pair_id": pair_id,
                "smiles_a": test.iloc[i]["canonical_smiles"],
                "smiles_b": test.iloc[j]["canonical_smiles"],
                "activity_a": test.iloc[i]["activity"],
                "activity_b": test.iloc[j]["activity"],
            }
        )
    return row, pair_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests = []
    qc_rows = []
    all_pairs = []
    fractions = config["splits"]["mmp_group_fractions"]
    seed = config["splits"]["group_seed"]
    for selected in selection["selected"]:
        dataset = selected["dataset"]
        frame = pd.read_csv(Path(args.processed_dir) / f"{dataset}.csv")
        original_labels, original_groups = original_split(
            frame,
            seed,
            config["splits"]["original_calibration_fraction_of_official_train"],
        )
        original = frame.copy()
        original["split"] = original_labels
        original["group_id"] = original_groups
        original["dataset"] = dataset
        original["environment"] = "original"
        strict_labels, strict_groups, scaffolds = strict_group_split(frame, fractions, seed)
        strict = frame.copy()
        strict["split"] = strict_labels
        strict["group_id"] = strict_groups
        strict["scaffold"] = scaffolds
        strict["dataset"] = dataset
        strict["environment"] = "mmp_series"
        for environment, split_frame in (("original", original), ("mmp_series", strict)):
            qc, pairs = split_qc(dataset, environment, split_frame)
            qc_rows.append(qc)
            all_pairs.extend(pairs)
            manifests.append(split_frame)
            print(
                f"SPLIT {dataset} {environment}: "
                f"train={qc['n_train']} cal={qc['n_calibration']} test={qc['n_test']} "
                f"test_pairs={qc['test_cliff_pairs']} largest_group={qc['largest_group']}",
                flush=True,
            )
    pd.concat(manifests, ignore_index=True).to_csv(output_dir / "split_manifest.csv", index=False)
    pd.DataFrame(qc_rows).to_csv(output_dir / "split_qc.csv", index=False)
    pd.DataFrame(all_pairs).to_csv(output_dir / "cliff_pairs.csv", index=False)


if __name__ == "__main__":
    main()

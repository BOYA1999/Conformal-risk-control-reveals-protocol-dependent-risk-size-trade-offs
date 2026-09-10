import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem


TASK_PROPS = {
    "rings-count": "rings",
    "rings-max": "largest_rings",
    "X": "X",
    "P": "P",
    "B": "B",
    "indole": "indole",
    "PAINS": "pains",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def atom_indices(mol, prop):
    value = mol.GetProp(prop).strip() if mol.HasProp(prop) else ""
    return [] if not value else [int(item) for item in value.split(",")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--sdf", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.csv)
    canons = []
    invalid_csv = []
    for i, smiles in enumerate(frame["smiles"]):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid_csv.append(i)
            canons.append(None)
        else:
            canons.append(Chem.MolToSmiles(mol, isomericSmiles=True))
    frame["_canonical"] = canons

    split_report = {}
    for column in [c for c in frame.columns if c.startswith("split_")]:
        overlap = frame.groupby("_canonical")[column].nunique(dropna=False)
        split_report[column] = {
            "counts": frame[column].value_counts(dropna=False).to_dict(),
            "canonical_molecules_crossing_partitions": int((overlap > 1).sum()),
        }

    supplier = Chem.SDMolSupplier(args.sdf, removeHs=False, sanitize=False)
    parse_errors = []
    sanitize_errors = []
    canonical_mismatches = []
    atom_count_mismatches = []
    duplicate_indices = {task: [] for task in TASK_PROPS}
    out_of_range_indices = {task: [] for task in TASK_PROPS}
    rationale_sizes = {task: [] for task in TASK_PROPS}
    rationale_label_mismatches = {task: 0 for task in TASK_PROPS}
    sdf_count = 0
    for i, mol in enumerate(supplier):
        sdf_count += 1
        if mol is None:
            parse_errors.append(i)
            continue
        if i >= len(frame):
            continue
        csv_mol = Chem.MolFromSmiles(frame.at[i, "smiles"])
        sanitized = Chem.Mol(mol)
        try:
            Chem.SanitizeMol(sanitized)
        except Exception:
            sanitize_errors.append(i)
            sanitized = None
        if csv_mol is not None and sanitized is not None:
            sdf_canon = Chem.MolToSmiles(sanitized, isomericSmiles=True)
            if sdf_canon != frame.at[i, "_canonical"]:
                canonical_mismatches.append(i)
            if mol.GetNumAtoms() != csv_mol.GetNumAtoms():
                atom_count_mismatches.append(i)
        for task, prop in TASK_PROPS.items():
            indices = atom_indices(mol, prop)
            rationale_sizes[task].append(len(set(indices)))
            if len(indices) != len(set(indices)):
                duplicate_indices[task].append(i)
            if any(j < 0 or j >= mol.GetNumAtoms() for j in indices):
                out_of_range_indices[task].append(i)
            expected = bool(frame.at[i, task] > 0)
            if bool(indices) != expected:
                rationale_label_mismatches[task] += 1

    rationale_report = {}
    for task, sizes in rationale_sizes.items():
        values = np.asarray(sizes, dtype=float)
        rationale_report[task] = {
            "nonempty": int((values > 0).sum()),
            "mean_atoms": float(values.mean()),
            "median_atoms": float(np.median(values)),
            "max_atoms": int(values.max()),
            "duplicate_index_records": len(duplicate_indices[task]),
            "out_of_range_index_records": len(out_of_range_indices[task]),
            "label_presence_mismatches": rationale_label_mismatches[task],
        }

    def records(indices):
        return [
            {
                "row": int(i),
                "chembl_id": str(frame.at[i, "ChEMBL ID"]),
                "split_0": str(frame.at[i, "split_0"]),
            }
            for i in indices[:25]
            if i < len(frame)
        ]

    excluded = sorted(set(invalid_csv) | set(parse_errors) | set(sanitize_errors))
    usable_fraction = (len(frame) - len(excluded)) / len(frame)
    report = {
        "dataset": "B-XAIC",
        "files": {"data.csv": sha256(args.csv), "explanations.sdf": sha256(args.sdf)},
        "rows": len(frame),
        "sdf_records": sdf_count,
        "columns": [c for c in frame.columns if c != "_canonical"],
        "duplicate_chembl_ids": int(frame["ChEMBL ID"].duplicated().sum()),
        "duplicate_canonical_rows": int(frame["_canonical"].duplicated().sum()),
        "invalid_csv_smiles": len(invalid_csv),
        "sdf_parse_errors": len(parse_errors),
        "sdf_sanitize_errors": len(sanitize_errors),
        "canonical_mismatches": len(canonical_mismatches),
        "atom_count_mismatches": len(atom_count_mismatches),
        "diagnostic_records": {
            "invalid_or_unsanitizable": records(excluded),
            "canonical_mismatch_examples": records(canonical_mismatches),
        },
        "recommended_fixed_exclusions": records(excluded),
        "usable_mapping_fraction": usable_fraction,
        "splits": split_report,
        "rationales": rationale_report,
    }
    report["gate"] = "PASS_WITH_DECLARED_EXCLUSION" if all([
        len(frame) == sdf_count,
        not parse_errors,
        not atom_count_mismatches,
        all(not rows for rows in out_of_range_indices.values()),
        usable_fraction >= 0.995,
    ]) else "FAIL"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem


TASKS = ["benzene", "logic7", "logic8", "logic10"]


def sha256(path):
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return digest.upper()


def stratified_partitions(train_index, labels, ids, calibration_fraction=0.2, dev_fraction=0.1):
    calibration, dev = [], []
    for label in sorted(set(labels[train_index])):
        group = [i for i in train_index if labels[i] == label]
        group.sort(key=lambda i: hashlib.sha256(str(ids[i]).encode()).hexdigest())
        n_cal = round(calibration_fraction * len(group))
        n_dev = round(dev_fraction * len(group))
        calibration.extend(group[:n_cal])
        dev.extend(group[n_cal:n_cal + n_dev])
    calibration = np.asarray(sorted(calibration), dtype=int)
    dev = np.asarray(sorted(dev), dtype=int)
    fit = np.setdiff1d(train_index, np.r_[calibration, dev])
    return fit, dev, calibration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    report = {"dataset_family": "google-research/graph-attribution", "tasks": {}}
    all_pass = True

    for task in TASKS:
        folder = root / task
        csv_path = folder / f"{task}_smiles.csv"
        split_path = folder / f"{task}_traintest_indices.npz"
        graph_path = folder / "x_true.npz"
        label_path = folder / "y_true.npz"
        rationale_path = folder / "true_raw_attribution_datadicts.npz"
        frame = pd.read_csv(csv_path)
        splits = np.load(split_path)
        train = splits["train_index"].astype(int)
        test = splits["test_index"].astype(int)
        graphs = np.load(graph_path, allow_pickle=True)["datadict_list"].reshape(-1)
        rationales = np.load(rationale_path, allow_pickle=True)["datadict_list"].reshape(-1)
        labels = np.load(label_path)["y"].reshape(-1).astype(int)
        csv_labels = frame["label"].to_numpy(dtype=int)
        ids = frame["mol_id"].astype(str).to_numpy()
        fit, dev, calibration = stratified_partitions(train, labels, ids)

        invalid_smiles = sum(Chem.MolFromSmiles(s) is None for s in frame["smiles"])
        mapping_errors = 0
        nonbinary_masks = 0
        union_mask_errors = 0
        nonempty = 0
        positive_without_rationale = 0
        negative_with_rationale = 0
        valid_rationale_counts = []
        for label, graph, rationale in zip(labels, graphs, rationales):
            mask = np.asarray(rationale["nodes"])
            if mask.ndim == 1:
                mask = mask[:, None]
            if len(graph["nodes"]) != len(mask):
                mapping_errors += 1
            if not np.isin(mask, [0, 1]).all():
                nonbinary_masks += 1
            if np.any(mask[:, :-1] > mask[:, [-1]]):
                union_mask_errors += 1
            count = int(np.sum(mask.any(axis=0)))
            valid_rationale_counts.append(count)
            has_rationale = bool(mask[:, -1].any())
            nonempty += has_rationale
            positive_without_rationale += bool(label and not has_rationale)
            negative_with_rationale += bool(not label and has_rationale)

        complete = np.array_equal(np.sort(np.r_[train, test]), np.arange(len(frame)))
        disjoint = not np.intersect1d(train, test).size
        derived = np.r_[fit, dev, calibration]
        calibration_disjoint = len(np.unique(derived)) == len(train) and not np.intersect1d(derived, test).size
        task_pass = all([
            len(frame) == len(graphs) == len(rationales),
            complete,
            disjoint,
            calibration_disjoint,
            invalid_smiles == 0,
            mapping_errors == 0,
            nonbinary_masks == 0,
            union_mask_errors == 0,
            positive_without_rationale == 0,
            negative_with_rationale == 0,
        ])
        all_pass &= task_pass
        report["tasks"][task] = {
            "rows": len(frame),
            "fit": len(fit),
            "dev": len(dev),
            "calibration": len(calibration),
            "test": len(test),
            "positive": int(labels.sum()),
            "csv_label_agreement_with_y_true": float(np.mean(csv_labels == labels)),
            "nonempty_rationale": int(nonempty),
            "max_valid_rationales_per_molecule": int(max(valid_rationale_counts)),
            "invalid_smiles": invalid_smiles,
            "mapping_errors": mapping_errors,
            "nonbinary_masks": nonbinary_masks,
            "union_mask_errors": union_mask_errors,
            "positive_without_rationale": positive_without_rationale,
            "negative_with_rationale": negative_with_rationale,
            "official_split_complete": bool(complete),
            "official_split_disjoint": bool(disjoint),
            "derived_calibration_disjoint": bool(calibration_disjoint),
            "hashes": {
                csv_path.name: sha256(csv_path),
                split_path.name: sha256(split_path),
                graph_path.name: sha256(graph_path),
                label_path.name: sha256(label_path),
                rationale_path.name: sha256(rationale_path),
            },
            "gate": "PASS" if task_pass else "FAIL",
        }

    report["gate"] = "PASS" if all_pass else "FAIL"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCN, GIN, global_add_pool

from audit_graph_attribution import stratified_partitions
from molxai_crc import calibrate_crc, loss_table, top_fraction_set
from run_bxaic_probe import PROPS, SYMBOLS, evaluate, gradient_scores, hash_order, set_seed, train_model


BXAIC_TASKS = ["B", "P", "X", "indole", "PAINS", "rings-count", "rings-max"]
GOOGLE_TASKS = ["benzene", "logic7", "logic8", "logic10"]
ALPHAS = [0.05, 0.10, 0.20]
FRACTIONS = np.linspace(0, 1, 101)


class GraphClassifier(nn.Module):
    def __init__(self, kind, in_channels, hidden=32, layers=3):
        super().__init__()
        encoder = GIN if kind == "gin" else GCN
        kwargs = {"norm": "batch_norm"} if kind == "gin" else {}
        self.encoder = encoder(in_channels, hidden, num_layers=layers, out_channels=32, **kwargs)
        self.head = nn.Linear(32, 2)

    def forward(self, x, edge_index, batch):
        return self.head(global_add_pool(self.encoder(x, edge_index), batch))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def bxaic_partitions(csv_path, sdf_path, task):
    frame = pd.read_csv(csv_path)
    train_rows = frame.index[frame["split_0"] == "train"].tolist()
    dev_rows = []
    for label in sorted(frame.loc[train_rows, task].unique()):
        group = [i for i in train_rows if frame.at[i, task] == label]
        group.sort(key=lambda i: hash_order(frame.at[i, "ChEMBL ID"]))
        dev_rows.extend(group[: round(0.125 * len(group))])
    dev_rows = set(dev_rows)
    partitions = {"fit": [], "dev": [], "calibration": [], "test": []}
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    for i, mol in enumerate(supplier):
        if i == 997:
            continue
        split = frame.at[i, "split_0"]
        destination = "dev" if split == "train" and i in dev_rows else "fit" if split == "train" else "calibration" if split == "valid" else "test"
        atom_ids = [SYMBOLS.index(a.GetSymbol()) if a.GetSymbol() in SYMBOLS else 10 for a in mol.GetAtoms()]
        x = F.one_hot(torch.tensor(atom_ids), len(SYMBOLS)).float()
        edges = []
        for bond in mol.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edges.extend([(a, b), (b, a)])
        edge_index = torch.tensor(edges, dtype=torch.long).T.contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
        prop = PROPS[task]
        raw = mol.GetProp(prop).strip() if mol.HasProp(prop) else ""
        rationale = torch.zeros(len(atom_ids), dtype=torch.bool)
        if raw:
            rationale[torch.tensor(sorted({int(v) for v in raw.split(",")}), dtype=torch.long)] = True
        partitions[destination].append(Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor(int(frame.at[i, task]), dtype=torch.long),
            rationale_mask=rationale,
            source_index=torch.tensor(i, dtype=torch.long),
        ))
    return partitions


def google_partitions(root, task):
    folder = Path(root) / task
    frame = pd.read_csv(folder / f"{task}_smiles.csv")
    official = np.load(folder / f"{task}_traintest_indices.npz")
    train, test = official["train_index"].astype(int), official["test_index"].astype(int)
    labels = np.load(folder / "y_true.npz")["y"].reshape(-1).astype(int)
    graphs = np.load(folder / "x_true.npz", allow_pickle=True)["datadict_list"].reshape(-1)
    rationales = np.load(folder / "true_raw_attribution_datadicts.npz", allow_pickle=True)["datadict_list"].reshape(-1)
    fit, dev, calibration = stratified_partitions(train, labels, frame["mol_id"].astype(str).to_numpy())

    def convert(index):
        graph, rationale = graphs[index], rationales[index]
        mask = np.asarray(rationale["nodes"])
        if mask.ndim == 1:
            mask = mask[:, None]
        return Data(
            x=torch.tensor(graph["nodes"], dtype=torch.float32),
            edge_index=torch.tensor(np.stack([graph["senders"], graph["receivers"]]), dtype=torch.long),
            y=torch.tensor(labels[index], dtype=torch.long),
            rationale_mask=torch.tensor(mask[:, -1], dtype=torch.bool),
            source_index=torch.tensor(index, dtype=torch.long),
        )

    return {name: [convert(i) for i in rows] for name, rows in {
        "fit": fit, "dev": dev, "calibration": calibration, "test": test
    }.items()}


def set_metrics(scores, truths, fraction):
    losses, precisions, ious, realized = [], [], [], []
    for score, truth in zip(scores, truths):
        selected = top_fraction_set(score, fraction)
        intersection = len(selected & truth)
        losses.append(1 - intersection / len(truth))
        precisions.append(intersection / len(selected) if selected else 0.0)
        ious.append(intersection / len(selected | truth))
        realized.append(len(selected) / len(score))
    return {
        "risk": float(np.mean(losses)),
        "mean_atom_fraction": float(np.mean(realized)),
        "median_atom_fraction": float(np.median(realized)),
        "precision": float(np.mean(precisions)),
        "iou": float(np.mean(ious)),
    }


def explanation_metrics(cal_scores, cal_truths, test_scores, test_truths, seed):
    cal_nonempty = [(s, y) for s, y in zip(cal_scores, cal_truths) if y]
    test_nonempty = [(s, y) for s, y in zip(test_scores, test_truths) if y]
    cal_scores_ne, cal_truths_ne = map(list, zip(*cal_nonempty))
    test_scores_ne, test_truths_ne = map(list, zip(*test_nonempty))
    cal_table = loss_table(cal_scores_ne, cal_truths_ne, FRACTIONS)
    test_table = loss_table(test_scores_ne, test_truths_ne, FRACTIONS)
    rng = np.random.default_rng(seed)
    random_cal = [rng.normal(size=len(s)) for s in cal_scores_ne]
    random_test = [rng.normal(size=len(s)) for s in test_scores_ne]
    random_cal_table = loss_table(random_cal, cal_truths_ne, FRACTIONS)
    random_test_table = loss_table(random_test, test_truths_ne, FRACTIONS)
    results = {
        "n_calibration_rationale": len(cal_truths_ne),
        "n_test_rationale": len(test_truths_ne),
        "n_calibration_null": len(cal_truths) - len(cal_truths_ne),
        "n_test_null": len(test_truths) - len(test_truths_ne),
        "fixed": {
            "0.20": set_metrics(test_scores_ne, test_truths_ne, 0.20),
            "0.50": set_metrics(test_scores_ne, test_truths_ne, 0.50),
        },
        "alpha": {},
    }
    for alpha in ALPHAS:
        crc = calibrate_crc(cal_table, FRACTIONS, alpha)
        naive_idx = int(np.flatnonzero(cal_table.mean(axis=0) <= alpha)[0])
        random_crc = calibrate_crc(random_cal_table, FRACTIONS, alpha)
        results["alpha"][f"{alpha:.2f}"] = {
            "crc": {**crc, **set_metrics(test_scores_ne, test_truths_ne, crc["fraction"])},
            "naive_calibration": {
                "fraction": float(FRACTIONS[naive_idx]),
                "calibration_risk": float(cal_table[:, naive_idx].mean()),
                **set_metrics(test_scores_ne, test_truths_ne, float(FRACTIONS[naive_idx])),
            },
            "random_crc": {
                **random_crc,
                "risk": float(random_test_table[:, random_crc["index"]].mean()),
                "mean_atom_fraction": set_metrics(random_test, test_truths_ne, random_crc["fraction"])["mean_atom_fraction"],
            },
        }
    return results


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_cell(family, task, model_kind, seed, partitions, args, contract_hash, code_hash):
    cell_id = f"{family}__{task}__{model_kind}__seed{seed}"
    result_path = Path(args.out_dir) / "cells" / f"{cell_id}.json"
    if result_path.exists() and json.loads(result_path.read_text(encoding="utf-8")).get("status") == "complete":
        print(f"skip_complete={cell_id}", flush=True)
        return
    set_seed(seed)
    model = GraphClassifier(model_kind, partitions["fit"][0].x.shape[1]).to("cuda")
    training = train_model(model, partitions["fit"], partitions["dev"], torch.device("cuda"), args.batch_size, args.epochs)
    calibration, test = partitions["calibration"], partitions["test"]
    if args.surface == "dev":
        ordered = sorted(partitions["dev"], key=lambda d: hash_order(int(d.source_index)))
        midpoint = len(ordered) // 2
        calibration, test = ordered[:midpoint], ordered[midpoint:]
    predictor = {
        "calibration": evaluate(model, torch_geometric_loader(calibration, args.batch_size), torch.device("cuda")),
        "test": evaluate(model, torch_geometric_loader(test, args.batch_size), torch.device("cuda")),
    }
    explainers = {}
    for method in ["gradinput", "ig"]:
        cal_scores, cal_truths, cal_timing = gradient_scores(model, calibration, torch.device("cuda"), method, args.batch_size if method == "gradinput" else max(16, args.batch_size // 4))
        test_scores, test_truths, test_timing = gradient_scores(model, test, torch.device("cuda"), method, args.batch_size if method == "gradinput" else max(16, args.batch_size // 4))
        explainers[method] = {
            "calibration_timing": cal_timing,
            "test_timing": test_timing,
            "metrics": explanation_metrics(cal_scores, cal_truths, test_scores, test_truths, seed),
        }
    checkpoint = Path(args.out_dir) / "checkpoints" / f"{cell_id}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "family": family, "task": task, "model": model_kind, "seed": seed}, checkpoint)
    payload = {
        "status": "complete",
        "surface": args.surface,
        "cell_id": cell_id,
        "family": family,
        "task": task,
        "model": model_kind,
        "seed": seed,
        "contract_sha256": contract_hash,
        "code_sha256": code_hash,
        "training": training,
        "predictor": predictor,
        "predictor_competence_pass": predictor["test"]["auroc"] > 0.65,
        "explainers": explainers,
        "checkpoint": str(checkpoint),
    }
    atomic_json(result_path, payload)
    print(f"cell_complete={cell_id}", flush=True)


def torch_geometric_loader(graphs, batch_size):
    from torch_geometric.loader import DataLoader
    return DataLoader(graphs, batch_size=batch_size, shuffle=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bxaic-csv", required=True)
    parser.add_argument("--bxaic-sdf", required=True)
    parser.add_argument("--google-root", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--surface", choices=["dev", "final"], default="dev")
    parser.add_argument("--confirm-final", default="")
    parser.add_argument("--families", default="bxaic,google")
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--models", default="gin,gcn")
    parser.add_argument("--seeds", default="42,123,2026")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.surface == "final" and args.confirm_final != "FROZEN_20260809_V0_1":
        raise ValueError("final surface requires the frozen-contract confirmation token")
    contract_hash = sha256(args.contract)
    code_hash = sha256(__file__)
    families = args.families.split(",")
    selected_tasks = None if args.tasks == "all" else set(args.tasks.split(","))
    models = args.models.split(",")
    seeds = [int(value) for value in args.seeds.split(",")]
    started = time.time()
    for family in families:
        tasks = BXAIC_TASKS if family == "bxaic" else GOOGLE_TASKS
        for task in tasks:
            if selected_tasks is not None and task not in selected_tasks:
                continue
            print(f"load_task={family}/{task}", flush=True)
            partitions = bxaic_partitions(args.bxaic_csv, args.bxaic_sdf, task) if family == "bxaic" else google_partitions(args.google_root, task)
            for model_kind in models:
                for seed in seeds:
                    run_cell(family, task, model_kind, seed, partitions, args, contract_hash, code_hash)
    cells = sorted((Path(args.out_dir) / "cells").glob("*.json"))
    atomic_json(Path(args.out_dir) / "gradient_grid_manifest.json", {
        "status": "complete",
        "surface": args.surface,
        "started_unix": started,
        "finished_unix": time.time(),
        "contract_sha256": contract_hash,
        "code_sha256": code_hash,
        "cell_count": len(cells),
        "cells": [str(path) for path in cells],
    })


if __name__ == "__main__":
    main()

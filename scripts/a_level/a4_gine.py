import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_geometric
import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool


BXAIC_TASKS = ["B", "P", "X", "indole", "PAINS", "rings-count", "rings-max"]
GOOGLE_TASKS = ["benzene", "logic7", "logic8", "logic10"]
ALL_TASKS = [("bxaic", task) for task in BXAIC_TASKS] + [("google", task) for task in GOOGLE_TASKS]
SEEDS = [42, 123, 2026]
FRACTIONS = np.linspace(0.0, 1.0, 101)
ALPHA = 0.10
CONFIRM_TOKEN = "RUN_A4_GINE_20260904"
PROPS = {"B": "B", "P": "P", "X": "X", "indole": "indole", "PAINS": "pains", "rings-count": "rings", "rings-max": "largest_rings"}
ELEMENTS = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "Na", "Ca", "I", "B", "H"]
DEGREES = [0, 1, 2, 3, 4, 5]
CHARGES = [-2, -1, 0, 1, 2]
HYBRIDIZATIONS = ["SP", "SP2", "SP3", "SP3D", "SP3D2"]
HYDROGEN_COUNTS = [0, 1, 2, 3, 4]
CHIRAL_TAGS = ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW"]
BOND_TYPES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
BOND_STEREO = ["STEREONONE", "STEREOE", "STEREOZ", "STEREOANY"]
BOND_DIRS = ["NONE", "BEGINWEDGE", "BEGINDASH", "EITHERDOUBLE"]
ATOM_DIM = 45
EDGE_DIM = 17


LOG_PATH = None


def log(message):
    line = f"{datetime.now(timezone.utc).isoformat()} {message}"
    print(line, flush=True)
    if LOG_PATH is not None:
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def hash_order(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def one_hot_unknown(value, choices):
    return [float(value == choice) for choice in choices] + [float(value not in choices)]


def molecule_features(mol):
    atoms = []
    for atom in mol.GetAtoms():
        values = []
        values += one_hot_unknown(atom.GetSymbol(), ELEMENTS)
        values += one_hot_unknown(atom.GetDegree(), DEGREES)
        values += one_hot_unknown(atom.GetFormalCharge(), CHARGES)
        values += one_hot_unknown(str(atom.GetHybridization()), HYBRIDIZATIONS)
        values += one_hot_unknown(int(atom.GetTotalNumHs(includeNeighbors=True)), HYDROGEN_COUNTS)
        values += [float(atom.GetIsAromatic()), float(atom.IsInRing())]
        values += one_hot_unknown(str(atom.GetChiralTag()), CHIRAL_TAGS)
        atoms.append(values)
    directed_edges, edge_values = [], []
    for bond in mol.GetBonds():
        values = []
        values += one_hot_unknown(str(bond.GetBondType()), BOND_TYPES)
        values += one_hot_unknown(str(bond.GetStereo()), BOND_STEREO)
        values += [float(bond.GetIsConjugated()), float(bond.IsInRing())]
        values += one_hot_unknown(str(bond.GetBondDir()), BOND_DIRS)
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        directed_edges.extend([(a, b), (b, a)])
        edge_values.extend([values, values])
    x = torch.tensor(atoms, dtype=torch.float32)
    if directed_edges:
        edge_index = torch.tensor(directed_edges, dtype=torch.long).T.contiguous()
        edge_attr = torch.tensor(edge_values, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, EDGE_DIM), dtype=torch.float32)
    if x.ndim != 2 or x.shape[1] != ATOM_DIM or edge_attr.ndim != 2 or edge_attr.shape[1] != EDGE_DIM:
        raise ValueError(f"feature dimension mismatch: x={tuple(x.shape)}, edge_attr={tuple(edge_attr.shape)}")
    return x, edge_index, edge_attr


def indexed_signature(mol):
    atoms = tuple((atom.GetAtomicNum(), atom.GetFormalCharge(), atom.GetIsAromatic()) for atom in mol.GetAtoms())
    bonds = tuple(sorted((min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), str(b.GetBondType())) for b in mol.GetBonds()))
    return atoms, bonds


def stratified_partitions(train_index, labels, ids, calibration_fraction=0.2, dev_fraction=0.1):
    calibration, dev = [], []
    for label in sorted(set(labels[train_index])):
        group = [int(i) for i in train_index if labels[i] == label]
        group.sort(key=lambda i: hash_order(ids[i]))
        n_cal = round(calibration_fraction * len(group))
        n_dev = round(dev_fraction * len(group))
        calibration.extend(group[:n_cal])
        dev.extend(group[n_cal:n_cal + n_dev])
    calibration = np.asarray(sorted(calibration), dtype=int)
    dev = np.asarray(sorted(dev), dtype=int)
    fit = np.setdiff1d(train_index, np.r_[calibration, dev])
    return fit, dev, calibration


def source_paths(project_root):
    paths = {
        "bxaic_csv": project_root / "data/raw/bxaic/data.csv",
        "bxaic_sdf": project_root / "data/raw/bxaic/explanations.sdf",
        "google_featurization": project_root / "reference/graph-attribution/graph_attribution/featurization.py",
        "historical_contract": project_root / "artifacts/experiment/gradient_grid_main/run_contract.json",
    }
    for task in GOOGLE_TASKS:
        folder = project_root / "reference/graph-attribution/data" / task
        paths.update({
            f"google_{task}_smiles": folder / f"{task}_smiles.csv",
            f"google_{task}_split": folder / f"{task}_traintest_indices.npz",
            f"google_{task}_graphs": folder / "x_true.npz",
            f"google_{task}_labels": folder / "y_true.npz",
            f"google_{task}_rationales": folder / "true_raw_attribution_datadicts.npz",
        })
    return paths


def audit_google(project_root):
    tasks = {}
    for task in GOOGLE_TASKS:
        started = time.perf_counter()
        folder = project_root / "reference/graph-attribution/data" / task
        frame = pd.read_csv(folder / f"{task}_smiles.csv")
        graphs = np.load(folder / "x_true.npz", allow_pickle=True)["datadict_list"].reshape(-1)
        rationales = np.load(folder / "true_raw_attribution_datadicts.npz", allow_pickle=True)["datadict_list"].reshape(-1)
        labels = np.load(folder / "y_true.npz")["y"].reshape(-1).astype(int)
        csv_labels = frame["label"].to_numpy(dtype=int)
        counts = defaultdict(int)
        examples = []
        for index, (smiles, graph, rationale) in enumerate(zip(frame["smiles"], graphs, rationales)):
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                counts["invalid_smiles"] += 1
                if len(examples) < 5:
                    examples.append({"index": index, "error": "invalid_smiles"})
                continue
            x, edge_index, edge_attr = molecule_features(mol)
            official_nodes = np.asarray(graph["nodes"], dtype=np.float32)
            official_edges = np.asarray(graph["edges"], dtype=np.float32)
            official_edge_index = np.stack([graph["senders"], graph["receivers"]]).astype(np.int64)
            mask = np.asarray(rationale["nodes"])
            if mask.ndim == 1:
                mask = mask[:, None]
            checks = {
                "node_count": len(x) == len(official_nodes) == len(mask),
                "edge_count": edge_index.shape[1] == len(official_edges),
                "node_features": np.array_equal(x[:, :14].numpy(), official_nodes),
                "edge_features": np.array_equal(edge_attr[:, :5].numpy(), official_edges),
                "edge_order": np.array_equal(edge_index.numpy(), official_edge_index),
            }
            for name, passed in checks.items():
                if not passed:
                    counts[f"{name}_errors"] += 1
                    if len(examples) < 5:
                        examples.append({"index": index, "error": name})
            counts["molecules"] += 1
        splits = np.load(folder / f"{task}_traintest_indices.npz")
        train = splits["train_index"].astype(int)
        test = splits["test_index"].astype(int)
        ids = frame["mol_id"].astype(str).to_numpy()
        fit, dev, calibration = stratified_partitions(train, labels, ids)
        errors = sum(value for key, value in counts.items() if key.endswith("_errors")) + counts["invalid_smiles"]
        gate_checks = {
            "row_alignment": len(frame) == len(graphs) == len(rationales) == len(labels),
            "official_split_complete": np.array_equal(np.sort(np.r_[train, test]), np.arange(len(frame))),
            "official_split_disjoint": not np.intersect1d(train, test).size,
            "derived_split_complete": len(np.unique(np.r_[fit, dev, calibration])) == len(train),
            "derived_split_test_disjoint": not np.intersect1d(np.r_[fit, dev, calibration], test).size,
            "atom_edge_mapping": errors == 0,
        }
        task_pass = all(gate_checks.values())
        tasks[task] = {
            "status": "PASS" if task_pass else "FAIL",
            "n_molecules": len(frame),
            "partition_counts": {"fit": len(fit), "dev": len(dev), "calibration": len(calibration), "test": len(test)},
            "authoritative_label_source": "y_true.npz",
            "csv_label_agreement_with_y_true": float(np.mean(labels == csv_labels)),
            "gate_checks": gate_checks,
            "mapping_counts": dict(counts),
            "first_errors": examples,
            "seconds": time.perf_counter() - started,
        }
        log(f"google_mapping task={task} status={tasks[task]['status']} n={len(frame)}")
    return {"status": "PASS" if all(row["status"] == "PASS" for row in tasks.values()) else "FAIL", "tasks": tasks}


def audit_bxaic(project_root):
    started = time.perf_counter()
    csv_path = project_root / "data/raw/bxaic/data.csv"
    sdf_path = project_root / "data/raw/bxaic/explanations.sdf"
    frame = pd.read_csv(csv_path)
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    counts = defaultdict(int)
    examples = []
    for index, mol in enumerate(supplier):
        if index == 997:
            counts["expected_excluded_rows"] += 1
            continue
        counts["retained_rows"] += 1
        if mol is None:
            counts["sdf_parse_errors"] += 1
            if len(examples) < 5:
                examples.append({"index": index, "error": "sdf_parse"})
            continue
        try:
            Chem.SanitizeMol(mol)
        except Exception as error:
            counts["sdf_sanitize_errors"] += 1
            if len(examples) < 5:
                examples.append({"index": index, "error": "sdf_sanitize", "detail": str(error)})
            continue
        smiles_mol = Chem.MolFromSmiles(str(frame.at[index, "smiles"]))
        if smiles_mol is None:
            counts["csv_smiles_errors"] += 1
            if len(examples) < 5:
                examples.append({"index": index, "error": "csv_smiles"})
            continue
        if indexed_signature(mol) != indexed_signature(smiles_mol):
            counts["indexed_graph_mismatch_errors"] += 1
            if len(examples) < 5:
                examples.append({"index": index, "error": "indexed_graph_mismatch"})
        for task, prop in PROPS.items():
            raw = mol.GetProp(prop).strip() if mol.HasProp(prop) else ""
            rationale = {int(value) for value in raw.split(",")} if raw else set()
            if any(value < 0 or value >= mol.GetNumAtoms() for value in rationale):
                counts["rationale_index_errors"] += 1
                if len(examples) < 5:
                    examples.append({"index": index, "task": task, "error": "rationale_index"})
            label = int(frame.at[index, task])
            counts[f"{task}_nonempty"] += int(bool(rationale))
            counts[f"{task}_positive_without_rationale"] += int(label == 1 and not rationale)
            counts[f"{task}_negative_with_rationale"] += int(label == 0 and bool(rationale))
        if counts["retained_rows"] % 5000 == 0:
            log(f"bxaic_mapping progress={counts['retained_rows']}/49999")
    error_keys = [key for key in counts if key.endswith("_errors")]
    status = "PASS" if len(frame) == 50000 and len(supplier) == 50000 and counts["retained_rows"] == 49999 and counts["expected_excluded_rows"] == 1 and sum(counts[key] for key in error_keys) == 0 else "FAIL"
    log(f"bxaic_mapping status={status} retained={counts['retained_rows']} errors={sum(counts[key] for key in error_keys)}")
    return {
        "status": status,
        "n_csv_rows": len(frame),
        "n_sdf_records": len(supplier),
        "excluded_source_index": 997,
        "counts": dict(counts),
        "first_errors": examples,
        "seconds": time.perf_counter() - started,
    }


def run_mapping_audit(project_root, out_dir):
    inputs = source_paths(project_root)
    missing = [str(path) for path in inputs.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")
    audit = {
        "run_id": "EXP-A4-BOND-AWARE-GINE-20260904",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "feature_dimensions": {"atom": ATOM_DIM, "edge": EDGE_DIM},
        "google": audit_google(project_root),
        "bxaic": audit_bxaic(project_root),
    }
    audit["status"] = "PASS" if audit["google"]["status"] == "PASS" and audit["bxaic"]["status"] == "PASS" else "FAIL"
    audit["preferred_features_approved"] = audit["status"] == "PASS"
    audit["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(out_dir / "mapping_audit.json", audit)
    input_manifest = {name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)} for name, path in inputs.items()}
    write_json(out_dir / "input_manifest.json", {"status": "complete", "inputs": input_manifest})
    if audit["status"] != "PASS":
        raise RuntimeError("mapping audit failed; preferred rich-feature run is blocked")
    return audit


class ResidualGINE(nn.Module):
    def __init__(self):
        super().__init__()
        hidden = 128
        self.node_encoder = nn.Linear(ATOM_DIM, hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(4):
            mlp = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.ReLU(), nn.Linear(hidden * 2, hidden))
            self.convs.append(GINEConv(mlp, train_eps=True, edge_dim=EDGE_DIM))
            self.norms.append(nn.BatchNorm1d(hidden))
        self.head = nn.Linear(hidden, 2)

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.node_encoder(x)
        for conv, norm in zip(self.convs, self.norms):
            h = h + F.relu(norm(conv(h, edge_index, edge_attr)))
        return self.head(global_add_pool(h, batch))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@torch.no_grad()
def evaluate(model, graphs, device, batch_size):
    model.eval()
    labels, probabilities, predictions = [], [], []
    for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        labels.extend(batch.y.detach().cpu().tolist())
        probabilities.extend(logits.softmax(-1)[:, 1].detach().cpu().tolist())
        predictions.extend(logits.argmax(-1).detach().cpu().tolist())
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted")),
        "n": len(labels),
        "positive_prevalence": float(np.mean(labels)),
    }


def train_model(model, fit, dev, device, max_epochs, batch_size):
    labels = torch.tensor([int(graph.y) for graph in fit], dtype=torch.long)
    class_counts = torch.bincount(labels, minlength=2).float()
    if torch.any(class_counts == 0):
        raise ValueError(f"fit split missing class: {class_counts.tolist()}")
    class_weights = 1.0 / class_counts
    sampler = WeightedRandomSampler(class_weights[labels], len(labels), replacement=True)
    train_loader = DataLoader(fit, batch_size=batch_size, sampler=sampler)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state, best_metrics, best_epoch, stale = None, None, 0, 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss, seen = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = F.cross_entropy(logits, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch.y)
            seen += len(batch.y)
        metrics = evaluate(model, dev, device, batch_size)
        log(f"epoch={epoch} train_loss={total_loss/seen:.6f} dev_auroc={metrics['auroc']:.6f} dev_auprc={metrics['auprc']:.6f}")
        if best_metrics is None or metrics["auroc"] > best_metrics["auroc"] + 1e-5:
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            best_metrics, best_epoch, stale = metrics, epoch, 0
        else:
            stale += 1
        if epoch >= 10 and stale >= 5:
            break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    return {
        "seconds": time.perf_counter() - started,
        "epochs": epoch,
        "best_epoch": best_epoch,
        "dev_metrics": best_metrics,
        "peak_vram_mib": float(torch.cuda.max_memory_allocated(device) / 2**20),
    }


def integrated_gradients(model, graphs, device, batch_size, steps):
    model.eval()
    score_rows, rationale_rows = [], []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
        batch = batch.to(device)
        total_gradient = torch.zeros_like(batch.x)
        for step in range(1, steps + 1):
            x = (batch.x * (step / steps)).detach().requires_grad_(True)
            logits = model(x, batch.edge_index, batch.edge_attr, batch.batch)
            target = logits[torch.arange(len(batch.y), device=device), batch.y].sum()
            total_gradient += torch.autograd.grad(target, x)[0]
        atom_scores = (batch.x * total_gradient / steps).sum(-1).detach().cpu().numpy()
        mask = batch.rationale_mask.detach().cpu().numpy()
        ptr = batch.ptr.detach().cpu().tolist()
        for start, end in zip(ptr[:-1], ptr[1:]):
            row = atom_scores[start:end]
            if not np.isfinite(row).all():
                raise ValueError("non-finite IG score")
            truth = set(np.flatnonzero(mask[start:end]).tolist())
            if not truth:
                raise ValueError("IG input must have a nonempty rationale")
            score_rows.append(row)
            rationale_rows.append(truth)
    seconds = time.perf_counter() - started
    return score_rows, rationale_rows, {
        "seconds": seconds,
        "molecules": len(graphs),
        "seconds_per_molecule": seconds / len(graphs),
        "peak_vram_mib": float(torch.cuda.max_memory_allocated(device) / 2**20),
        "steps": steps,
    }


def top_fraction_set(scores, fraction):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not 0 <= fraction <= 1:
        raise ValueError("invalid score row or fraction")
    if fraction == 0 or not len(scores):
        return set()
    k = int(np.ceil(fraction * len(scores)))
    threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
    return set(np.flatnonzero(scores >= threshold).tolist())


def loss_table(score_rows, rationale_rows):
    table = np.empty((len(score_rows), len(FRACTIONS)), dtype=float)
    for row_index, (scores, rationale) in enumerate(zip(score_rows, rationale_rows)):
        scores = np.asarray(scores, dtype=float)
        rationale = np.asarray(sorted(rationale), dtype=int)
        order = np.argsort(-scores)
        ordered_scores = scores[order]
        rationale_mask = np.zeros(len(scores), dtype=int)
        rationale_mask[rationale] = 1
        recovered_by_rank = np.cumsum(rationale_mask[order])
        nominal_counts = np.where(FRACTIONS == 0, 0, np.ceil(FRACTIONS * len(scores))).astype(int)
        recovered = np.zeros(len(FRACTIONS), dtype=int)
        positive = nominal_counts > 0
        thresholds = ordered_scores[nominal_counts[positive] - 1]
        tie_counts = np.searchsorted(-ordered_scores, -thresholds, side="right")
        recovered[positive] = recovered_by_rank[tie_counts - 1]
        table[row_index] = 1.0 - recovered / len(rationale)
    if not np.isfinite(table).all() or np.any(np.diff(table, axis=1) > 1e-12):
        raise ValueError("invalid loss table")
    return table


def selected_count_table(score_rows):
    table = np.empty((len(score_rows), len(FRACTIONS)), dtype=np.int32)
    for row_index, scores in enumerate(score_rows):
        scores = np.asarray(scores, dtype=float)
        if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
            raise ValueError("invalid score row")
        ordered_scores = np.sort(scores)[::-1]
        nominal_counts = np.where(FRACTIONS == 0, 0, np.ceil(FRACTIONS * len(scores))).astype(int)
        counts = np.zeros(len(FRACTIONS), dtype=np.int32)
        positive = nominal_counts > 0
        thresholds = ordered_scores[nominal_counts[positive] - 1]
        counts[positive] = np.searchsorted(-ordered_scores, -thresholds, side="right")
        if np.any(np.diff(counts) < 0) or counts[-1] != len(scores):
            raise ValueError("invalid tie-inclusive selected-count schedule")
        table[row_index] = counts
    return table


def calibrate_crc(table):
    n = len(table)
    corrected = (n * table.mean(axis=0) + 1.0) / (n + 1)
    feasible = np.flatnonzero(corrected <= ALPHA)
    if not len(feasible):
        raise ValueError("no feasible CRC set")
    index = int(feasible[0])
    return {
        "index": index,
        "nominal_fraction": float(FRACTIONS[index]),
        "calibration_empirical_risk": float(table[:, index].mean()),
        "calibration_corrected_risk": float(corrected[index]),
        "n_calibration": n,
    }


def set_metrics(score_rows, rationale_rows, fraction):
    risks, realized, precision, iou, inflation = [], [], [], [], []
    for scores, truth in zip(score_rows, rationale_rows):
        selected = top_fraction_set(scores, fraction)
        intersection = len(selected & truth)
        risks.append(1.0 - intersection / len(truth))
        realized_fraction = len(selected) / len(scores)
        realized.append(realized_fraction)
        precision.append(intersection / len(selected) if selected else 0.0)
        iou.append(intersection / len(selected | truth))
        inflation.append(realized_fraction - fraction)
    return {
        "test_risk": float(np.mean(risks)),
        "test_mean_atom_fraction": float(np.mean(realized)),
        "test_median_atom_fraction": float(np.median(realized)),
        "test_precision": float(np.mean(precision)),
        "test_iou": float(np.mean(iou)),
        "test_mean_tie_inflation": float(np.mean(inflation)),
    }


def matched_oracle_loss_table(selected_counts, rationale_rows):
    if selected_counts.shape != (len(rationale_rows), len(FRACTIONS)):
        raise ValueError("oracle selected-count shape mismatch")
    rows = []
    for retained, truth in zip(selected_counts, rationale_rows):
        if not truth:
            raise ValueError("oracle input must have a nonempty rationale")
        rows.append(1.0 - np.minimum(retained, len(truth)) / len(truth))
    table = np.asarray(rows, dtype=float)
    if not np.isfinite(table).all() or np.any(np.diff(table, axis=1) > 1e-12):
        raise ValueError("invalid oracle loss table")
    return table


def matched_oracle_set_metrics(score_rows, rationale_rows, selected_counts, index):
    risks, realized, precision, iou, inflation = [], [], [], [], []
    for scores, truth, count_schedule in zip(score_rows, rationale_rows, selected_counts):
        n_atoms, n_rationale = len(scores), len(truth)
        retained = int(count_schedule[index])
        recovered = min(retained, n_rationale)
        risks.append(1.0 - recovered / n_rationale)
        realized.append(retained / n_atoms)
        precision.append(recovered / retained if retained else 0.0)
        iou.append(recovered / (retained + n_rationale - recovered))
        inflation.append(retained / n_atoms - FRACTIONS[index])
    return {
        "test_risk": float(np.mean(risks)),
        "test_mean_atom_fraction": float(np.mean(realized)),
        "test_median_atom_fraction": float(np.median(realized)),
        "test_precision": float(np.mean(precision)),
        "test_iou": float(np.mean(iou)),
        "test_mean_tie_inflation": float(np.mean(inflation)),
    }


def explanation_metrics(cal_scores, cal_truths, test_scores, test_truths):
    cal_table = loss_table(cal_scores, cal_truths)
    test_table = loss_table(test_scores, test_truths)
    cal_counts = selected_count_table(cal_scores)
    test_counts = selected_count_table(test_scores)
    crc = calibrate_crc(cal_table)
    measured = set_metrics(test_scores, test_truths, crc["nominal_fraction"])
    if not math.isclose(measured["test_risk"], float(test_table[:, crc["index"]].mean()), abs_tol=1e-12):
        raise ValueError("set metric and loss table disagree")
    for row, counts in zip(test_scores, test_counts):
        if len(top_fraction_set(row, crc["nominal_fraction"])) != int(counts[crc["index"]]):
            raise ValueError("set metric and selected-count schedule disagree")
    cal_oracle_table = matched_oracle_loss_table(cal_counts, cal_truths)
    test_oracle_table = matched_oracle_loss_table(test_counts, test_truths)
    if np.any(cal_oracle_table > cal_table + 1e-12) or np.any(test_oracle_table > test_table + 1e-12):
        raise ValueError("schedule-matched oracle is not pointwise loss-dominating")
    oracle = calibrate_crc(cal_oracle_table)
    if oracle["index"] > crc["index"]:
        raise ValueError("schedule-matched oracle CRC index is later than the actual CRC index")
    oracle_measured = matched_oracle_set_metrics(test_scores, test_truths, test_counts, oracle["index"])
    if not math.isclose(oracle_measured["test_risk"], float(test_oracle_table[:, oracle["index"]].mean()), abs_tol=1e-12):
        raise ValueError("oracle set metric and loss table disagree")
    excess = measured["test_mean_atom_fraction"] - oracle_measured["test_mean_atom_fraction"]
    if excess < -1e-12:
        raise ValueError("schedule-matched oracle capacity excess is negative")
    metrics = {
        "alpha": ALPHA,
        "target": "true_class_logit",
        "set_policy": "include_all_ties",
        "n_calibration_rationale": len(cal_truths),
        "n_test_rationale": len(test_truths),
        "crc": {**crc, **measured},
        "oracle": {"definition": "rationale_first_on_actual_include_all_ties_cardinality_schedule", **oracle, **oracle_measured},
        "oracle_excess_mean_atom_fraction": excess,
        "oracle_proof": {
            "calibration_pointwise_loss_le_actual": True,
            "test_pointwise_loss_le_actual": True,
            "oracle_crc_index_not_later_than_actual": True,
            "shared_per_molecule_cardinality_schedule": True,
            "capacity_excess_nonnegative": True,
        },
    }
    artifacts = {
        "calibration_selected_counts": cal_counts,
        "test_selected_counts": test_counts,
    }
    return metrics, artifacts


def ragged_arrays(rows, dtype):
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    arrays = []
    for index, row in enumerate(rows):
        array = np.asarray(sorted(row) if isinstance(row, set) else row, dtype=dtype).reshape(-1)
        arrays.append(array)
        offsets[index + 1] = offsets[index] + len(array)
    values = np.concatenate(arrays) if arrays else np.asarray([], dtype=dtype)
    return values, offsets


def write_score_artifact(path, cal_scores, cal_truths, test_scores, test_truths, cal_graphs, test_graphs, schedules, contract_hash, code_hash):
    cal_score_values, cal_score_offsets = ragged_arrays(cal_scores, np.float32)
    cal_truth_values, cal_truth_offsets = ragged_arrays(cal_truths, np.int32)
    test_score_values, test_score_offsets = ragged_arrays(test_scores, np.float32)
    test_truth_values, test_truth_offsets = ragged_arrays(test_truths, np.int32)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema_version=np.asarray("a4_gine_scores_v1"),
            contract_sha256=np.asarray(contract_hash),
            code_sha256=np.asarray(code_hash),
            fractions=FRACTIONS,
            calibration_score_values=cal_score_values,
            calibration_score_offsets=cal_score_offsets,
            calibration_truth_values=cal_truth_values,
            calibration_truth_offsets=cal_truth_offsets,
            calibration_source_indices=np.asarray([int(graph.source_index) for graph in cal_graphs], dtype=np.int64),
            calibration_selected_counts=schedules["calibration_selected_counts"],
            test_score_values=test_score_values,
            test_score_offsets=test_score_offsets,
            test_truth_values=test_truth_values,
            test_truth_offsets=test_truth_offsets,
            test_source_indices=np.asarray([int(graph.source_index) for graph in test_graphs], dtype=np.int64),
            test_selected_counts=schedules["test_selected_counts"],
        )
    temporary.replace(path)


def validate_score_artifact(path, expected_contract_hash, expected_code_hash):
    with np.load(path, allow_pickle=False) as artifact:
        required = {
            "schema_version", "contract_sha256", "code_sha256", "fractions",
            "calibration_score_values", "calibration_score_offsets", "calibration_truth_values", "calibration_truth_offsets",
            "calibration_source_indices", "calibration_selected_counts", "test_score_values", "test_score_offsets",
            "test_truth_values", "test_truth_offsets", "test_source_indices", "test_selected_counts",
        }
        if not required.issubset(artifact.files):
            return False
        return bool(
            str(artifact["schema_version"]) == "a4_gine_scores_v1"
            and str(artifact["contract_sha256"]) == expected_contract_hash
            and str(artifact["code_sha256"]) == expected_code_hash
            and np.array_equal(artifact["fractions"], FRACTIONS)
            and artifact["calibration_selected_counts"].shape == (len(artifact["calibration_source_indices"]), len(FRACTIONS))
            and artifact["test_selected_counts"].shape == (len(artifact["test_source_indices"]), len(FRACTIONS))
            and artifact["calibration_score_offsets"][-1] == len(artifact["calibration_score_values"])
            and artifact["calibration_truth_offsets"][-1] == len(artifact["calibration_truth_values"])
            and artifact["test_score_offsets"][-1] == len(artifact["test_score_values"])
            and artifact["test_truth_offsets"][-1] == len(artifact["test_truth_values"])
        )


def load_bxaic_base(project_root):
    frame = pd.read_csv(project_root / "data/raw/bxaic/data.csv")
    supplier = Chem.SDMolSupplier(str(project_root / "data/raw/bxaic/explanations.sdf"), removeHs=False, sanitize=False)
    records = []
    for index, mol in enumerate(supplier):
        if index == 997:
            continue
        if mol is None:
            raise ValueError(f"unexpected null B-XAIC molecule {index}")
        Chem.SanitizeMol(mol)
        x, edge_index, edge_attr = molecule_features(mol)
        labels = torch.tensor([int(frame.at[index, task]) for task in BXAIC_TASKS], dtype=torch.long)
        masks = torch.zeros((len(x), len(BXAIC_TASKS)), dtype=torch.bool)
        for task_index, task in enumerate(BXAIC_TASKS):
            prop = PROPS[task]
            raw = mol.GetProp(prop).strip() if mol.HasProp(prop) else ""
            if raw:
                masks[torch.tensor(sorted({int(value) for value in raw.split(",")}), dtype=torch.long), task_index] = True
        records.append((index, x, edge_index, edge_attr, labels, masks))
        if len(records) % 5000 == 0:
            log(f"bxaic_features progress={len(records)}/49999")
    return frame, records


def bxaic_task_partitions(frame, records, task):
    task_index = BXAIC_TASKS.index(task)
    train_rows = [int(index) for index in frame.index[frame["split_0"] == "train"] if index != 997]
    dev_rows = []
    for label in sorted(frame.loc[train_rows, task].unique()):
        group = [index for index in train_rows if frame.at[index, task] == label]
        group.sort(key=lambda index: hash_order(frame.at[index, "ChEMBL ID"]))
        dev_rows.extend(group[:round(0.125 * len(group))])
    dev_rows = set(dev_rows)
    partitions = {"fit": [], "dev": [], "calibration": [], "test": []}
    for source_index, x, edge_index, edge_attr, labels, masks in records:
        split = frame.at[source_index, "split_0"]
        destination = "dev" if split == "train" and source_index in dev_rows else "fit" if split == "train" else "calibration" if split == "valid" else "test"
        partitions[destination].append(Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=labels[task_index].clone(),
            rationale_mask=masks[:, task_index].clone(),
            source_index=torch.tensor(source_index, dtype=torch.long),
        ))
    return partitions


def google_task_partitions(project_root, task):
    folder = project_root / "reference/graph-attribution/data" / task
    frame = pd.read_csv(folder / f"{task}_smiles.csv")
    official = np.load(folder / f"{task}_traintest_indices.npz")
    train, test = official["train_index"].astype(int), official["test_index"].astype(int)
    labels = np.load(folder / "y_true.npz")["y"].reshape(-1).astype(int)
    rationales = np.load(folder / "true_raw_attribution_datadicts.npz", allow_pickle=True)["datadict_list"].reshape(-1)
    fit, dev, calibration = stratified_partitions(train, labels, frame["mol_id"].astype(str).to_numpy())
    destinations = {}
    for name, indices in {"fit": fit, "dev": dev, "calibration": calibration, "test": test}.items():
        for index in indices:
            destinations[int(index)] = name
    partitions = {"fit": [], "dev": [], "calibration": [], "test": []}
    for index, (smiles, rationale) in enumerate(zip(frame["smiles"], rationales)):
        mol = Chem.MolFromSmiles(str(smiles))
        x, edge_index, edge_attr = molecule_features(mol)
        mask = np.asarray(rationale["nodes"])
        if mask.ndim == 1:
            mask = mask[:, None]
        partitions[destinations[index]].append(Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor(labels[index], dtype=torch.long),
            rationale_mask=torch.tensor(mask[:, -1], dtype=torch.bool),
            source_index=torch.tensor(index, dtype=torch.long),
        ))
    return partitions


def subset_stratified(graphs, limit):
    if len(graphs) <= limit:
        return graphs
    groups = defaultdict(list)
    for graph in graphs:
        groups[int(graph.y)].append(graph)
    selected = []
    for label in sorted(groups):
        group = sorted(groups[label], key=lambda graph: hash_order(int(graph.source_index)))
        selected.extend(group[:max(1, round(limit * len(group) / len(graphs)))])
    return sorted(selected[:limit], key=lambda graph: int(graph.source_index))


def run_cell(family, task, seed, partitions, out_dir, contract_hash, code_hash, mode):
    cell_id = f"{family}__{task}__gine__seed{seed}"
    cell_dir = out_dir / "cells"
    result_path = cell_dir / f"{cell_id}.json"
    checkpoint_path = out_dir / "checkpoints" / f"{cell_id}.pt"
    score_artifact_path = out_dir / "score_artifacts" / f"{cell_id}.npz"
    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        artifact = existing.get("score_artifact", {})
        artifact_path = Path(artifact.get("path", ""))
        if (
            existing.get("status") == "complete"
            and existing.get("contract_sha256") == contract_hash
            and existing.get("code_sha256") == code_hash
            and artifact_path.is_file()
            and artifact.get("sha256") == sha256(artifact_path)
            and validate_score_artifact(artifact_path, contract_hash, code_hash)
        ):
            log(f"skip_complete={cell_id}")
            return existing
        raise RuntimeError(f"existing cell has incompatible provenance: {result_path}")
    set_seed(seed)
    device = torch.device("cuda")
    model = ResidualGINE().to(device)
    max_epochs = 2 if mode == "smoke" else 100
    batch_size = 64 if mode == "smoke" else 128
    ig_steps = 3 if mode == "smoke" else 20
    log(f"cell_start={cell_id} counts=" + json.dumps({name: len(rows) for name, rows in partitions.items()}))
    training = train_model(model, partitions["fit"], partitions["dev"], device, max_epochs, batch_size)
    predictor = {
        "calibration": evaluate(model, partitions["calibration"], device, batch_size),
        "test": evaluate(model, partitions["test"], device, batch_size),
    }
    cal_nonempty = [graph for graph in partitions["calibration"] if bool(graph.rationale_mask.any())]
    test_nonempty = [graph for graph in partitions["test"] if bool(graph.rationale_mask.any())]
    if len(cal_nonempty) < 10 or len(test_nonempty) == 0:
        raise ValueError(f"insufficient rationale examples: calibration={len(cal_nonempty)}, test={len(test_nonempty)}")
    cal_scores, cal_truths, cal_timing = integrated_gradients(model, cal_nonempty, device, max(8, batch_size // 4), ig_steps)
    test_scores, test_truths, test_timing = integrated_gradients(model, test_nonempty, device, max(8, batch_size // 4), ig_steps)
    explanation, schedules = explanation_metrics(cal_scores, cal_truths, test_scores, test_truths)
    explanation["n_calibration_null"] = len(partitions["calibration"]) - len(cal_nonempty)
    explanation["n_test_null"] = len(partitions["test"]) - len(test_nonempty)
    explanation["calibration_timing"] = cal_timing
    explanation["test_timing"] = test_timing
    write_score_artifact(
        score_artifact_path, cal_scores, cal_truths, test_scores, test_truths,
        cal_nonempty, test_nonempty, schedules, contract_hash, code_hash,
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "family": family,
        "task": task,
        "model": "residual_gine",
        "seed": seed,
        "contract_sha256": contract_hash,
        "code_sha256": code_hash,
        "atom_dim": ATOM_DIM,
        "edge_dim": EDGE_DIM,
    }, checkpoint_path)
    payload = {
        "status": "complete",
        "evidence_surface": "non_evidentiary_smoke" if mode == "smoke" else "reviewer_required_retrospective_main",
        "cell_id": cell_id,
        "family": family,
        "task": task,
        "model": "residual_gine",
        "seed": seed,
        "contract_sha256": contract_hash,
        "code_sha256": code_hash,
        "partition_counts": {name: len(rows) for name, rows in partitions.items()},
        "training_provenance": {
            "contract_sha256": contract_hash,
            "code_sha256": code_hash,
            "checkpoint_reused": False,
        },
        "evaluation_provenance": {
            "contract_sha256": contract_hash,
            "code_sha256": code_hash,
            "mode": mode,
        },
        "training": training,
        "predictor": predictor,
        "predictor_competence_pass": predictor["test"]["auroc"] > 0.65,
        "integrated_gradients": explanation,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "score_artifact": {
            "path": str(score_artifact_path),
            "sha256": sha256(score_artifact_path),
            "schema": "a4_gine_scores_v1",
        },
    }
    write_json(result_path, payload)
    log(f"cell_complete={cell_id} test_auroc={predictor['test']['auroc']:.6f} risk={explanation['crc']['test_risk']:.6f} retained={explanation['crc']['test_mean_atom_fraction']:.6f}")
    del model
    torch.cuda.empty_cache()
    return payload


def repair_cell(previous, partitions, out_dir, contract_hash, code_hash):
    cell_id = previous["cell_id"]
    checkpoint_path = Path(previous["checkpoint"])
    checkpoint_hash = sha256(checkpoint_path)
    if checkpoint_hash != previous["checkpoint_sha256"]:
        raise ValueError(f"checkpoint hash drift for {cell_id}")
    device = torch.device("cuda")
    model = ResidualGINE().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    batch_size = 128
    predictor = {
        "calibration": evaluate(model, partitions["calibration"], device, batch_size),
        "test": evaluate(model, partitions["test"], device, batch_size),
    }
    cal_nonempty = [graph for graph in partitions["calibration"] if bool(graph.rationale_mask.any())]
    test_nonempty = [graph for graph in partitions["test"] if bool(graph.rationale_mask.any())]
    cal_scores, cal_truths, cal_timing = integrated_gradients(model, cal_nonempty, device, 32, 20)
    test_scores, test_truths, test_timing = integrated_gradients(model, test_nonempty, device, 32, 20)
    explanation, schedules = explanation_metrics(cal_scores, cal_truths, test_scores, test_truths)
    explanation["n_calibration_null"] = len(partitions["calibration"]) - len(cal_nonempty)
    explanation["n_test_null"] = len(partitions["test"]) - len(test_nonempty)
    explanation["calibration_timing"] = cal_timing
    explanation["test_timing"] = test_timing
    score_artifact_path = out_dir / "score_artifacts" / f"{cell_id}.npz"
    write_score_artifact(
        score_artifact_path, cal_scores, cal_truths, test_scores, test_truths,
        cal_nonempty, test_nonempty, schedules, contract_hash, code_hash,
    )
    repaired = copy.deepcopy(previous)
    repaired["contract_sha256"] = contract_hash
    repaired["code_sha256"] = code_hash
    training_provenance = copy.deepcopy(previous.get("training_provenance", {}))
    if not training_provenance:
        training_provenance = {
            "contract_sha256": previous["contract_sha256"],
            "code_sha256": previous["code_sha256"],
        }
    training_provenance.update({
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_reused": True,
        "compatibility_basis": "amendment_002 changes only oracle postprocessing and reusable score persistence; model, training, split, and IG are unchanged",
    })
    repaired["training_provenance"] = training_provenance
    repaired["evaluation_provenance"] = {
        "contract_sha256": contract_hash,
        "code_sha256": code_hash,
        "mode": "repair",
    }
    repaired["predictor"] = predictor
    repaired["predictor_competence_pass"] = predictor["test"]["auroc"] > 0.65
    repaired["integrated_gradients"] = explanation
    repaired["score_artifact"] = {
        "path": str(score_artifact_path),
        "sha256": sha256(score_artifact_path),
        "schema": "a4_gine_scores_v1",
    }
    amendment_history = copy.deepcopy(previous.get("amendment_history", []))
    if not any(item.get("amendment") == "contract_amendment_002.json" for item in amendment_history):
        amendment_history.append({
            "amendment": "contract_amendment_002.json",
            "previous_contract_sha256": previous["contract_sha256"],
            "previous_code_sha256": previous["code_sha256"],
            "previous_oracle_definition": previous["integrated_gradients"]["oracle"]["definition"],
            "previous_oracle_excess_mean_atom_fraction": previous["integrated_gradients"]["oracle_excess_mean_atom_fraction"],
            "checkpoint_reused": True,
        })
    repaired["amendment_history"] = amendment_history
    result_path = out_dir / "cells" / f"{cell_id}.json"
    write_json(result_path, repaired)
    log(
        f"cell_repaired={cell_id} checkpoint_sha256={checkpoint_hash} "
        f"risk={explanation['crc']['test_risk']:.6f} retained={explanation['crc']['test_mean_atom_fraction']:.6f} "
        f"oracle_excess={explanation['oracle_excess_mean_atom_fraction']:.6f}"
    )
    del model
    torch.cuda.empty_cache()
    return repaired


def flatten_cell(cell):
    crc = cell["integrated_gradients"]["crc"]
    oracle = cell["integrated_gradients"]["oracle"]
    return {
        "cell_id": cell["cell_id"],
        "family": cell["family"],
        "task": cell["task"],
        "model": "gine",
        "seed": cell["seed"],
        "test_auroc": cell["predictor"]["test"]["auroc"],
        "test_auprc": cell["predictor"]["test"]["auprc"],
        "test_weighted_f1": cell["predictor"]["test"]["weighted_f1"],
        "predictor_competence_pass": cell["predictor_competence_pass"],
        "n_calibration_rationale": cell["integrated_gradients"]["n_calibration_rationale"],
        "n_test_rationale": cell["integrated_gradients"]["n_test_rationale"],
        "crc_nominal_fraction": crc["nominal_fraction"],
        "test_risk": crc["test_risk"],
        "mean_atom_fraction": crc["test_mean_atom_fraction"],
        "median_atom_fraction": crc["test_median_atom_fraction"],
        "precision": crc["test_precision"],
        "iou": crc["test_iou"],
        "tie_inflation": crc["test_mean_tie_inflation"],
        "oracle_nominal_fraction": oracle["nominal_fraction"],
        "oracle_mean_atom_fraction": oracle["test_mean_atom_fraction"],
        "oracle_excess_mean_atom_fraction": cell["integrated_gradients"]["oracle_excess_mean_atom_fraction"],
        "risk_pass": crc["test_risk"] <= ALPHA,
    }


def historical_rows(project_root):
    rows = []
    cell_dir = project_root / "artifacts/experiment/gradient_grid_main/cells"
    for path in sorted(cell_dir.glob("*.json")):
        cell = json.loads(path.read_text(encoding="utf-8"))
        metrics = cell["explainers"]["ig"]["metrics"]["alpha"]["0.10"]["crc"]
        rows.append({
            "family": cell["family"],
            "task": cell["task"],
            "model": cell["model"],
            "seed": cell["seed"],
            "test_auroc": cell["predictor"]["test"]["auroc"],
            "test_auprc": cell["predictor"]["test"]["auprc"],
            "test_weighted_f1": cell["predictor"]["test"]["weighted_f1"],
            "test_risk": metrics["risk"],
            "mean_atom_fraction": metrics["mean_atom_fraction"],
            "precision": metrics["precision"],
            "iou": metrics["iou"],
        })
    if len(rows) != 66:
        raise ValueError(f"expected 66 historical cells, observed {len(rows)}")
    return pd.DataFrame(rows)


def summarize_full(project_root, out_dir, contract_hash, code_hash):
    cells = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((out_dir / "cells").glob("*.json"))]
    flat = pd.DataFrame([flatten_cell(cell) for cell in cells])
    flat.to_csv(out_dir / "cells.csv", index=False)
    metric_columns = ["test_auroc", "test_auprc", "test_weighted_f1", "test_risk", "mean_atom_fraction", "median_atom_fraction", "precision", "iou", "tie_inflation", "oracle_mean_atom_fraction", "oracle_excess_mean_atom_fraction"]
    tasks = flat.groupby(["family", "task"], as_index=False)[metric_columns].mean()
    tasks["n_seeds"] = 3
    tasks.to_csv(out_dir / "task_summary.csv", index=False)
    family = tasks.groupby("family", as_index=False)[metric_columns].mean()
    family["n_tasks"] = family["family"].map({"bxaic": 7, "google": 4})
    family.to_csv(out_dir / "family_summary.csv", index=False)
    historical = historical_rows(project_root)
    historical_task = historical.groupby(["family", "task", "model"], as_index=False)[["test_auroc", "test_auprc", "test_weighted_f1", "test_risk", "mean_atom_fraction", "precision", "iou"]].mean()
    comparisons = historical_task.merge(tasks, on=["family", "task"], suffixes=("_historical", "_gine"), validate="many_to_one")
    for metric in ["test_auroc", "test_auprc", "test_weighted_f1", "test_risk", "mean_atom_fraction", "precision", "iou"]:
        comparisons[f"delta_gine_minus_historical_{metric}"] = comparisons[f"{metric}_gine"] - comparisons[f"{metric}_historical"]
    comparisons["predictor_metrics_directly_comparable"] = True
    comparisons["explanation_metrics_directly_comparable"] = False
    comparisons["explanation_comparison_caveat"] = "Historical IG used deterministic atom-index tie breaking; A4 uses include-all-ties and jointly changes architecture, capacity, atom features, and bond features."
    comparisons.to_csv(out_dir / "historical_comparison.csv", index=False)
    summary = {
        "status": "complete",
        "run_id": "EXP-A4-BOND-AWARE-GINE-20260904",
        "contract_sha256": contract_hash,
        "code_sha256": code_hash,
        "expected_cells": 33,
        "observed_cells": len(flat),
        "n_tasks": len(tasks),
        "n_seeds": 3,
        "primary_alpha": ALPHA,
        "predictor_competence_pass_cells": int(flat["predictor_competence_pass"].sum()),
        "risk_pass_cells": int(flat["risk_pass"].sum()),
        "task_macro": {column: float(tasks[column].mean()) for column in metric_columns},
        "family_task_macro": {row["family"]: {column: float(row[column]) for column in metric_columns} for _, row in family.iterrows()},
        "minimum_test_auroc_cell": flat.loc[flat["test_auroc"].idxmin()].to_dict(),
        "maximum_test_risk_cell": flat.loc[flat["test_risk"].idxmax()].to_dict(),
        "comparison_boundary": "Predictor deltas are same-split descriptive comparisons. Explanation deltas are not directly comparable because the historical cells used deterministic index tie breaking and A4 uses include-all-ties; the architecture/feature changes are joint, not a bond-only ablation.",
        "claim_update": "pending validation",
    }
    write_json(out_dir / "summary.json", summary)
    return flat, tasks, comparisons, summary


def validate_full(out_dir, flat, tasks, comparisons, contract_hash, code_hash):
    cell_payloads = [json.loads(path.read_text(encoding="utf-8")) for path in (out_dir / "cells").glob("*.json")]
    artifact_checks = []
    for cell in cell_payloads:
        artifact = cell.get("score_artifact", {})
        path = Path(artifact.get("path", ""))
        artifact_checks.append(bool(
            path.is_file()
            and artifact.get("sha256") == sha256(path)
            and validate_score_artifact(path, contract_hash, code_hash)
        ))
    required_numeric = ["test_auroc", "test_auprc", "test_weighted_f1", "test_risk", "mean_atom_fraction", "precision", "iou", "oracle_mean_atom_fraction", "oracle_excess_mean_atom_fraction"]
    checks = {
        "mapping_audit_pass": json.loads((out_dir / "mapping_audit.json").read_text(encoding="utf-8"))["status"] == "PASS",
        "cell_count_33": len(flat) == 33,
        "unique_cell_ids_33": flat["cell_id"].nunique() == 33,
        "task_count_11": len(tasks) == 11,
        "three_seeds_each": bool(flat.groupby(["family", "task"])["seed"].nunique().eq(3).all()),
        "seed_set_exact": set(flat["seed"]) == set(SEEDS),
        "contract_hash_match": all(cell["contract_sha256"] == contract_hash for cell in cell_payloads),
        "code_hash_match": all(cell["code_sha256"] == code_hash for cell in cell_payloads),
        "evaluation_provenance_match": all(
            cell.get("evaluation_provenance", {}).get("contract_sha256") == contract_hash
            and cell.get("evaluation_provenance", {}).get("code_sha256") == code_hash
            for cell in cell_payloads
        ),
        "training_provenance_declared": all("training_provenance" in cell for cell in cell_payloads),
        "score_artifacts_valid": len(artifact_checks) == 33 and all(artifact_checks),
        "schedule_matched_oracle_definition": all(
            cell["integrated_gradients"]["oracle"]["definition"] == "rationale_first_on_actual_include_all_ties_cardinality_schedule"
            for cell in cell_payloads
        ),
        "oracle_proofs_pass": all(all(cell["integrated_gradients"]["oracle_proof"].values()) for cell in cell_payloads),
        "finite_required_metrics": bool(np.isfinite(flat[required_numeric].to_numpy(dtype=float)).all()),
        "metric_ranges": bool(((flat[["test_auroc", "test_auprc", "test_weighted_f1", "test_risk", "mean_atom_fraction", "precision", "iou"]] >= 0).all().all()) and ((flat[["test_auroc", "test_auprc", "test_weighted_f1", "test_risk", "mean_atom_fraction", "precision", "iou"]] <= 1).all().all())),
        "oracle_excess_nonnegative": bool((flat["oracle_excess_mean_atom_fraction"] >= -1e-12).all()),
        "partition_counts_seed_invariant": bool(flat.merge(pd.DataFrame([{"cell_id": p.stem, **json.loads(p.read_text(encoding="utf-8"))["partition_counts"]} for p in (out_dir / "cells").glob("*.json")]), on="cell_id").groupby(["family", "task"])[["fit", "dev", "calibration", "test"]].nunique().eq(1).all().all()),
        "historical_comparison_rows_22": len(comparisons) == 22,
        "all_checkpoints_exist": all(Path(json.loads(path.read_text(encoding="utf-8"))["checkpoint"]).exists() for path in (out_dir / "cells").glob("*.json")),
        "all_negative_results_retained": len(flat) == 33,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    validation = {"status": status, "checks": checks, "failed_checks": [key for key, value in checks.items() if not value]}
    write_json(out_dir / "validation.json", validation)
    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["validation_status"] = status
    summary["claim_update"] = "supported as a bounded architecture-and-representation robustness extension" if status == "PASS" else "inconclusive because validation failed"
    write_json(summary_path, summary)
    if status != "PASS":
        raise RuntimeError(f"full validation failed: {validation['failed_checks']}")
    return validation


def run_oracle_unit_validation(out_dir, contract_hash, code_hash):
    rng = np.random.default_rng(20260904)
    score_rows, rationale_rows = [], []
    for _ in range(240):
        n_atoms = int(rng.integers(3, 48))
        scores = np.round(rng.normal(size=n_atoms), decimals=int(rng.integers(0, 3)))
        rationale_size = int(rng.integers(1, n_atoms + 1))
        truth = set(rng.choice(n_atoms, size=rationale_size, replace=False).tolist())
        score_rows.append(scores)
        rationale_rows.append(truth)
    counts = selected_count_table(score_rows)
    actual = loss_table(score_rows, rationale_rows)
    oracle = matched_oracle_loss_table(counts, rationale_rows)
    metrics, schedules = explanation_metrics(score_rows[:120], rationale_rows[:120], score_rows[120:], rationale_rows[120:])
    proof_artifact = out_dir / "oracle_proof_score_artifact.npz"
    proof_cal_graphs = [Data(source_index=torch.tensor(index)) for index in range(120)]
    proof_test_graphs = [Data(source_index=torch.tensor(index + 120)) for index in range(120)]
    write_score_artifact(
        proof_artifact,
        score_rows[:120], rationale_rows[:120], score_rows[120:], rationale_rows[120:],
        proof_cal_graphs, proof_test_graphs, schedules, contract_hash, code_hash,
    )
    exact_count_matches = all(
        len(top_fraction_set(scores, float(fraction))) == int(counts[row_index, fraction_index])
        for row_index, scores in enumerate(score_rows)
        for fraction_index, fraction in enumerate(FRACTIONS)
    )
    checks = {
        "tie_inclusive_counts_exact": exact_count_matches,
        "selected_counts_monotone": bool((np.diff(counts, axis=1) >= 0).all()),
        "oracle_pointwise_loss_le_actual": bool((oracle <= actual + 1e-12).all()),
        "oracle_crc_index_not_later": metrics["oracle"]["index"] <= metrics["crc"]["index"],
        "oracle_capacity_excess_nonnegative": metrics["oracle_excess_mean_atom_fraction"] >= -1e-12,
        "returned_schedules_exact_shape": schedules["calibration_selected_counts"].shape == (120, 101) and schedules["test_selected_counts"].shape == (120, 101),
        "ragged_score_artifact_roundtrip": validate_score_artifact(proof_artifact, contract_hash, code_hash),
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "contract_sha256": contract_hash,
        "code_sha256": code_hash,
        "design": "240 deterministic synthetic molecules with deliberately quantized scores to exercise ties",
        "checks": checks,
        "observed_crc_index": metrics["crc"]["index"],
        "observed_oracle_index": metrics["oracle"]["index"],
        "observed_capacity_excess": metrics["oracle_excess_mean_atom_fraction"],
    }
    write_json(out_dir / "oracle_unit_validation.json", payload)
    if payload["status"] != "PASS":
        raise RuntimeError("oracle unit validation failed")
    return payload


def run_repair(project_root, out_dir, contract_hash, code_hash):
    paths = sorted((out_dir / "cells").glob("*.json"))
    previous_cells = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if not previous_cells:
        raise ValueError("repair requires at least one completed cell")
    before_hashes = {cell["cell_id"]: cell["checkpoint_sha256"] for cell in previous_cells}
    by_task = defaultdict(list)
    for cell in previous_cells:
        by_task[(cell["family"], cell["task"])].append(cell)
    frame = base_records = None
    if any(family == "bxaic" for family, _ in by_task):
        frame, base_records = load_bxaic_base(project_root)
    repaired = []
    for family, task in ALL_TASKS:
        group = by_task.get((family, task), [])
        if not group:
            continue
        partitions = bxaic_task_partitions(frame, base_records, task) if family == "bxaic" else google_task_partitions(project_root, task)
        for previous in sorted(group, key=lambda cell: cell["seed"]):
            artifact = previous.get("score_artifact", {})
            artifact_path = Path(artifact.get("path", ""))
            already_current = bool(
                previous.get("contract_sha256") == contract_hash
                and previous.get("code_sha256") == code_hash
                and previous.get("evaluation_provenance", {}).get("contract_sha256") == contract_hash
                and previous.get("evaluation_provenance", {}).get("code_sha256") == code_hash
                and artifact_path.is_file()
                and artifact.get("sha256") == sha256(artifact_path)
                and validate_score_artifact(artifact_path, contract_hash, code_hash)
            )
            if already_current:
                log(f"skip_repaired={previous['cell_id']}")
                repaired.append(previous)
            else:
                repaired.append(repair_cell(previous, partitions, out_dir, contract_hash, code_hash))
        del partitions
    checks = {
        "repaired_count_matches_input": len(repaired) == len(previous_cells),
        "unique_cells_preserved": len({cell["cell_id"] for cell in repaired}) == len(previous_cells),
        "checkpoint_hashes_unchanged": all(cell["checkpoint_sha256"] == before_hashes[cell["cell_id"]] for cell in repaired),
        "evaluation_hashes_current": all(
            cell["contract_sha256"] == contract_hash
            and cell["code_sha256"] == code_hash
            and cell["evaluation_provenance"]["contract_sha256"] == contract_hash
            and cell["evaluation_provenance"]["code_sha256"] == code_hash
            for cell in repaired
        ),
        "checkpoint_reuse_declared": all(cell["training_provenance"]["checkpoint_reused"] for cell in repaired),
        "oracle_excess_nonnegative": all(cell["integrated_gradients"]["oracle_excess_mean_atom_fraction"] >= -1e-12 for cell in repaired),
        "score_artifacts_valid": all(
            Path(cell["score_artifact"]["path"]).is_file()
            and cell["score_artifact"]["sha256"] == sha256(Path(cell["score_artifact"]["path"]))
            and validate_score_artifact(Path(cell["score_artifact"]["path"]), contract_hash, code_hash)
            for cell in repaired
        ),
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "contract_sha256": contract_hash,
        "code_sha256": code_hash,
        "input_cells": len(previous_cells),
        "repaired_cells": len(repaired),
        "checks": checks,
    }
    write_json(out_dir / "repair_validation.json", payload)
    if payload["status"] != "PASS":
        raise RuntimeError(f"repair validation failed: {[key for key, value in checks.items() if not value]}")
    return payload


def environment_snapshot(out_dir, contract_hash, code_hash, command):
    gpu = None
    try:
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True, timeout=15).strip()
    except Exception as error:
        gpu = f"unavailable: {error}"
    payload = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_geometric": torch_geometric.__version__,
        "rdkit": rdBase.rdkitVersion,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu,
        "contract_sha256": contract_hash,
        "code_sha256": code_hash,
    }
    write_json(out_dir / "environment.json", payload)


def write_manifest(out_dir):
    paths = sorted(path for path in out_dir.rglob("*") if path.is_file() and path.name != "manifest.sha256")
    lines = [f"{sha256(path)}  {path.relative_to(out_dir).as_posix()}" for path in paths]
    (out_dir / "manifest.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize_authoritative_logs(out_dir, contract_hash, code_hash):
    history_source = out_dir / "full.log"
    history_copy = out_dir / "full_attempt_history.log"
    shutil.copyfile(history_source, history_copy)
    cells = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((out_dir / "cells").glob("*.json"))]
    lines = [
        f"authoritative_cell_snapshot contract_sha256={contract_hash} code_sha256={code_hash} source=active_cells",
    ]
    for cell in cells:
        lines.append(
            f"cell_complete={cell['cell_id']} contract_sha256={cell['contract_sha256']} code_sha256={cell['code_sha256']} "
            f"checkpoint_sha256={cell['checkpoint_sha256']} score_artifact_sha256={cell['score_artifact']['sha256']}"
        )
    lines.append(f"authoritative_complete cells={len(cells)} contract_sha256={contract_hash} code_sha256={code_hash}")
    authoritative_path = out_dir / "full_authoritative.log"
    authoritative_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cell_lines = [line for line in lines if line.startswith("cell_complete=")]
    cell_ids = [line.split()[0].split("=", 1)[1] for line in cell_lines]
    checks = {
        "attempt_history_exact_copy": sha256(history_source) == sha256(history_copy),
        "authoritative_cell_lines_33": len(cell_lines) == 33,
        "authoritative_unique_cell_ids_33": len(set(cell_ids)) == 33,
        "authoritative_matches_active_cells": set(cell_ids) == {cell["cell_id"] for cell in cells},
        "only_final_contract_hash": all(f"contract_sha256={contract_hash}" in line for line in lines),
        "only_final_code_hash": all(f"code_sha256={code_hash}" in line for line in lines),
        "all_active_cell_hashes_final": all(cell["contract_sha256"] == contract_hash and cell["code_sha256"] == code_hash for cell in cells),
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "contract_sha256": contract_hash,
        "code_sha256": code_hash,
        "raw_attempt_history": str(history_copy),
        "raw_attempt_history_sha256": sha256(history_copy),
        "authoritative_log": str(authoritative_path),
        "authoritative_log_sha256": sha256(authoritative_path),
        "checks": checks,
    }
    write_json(out_dir / "log_validation.json", payload)
    if payload["status"] != "PASS":
        raise RuntimeError(f"authoritative log validation failed: {[key for key, value in checks.items() if not value]}")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["audit", "proof", "smoke", "repair", "full"], required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    global LOG_PATH
    LOG_PATH = out_dir / f"{args.mode}.log"
    contract_path = out_dir / "run_contract.json"
    if not contract_path.exists():
        raise FileNotFoundError(f"run contract missing: {contract_path}")
    contract_hash = sha256(contract_path)
    code_hash = sha256(__file__)
    command = subprocess.list2cmdline(sys.argv)
    log(f"mode={args.mode} contract_sha256={contract_hash} code_sha256={code_hash}")
    environment_snapshot(out_dir, contract_hash, code_hash, command)
    if args.mode == "audit":
        run_mapping_audit(project_root, out_dir)
        log("audit_complete status=PASS")
        write_manifest(out_dir)
        return
    if args.mode == "proof":
        result = run_oracle_unit_validation(out_dir, contract_hash, code_hash)
        log(f"oracle_proof_complete status={result['status']}")
        write_manifest(out_dir)
        return
    if args.confirm != CONFIRM_TOKEN:
        raise ValueError(f"{args.mode} requires --confirm {CONFIRM_TOKEN}")
    audit_path = out_dir / "mapping_audit.json"
    if not audit_path.exists() or json.loads(audit_path.read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("a PASS mapping_audit.json is required before model execution")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen A4 execution envelope")
    run_oracle_unit_validation(out_dir, contract_hash, code_hash)
    if args.mode == "smoke":
        smoke_dir = out_dir / "smoke"
        smoke_dir.mkdir(parents=True, exist_ok=True)
        partitions = google_task_partitions(project_root, "benzene")
        partitions = {name: subset_stratified(rows, 512) for name, rows in partitions.items()}
        result = run_cell("google", "benzene", 42, partitions, smoke_dir, contract_hash, code_hash, "smoke")
        checks = {
            "cell_complete": result["status"] == "complete",
            "non_evidentiary_surface": result["evidence_surface"] == "non_evidentiary_smoke",
            "finite_test_auroc": math.isfinite(result["predictor"]["test"]["auroc"]),
            "finite_explanation_metrics": all(math.isfinite(result["integrated_gradients"]["crc"][key]) for key in ["test_risk", "test_mean_atom_fraction", "test_precision", "test_iou"]),
            "checkpoint_exists": Path(result["checkpoint"]).exists(),
            "schedule_matched_oracle": result["integrated_gradients"]["oracle"]["definition"] == "rationale_first_on_actual_include_all_ties_cardinality_schedule",
            "oracle_excess_nonnegative": result["integrated_gradients"]["oracle_excess_mean_atom_fraction"] >= -1e-12,
            "score_artifact_valid": validate_score_artifact(Path(result["score_artifact"]["path"]), contract_hash, code_hash),
        }
        write_json(smoke_dir / "validation.json", {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks})
        log(f"smoke_complete status={'PASS' if all(checks.values()) else 'FAIL'}")
        write_manifest(out_dir)
        if not all(checks.values()):
            raise RuntimeError("smoke validation failed")
        return
    if args.mode == "repair":
        validation = run_repair(project_root, out_dir, contract_hash, code_hash)
        log(f"repair_complete status={validation['status']} cells={validation['repaired_cells']}")
        write_manifest(out_dir)
        return
    frame, base_records = load_bxaic_base(project_root)
    all_cells = []
    for family, task in ALL_TASKS:
        if family == "bxaic":
            partitions = bxaic_task_partitions(frame, base_records, task)
        else:
            if base_records is not None:
                del base_records
                base_records = None
                frame = None
            partitions = google_task_partitions(project_root, task)
        for seed in SEEDS:
            all_cells.append(run_cell(family, task, seed, partitions, out_dir, contract_hash, code_hash, "full"))
        del partitions
    flat, tasks, comparisons, _ = summarize_full(project_root, out_dir, contract_hash, code_hash)
    validation = validate_full(out_dir, flat, tasks, comparisons, contract_hash, code_hash)
    log(f"full_complete status={validation['status']} cells={len(flat)}")
    finalize_authoritative_logs(out_dir, contract_hash, code_hash)
    write_manifest(out_dir)


if __name__ == "__main__":
    main()

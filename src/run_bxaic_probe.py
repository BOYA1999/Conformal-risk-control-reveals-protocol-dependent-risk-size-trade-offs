import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GIN, global_add_pool

from molxai_crc import calibrate_crc, loss_table, top_fraction_set


SYMBOLS = ["C", "N", "O", "F", "Cl", "Br", "P", "S", "B", "I", "Unk"]
PROPS = {"B": "B", "P": "P", "X": "X", "indole": "indole", "PAINS": "pains", "rings-count": "rings", "rings-max": "largest_rings"}


class GINClassifier(nn.Module):
    def __init__(self, in_channels=11, hidden=32, layers=3):
        super().__init__()
        self.encoder = GIN(in_channels, hidden, num_layers=layers, out_channels=32, norm="batch_norm")
        self.head = nn.Linear(32, 2)

    def forward(self, x, edge_index, batch):
        return self.head(global_add_pool(self.encoder(x, edge_index), batch))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def hash_order(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def build_train_graphs(csv_path, sdf_path, task):
    frame = pd.read_csv(csv_path)
    train_rows = frame.index[frame["split_0"] == "train"].tolist()
    dev_rows = []
    for label in sorted(frame.loc[train_rows, task].unique()):
        group = [i for i in train_rows if frame.at[i, task] == label]
        group.sort(key=lambda i: hash_order(frame.at[i, "ChEMBL ID"]))
        dev_rows.extend(group[: round(0.125 * len(group))])
    dev_rows = set(dev_rows)
    wanted = set(train_rows)
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    fit, dev = [], []
    prop = PROPS[task]
    for i, mol in enumerate(supplier):
        if i not in wanted:
            continue
        atom_ids = [SYMBOLS.index(a.GetSymbol()) if a.GetSymbol() in SYMBOLS else 10 for a in mol.GetAtoms()]
        x = F.one_hot(torch.tensor(atom_ids), len(SYMBOLS)).float()
        edges = []
        for bond in mol.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edges.extend([(a, b), (b, a)])
        edge_index = torch.tensor(edges, dtype=torch.long).T.contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
        raw = mol.GetProp(prop).strip() if mol.HasProp(prop) else ""
        rationale = torch.zeros(len(atom_ids), dtype=torch.bool)
        if raw:
            rationale[torch.tensor(sorted({int(v) for v in raw.split(",")}), dtype=torch.long)] = True
        data = Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor(int(frame.at[i, task]), dtype=torch.long),
            rationale_mask=rationale,
            source_index=torch.tensor(i, dtype=torch.long),
        )
        (dev if i in dev_rows else fit).append(data)
    return fit, dev


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    labels, probs, preds = [], [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        probability = logits.softmax(-1)[:, 1]
        labels.extend(batch.y.cpu().tolist())
        probs.extend(probability.cpu().tolist())
        preds.extend(logits.argmax(-1).cpu().tolist())
    return {
        "auroc": float(roc_auc_score(labels, probs)),
        "auprc": float(average_precision_score(labels, probs)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted")),
    }


def train_model(model, fit, dev, device, batch_size, epochs):
    labels = torch.tensor([int(d.y) for d in fit])
    class_weights = 1.0 / torch.bincount(labels).float()
    sampler = WeightedRandomSampler(class_weights[labels], len(labels), replacement=True)
    train_loader = DataLoader(fit, batch_size=batch_size, sampler=sampler)
    dev_loader = DataLoader(dev, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state, best_metrics, best_epoch, stale = None, None, 0, 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(batch.x, batch.edge_index, batch.batch), batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        metrics = evaluate(model, dev_loader, device)
        print(f"epoch={epoch} dev_auroc={metrics['auroc']:.5f} dev_f1={metrics['weighted_f1']:.5f}", flush=True)
        if best_metrics is None or metrics["auroc"] > best_metrics["auroc"] + 1e-5:
            best_state, best_metrics, best_epoch, stale = copy.deepcopy(model.state_dict()), metrics, epoch, 0
        else:
            stale += 1
        if epoch >= 10 and stale >= 5:
            break
    model.load_state_dict(best_state)
    return {
        "seconds": time.perf_counter() - started,
        "epochs": epoch,
        "best_epoch": best_epoch,
        "dev_metrics": best_metrics,
        "peak_vram_mib": torch.cuda.max_memory_allocated(device) / 2**20,
    }


def gradient_scores(model, graphs, device, method, batch_size=32, ig_steps=20):
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    rows, rationales = [], []
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for batch in loader:
        batch = batch.to(device)
        if method == "gradinput":
            x = batch.x.detach().clone().requires_grad_(True)
            logits = model(x, batch.edge_index, batch.batch)
            target = logits[torch.arange(len(batch.y), device=device), batch.y].sum()
            grad = torch.autograd.grad(target, x)[0]
            atom_scores = (grad * x).sum(-1)
        else:
            total_grad = torch.zeros_like(batch.x)
            for step in range(1, ig_steps + 1):
                x = (batch.x * step / ig_steps).detach().requires_grad_(True)
                logits = model(x, batch.edge_index, batch.batch)
                target = logits[torch.arange(len(batch.y), device=device), batch.y].sum()
                total_grad += torch.autograd.grad(target, x)[0]
            atom_scores = (batch.x * total_grad / ig_steps).sum(-1)
        ptr = batch.ptr.cpu().tolist()
        atom_scores = atom_scores.detach().cpu().numpy()
        mask = batch.rationale_mask.cpu().numpy()
        for start, end in zip(ptr[:-1], ptr[1:]):
            rows.append(atom_scores[start:end])
            rationales.append(set(np.flatnonzero(mask[start:end]).tolist()))
    seconds = time.perf_counter() - started
    return rows, rationales, {
        "seconds": seconds,
        "molecules": len(graphs),
        "seconds_per_molecule": seconds / len(graphs),
        "peak_vram_mib": torch.cuda.max_memory_allocated(device) / 2**20,
    }


def crc_dev_diagnostic(scores, rationales):
    size = min(1000, len(scores))
    n_calibration = size // 2
    if n_calibration < 20:
        raise ValueError("development diagnostic needs at least 40 rationale-bearing molecules")
    fractions = np.linspace(0, 1, 101)
    calibration = loss_table(scores[:n_calibration], rationales[:n_calibration], fractions)
    evaluation = loss_table(scores[n_calibration:size], rationales[n_calibration:size], fractions)
    result = calibrate_crc(calibration, fractions, alpha=0.1)
    index, fraction = result["index"], result["fraction"]
    precisions, ious, realized = [], [], []
    for score, truth in zip(scores[n_calibration:size], rationales[n_calibration:size]):
        selected = top_fraction_set(score, fraction)
        intersection = len(selected & truth)
        precisions.append(intersection / len(selected) if selected else 0.0)
        ious.append(intersection / len(selected | truth))
        realized.append(len(selected) / len(score))
    result.update({
        "dev_evaluation_risk": float(evaluation[:, index].mean()),
        "mean_realized_atom_fraction": float(np.mean(realized)),
        "mean_precision": float(np.mean(precisions)),
        "mean_iou": float(np.mean(ious)),
        "fixed_20pct_risk": float(evaluation[:, 20].mean()),
        "fixed_50pct_risk": float(evaluation[:, 50].mean()),
    })
    return result


def gnnexplainer_timing(model, graphs, device, epochs=50):
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="phenomenon",
        node_mask_type="attributes",
        edge_mask_type=None,
        model_config=dict(mode="multiclass_classification", task_level="graph", return_type="raw"),
    )
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for data in graphs:
        data = data.to(device)
        batch = torch.zeros(len(data.x), dtype=torch.long, device=device)
        explainer(data.x, data.edge_index, batch=batch, target=data.y.view(1))
    seconds = time.perf_counter() - started
    return {
        "epochs": epochs,
        "seconds": seconds,
        "molecules": len(graphs),
        "seconds_per_molecule": seconds / len(graphs),
        "peak_vram_mib": torch.cuda.max_memory_allocated(device) / 2**20,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--sdf", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", default="indole")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda")
    load_start = time.perf_counter()
    fit, dev = build_train_graphs(args.csv, args.sdf, args.task)
    load_seconds = time.perf_counter() - load_start
    positive = sorted([d for d in dev if d.rationale_mask.any()], key=lambda d: hash_order(int(d.source_index)))[:1000]
    if len(positive) < 1000:
        raise RuntimeError("fewer than 1000 development molecules have rationales")
    model = GINClassifier().to(device)
    training = train_model(model, fit, dev, device, args.batch_size, args.epochs)
    grad_scores, truths, grad_timing = gradient_scores(model, positive, device, "gradinput", args.batch_size)
    ig_scores, ig_truths, ig_timing = gradient_scores(model, positive, device, "ig", max(16, args.batch_size // 4))
    if truths != ig_truths:
        raise RuntimeError("explainer rationale order mismatch")
    gnn_timing = gnnexplainer_timing(model, positive[:20], device)
    explanation_units = 80235 * 2 * 3
    model_cells = 11 * 2 * 3
    projected_seconds_50 = 1.5 * (
        model_cells * training["seconds"]
        + explanation_units * (
            grad_timing["seconds_per_molecule"]
            + ig_timing["seconds_per_molecule"]
            + gnn_timing["seconds_per_molecule"]
        )
    )
    projected_seconds_200 = projected_seconds_50 + 1.5 * explanation_units * 3 * gnn_timing["seconds_per_molecule"]
    result = {
        "run_id": "feasibility_20260809",
        "tier": "auxiliary/dev",
        "task": args.task,
        "seed": args.seed,
        "device": torch.cuda.get_device_name(0),
        "fit_molecules": len(fit),
        "dev_molecules": len(dev),
        "explained_dev_molecules": len(positive),
        "data_load_seconds": load_seconds,
        "training": training,
        "gradinput": {"timing": grad_timing, "diagnostic": crc_dev_diagnostic(grad_scores, truths)},
        "integrated_gradients": {"steps": 20, "timing": ig_timing, "diagnostic": crc_dev_diagnostic(ig_scores, truths)},
        "gnnexplainer": {"timing_only": gnn_timing},
        "projection": {
            "model_cells": model_cells,
            "explanation_units_per_explainer": explanation_units,
            "safety_factor": 1.5,
            "days_with_gnnexplainer_50_epochs": projected_seconds_50 / 86400,
            "days_with_gnnexplainer_200_epochs_linear": projected_seconds_200 / 86400,
            "requires_rental_report_at_200_epochs": projected_seconds_200 / 86400 > 30,
            "assumption": "Indole GIN timing applied conservatively to 11 tasks x 2 backbones x 3 seeds; the 200-epoch estimate linearly scales the measured 50-epoch GNNExplainer time.",
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save({"state_dict": model.state_dict(), "task": args.task, "seed": args.seed}, args.checkpoint)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from run_bxaic_probe import gradient_scores, hash_order, set_seed
from run_gnnexplainer_dev_gate import gnn_scores
from run_gradient_grid import (
    BXAIC_TASKS,
    GOOGLE_TASKS,
    GraphClassifier,
    bxaic_partitions,
    explanation_metrics,
    google_partitions,
)


CHECKPOINTS = ROOT / "artifacts" / "experiment" / "gradient_grid_main" / "checkpoints"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def truths(graphs):
    return [set(torch.nonzero(graph.rationale_mask, as_tuple=False).flatten().tolist()) for graph in graphs]


def select_positive(graphs, limit):
    selected = [graph for graph in graphs if bool(graph.rationale_mask.any())]
    selected.sort(key=lambda graph: hash_order(int(graph.source_index)))
    return selected[:limit]


def saliency_scores(model, graphs, device, batch_size):
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    rows = []
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for batch in loader:
        batch = batch.to(device)
        x = batch.x.detach().clone().requires_grad_(True)
        logits = model(x, batch.edge_index, batch.batch)
        target = logits[torch.arange(len(batch.y), device=device), batch.y].sum()
        scores = torch.autograd.grad(target, x)[0].abs().sum(dim=-1).detach().cpu().numpy()
        ptr = batch.ptr.cpu().tolist()
        rows.extend(scores[start:end] for start, end in zip(ptr[:-1], ptr[1:]))
    seconds = time.perf_counter() - started
    return rows, {
        "seconds": seconds,
        "molecules": len(graphs),
        "seconds_per_molecule": seconds / len(graphs),
        "peak_vram_mib": torch.cuda.max_memory_allocated(device) / 2**20,
    }


@torch.no_grad()
def occlusion_scores(model, graphs, device):
    rows = []
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for graph in graphs:
        source = graph.to(device)
        batch_index = torch.zeros(source.num_nodes, dtype=torch.long, device=device)
        base = model(source.x, source.edge_index, batch_index)[0, int(source.y)]
        variants = []
        for atom in range(source.num_nodes):
            variant = source.clone()
            variant.x = source.x.clone()
            variant.x[atom] = 0
            variants.append(variant)
        batch = Batch.from_data_list(variants)
        perturbed = model(batch.x, batch.edge_index, batch.batch)[:, int(source.y)]
        rows.append((base - perturbed).detach().cpu().numpy())
    seconds = time.perf_counter() - started
    return rows, {
        "seconds": seconds,
        "molecules": len(graphs),
        "seconds_per_molecule": seconds / len(graphs),
        "peak_vram_mib": torch.cuda.max_memory_allocated(device) / 2**20,
    }


def method_payload(cal_scores, test_scores, cal_graphs, test_graphs, seed, cal_timing, test_timing):
    return {
        "calibration_timing": cal_timing,
        "test_timing": test_timing,
        "metrics": explanation_metrics(
            cal_scores,
            truths(cal_graphs),
            test_scores,
            truths(test_graphs),
            seed,
        ),
    }


def run_cell(family, task, model_kind, seed, partitions, limit, gnn_epochs, batch_size, out_dir):
    cell_id = f"{family}__{task}__{model_kind}__seed{seed}"
    result_path = out_dir / "cells" / f"{cell_id}.json"
    if result_path.exists() and json.loads(result_path.read_text(encoding="utf-8")).get("status") == "complete":
        print(f"skip_complete={cell_id}", flush=True)
        return
    calibration = select_positive(partitions["calibration"], limit)
    test = select_positive(partitions["test"], limit)
    if min(len(calibration), len(test)) < 2:
        raise ValueError(f"insufficient rationale-bearing molecules for {cell_id}")
    checkpoint = CHECKPOINTS / f"{cell_id}.pt"
    device = torch.device("cuda")
    model = GraphClassifier(model_kind, calibration[0].x.shape[1]).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["state_dict"])
    model.eval()
    methods = {}

    for method in ("gradinput", "ig"):
        cal_scores, _, cal_timing = gradient_scores(model, calibration, device, method, batch_size)
        test_scores, _, test_timing = gradient_scores(model, test, device, method, batch_size)
        methods[method] = method_payload(cal_scores, test_scores, calibration, test, seed, cal_timing, test_timing)

    cal_scores, cal_timing = saliency_scores(model, calibration, device, batch_size)
    test_scores, test_timing = saliency_scores(model, test, device, batch_size)
    methods["saliency"] = method_payload(cal_scores, test_scores, calibration, test, seed, cal_timing, test_timing)

    cal_scores, cal_timing = occlusion_scores(model, calibration, device)
    test_scores, test_timing = occlusion_scores(model, test, device)
    methods["atom_occlusion"] = method_payload(cal_scores, test_scores, calibration, test, seed, cal_timing, test_timing)

    set_seed(seed)
    cal_scores, cal_timing = gnn_scores(model, calibration, device, gnn_epochs)
    test_scores, test_timing = gnn_scores(model, test, device, gnn_epochs)
    methods["gnnexplainer"] = method_payload(cal_scores, test_scores, calibration, test, seed, cal_timing, test_timing)

    write_json(result_path, {
        "status": "complete",
        "surface": "post_rejection_official_split_matched_subset",
        "cell_id": cell_id,
        "family": family,
        "task": task,
        "model": model_kind,
        "seed": seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "sample": {
            "limit_per_split": limit,
            "calibration_count": len(calibration),
            "test_count": len(test),
            "calibration_source_indices": [int(graph.source_index) for graph in calibration],
            "test_source_indices": [int(graph.source_index) for graph in test],
        },
        "gnnexplainer_epochs": gnn_epochs,
        "methods": methods,
    })
    print(f"cell_complete={cell_id}", flush=True)


def aggregate(out_dir, runtime_seconds):
    rows = []
    for path in sorted((out_dir / "cells").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for method, result in payload["methods"].items():
            metrics = result["metrics"]["alpha"]["0.10"]["crc"]
            rows.append({
                "cell_id": payload["cell_id"],
                "family": payload["family"],
                "task": payload["task"],
                "model": payload["model"],
                "seed": payload["seed"],
                "method": method,
                "n_calibration": payload["sample"]["calibration_count"],
                "n_test": payload["sample"]["test_count"],
                "risk": metrics["risk"],
                "atom_fraction": metrics["mean_atom_fraction"],
                "precision": metrics["precision"],
                "iou": metrics["iou"],
                "selected_grid_fraction": metrics["fraction"],
                "risk_pass": metrics["risk"] <= 0.10,
                "efficiency_pass": metrics["mean_atom_fraction"] < 0.80,
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "cells.csv", index=False)
    summary = []
    for method, group in frame.groupby("method", sort=True):
        task_means = group.groupby(["family", "task"])[["risk", "atom_fraction", "precision", "iou"]].mean()
        summary.append({
            "method": method,
            "cells": int(len(group)),
            "macro_task_mean_risk": float(task_means["risk"].mean()),
            "macro_task_mean_atom_fraction": float(task_means["atom_fraction"].mean()),
            "macro_task_mean_precision": float(task_means["precision"].mean()),
            "macro_task_mean_iou": float(task_means["iou"].mean()),
            "risk_pass_cells": int(group["risk_pass"].sum()),
            "efficiency_pass_cells": int(group["efficiency_pass"].sum()),
        })
    write_json(out_dir / "summary.json", {
        "status": "PASS" if len(frame) else "FAIL",
        "surface": "post_rejection_official_split_matched_subset",
        "runtime_seconds": runtime_seconds,
        "cell_method_rows": int(len(frame)),
        "unique_cells": int(frame["cell_id"].nunique()),
        "methods": summary,
        "interpretation": "Reviewer-driven matched-subset benchmark; not a restoration of the original untouched final-test gate.",
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "established_explainers_20260901"))
    parser.add_argument("--limit-per-split", type=int, default=100)
    parser.add_argument("--gnn-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    limit = 24 if args.smoke else args.limit_per_split
    gnn_epochs = 2 if args.smoke else args.gnn_epochs
    jobs = [("bxaic", task) for task in BXAIC_TASKS] + [("google", task) for task in GOOGLE_TASKS]
    if args.smoke:
        jobs = jobs[:1]
    started = time.time()
    write_json(out_dir / "run_contract.json", {
        "reviewer_item": "DEC-C2",
        "surface": "post_rejection_official_split_matched_subset",
        "methods": ["gradinput", "ig", "saliency", "atom_occlusion", "gnnexplainer"],
        "limit_per_split": limit,
        "gnnexplainer_epochs": gnn_epochs,
        "selection": "rationale-bearing molecules ordered by SHA-256 of source index within each official calibration/test split",
        "fixed": "original checkpoints, splits, rationale masks, CRC grid, alpha=0.10, risk and efficiency definitions",
        "script_sha256": sha256(__file__),
    })
    for family, task in jobs:
        print(f"load_task={family}/{task}", flush=True)
        if family == "bxaic":
            partitions = bxaic_partitions(ROOT / "data/raw/bxaic/data.csv", ROOT / "data/raw/bxaic/explanations.sdf", task)
        else:
            partitions = google_partitions(ROOT / "reference/graph-attribution/data", task)
        models = ["gin"] if args.smoke else ["gin", "gcn"]
        seeds = [42] if args.smoke else [42, 123, 2026]
        for model_kind in models:
            for seed in seeds:
                run_cell(family, task, model_kind, seed, partitions, limit, gnn_epochs, args.batch_size, out_dir)
    aggregate(out_dir, time.time() - started)


if __name__ == "__main__":
    main()

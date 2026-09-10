import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.explain import Explainer, GNNExplainer

from molxai_crc import calibrate_crc, loss_table
from run_bxaic_probe import gradient_scores, hash_order, set_seed
from run_gradient_grid import (
    BXAIC_TASKS,
    FRACTIONS,
    GOOGLE_TASKS,
    GraphClassifier,
    bxaic_partitions,
    google_partitions,
    set_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "artifacts" / "experiment" / "gradient_grid_main"
OUT = ROOT / "artifacts" / "analysis" / "phase_a_followup_20260809" / "gnnexplainer_dev_gate_v2"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def gnn_scores(model, graphs, device, epochs):
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="phenomenon",
        node_mask_type="object",
        edge_mask_type=None,
        model_config=dict(mode="multiclass_classification", task_level="graph", return_type="raw"),
    )
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    scores = []
    for data in graphs:
        data = data.to(device)
        batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
        explanation = explainer(data.x, data.edge_index, batch=batch, target=data.y.view(1))
        scores.append(explanation.node_mask.detach().reshape(data.num_nodes, -1).mean(dim=1).cpu().numpy())
    return scores, {
        "seconds": time.perf_counter() - started,
        "molecules": len(graphs),
        "epochs": epochs,
        "peak_vram_mib": torch.cuda.max_memory_allocated(device) / 2**20,
    }


def gate_metrics(cal_scores, cal_truths, eval_scores, eval_truths):
    cal_table = loss_table(cal_scores, cal_truths, FRACTIONS)
    crc = calibrate_crc(cal_table, FRACTIONS, 0.10)
    return {
        "n_calibration_rationale": len(cal_truths),
        "n_evaluation_rationale": len(eval_truths),
        "alpha": {"0.10": {"crc": {**crc, **set_metrics(eval_scores, eval_truths, crc["fraction"])}}},
    }


def run_cell(family, task, model_kind, partitions, epochs, max_molecules, out_dir):
    cell_id = f"{family}__{task}__{model_kind}__seed42"
    result_path = out_dir / "cells" / f"{cell_id}.json"
    if result_path.exists() and json.loads(result_path.read_text(encoding="utf-8")).get("status") == "complete":
        print(f"skip_complete={cell_id}", flush=True)
        return
    positive = [data for data in partitions["dev"] if bool(data.rationale_mask.any())]
    positive.sort(key=lambda data: hash_order(int(data.source_index)))
    selected = positive[:max_molecules]
    midpoint = len(selected) // 2
    calibration, evaluation = selected[:midpoint], selected[midpoint:]
    if len(calibration) < 2 or len(evaluation) < 2:
        raise ValueError(f"insufficient development rationales for {cell_id}")

    device = torch.device("cuda")
    checkpoint = MAIN / "checkpoints" / f"{cell_id}.pt"
    model = GraphClassifier(model_kind, selected[0].x.shape[1]).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["state_dict"])
    model.eval()
    set_seed(42)
    gnn_cal, gnn_cal_timing = gnn_scores(model, calibration, device, epochs)
    gnn_eval, gnn_eval_timing = gnn_scores(model, evaluation, device, epochs)
    ig_cal, _, ig_cal_timing = gradient_scores(model, calibration, device, "ig", 16)
    ig_eval, _, ig_eval_timing = gradient_scores(model, evaluation, device, "ig", 16)
    cal_truths = [set(torch.nonzero(data.rationale_mask, as_tuple=False).flatten().tolist()) for data in calibration]
    eval_truths = [set(torch.nonzero(data.rationale_mask, as_tuple=False).flatten().tolist()) for data in evaluation]
    payload = {
        "status": "complete",
        "surface": "development_compute_gate",
        "cell_id": cell_id,
        "family": family,
        "task": task,
        "model": model_kind,
        "seed": 42,
        "script_sha256": sha256(__file__),
        "checkpoint_sha256": sha256(checkpoint),
        "epochs": epochs,
        "sample": {
            "available_nonempty_dev": len(positive),
            "calibration_source_indices": [int(data.source_index) for data in calibration],
            "evaluation_source_indices": [int(data.source_index) for data in evaluation],
        },
        "explainers": {
            "gnnexplainer": {
                "calibration_timing": gnn_cal_timing,
                "evaluation_timing": gnn_eval_timing,
                "metrics": gate_metrics(gnn_cal, cal_truths, gnn_eval, eval_truths),
            },
            "ig": {
                "calibration_timing": ig_cal_timing,
                "evaluation_timing": ig_eval_timing,
                "metrics": gate_metrics(ig_cal, cal_truths, ig_eval, eval_truths),
            },
        },
    }
    atomic_json(result_path, payload)
    print(f"cell_complete={cell_id}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    epochs, max_molecules = (1, 40) if args.smoke else (100, 200)
    out_dir = OUT / "smoke" if args.smoke else OUT
    jobs = [("bxaic", task) for task in BXAIC_TASKS] + [("google", task) for task in GOOGLE_TASKS]
    if args.smoke:
        jobs = jobs[:1]
    started = time.time()
    for family, task in jobs:
        print(f"load_task={family}/{task}", flush=True)
        partitions = bxaic_partitions(ROOT / "data/raw/bxaic/data.csv", ROOT / "data/raw/bxaic/explanations.sdf", task) if family == "bxaic" else google_partitions(ROOT / "reference/graph-attribution/data", task)
        for model_kind in (["gin"] if args.smoke else ["gin", "gcn"]):
            run_cell(family, task, model_kind, partitions, epochs, max_molecules, out_dir)
    cells = sorted((out_dir / "cells").glob("*.json"))
    atomic_json(out_dir / "manifest.json", {
        "status": "complete",
        "surface": "development_compute_gate",
        "script_sha256": sha256(__file__),
        "epochs": epochs,
        "max_molecules_per_cell": max_molecules,
        "cell_count": len(cells),
        "seconds": time.time() - started,
        "cells": [str(path.relative_to(ROOT)) for path in cells],
    })


if __name__ == "__main__":
    main()

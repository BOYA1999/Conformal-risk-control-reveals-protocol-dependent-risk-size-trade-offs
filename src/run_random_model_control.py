import json
import time
from pathlib import Path

import torch

from run_bxaic_probe import evaluate, gradient_scores, set_seed
from run_gradient_grid import (
    BXAIC_TASKS,
    GOOGLE_TASKS,
    GraphClassifier,
    atomic_json,
    bxaic_partitions,
    explanation_metrics,
    google_partitions,
    sha256,
    torch_geometric_loader,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "analysis" / "phase_a_followup_20260809" / "random_model_control"


def run_cell(family, task, model_kind, seed, partitions, code_hash):
    cell_id = f"{family}__{task}__{model_kind}__seed{seed}"
    result_path = OUT / "cells" / f"{cell_id}.json"
    if result_path.exists() and json.loads(result_path.read_text(encoding="utf-8")).get("status") == "complete":
        print(f"skip_complete={cell_id}", flush=True)
        return
    set_seed(seed)
    device = torch.device("cuda")
    model = GraphClassifier(model_kind, partitions["fit"][0].x.shape[1]).to(device)
    calibration, test = partitions["calibration"], partitions["test"]
    predictor = {
        "calibration": evaluate(model, torch_geometric_loader(calibration, 128), device),
        "test": evaluate(model, torch_geometric_loader(test, 128), device),
    }
    explainers = {}
    for method in ["gradinput", "ig"]:
        batch_size = 128 if method == "gradinput" else 32
        cal_scores, cal_truths, cal_timing = gradient_scores(model, calibration, device, method, batch_size)
        test_scores, test_truths, test_timing = gradient_scores(model, test, device, method, batch_size)
        explainers[method] = {
            "calibration_timing": cal_timing,
            "test_timing": test_timing,
            "metrics": explanation_metrics(cal_scores, cal_truths, test_scores, test_truths, seed),
        }
    atomic_json(result_path, {
        "status": "complete",
        "surface": "final_random_model_control",
        "control": "untrained_network_at_matched_main_seed",
        "cell_id": cell_id,
        "family": family,
        "task": task,
        "model": model_kind,
        "seed": seed,
        "code_sha256": code_hash,
        "predictor": predictor,
        "explainers": explainers,
    })
    print(f"cell_complete={cell_id}", flush=True)


def main():
    code_hash = sha256(__file__)
    started = time.time()
    for family, tasks in [("bxaic", BXAIC_TASKS), ("google", GOOGLE_TASKS)]:
        for task in tasks:
            print(f"load_task={family}/{task}", flush=True)
            partitions = bxaic_partitions(ROOT / "data/raw/bxaic/data.csv", ROOT / "data/raw/bxaic/explanations.sdf", task) if family == "bxaic" else google_partitions(ROOT / "reference/graph-attribution/data", task)
            for model_kind in ["gin", "gcn"]:
                for seed in [42, 123, 2026]:
                    run_cell(family, task, model_kind, seed, partitions, code_hash)
    cells = sorted((OUT / "cells").glob("*.json"))
    atomic_json(OUT / "manifest.json", {
        "status": "complete",
        "surface": "final_random_model_control",
        "code_sha256": code_hash,
        "cell_count": len(cells),
        "seconds": time.time() - started,
        "cells": [str(path.relative_to(ROOT)) for path in cells],
    })


if __name__ == "__main__":
    main()

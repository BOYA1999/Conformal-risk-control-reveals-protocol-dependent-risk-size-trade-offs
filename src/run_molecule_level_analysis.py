import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from molxai_crc import top_fraction_set
from run_bxaic_probe import gradient_scores
from run_gradient_grid import (
    BXAIC_TASKS,
    GOOGLE_TASKS,
    GraphClassifier,
    atomic_json,
    bxaic_partitions,
    google_partitions,
    sha256,
)


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "artifacts" / "experiment" / "gradient_grid_main"
OUT = ROOT / "artifacts" / "analysis" / "phase_a_followup_20260809" / "molecule_level_ig_seed42"


def logits(model, graphs, device):
    rows = []
    with torch.no_grad():
        for batch in DataLoader(graphs, batch_size=128, shuffle=False):
            batch = batch.to(device)
            rows.append(model(batch.x, batch.edge_index, batch.batch).cpu().numpy())
    return np.concatenate(rows)


def masked_graph(data, indices):
    result = data.clone()
    result.x = data.x.clone()
    if indices:
        result.x[torch.tensor(sorted(indices), dtype=torch.long)] = 0
    return result


def random_set(cell_id, source_index, n_atoms, size):
    digest = hashlib.sha256(f"{cell_id}|{source_index}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    return set(rng.choice(n_atoms, size=size, replace=False).tolist())


def atom_size_group(n_atoms):
    return "small" if n_atoms <= 22 else "medium" if n_atoms <= 30 else "large"


def rationale_group(fraction):
    if fraction == 0:
        return "null"
    return "focal" if fraction <= 0.10 else "medium" if fraction <= 0.40 else "diffuse"


def atomic_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run_cell(family, task, model_kind, partitions, code_hash):
    cell_id = f"{family}__{task}__{model_kind}__seed42"
    table_path = OUT / "tables" / f"{cell_id}.csv"
    result_path = OUT / "cells" / f"{cell_id}.json"
    if table_path.exists() and result_path.exists():
        print(f"skip_complete={cell_id}", flush=True)
        return
    device = torch.device("cuda")
    checkpoint = MAIN / "checkpoints" / f"{cell_id}.pt"
    model = GraphClassifier(model_kind, partitions["fit"][0].x.shape[1]).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["state_dict"])
    model.eval()
    test = partitions["test"]
    started = time.perf_counter()
    scores, truths, timing = gradient_scores(model, test, device, "ig", 32)
    main_cell = json.loads((MAIN / "cells" / f"{cell_id}.json").read_text(encoding="utf-8"))
    fraction = main_cell["explainers"]["ig"]["metrics"]["alpha"]["0.10"]["crc"]["fraction"]
    selected_sets = [top_fraction_set(score, fraction) for score in scores]
    random_sets = [random_set(cell_id, int(data.source_index), data.num_nodes, len(selected)) for data, selected in zip(test, selected_sets)]
    original_logits = logits(model, test, device)
    selected_logits = logits(model, [masked_graph(data, selected) for data, selected in zip(test, selected_sets)], device)
    random_logits = logits(model, [masked_graph(data, selected) for data, selected in zip(test, random_sets)], device)

    rows = []
    for i, (data, truth, selected, random_selected) in enumerate(zip(test, truths, selected_sets, random_sets)):
        n_atoms, n_rationale = data.num_nodes, len(truth)
        intersection = len(selected & truth)
        union = len(selected | truth)
        target = int(data.y)
        drop = float(original_logits[i, target] - selected_logits[i, target])
        random_drop = float(original_logits[i, target] - random_logits[i, target])
        rows.append({
            "cell_id": cell_id,
            "family": family,
            "task": task,
            "model": model_kind,
            "seed": 42,
            "source_index": int(data.source_index),
            "label": target,
            "prediction_correct": int(original_logits[i].argmax()) == target,
            "n_atoms": n_atoms,
            "atom_size_group": atom_size_group(n_atoms),
            "n_rationale": n_rationale,
            "rationale_fraction": n_rationale / n_atoms,
            "rationale_group": rationale_group(n_rationale / n_atoms),
            "crc_fraction": fraction,
            "n_selected": len(selected),
            "selected_fraction": len(selected) / n_atoms,
            "miss_loss": None if not truth else 1 - intersection / n_rationale,
            "precision": None if not truth else intersection / len(selected),
            "iou": None if not truth else intersection / union,
            "true_logit_drop_selected": drop,
            "true_logit_drop_random": random_drop,
            "fidelity_advantage": drop - random_drop,
            "rationale_indices": ";".join(map(str, sorted(truth))),
            "selected_indices": ";".join(map(str, sorted(selected))),
            "random_indices": ";".join(map(str, sorted(random_selected))),
        })
    atomic_csv(table_path, rows)
    atomic_json(result_path, {
        "status": "complete",
        "surface": "final_molecule_level_followup",
        "cell_id": cell_id,
        "code_sha256": code_hash,
        "checkpoint_sha256": sha256(checkpoint),
        "crc_fraction": fraction,
        "molecules": len(rows),
        "rationale_molecules": sum(bool(truth) for truth in truths),
        "null_molecules": sum(not truth for truth in truths),
        "gradient_timing": timing,
        "total_seconds": time.perf_counter() - started,
        "table": str(table_path.relative_to(ROOT)),
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
                run_cell(family, task, model_kind, partitions, code_hash)
    cells = sorted((OUT / "cells").glob("*.json"))
    atomic_json(OUT / "manifest.json", {
        "status": "complete",
        "surface": "final_molecule_level_followup",
        "code_sha256": code_hash,
        "cell_count": len(cells),
        "seconds": time.time() - started,
        "cells": [str(path.relative_to(ROOT)) for path in cells],
    })


if __name__ == "__main__":
    main()

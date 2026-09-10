import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from audit_graph_attribution import stratified_partitions
from run_bxaic_probe import GINClassifier, crc_dev_diagnostic, gradient_scores, gnnexplainer_timing, hash_order, set_seed, train_model


def load_graphs(root, task):
    folder = Path(root) / task
    frame = pd.read_csv(folder / f"{task}_smiles.csv")
    train = np.load(folder / f"{task}_traintest_indices.npz")["train_index"].astype(int)
    labels = np.load(folder / "y_true.npz")["y"].reshape(-1).astype(int)
    graph_rows = np.load(folder / "x_true.npz", allow_pickle=True)["datadict_list"].reshape(-1)
    rationale_rows = np.load(folder / "true_raw_attribution_datadicts.npz", allow_pickle=True)["datadict_list"].reshape(-1)
    fit_idx, dev_idx, _ = stratified_partitions(train, labels, frame["mol_id"].astype(str).to_numpy())

    def convert(index):
        graph, rationale = graph_rows[index], rationale_rows[index]
        edge_index = torch.tensor(np.stack([graph["senders"], graph["receivers"]]), dtype=torch.long)
        mask = np.asarray(rationale["nodes"])
        if mask.ndim == 1:
            mask = mask[:, None]
        return Data(
            x=torch.tensor(graph["nodes"], dtype=torch.float32),
            edge_index=edge_index,
            y=torch.tensor(labels[index], dtype=torch.long),
            rationale_mask=torch.tensor(mask[:, -1], dtype=torch.bool),
            source_index=torch.tensor(index, dtype=torch.long),
        )

    return [convert(i) for i in fit_idx], [convert(i) for i in dev_idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", default="benzene")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda")
    started = time.perf_counter()
    fit, dev = load_graphs(args.root, args.task)
    load_seconds = time.perf_counter() - started
    positive = sorted([d for d in dev if d.rationale_mask.any()], key=lambda d: hash_order(int(d.source_index)))
    model = GINClassifier(in_channels=fit[0].x.shape[1]).to(device)
    training = train_model(model, fit, dev, device, args.batch_size, args.epochs)
    grad_scores, truths, grad_timing = gradient_scores(model, positive, device, "gradinput", args.batch_size)
    ig_scores, ig_truths, ig_timing = gradient_scores(model, positive, device, "ig", max(16, args.batch_size // 4))
    if truths != ig_truths:
        raise RuntimeError("explainer rationale order mismatch")
    gnn_timing = gnnexplainer_timing(model, positive[:20], device)
    result = {
        "run_id": "graph_attribution_feasibility_20260809",
        "tier": "auxiliary/dev",
        "task": args.task,
        "seed": args.seed,
        "device": torch.cuda.get_device_name(0),
        "fit_molecules": len(fit),
        "dev_molecules": len(dev),
        "positive_dev_molecules": len(positive),
        "data_load_seconds": load_seconds,
        "training": training,
        "gradinput": {"timing": grad_timing, "diagnostic": crc_dev_diagnostic(grad_scores, truths)},
        "integrated_gradients": {"steps": 20, "timing": ig_timing, "diagnostic": crc_dev_diagnostic(ig_scores, truths)},
        "gnnexplainer": {"timing_only": gnn_timing},
    }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save({"state_dict": model.state_dict(), "task": args.task, "seed": args.seed}, args.checkpoint)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

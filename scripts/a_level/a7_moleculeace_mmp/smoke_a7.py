import json

import torch
import numpy as np
import pandas as pd

from attribute_and_crc import calibrate, loss_table, molecule_attributions, top_fraction_set
from gine_model import EDGE_DIM, NODE_DIM, GINERegressor, graph_from_smiles


def main():
    torch.manual_seed(20260904)
    row = pd.read_csv("mmp_pairs.csv", nrows=1).iloc[0]
    graph_a = graph_from_smiles(row.smiles_a)
    graph_b = graph_from_smiles(row.smiles_b)
    model = GINERegressor(hidden=32, layers=2, dropout=0.0).eval()
    with torch.no_grad():
        prediction_a = float(model(graph_a.x, graph_a.edge_index, graph_a.edge_attr, torch.zeros(graph_a.num_nodes, dtype=torch.long)))
        prediction_b = float(model(graph_b.x, graph_b.edge_index, graph_b.edge_attr, torch.zeros(graph_b.num_nodes, dtype=torch.long)))
    batched = molecule_attributions(model, row.smiles_a, torch.device("cpu"), 0.0, 1.0, 8)
    original_x = graph_a.x.detach()
    gradient_sum = torch.zeros_like(original_x)
    for scale in torch.linspace(1.0 / 8, 1.0, 8):
        interpolated = (scale * original_x).detach().requires_grad_(True)
        output = model(interpolated, graph_a.edge_index, graph_a.edge_attr, torch.zeros(graph_a.num_nodes, dtype=torch.long))
        gradient_sum += torch.autograd.grad(output.sum(), interpolated)[0].detach()
    sequential = np.abs((original_x * gradient_sum / 8).sum(dim=1).numpy())
    ig_error = float(np.max(np.abs(sequential - batched["integrated_gradients"])))
    tied = top_fraction_set(np.asarray([1.0, 0.5, 0.5, 0.0]), 0.5)
    losses = loss_table(
        [np.asarray([1.0, 0.5, 0.5, 0.0])] * 30,
        [{0, 2}] * 30,
    )
    crc_index, empirical, corrected = calibrate(losses)
    checks = {
        "node_dim": NODE_DIM == 17,
        "edge_dim": EDGE_DIM == 11,
        "real_pair_forward": np.isfinite(prediction_a - prediction_b),
        "score_lengths": len(batched["integrated_gradients"]) == graph_a.num_nodes and len(batched["atom_occlusion"]) == graph_a.num_nodes,
        "batched_sequential_ig_equivalence": ig_error <= 1e-6,
        "include_all_ties": tied == {0, 1, 2},
        "loss_monotone": bool((np.diff(losses, axis=1) <= 1e-12).all()),
        "crc_finite_correction": corrected <= 0.1 and empirical <= corrected,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    output = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "device": "cpu",
        "molecules": 2,
        "atoms_a": graph_a.num_nodes,
        "atoms_b": graph_b.num_nodes,
        "directed_edges_a": graph_a.edge_index.shape[1],
        "directed_edges_b": graph_b.edge_index.shape[1],
        "ig_batched_sequential_max_absolute_error": ig_error,
        "tie_selected_indices": sorted(tied),
        "crc_fraction": float(crc_index / 100),
        "crc_empirical_risk": empirical,
        "crc_corrected_risk": corrected,
        "checks": checks,
    }
    print(json.dumps(output, indent=2))
    if output["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

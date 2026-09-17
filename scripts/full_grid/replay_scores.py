import os
import json
import sys
from pathlib import Path

import torch
import numpy as np

CODE = Path(__file__).resolve().parent
PACKAGE = CODE.parents[1]
WORK = Path(os.environ.get('MOLXAI_WORK_ROOT', '../molxai-work')).resolve()
HERE = WORK / 'full_grid'
SCIENCE = WORK
CONTRACT = PACKAGE / 'contracts/full_grid/run_contract.json'
sys.path.insert(0, str(PACKAGE / 'src'))
from run_gradient_grid import BXAIC_TASKS, GOOGLE_TASKS, GraphClassifier, bxaic_partitions, google_partitions


def main():
    torch.set_num_threads(1)
    rows, source_checks = [], []
    for family, tasks in [("bxaic", BXAIC_TASKS), ("google", GOOGLE_TASKS)]:
        for task in tasks:
            if family == "bxaic":
                parts = bxaic_partitions(SCIENCE / "data/raw/bxaic/data.csv", SCIENCE / "data/raw/bxaic/explanations.sdf", task)
            else:
                parts = google_partitions(SCIENCE / "reference/graph-attribution/data", task)
            by_id = {int(graph.source_index): graph for graph in parts["test"]}
            for kind in ["gin", "gcn"]:
                for seed in [42, 123, 2026]:
                    cell_id = f"{family}__{task}__{kind}__seed{seed}"
                    model = GraphClassifier(kind, parts["test"][0].x.shape[1]).cuda().eval()
                    checkpoint = torch.load(SCIENCE / "artifacts/experiment/gradient_grid_main/checkpoints" / f"{cell_id}.pt", map_location="cuda", weights_only=True)
                    model.load_state_dict(checkpoint["state_dict"])
                    with np.load(HERE / "scores" / f"{cell_id}.npz", allow_pickle=False) as z:
                        for split in ["calibration", "test"]:
                            full = parts[split]
                            selected = [g for g in full if bool(g.rationale_mask.any())]
                            assert np.array_equal(z[f"{split}__all_source_ids"], [int(g.source_index) for g in full])
                            assert np.array_equal(z[f"{split}__source_ids"], [int(g.source_index) for g in selected])
                            assert np.array_equal(z[f"{split}__labels"], [int(g.y) for g in selected])
                            assert np.array_equal(np.diff(z[f"{split}__offsets"]), [g.num_nodes for g in selected])
                            assert np.array_equal(z[f"{split}__rationale"], np.concatenate([g.rationale_mask.numpy() for g in selected]))
                            source_checks.append({"cell_id": cell_id, "split": split, "all_ids": len(full), "eligible_ids": len(selected), "labels_masks_atom_counts_exact": True})
                        ids, ptr = z["test__source_ids"], z["test__offsets"]
                        for index in sorted({0, len(ids) // 2, len(ids) - 1}):
                            graph = by_id[int(ids[index])].clone().cuda()
                            target = int(graph.y)
                            batch_index = torch.zeros(graph.num_nodes, dtype=torch.long, device="cuda")
                            x = graph.x.detach().clone().requires_grad_(True)
                            logit = model(x, graph.edge_index, batch_index)[0, target]
                            saliency = torch.autograd.grad(logit, x)[0].abs().sum(1).cpu().numpy()
                            with torch.no_grad():
                                base = model(graph.x, graph.edge_index, batch_index)[0, target]
                                values = []
                                for atom in range(graph.num_nodes):
                                    replaced = graph.x.clone()
                                    replaced[atom] = 0
                                    value = model(replaced, graph.edge_index, batch_index)[0, target]
                                    values.append(float((base - value).cpu()))
                            actual = {"saliency": saliency, "atom_occlusion": np.asarray(values)}
                            for method, scores in actual.items():
                                cached = z[f"test__{method}"][ptr[index]:ptr[index + 1]]
                                passed = bool(np.allclose(scores, cached, atol=1e-4, rtol=1e-5))
                                rows.append({"cell_id": cell_id, "source_id": int(ids[index]), "method": method, "atoms": graph.num_nodes, "max_abs_difference": float(np.max(np.abs(scores - cached))), "pass": passed})
                    print(f"score_replay {cell_id}", flush=True)
            del parts, by_id, model
    passed = all(row["pass"] for row in rows)
    result = {"status": "PASS" if passed else "FAIL", "cell_count": len({row["cell_id"] for row in rows}), "molecule_count": len(rows) // 2, "method_molecule_checks": len(rows), "selection": "first, middle and last eligible test source IDs per frozen cell", "atol": 1e-4, "rtol": 1e-5, "max_abs_difference": max(row["max_abs_difference"] for row in rows), "rows": rows, "full_source_correspondence": source_checks, "boundary": "all original source IDs, labels, masks and atom counts checked; sampled numerical solo-inference score consistency is not byte-identical set replay or independent chemical validation"}
    (HERE / "score_replay_validation.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    assert passed, "score replay failed; keep all diagnostic rows"


if __name__ == "__main__":
    main()

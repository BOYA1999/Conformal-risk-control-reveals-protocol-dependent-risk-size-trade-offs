import os
import hashlib
import itertools
import json
import math
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, roc_auc_score
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCN, GIN, global_add_pool

from audit_liver_inputs import read_masks

CODE = Path(__file__).resolve().parent
PACKAGE = CODE.parents[1]
WORK = Path(os.environ.get('MOLXAI_WORK_ROOT', '../molxai-work')).resolve()
HERE = WORK / 'independent_reference'
OUT = HERE / "results"
torch.set_num_threads(2)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    rows = []
    choices = [list("BCNOF") + ["Si", "P", "S", "Cl", "Br", "I"], list(range(6)), [-2, -1, 0, 1, 2], list(range(5))]
    for atom in mol.GetAtoms():
        values = [atom.GetSymbol(), atom.GetDegree(), atom.GetFormalCharge(), atom.GetTotalNumHs()]
        row = [int(value == option) for value, options in zip(values, choices) for option in options + [None]]
        offset = 0
        for value, options in zip(values, choices):
            row[offset + len(options)] = int(value not in options)
            offset += len(options) + 1
        rows.append(row + [int(atom.GetIsAromatic()), int(atom.IsInRing())])
    edges = [(a, b) for bond in mol.GetBonds() for a, b in [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()), (bond.GetEndAtomIdx(), bond.GetBeginAtomIdx())]]
    return Data(x=torch.tensor(rows, dtype=torch.float32), edge_index=torch.tensor(edges, dtype=torch.long).T.contiguous() if edges else torch.empty((2, 0), dtype=torch.long))


class Predictor(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.encoder = (GIN if kind == "gin" else GCN)(33, 32, num_layers=3, out_channels=32, **({"norm": "batch_norm"} if kind == "gin" else {}))
        self.head = nn.Linear(32, 3)

    def forward(self, x, edge, batch):
        return self.head(global_add_pool(self.encoder(x, edge), batch))


def score_one(model, item, target, method):
    batch = torch.zeros(item.num_nodes, dtype=torch.long)
    if method == "atom_occlusion":
        with torch.no_grad():
            original = model(item.x, item.edge_index, batch)[0, target]
            values = []
            for i in range(item.num_nodes):
                masked = item.x.clone()
                masked[i] = 0
                values.append(float(original - model(masked, item.edge_index, batch)[0, target]))
        return np.asarray(values)
    total = torch.zeros_like(item.x)
    steps = 20 if method == "ig" else 1
    for step in range(1, steps + 1):
        x = (item.x * step / steps).detach().requires_grad_()
        total += torch.autograd.grad(model(x, item.edge_index, batch)[0, target], x)[0]
    value = total.abs().sum(1) if method == "saliency" else (item.x * total / steps).sum(1)
    return value.detach().numpy()


def unpack(path, ids, references):
    with np.load(path, allow_pickle=False) as data:
        assert data["source_ids"].tolist() == ids
        pointers = data["ptr"]
        assert pointers[0] == 0 and pointers[-1] == len(data["scores"]) and (np.diff(pointers) > 0).all()
        scores = [data["scores"][a:b].copy() for a, b in zip(pointers[:-1], pointers[1:])]
        masks = [data["masks"][a:b].copy() for a, b in zip(pointers[:-1], pointers[1:])]
    for i, values, mask in zip(ids, scores, masks):
        assert np.isfinite(values).all() and np.array_equal(mask, references[i]["node_atts"].astype(bool)) and mask.any()
    return scores, masks


def table(scores, masks, inclusive):
    records = []
    for values, truth in zip(scores, masks):
        n = len(values)
        order = sorted(range(n), key=lambda i: (-float(values[i]), i))
        rows, previous = [], set()
        for j in range(101):
            k = math.ceil((j * .01) * n)
            selected = set(order[:k])
            if inclusive and k:
                cutoff = values[order[k - 1]]
                selected = {i for i in range(n) if values[i] >= cutoff}
            assert previous <= selected
            previous = selected
            size, hits, r = len(selected), sum(bool(truth[i]) for i in selected), int(truth.sum())
            rows.append([1 - hits / r, size / n, hits / size if size else 0., hits / (size + r - hits), (size - k) / n])
        records.append(rows)
    return np.asarray(records)


def main():
    source = pd.read_csv(HERE / "data/raw/Liver.csv")
    refs = read_masks()
    manifest = pd.read_csv(OUT / "split_manifest.csv")
    contract = json.loads((OUT / "run_manifest.json").read_text())
    assert len(manifest) == 552 and manifest.source_index.is_unique
    for key, path in [("code_sha256", CODE / "run_liver.py"), ("adapter_sha256", CODE / "prepare_liver.py"), ("safe_reader_sha256", CODE / "audit_liver_inputs.py"), ("plan_sha256", PACKAGE / "contracts/independent_reference/run_contract.json"), ("metric_code_sha256", CODE.parent / "full_grid/run_full_grid.py")]:
        assert sha(path) == contract[key], f"Hash mismatch {key}"
    assert sha(OUT / "split_manifest.csv") == contract["data"]["split_sha256"]
    for filename, value in contract["data"]["source_hashes"].items():
        assert sha(HERE / "data/raw" / filename) == value
    identities, largest = {}, {}
    for row in manifest.itertuples():
        mol = Chem.MolFromSmiles(source.SMILES[row.source_index])
        key = Chem.MolToSmiles(mol, isomericSmiles=False)
        fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        top = min(fragments, key=lambda m: (-m.GetNumHeavyAtoms(), Chem.MolToSmiles(m, isomericSmiles=False)))
        large = Chem.MolToSmiles(top, isomericSmiles=False)
        assert key not in identities and large not in largest
        identities[key], largest[large] = row.source_index, row.source_index
        assert hashlib.sha256(key.encode()).hexdigest() == row.identity_sha256
        assert source.label[row.source_index] == row.label == refs[row.source_index]["label"] + 1
        assert source.SPLIT[row.source_index] == ("test" if row.split == "test" else "train")
    for label in [0, 1, 2]:
        group = manifest[(manifest.label == label) & (manifest.split != "test")]
        order = sorted(group.itertuples(), key=lambda r: hashlib.sha256(("molxai_crc_liver_20260912|" + next(k for k, i in identities.items() if i == r.source_index)).encode()).hexdigest())
        a, b = round(.20 * len(order)), round(.15 * len(order))
        assert all(r.split == ("calibration" if j < a else "dev" if j < a + b else "fit") for j, r in enumerate(order))
    expected = pd.read_csv(OUT / "cell_metrics.csv")
    molecule_expected = pd.read_csv(OUT / "molecule_metrics.csv")
    assert len(expected) == 288 and len(molecule_expected) == 2976
    columns = ["risk", "mean_atom_fraction", "precision", "iou", "tie_inflation"]
    molecules, predictors, differences, epoch_counts, cache_hashes = [], [], [], {}, {}
    ids_by_split = {s: manifest[(manifest.split == s) & (manifest.reference_atoms > 0)].source_index.tolist() for s in ["calibration", "test"]}
    graphs = {i: graph(source.SMILES[i]) for i in manifest.source_index}
    for kind, seed in itertools.product(["gin", "gcn"], [42, 123, 2026]):
        cell = f"liver__{kind}__seed{seed}"
        record = json.loads((OUT / "cells" / f"{cell}.json").read_text())
        assert record["run_code_sha256"] == contract["code_sha256"] and record["checkpoint_sha256"] == sha(OUT / f"{cell}.pt")
        history = record["training"]["history"]
        assert len(history) == record["training"]["epochs"] >= 20
        assert [r["epoch"] for r in history] == list(range(1, len(history) + 1))
        best, chosen = -1., None
        for row in history:
            if row["macro_ovr_auroc"] > best + 1e-5:
                best, chosen = row["macro_ovr_auroc"], row["epoch"]
        assert chosen == record["training"]["best_epoch"]
        epoch_counts[cell] = {"epochs": len(history), "best_epoch": chosen}
        checkpoint = torch.load(OUT / f"{cell}.pt", map_location="cpu", weights_only=True)
        model = Predictor(kind)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        for split in ["dev", "calibration", "test"]:
            data = np.load(OUT / f"{cell}_{split}_predictions.npz", allow_pickle=False)
            ids = manifest[manifest.split == split].source_index.tolist()
            assert data["source_ids"].tolist() == ids and np.array_equal(data["labels"], source.label[ids])
            batch = Batch.from_data_list([graphs[i] for i in ids])
            with torch.no_grad():
                replay = model(batch.x, batch.edge_index, batch.batch).softmax(1).numpy()
            assert np.allclose(replay, data["probabilities"], atol=2e-6, rtol=2e-5)
            y, p = data["labels"], data["probabilities"]
            metrics = {"accuracy": accuracy_score(y, p.argmax(1)), "balanced_accuracy": balanced_accuracy_score(y, p.argmax(1)), "macro_ovr_auroc": roc_auc_score(y, p, multi_class="ovr", average="macro"), "macro_ap": average_precision_score(np.eye(3)[y], p, average="macro"), "majority_accuracy": np.bincount(y).max() / len(y)}
            assert all(np.isclose(v, record["predictor"][split][k], atol=1e-12) for k, v in metrics.items())
            if split == "test":
                predictors.append({"cell_id": cell, "model": kind, "seed": seed, "test_molecules": len(y), **metrics, **epoch_counts[cell], "seconds": record["seconds"]})
        for target, method in itertools.product(["observed_class", "hepatotoxic_class"], ["gradinput", "ig", "saliency", "atom_occlusion"]):
            cached = {}
            for split, ids in ids_by_split.items():
                path = OUT / f"{cell}_{target}_{method}_{split}_scores.npz"
                values, masks = unpack(path, ids, refs)
                cached[split] = values, masks
                cache_hashes[path.name] = sha(path)
                for j in [0, len(ids) - 1]:
                    i = ids[j]
                    label = int(source.label[i]) if target == "observed_class" else 2
                    actual = score_one(model, graphs[i], label, method)
                    differences.append(float(np.max(np.abs(actual - values[j]))))
                    assert np.allclose(actual, values[j], atol=2e-5, rtol=1e-4), (cell, target, method, split, i, differences[-1])
            for policy in ["index_tiebreak_v1", "include_all_exact_ties_v2"]:
                cal = table(*cached["calibration"], policy.endswith("v2"))
                test = table(*cached["test"], policy.endswith("v2"))
                corrected = (cal[:, :, 0].sum(0) + 1) / (len(cal) + 1)
                for alpha in [.05, .1, .2]:
                    choices = np.flatnonzero(corrected <= alpha)
                    j = int(choices[0]) if len(choices) else 100
                    values = dict(zip(columns, test[:, j].mean(0)))
                    sizes = np.asarray([len(s) for s in cached["test"][0]])
                    counts = np.rint(test[:, j, 1] * sizes).astype(int)
                    refcounts = np.asarray([m.sum() for m in cached["test"][1]])
                    random = 1 - counts / sizes
                    oracle = 1 - np.minimum(counts, refcounts) / refcounts
                    values.update(selected_fraction=j * .01, corrected_calibration_risk=corrected[j], equal_size_random_risk=random.mean(), reference_first_schedule_risk=oracle.mean())
                    for fraction in [20, 50]:
                        values.update({f"fixed_{fraction}_{k}": v for k, v in zip(columns[:4], test[:, fraction].mean(0)[:4])})
                    wanted = expected[(expected.cell_id == cell) & (expected.target == target) & (expected.method == method) & (expected.policy == policy) & (expected.alpha == alpha)].iloc[0]
                    assert all(np.isclose(v, wanted[k], atol=1e-12) for k, v in values.items()), (cell, target, method, policy, alpha)
                    assert bool(wanted.fallback_full) == (len(choices) == 0) and wanted.n_calibration == 33 and wanted.n_test == 31
                    if alpha == .1:
                        for z, i in enumerate(ids_by_split["test"]):
                            molecules.append({"cell_id": cell, "model": kind, "seed": seed, "target": target, "method": method, "policy": policy, "source_index": i, "label": int(source.label[i]), **dict(zip(columns, test[z, j])), "equal_size_random_risk": random[z], "risk_minus_random": test[z, j, 0] - random[z], "selected_fraction": j * .01})
    replay = pd.DataFrame(molecules)
    join = replay.merge(molecule_expected, on=["cell_id", "target", "method", "policy", "source_index"], suffixes=("_replay", "_original"), validate="one_to_one")
    assert len(join) == 2976
    for key in columns + ["equal_size_random_risk"]:
        assert np.allclose(join[key + "_replay"], join[key + "_original"], atol=1e-12)
    unique_ids = sorted(replay.source_index.unique())
    rng = np.random.default_rng(20260912)
    draws = rng.integers(0, len(unique_ids), size=(2000, len(unique_ids)))
    summaries, strata = [], []
    for keys, group in replay.groupby(["target", "policy", "method"]):
        row = dict(zip(["target", "policy", "method"], keys))
        row.update(alpha=.1, models=6, test_molecules=31, calibration_molecules=33, bootstrap_replicates=2000)
        means = group.groupby("source_index")[["risk", "mean_atom_fraction", "equal_size_random_risk", "risk_minus_random", "precision", "iou", "tie_inflation"]].mean().reindex(unique_ids)
        for key in means:
            values = means[key].to_numpy()
            interval = np.quantile(values[draws].mean(1), [.025, .975])
            row[key] = values.mean()
            row[key + "_ci_low"], row[key + "_ci_high"] = interval
        row["risk_pass_cells"] = int((group.groupby("cell_id").risk.mean() <= .1).sum())
        row["efficiency_pass_cells"] = int((group.groupby("cell_id").mean_atom_fraction.mean() < .8).sum())
        row["mean_selected_fraction"] = group.groupby("cell_id").selected_fraction.first().mean()
        summaries.append(row)
        for label in [0, 1, 2]:
            subset = means.loc[[i for i in unique_ids if source.label[i] == label]]
            strata.append({**dict(zip(["target", "policy", "method"], keys)), "label": label, "test_molecules": len(subset), "calibration_rule": "shared_all_nonempty_references", **subset.mean().to_dict()})
    pd.DataFrame(summaries).to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(predictors).to_csv(OUT / "predictor_model_table.csv", index=False)
    pd.DataFrame(strata).to_csv(OUT / "class_strata.csv", index=False)
    report = {"status": "PASS", "cells": 6, "all_metric_rows_replayed": len(expected), "molecule_rows_replayed": len(join), "cached_score_arrays_checked": len(cache_hashes), "independent_checkpoint_score_checks": len(differences), "maximum_score_absolute_difference": max(differences), "training_epochs": epoch_counts, "unique_test_reference_molecules": 31, "bootstrap": "2000 shared molecule draws; seed20260912; average six fixed models per molecule before resampling; percentile95%; conditional on calibration/checkpoints; no subgroup recalibration or multiple-comparison-adjusted significance claim", "source_reference_boundary": "source-provided binary alert masks;48/587 differ from independent12-SMARTS union; operational reference fidelity is verified, exact SMARTS reconstruction and experimental causal truth are not established", "cache_sha256": cache_hashes, "validator_sha256": sha(__file__)}
    (OUT / "independent_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "cache_sha256"}, indent=2))
    print(pd.DataFrame(summaries)[["target", "policy", "method", "risk", "mean_atom_fraction", "risk_minus_random", "risk_minus_random_ci_low", "risk_minus_random_ci_high"]].to_string(index=False))


if __name__ == "__main__":
    main()

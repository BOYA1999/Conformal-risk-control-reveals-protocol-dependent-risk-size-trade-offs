import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_geometric
import numpy as np
import pandas as pd
import sklearn
from rdkit import Chem, rdBase
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCN, GIN, global_add_pool


BXAIC_TASKS = ["B", "P", "X", "indole", "PAINS", "rings-count", "rings-max"]
GOOGLE_TASKS = ["benzene", "logic7", "logic8", "logic10"]
MODELS = ["gin", "gcn"]
SEEDS = [42, 123, 2026]
SYMBOLS = ["C", "N", "O", "F", "Cl", "Br", "P", "S", "B", "I", "Unk"]
PROPS = {"B": "B", "P": "P", "X": "X", "indole": "indole", "PAINS": "pains", "rings-count": "rings", "rings-max": "largest_rings"}
SOURCE_HASHES = {
    "src/run_gradient_grid.py": "5CE5AFA8E59CA6F7F8021B4D55FCCFE9A5DFA383349AE404BD196F195A5EB5E4",
    "src/run_bxaic_probe.py": "DD7ED39D8D2F9F612343381C2C27A3EA25E5FA1C379450C75A13230682340F14",
    "src/audit_graph_attribution.py": "F12F3F1D78DAA6D90AC7D1DDFC67D77A5DC94616A77922DE15F786EF17D7E67F",
}


class GraphClassifier(nn.Module):
    def __init__(self, kind, in_channels, hidden=32, layers=3):
        super().__init__()
        encoder = GIN if kind == "gin" else GCN
        kwargs = {"norm": "batch_norm"} if kind == "gin" else {}
        self.encoder = encoder(in_channels, hidden, num_layers=layers, out_channels=32, **kwargs)
        self.head = nn.Linear(32, 2)

    def forward(self, x, edge_index, batch):
        return self.head(global_add_pool(self.encoder(x, edge_index), batch))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean_json(payload), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path, frame):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def expected_cell_ids():
    rows = []
    for family, tasks in [("bxaic", BXAIC_TASKS), ("google", GOOGLE_TASKS)]:
        for task in tasks:
            for model in MODELS:
                for seed in SEEDS:
                    rows.append(f"{family}__{task}__{model}__seed{seed}")
    return rows


def validate_sources(root):
    observed = {name: sha256(root / name) for name in SOURCE_HASHES}
    mismatch = {name: {"expected": SOURCE_HASHES[name], "observed": value} for name, value in observed.items() if value != SOURCE_HASHES[name]}
    if mismatch:
        raise RuntimeError(f"Frozen source mismatch: {mismatch}")
    checkpoint_dir = root / "artifacts/experiment/gradient_grid_main/checkpoints"
    observed_ids = sorted(p.stem for p in checkpoint_dir.glob("*.pt"))
    expected_ids = sorted(expected_cell_ids())
    if observed_ids != expected_ids:
        raise RuntimeError(f"Checkpoint inventory mismatch: expected {len(expected_ids)}, observed {len(observed_ids)}")
    return observed


def build_input_manifest(root, contract):
    files = [
        contract,
        root / "artifacts/experiment/gradient_grid_main/run_contract.json",
        root / "artifacts/experiment/gradient_grid_main/gradient_grid_manifest.json",
        root / "data/raw/bxaic/data.csv",
        root / "data/raw/bxaic/explanations.sdf",
    ]
    files.extend(root / name for name in SOURCE_HASHES)
    files.extend(sorted((root / "artifacts/experiment/gradient_grid_main/checkpoints").glob("*.pt")))
    for task in GOOGLE_TASKS:
        folder = root / "reference/graph-attribution/data" / task
        files.extend([
            folder / f"{task}_smiles.csv",
            folder / f"{task}_traintest_indices.npz",
            folder / "y_true.npz",
            folder / "x_true.npz",
            folder / "true_raw_attribution_datadicts.npz",
        ])
    return {
        "algorithm": "SHA-256",
        "file_count": len(files),
        "files": [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files],
    }


def bxaic_test_graphs(root, tasks):
    frame = pd.read_csv(root / "data/raw/bxaic/data.csv")
    wanted = set(frame.index[frame["split_0"] == "test"].tolist())
    wanted.discard(997)
    result = {task: [] for task in tasks}
    supplier = Chem.SDMolSupplier(str(root / "data/raw/bxaic/explanations.sdf"), removeHs=False, sanitize=False)
    for index, mol in enumerate(supplier):
        if index not in wanted:
            continue
        if mol is None:
            raise RuntimeError(f"B-XAIC SDF molecule {index} could not be loaded")
        atom_ids = [SYMBOLS.index(atom.GetSymbol()) if atom.GetSymbol() in SYMBOLS else 10 for atom in mol.GetAtoms()]
        x = F.one_hot(torch.tensor(atom_ids), len(SYMBOLS)).float()
        edges = []
        for bond in mol.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edges.extend([(a, b), (b, a)])
        edge_index = torch.tensor(edges, dtype=torch.long).T.contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
        for task in tasks:
            raw = mol.GetProp(PROPS[task]).strip() if mol.HasProp(PROPS[task]) else ""
            mask = torch.zeros(len(atom_ids), dtype=torch.bool)
            if raw:
                mask[torch.tensor(sorted({int(v) for v in raw.split(",")}), dtype=torch.long)] = True
            result[task].append(Data(
                x=x,
                edge_index=edge_index,
                y=torch.tensor(int(frame.at[index, task]), dtype=torch.long),
                mask_reference=mask,
                source_index=torch.tensor(index, dtype=torch.long),
            ))
    for task in tasks:
        if len(result[task]) != len(wanted):
            raise RuntimeError(f"B-XAIC {task}: expected {len(wanted)} test graphs, loaded {len(result[task])}")
    return result


def jaccard(a, b):
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def google_test_graphs(root, task):
    folder = root / "reference/graph-attribution/data" / task
    frame = pd.read_csv(folder / f"{task}_smiles.csv")
    official = np.load(folder / f"{task}_traintest_indices.npz")
    test = official["test_index"].astype(int)
    labels = np.load(folder / "y_true.npz")["y"].reshape(-1).astype(int)
    graphs = np.load(folder / "x_true.npz", allow_pickle=True)["datadict_list"].reshape(-1)
    rationales = np.load(folder / "true_raw_attribution_datadicts.npz", allow_pickle=True)["datadict_list"].reshape(-1)
    data_rows, mask_rows, channel_rows, channel_graphs, base_positions = [], [], [], [], []
    for base_position, index in enumerate(test):
        graph = graphs[index]
        channels = np.asarray(rationales[index]["nodes"])
        if channels.ndim == 1:
            channels = channels[:, None]
        channels = channels.astype(bool)
        last = channels[:, -1]
        union = channels.any(axis=1)
        intersection = channels.all(axis=1)
        x = torch.tensor(graph["nodes"], dtype=torch.float32)
        edge_index = torch.tensor(np.stack([graph["senders"], graph["receivers"]]), dtype=torch.long)
        data_rows.append(Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor(labels[index], dtype=torch.long),
            mask_last=torch.tensor(last, dtype=torch.bool),
            mask_union=torch.tensor(union, dtype=torch.bool),
            mask_intersection=torch.tensor(intersection, dtype=torch.bool),
            source_index=torch.tensor(index, dtype=torch.long),
        ))
        channel_sizes = channels.sum(axis=0)
        if channels.shape[1] == 1:
            pairwise_mean = 1.0
        else:
            inter = channels.T.astype(int) @ channels.astype(int)
            union_counts = channel_sizes[:, None] + channel_sizes[None, :] - inter
            pairwise = np.divide(inter, union_counts, out=np.ones_like(inter, dtype=float), where=union_counts != 0)
            pairwise_mean = float(pairwise[np.triu_indices(channels.shape[1], 1)].mean())
        source_id = str(frame.iloc[index]["mol_id"]) if "mol_id" in frame.columns else str(index)
        mask_rows.append({
            "task": task,
            "source_index": int(index),
            "source_id": source_id,
            "label": int(labels[index]),
            "n_atoms": int(len(last)),
            "n_channels": int(channels.shape[1]),
            "last_atom_count": int(last.sum()),
            "union_atom_count": int(union.sum()),
            "intersection_atom_count": int(intersection.sum()),
            "last_atom_fraction": float(last.mean()),
            "union_atom_fraction": float(union.mean()),
            "intersection_atom_fraction": float(intersection.mean()),
            "mean_channel_atom_count": float(channel_sizes.mean()),
            "min_channel_atom_count": int(channel_sizes.min()),
            "max_channel_atom_count": int(channel_sizes.max()),
            "last_union_jaccard": jaccard(last, union),
            "last_intersection_jaccard": jaccard(last, intersection),
            "pairwise_channel_jaccard_mean": pairwise_mean,
        })
        for channel_index in range(channels.shape[1]):
            channel = channels[:, channel_index]
            channel_rows.append({
                "task": task,
                "source_index": int(index),
                "source_id": source_id,
                "label": int(labels[index]),
                "channel_index": int(channel_index),
                "n_channels": int(channels.shape[1]),
                "is_historical_last_channel": bool(channel_index == channels.shape[1] - 1),
                "n_atoms": int(len(channel)),
                "channel_atom_count": int(channel.sum()),
                "channel_atom_fraction": float(channel.mean()),
                "jaccard_with_historical_last": jaccard(channel, last),
                "jaccard_with_union": jaccard(channel, union),
                "jaccard_with_intersection": jaccard(channel, intersection),
            })
            channel_graphs.append(Data(
                x=x,
                edge_index=edge_index,
                y=torch.tensor(labels[index], dtype=torch.long),
                mask_channel=torch.tensor(channel, dtype=torch.bool),
                source_index=torch.tensor(index, dtype=torch.long),
            ))
            base_positions.append(base_position)
    return data_rows, mask_rows, channel_rows, channel_graphs, np.asarray(base_positions, dtype=int)


@torch.inference_mode()
def predict(model, graphs, device, batch_size, condition="intact", mask_attr=None):
    model.eval()
    outputs = []
    for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
        batch = batch.to(device)
        x = batch.x
        if condition != "intact":
            mask = getattr(batch, mask_attr).bool()
            keep = mask if condition == "rationale_only" else ~mask
            x = x * keep.unsqueeze(-1).to(x.dtype)
        outputs.append(model(x, batch.edge_index, batch.batch).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def probabilities(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def classification_metrics(labels, logits):
    probs = probabilities(logits)[:, 1]
    preds = logits.argmax(axis=1)
    return {
        "auroc": float(roc_auc_score(labels, probs)),
        "auprc": float(average_precision_score(labels, probs)),
        "accuracy": float(accuracy_score(labels, preds)),
        "positive_probability_mean": float(probs.mean()),
        "predicted_positive_fraction": float((preds == 1).mean()),
    }


def mean_or_nan(values):
    return float(np.mean(values)) if len(values) else np.nan


def paired_metrics(labels, masked_logits, intact_logits, active_masks):
    labels = np.asarray(labels, dtype=int)
    rows = np.arange(len(labels))
    masked_probs = probabilities(masked_logits)
    intact_probs = probabilities(intact_logits)
    logit_delta = masked_logits[rows, labels] - intact_logits[rows, labels]
    probability_delta = masked_probs[rows, labels] - intact_probs[rows, labels]
    nonempty = np.asarray([bool(mask.any()) for mask in active_masks])
    return {
        "mean_true_class_logit_delta": float(logit_delta.mean()),
        "median_true_class_logit_delta": float(np.median(logit_delta)),
        "mean_true_class_probability_delta": float(probability_delta.mean()),
        "median_true_class_probability_delta": float(np.median(probability_delta)),
        "prediction_flip_rate": float((masked_logits.argmax(axis=1) != intact_logits.argmax(axis=1)).mean()),
        "mean_true_class_probability_delta_mask_nonempty": mean_or_nan(probability_delta[nonempty]),
        "mean_true_class_probability_delta_mask_empty": mean_or_nan(probability_delta[~nonempty]),
    }


def active_masks(graphs, mask_attr):
    return [getattr(graph, mask_attr).numpy().astype(bool) for graph in graphs]


def cell_rows(root, family, task, model_kind, seed, graphs, mask_definitions, device, batch_size):
    cell_id = f"{family}__{task}__{model_kind}__seed{seed}"
    checkpoint_path = root / "artifacts/experiment/gradient_grid_main/checkpoints" / f"{cell_id}.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_meta = {"family": family, "task": task, "model": model_kind, "seed": seed}
    observed_meta = {key: checkpoint[key] for key in expected_meta}
    if observed_meta != expected_meta:
        raise RuntimeError(f"Checkpoint metadata mismatch for {cell_id}: {observed_meta}")
    model = GraphClassifier(model_kind, graphs[0].x.shape[1]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    labels = np.asarray([int(graph.y) for graph in graphs], dtype=int)
    intact_logits = predict(model, graphs, device, batch_size)
    historical = json.loads((root / "artifacts/experiment/gradient_grid_main/cells" / f"{cell_id}.json").read_text(encoding="utf-8"))["predictor"]["test"]
    intact_metrics = classification_metrics(labels, intact_logits)
    rows = [{
        "cell_id": cell_id,
        "family": family,
        "task": task,
        "model": model_kind,
        "seed": seed,
        "mask_definition": "none",
        "condition": "intact",
        "n_test": len(labels),
        "n_positive": int(labels.sum()),
        "n_reference_mask_nonempty": np.nan,
        "mean_reference_mask_fraction": np.nan,
        "mean_kept_node_fraction": 1.0,
        **intact_metrics,
        "mean_true_class_logit_delta": 0.0,
        "median_true_class_logit_delta": 0.0,
        "mean_true_class_probability_delta": 0.0,
        "median_true_class_probability_delta": 0.0,
        "prediction_flip_rate": 0.0,
        "mean_true_class_probability_delta_mask_nonempty": np.nan,
        "mean_true_class_probability_delta_mask_empty": np.nan,
        "historical_auroc": float(historical["auroc"]),
        "historical_auprc": float(historical["auprc"]),
        "historical_auroc_abs_diff": abs(intact_metrics["auroc"] - float(historical["auroc"])),
        "historical_auprc_abs_diff": abs(intact_metrics["auprc"] - float(historical["auprc"])),
    }]
    for definition, attr in mask_definitions:
        masks = active_masks(graphs, attr)
        fractions = np.asarray([mask.mean() for mask in masks])
        for condition in ["rationale_only", "complement_only"]:
            logits = predict(model, graphs, device, batch_size, condition, attr)
            rows.append({
                "cell_id": cell_id,
                "family": family,
                "task": task,
                "model": model_kind,
                "seed": seed,
                "mask_definition": definition,
                "condition": condition,
                "n_test": len(labels),
                "n_positive": int(labels.sum()),
                "n_reference_mask_nonempty": int(sum(mask.any() for mask in masks)),
                "mean_reference_mask_fraction": float(fractions.mean()),
                "mean_kept_node_fraction": float(fractions.mean() if condition == "rationale_only" else (1 - fractions).mean()),
                **classification_metrics(labels, logits),
                **paired_metrics(labels, logits, intact_logits, masks),
                "historical_auroc": np.nan,
                "historical_auprc": np.nan,
                "historical_auroc_abs_diff": np.nan,
                "historical_auprc_abs_diff": np.nan,
            })
    return rows, model, intact_logits, labels


def channel_sensitivity_rows(cell_id, family, task, model_kind, seed, model, intact_logits, labels, channel_graphs, base_positions, device, batch_size):
    expanded_labels = labels[base_positions]
    intact_expanded = intact_logits[base_positions]
    masks = active_masks(channel_graphs, "mask_channel")
    rows = []
    for condition in ["rationale_only", "complement_only"]:
        logits = predict(model, channel_graphs, device, batch_size, condition, "mask_channel")
        variant = classification_metrics(expanded_labels, logits)
        aggregate_logits = np.zeros_like(intact_logits)
        counts = np.zeros(len(intact_logits), dtype=int)
        np.add.at(aggregate_logits, base_positions, logits)
        np.add.at(counts, base_positions, 1)
        aggregate_logits /= counts[:, None]
        paired_variant = paired_metrics(expanded_labels, logits, intact_expanded, masks)
        paired_molecule = paired_metrics(labels, aggregate_logits, intact_logits, [np.ones(1, dtype=bool)] * len(labels))
        aggregate = classification_metrics(labels, aggregate_logits)
        rows.append({
            "cell_id": cell_id,
            "family": family,
            "task": task,
            "model": model_kind,
            "seed": seed,
            "mask_definition": "each_independent_channel",
            "condition": condition,
            "n_molecules": len(labels),
            "n_channel_variants": len(expanded_labels),
            "variant_weighted_auroc": variant["auroc"],
            "variant_weighted_auprc": variant["auprc"],
            "variant_weighted_accuracy": variant["accuracy"],
            "molecule_mean_auroc": aggregate["auroc"],
            "molecule_mean_auprc": aggregate["auprc"],
            "molecule_mean_accuracy": aggregate["accuracy"],
            "variant_mean_true_class_logit_delta": paired_variant["mean_true_class_logit_delta"],
            "variant_median_true_class_logit_delta": paired_variant["median_true_class_logit_delta"],
            "variant_mean_true_class_probability_delta": paired_variant["mean_true_class_probability_delta"],
            "variant_median_true_class_probability_delta": paired_variant["median_true_class_probability_delta"],
            "variant_prediction_flip_rate": paired_variant["prediction_flip_rate"],
            "molecule_mean_true_class_logit_delta": paired_molecule["mean_true_class_logit_delta"],
            "molecule_mean_true_class_probability_delta": paired_molecule["mean_true_class_probability_delta"],
            "molecule_mean_prediction_flip_rate": paired_molecule["prediction_flip_rate"],
        })
    return rows


def records(frame, columns):
    return clean_json(frame[columns].to_dict(orient="records"))


def build_summary(cell_frame, mask_frame, channel_sensitivity, elapsed, device, mode):
    group_cols = ["family", "mask_definition", "condition"]
    value_cols = ["auroc", "auprc", "accuracy", "mean_true_class_probability_delta", "prediction_flip_rate", "mean_reference_mask_fraction", "mean_kept_node_fraction"]
    grouped = cell_frame.groupby(group_cols, dropna=False)[value_cols].mean().reset_index()
    paired = cell_frame[cell_frame["condition"] != "intact"].pivot_table(
        index=["cell_id", "family", "task", "model", "seed", "mask_definition"],
        columns="condition",
        values=["auroc", "auprc", "accuracy", "mean_true_class_probability_delta"],
    )
    paired.columns = [f"{metric}__{condition}" for metric, condition in paired.columns]
    paired = paired.reset_index()
    for metric in ["auroc", "auprc", "accuracy", "mean_true_class_probability_delta"]:
        paired[f"{metric}__rationale_minus_complement"] = paired[f"{metric}__rationale_only"] - paired[f"{metric}__complement_only"]
    contrast_cols = ["auroc__rationale_minus_complement", "auprc__rationale_minus_complement", "accuracy__rationale_minus_complement", "mean_true_class_probability_delta__rationale_minus_complement"]
    contrasts = paired.groupby(["family", "mask_definition"])[contrast_cols].agg(["mean", "median"]).reset_index()
    contrasts.columns = ["__".join([str(v) for v in col if str(v)]) if isinstance(col, tuple) else col for col in contrasts.columns]
    mask_summary = []
    if len(mask_frame):
        for task, group in mask_frame.groupby("task"):
            informative = group[group["union_atom_count"] > 0]
            mask_summary.append({
                "task": task,
                "n_test": len(group),
                "n_positive": int(group["label"].sum()),
                "n_union_nonempty": int((group["union_atom_count"] > 0).sum()),
                "n_multichannel": int((group["n_channels"] > 1).sum()),
                "total_channels": int(group["n_channels"].sum()),
                "maximum_channels": int(group["n_channels"].max()),
                "mean_last_atom_fraction_when_union_nonempty": mean_or_nan(informative["last_atom_fraction"].to_numpy()),
                "mean_union_atom_fraction_when_union_nonempty": mean_or_nan(informative["union_atom_fraction"].to_numpy()),
                "mean_intersection_atom_fraction_when_union_nonempty": mean_or_nan(informative["intersection_atom_fraction"].to_numpy()),
                "mean_last_union_jaccard_when_union_nonempty": mean_or_nan(informative["last_union_jaccard"].to_numpy()),
                "mean_last_intersection_jaccard_when_union_nonempty": mean_or_nan(informative["last_intersection_jaccard"].to_numpy()),
                "mean_pairwise_channel_jaccard_when_union_nonempty": mean_or_nan(informative["pairwise_channel_jaccard_mean"].to_numpy()),
            })
    return {
        "status": "complete_pending_validation",
        "mode": mode,
        "elapsed_seconds": elapsed,
        "device": str(device),
        "cell_count": int(cell_frame["cell_id"].nunique()),
        "cell_metric_row_count": len(cell_frame),
        "google_channel_sensitivity_row_count": len(channel_sensitivity),
        "cell_mean_metrics": records(grouped, group_cols + value_cols),
        "rationale_minus_complement_contrasts": clean_json(contrasts.to_dict(orient="records")),
        "google_mask_definition_summary": clean_json(mask_summary),
        "interpretation_boundary": "All masked conditions zero node features while retaining graph topology and edges. They are label-informed computational perturbation diagnostics, not chemically valid molecular edits, causal sufficiency/necessity tests, or mechanistic validation.",
    }


def build_validation(root, contract, cell_frame, mask_frame, channel_frame, channel_sensitivity, mode, source_hashes):
    expected_cells = 2 if mode == "smoke" else 66
    expected_cell_rows = 10 if mode == "smoke" else 294
    expected_channel_rows = 2 if mode == "smoke" else 48
    historical = cell_frame[cell_frame["condition"] == "intact"]
    required = ["auroc", "auprc", "accuracy", "mean_true_class_logit_delta", "mean_true_class_probability_delta", "prediction_flip_rate"]
    checks = {
        "contract_json_valid": bool(json.loads(contract.read_text(encoding="utf-8"))),
        "source_hashes_match_frozen_baseline": source_hashes == SOURCE_HASHES,
        "checkpoint_inventory_is_exactly_66": len(list((root / "artifacts/experiment/gradient_grid_main/checkpoints").glob("*.pt"))) == 66,
        "bxaic_test_count_matches_frozen_loader": bool((cell_frame.loc[cell_frame["family"] == "bxaic", "n_test"] == 4999).all()),
        "evaluated_cell_count_matches_mode": int(cell_frame["cell_id"].nunique()) == expected_cells,
        "cell_metric_row_count_matches_mode": len(cell_frame) == expected_cell_rows,
        "google_channel_sensitivity_row_count_matches_mode": len(channel_sensitivity) == expected_channel_rows,
        "required_cell_metrics_finite": bool(np.isfinite(cell_frame[required].to_numpy(dtype=float)).all()),
        "historical_intact_auroc_max_abs_diff_le_2e_5": float(historical["historical_auroc_abs_diff"].max()) <= 2e-5,
        "historical_intact_auprc_max_abs_diff_le_5e_5": float(historical["historical_auprc_abs_diff"].max()) <= 5e-5,
        "google_mask_rows_match_test_molecules": len(mask_frame) == (2000 if mode == "smoke" else 4373),
        "google_channel_rows_match_all_independent_channels": len(channel_frame) == (2822 if mode == "smoke" else 6393),
        "google_channel_metrics_finite": bool(np.isfinite(channel_sensitivity.select_dtypes(include=[np.number]).drop(columns=["seed"], errors="ignore").to_numpy(dtype=float)).all()),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "mode": mode,
        "checks": checks,
        "observed": {
            "cell_count": int(cell_frame["cell_id"].nunique()),
            "cell_metric_rows": len(cell_frame),
            "google_mask_rows": len(mask_frame),
            "google_channel_rows": len(channel_frame),
            "google_channel_sensitivity_rows": len(channel_sensitivity),
            "maximum_historical_auroc_abs_diff": float(historical["historical_auroc_abs_diff"].max()),
            "maximum_historical_auprc_abs_diff": float(historical["historical_auprc_abs_diff"].max()),
        },
        "interpretation_boundary_present": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=r"<project-root>")
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--contract", default=str(Path(__file__).resolve().parent / "run_contract.json"))
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    root, out_dir, contract = Path(args.project_root), Path(args.out_dir), Path(args.contract)
    out_dir.mkdir(parents=True, exist_ok=True)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name)
    started = time.time()
    log = []

    def emit(message):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}"
        log.append(line)
        print(line, flush=True)

    emit(f"start mode={args.mode} device={device} batch_size={args.batch_size}")
    observed_sources = validate_sources(root)
    emit("frozen_source_and_checkpoint_inventory=PASS")
    write_json(out_dir / "input_manifest.json", build_input_manifest(root, contract))
    cell_output, mask_output, channel_output, sensitivity_output = [], [], [], []
    bxaic_tasks = ["B"] if args.mode == "smoke" else BXAIC_TASKS
    google_tasks = ["benzene"] if args.mode == "smoke" else GOOGLE_TASKS
    bxaic = bxaic_test_graphs(root, bxaic_tasks)
    emit(f"bxaic_test_graphs_loaded tasks={len(bxaic_tasks)} n_per_task={len(next(iter(bxaic.values())))}")
    for task in bxaic_tasks:
        cell_specs = [("gin", 42)] if args.mode == "smoke" else [(model, seed) for model in MODELS for seed in SEEDS]
        for model_kind, seed in cell_specs:
            rows, _, _, _ = cell_rows(root, "bxaic", task, model_kind, seed, bxaic[task], [("reference", "mask_reference")], device, args.batch_size)
            cell_output.extend(rows)
            emit(f"cell_complete=bxaic__{task}__{model_kind}__seed{seed}")
    del bxaic
    for task in google_tasks:
        graphs, mask_rows, channel_rows, channel_graphs, base_positions = google_test_graphs(root, task)
        mask_output.extend(mask_rows)
        channel_output.extend(channel_rows)
        emit(f"google_task_loaded={task} test={len(graphs)} channels={len(channel_graphs)}")
        cell_specs = [("gin", 42)] if args.mode == "smoke" else [(model, seed) for model in MODELS for seed in SEEDS]
        for model_kind, seed in cell_specs:
            rows, model, intact_logits, labels = cell_rows(
                root,
                "google",
                task,
                model_kind,
                seed,
                graphs,
                [("historical_last_channel", "mask_last"), ("union", "mask_union"), ("intersection", "mask_intersection")],
                device,
                args.batch_size,
            )
            cell_output.extend(rows)
            cell_id = f"google__{task}__{model_kind}__seed{seed}"
            sensitivity_output.extend(channel_sensitivity_rows(cell_id, "google", task, model_kind, seed, model, intact_logits, labels, channel_graphs, base_positions, device, args.batch_size))
            emit(f"cell_complete={cell_id}")
            del model
        del graphs, channel_graphs
    cell_frame = pd.DataFrame(cell_output)
    mask_frame = pd.DataFrame(mask_output)
    channel_frame = pd.DataFrame(channel_output)
    sensitivity_frame = pd.DataFrame(sensitivity_output)
    write_csv(out_dir / "cell_metrics.csv", cell_frame)
    write_csv(out_dir / "google_mask_definitions.csv", mask_frame)
    write_csv(out_dir / "google_channel_masks.csv", channel_frame)
    write_csv(out_dir / "google_channel_sensitivity.csv", sensitivity_frame)
    summary = build_summary(cell_frame, mask_frame, sensitivity_frame, time.time() - started, device, args.mode)
    validation = build_validation(root, contract, cell_frame, mask_frame, channel_frame, sensitivity_frame, args.mode, observed_sources)
    summary["status"] = "complete_validated" if validation["status"] == "PASS" else "complete_validation_failed"
    summary["validation_status"] = validation["status"]
    summary["command"] = " ".join(sys.argv)
    summary["environment"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "scikit_learn": sklearn.__version__,
        "rdkit": rdBase.rdkitVersion,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
    }
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "validation.json", validation)
    emit(f"validation={validation['status']} elapsed_seconds={time.time() - started:.3f}")
    (out_dir / "run.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    output_files = [
        Path(__file__), contract, out_dir / "input_manifest.json", out_dir / "cell_metrics.csv",
        out_dir / "google_mask_definitions.csv", out_dir / "google_channel_masks.csv",
        out_dir / "google_channel_sensitivity.csv", out_dir / "summary.json", out_dir / "validation.json", out_dir / "run.log",
        Path(__file__).resolve().parent / "attempt_history.json",
        Path(__file__).resolve().parent / "A2_EVIDENCE_REPORT.md",
        Path(__file__).resolve().parent / "dingzhen.md",
    ]
    write_json(out_dir / "output_manifest.json", {
        "algorithm": "SHA-256",
        "file_count": len(output_files),
        "files": [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in output_files],
    })
    if validation["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import torch
import torch_geometric
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from torch_geometric.loader import DataLoader

from gine_model import GINERegressor, graph_from_smiles


DEFAULT_TARGETS = [
    "CHEMBL204_Ki",
    "CHEMBL214_Ki",
    "CHEMBL228_Ki",
    "CHEMBL233_Ki",
    "CHEMBL234_Ki",
    "CHEMBL235_EC50",
    "CHEMBL236_Ki",
    "CHEMBL237_Ki",
    "CHEMBL244_Ki",
    "CHEMBL264_Ki",
    "CHEMBL4792_Ki",
]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def stable_group_split(frame, seed, fraction=0.2):
    sizes = frame.groupby("group_id").size().to_dict()
    ordered = sorted(sizes, key=lambda value: hashlib.sha256(f"{seed}|{value}".encode()).hexdigest())
    target = fraction * len(frame)
    reachable = {0: ()}
    for group in ordered:
        additions = {}
        for count, selected in list(reachable.items()):
            proposed = count + sizes[group]
            if proposed not in reachable and proposed not in additions:
                additions[proposed] = selected + (group,)
        reachable.update(additions)
    feasible = [count for count in reachable if 0 < count < len(frame)]
    closest = min(feasible, key=lambda count: (abs(count - target), count))
    selected = reachable[closest]
    dev = frame.group_id.isin(selected).to_numpy()
    if not dev.any() or dev.all():
        raise RuntimeError("training-only component development split failed")
    return dev, selected


def predict(model, graphs, device, y_mean, y_std, batch_size):
    model.eval()
    values = []
    with torch.no_grad():
        for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
            batch = batch.to(device)
            values.extend((model(batch.x, batch.edge_index, batch.edge_attr, batch.batch) * y_std + y_mean).cpu().tolist())
    return np.asarray(values, dtype=float)


def metrics(y_true, y_pred):
    rho = spearmanr(y_true, y_pred).statistic if len(y_true) > 1 else np.nan
    return float(math.sqrt(mean_squared_error(y_true, y_pred))), float(rho)


def train_one(frame, graphs, dev_mask, target, seed, output_dir, device, epochs, patience, batch_size):
    seed_everything(seed)
    train_indices = np.flatnonzero(~dev_mask & frame.split.eq("train").to_numpy())
    dev_indices = np.flatnonzero(dev_mask & frame.split.eq("train").to_numpy())
    training = [graphs[index] for index in train_indices]
    development = [graphs[index] for index in dev_indices]
    y_train = frame.iloc[train_indices].activity.to_numpy(float)
    y_mean = float(y_train.mean())
    y_std = float(y_train.std(ddof=0))
    if not y_std:
        raise RuntimeError("zero training outcome variance")
    model = GINERegressor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best = None
    stale = 0
    history = []
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        loader = DataLoader(training, batch_size=batch_size, shuffle=True, generator=generator)
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            target_scaled = (batch.y.view(-1) - y_mean) / y_std
            loss = torch.nn.functional.mse_loss(prediction, target_scaled)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        dev_true = frame.iloc[dev_indices].activity.to_numpy(float)
        dev_pred = predict(model, development, device, y_mean, y_std, batch_size)
        dev_rmse, dev_rho = metrics(dev_true, dev_pred)
        history.append({"epoch": epoch, "train_mse_scaled": float(np.mean(losses)), "dev_rmse": dev_rmse, "dev_spearman": dev_rho})
        if best is None or dev_rmse < best["dev_rmse"] - 1e-6:
            best = {"dev_rmse": dev_rmse, "epoch": epoch, "state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}}
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"{target} seed={seed} epoch={epoch} dev_rmse={dev_rmse:.4f}", flush=True)
        if stale >= patience:
            break
    model.load_state_dict(best["state"])
    model_dir = output_dir / "models" / target
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = model_dir / f"seed_{seed}.pt"
    torch.save(
        {
            "state_dict": best["state"],
            "target": target,
            "seed": seed,
            "y_mean": y_mean,
            "y_std": y_std,
            "best_epoch": best["epoch"],
            "architecture": {"hidden": 96, "layers": 4, "dropout": 0.1},
        },
        checkpoint,
    )
    history_dir = output_dir / "training_history" / target
    history_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_dir / f"seed_{seed}.csv", index=False)
    predictions = predict(model, graphs, device, y_mean, y_std, batch_size)
    return model, predictions, checkpoint, best["epoch"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--seeds", default="17,29,43")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = args.targets.split(",")
    seeds = [int(value) for value in args.seeds.split(",")]
    usecols = ["canonical_smiles", "activity", "split", "group_id", "dataset", "environment"]
    manifest = pd.read_csv(args.manifest, usecols=usecols)
    manifest = manifest[(manifest.environment == "mmp_series") & manifest.dataset.isin(targets)].copy()
    pairs = pd.read_csv(args.pairs)
    prediction_rows = []
    metric_rows = []
    split_rows = []
    membership_rows = []
    device = torch.device(args.device)
    for target in targets:
        frame = manifest[manifest.dataset == target].reset_index(drop=True)
        graphs = [graph_from_smiles(row.canonical_smiles, row.activity) for row in frame.itertuples()]
        train_frame = frame[frame.split == "train"].reset_index()
        for seed in seeds:
            train_dev_local, dev_groups = stable_group_split(train_frame, seed)
            dev_mask = np.zeros(len(frame), dtype=bool)
            dev_mask[train_frame.loc[train_dev_local, "index"].to_numpy(int)] = True
            train_groups = set(frame.loc[(frame.split == "train") & ~dev_mask, "group_id"])
            if train_groups & set(dev_groups):
                raise RuntimeError("train/development group overlap")
            for row in train_frame.itertuples(index=False):
                membership_rows.append(
                    {
                        "dataset": target,
                        "seed": seed,
                        "canonical_smiles": row.canonical_smiles,
                        "group_id": row.group_id,
                        "role": "development" if row.group_id in set(dev_groups) else "model_train",
                    }
                )
            split_rows.append(
                {
                    "dataset": target,
                    "seed": seed,
                    "n_model_train": int(((frame.split == "train") & ~dev_mask).sum()),
                    "n_development": int(dev_mask.sum()),
                    "development_fraction_of_training_pool": float(dev_mask.sum() / (frame.split == "train").sum()),
                    "n_train_groups": len(train_groups),
                    "n_development_groups": len(dev_groups),
                    "group_overlap": 0,
                }
            )
            _, predictions, checkpoint, best_epoch = train_one(
                frame, graphs, dev_mask, target, seed, output_dir, device, args.epochs, args.patience, args.batch_size
            )
            for row, prediction in zip(frame.itertuples(), predictions):
                prediction_rows.append(
                    {
                        "dataset": target,
                        "seed": seed,
                        "split": row.split,
                        "canonical_smiles": row.canonical_smiles,
                        "activity": float(row.activity),
                        "prediction": float(prediction),
                    }
                )
            row_metrics = {"dataset": target, "seed": seed, "best_epoch": best_epoch, "checkpoint": str(checkpoint)}
            for split, mask in {
                "development": dev_mask,
                "calibration": frame.split.eq("calibration").to_numpy(),
                "test": frame.split.eq("test").to_numpy(),
            }.items():
                rmse, rho = metrics(frame.loc[mask, "activity"].to_numpy(float), predictions[mask])
                row_metrics[f"{split}_rmse"] = rmse
                row_metrics[f"{split}_spearman"] = rho
                row_metrics[f"n_{split}"] = int(mask.sum())
            target_test_pairs = pairs[(pairs.dataset == target) & (pairs.split == "test")]
            prediction_map = dict(zip(frame.canonical_smiles, predictions))
            correct = [
                np.sign(prediction_map[row.smiles_a] - prediction_map[row.smiles_b])
                == np.sign(row.measured_delta_a_minus_b)
                for row in target_test_pairs.itertuples()
            ]
            row_metrics["test_pair_direction_accuracy"] = float(np.mean(correct))
            row_metrics["n_test_pairs"] = len(correct)
            metric_rows.append(row_metrics)
            print(f"COMPLETED {target} seed={seed} test_rmse={row_metrics['test_rmse']:.4f} pair_direction={row_metrics['test_pair_direction_accuracy']:.3f}", flush=True)
    pd.DataFrame(prediction_rows).to_csv(output_dir / "predictions.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(output_dir / "model_metrics.csv", index=False)
    pd.DataFrame(split_rows).to_csv(output_dir / "train_development_splits.csv", index=False)
    pd.DataFrame(membership_rows).to_csv(output_dir / "training_membership.csv", index=False)
    environment = {
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "device_requested": args.device,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "targets": targets,
        "seeds": seeds,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
    }
    (output_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

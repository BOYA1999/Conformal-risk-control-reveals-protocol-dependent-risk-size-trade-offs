import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.models import MPNN


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashed_indices(smiles: pd.Series, limit: int) -> list[int]:
    return sorted(
        range(len(smiles)),
        key=lambda index: hashlib.sha256(
            f"{index}|{smiles.iloc[index]}".encode("utf-8")
        ).hexdigest(),
    )[:limit]


def graph(smiles: str, device: torch.device):
    dataset = MoleculeDataset([MoleculeDatapoint.from_smi(smiles)])
    batch = next(iter(build_dataloader(dataset, batch_size=1, shuffle=False)))
    batch.bmg.to(device)
    return batch.bmg


def prediction(model: MPNN, bmg) -> float:
    with torch.no_grad():
        return float(model(bmg).squeeze().cpu())


def saliency(model: MPNN, bmg) -> np.ndarray:
    original = bmg.V.detach().clone()
    bmg.V = original.clone().requires_grad_(True)
    value = model(bmg).sum()
    scores = torch.autograd.grad(value, bmg.V)[0].abs().sum(dim=1)
    bmg.V = original
    return scores.detach().cpu().numpy()


def integrated_gradients(model: MPNN, bmg, steps: int) -> np.ndarray:
    original = bmg.V.detach().clone()
    total = torch.zeros_like(original)
    for scale in torch.linspace(1 / steps, 1, steps, device=original.device):
        bmg.V = (original * scale).detach().requires_grad_(True)
        total += torch.autograd.grad(model(bmg).sum(), bmg.V)[0]
    bmg.V = original
    return (original * total / steps).abs().sum(dim=1).detach().cpu().numpy()


def masked_change(model: MPNN, bmg, base: float, atoms: np.ndarray) -> float:
    original = bmg.V.detach().clone()
    masked = original.clone()
    masked[torch.as_tensor(atoms, dtype=torch.long, device=original.device)] = 0
    bmg.V = masked
    value = abs(base - prediction(model, bmg))
    bmg.V = original
    return value


def bootstrap_interval(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    samples = np.empty(2000)
    for i in range(len(samples)):
        samples[i] = rng.choice(values, len(values), replace=True).mean()
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--ig-steps", type=int, default=20)
    parser.add_argument("--random-repeats", type=int, default=20)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    model = MPNN.load_from_file(args.model, map_location=device).to(device).eval()
    test = pd.read_csv(args.test_csv)
    predictions = pd.read_csv(args.predictions_csv)
    selected_indices = hashed_indices(test["smiles"], args.limit)
    rows = []
    attribution_rows = []
    started = time.perf_counter()
    for index in selected_indices:
        smiles = test.at[index, "smiles"]
        bmg = graph(smiles, device)
        base = prediction(model, bmg)
        n_atoms = len(bmg.V)
        k = max(1, math.ceil(0.2 * n_atoms))
        saliency_scores = saliency(model, bmg)
        ig_scores = integrated_gradients(model, bmg, args.ig_steps)
        saliency_atoms = np.argsort(-saliency_scores, kind="stable")[:k]
        ig_atoms = np.argsort(-ig_scores, kind="stable")[:k]
        rng = np.random.default_rng(
            int(hashlib.sha256(smiles.encode("utf-8")).hexdigest()[:16], 16)
        )
        random_changes = [
            masked_change(model, bmg, base, rng.choice(n_atoms, k, replace=False))
            for _ in range(args.random_repeats)
        ]
        cli_prediction = float(predictions.at[index, "LOG_HLM_CLint"])
        rows.append(
            {
                "test_index": index,
                "smiles": smiles,
                "n_atoms": n_atoms,
                "masked_atoms": k,
                "base_prediction": base,
                "cli_prediction": cli_prediction,
                "prediction_abs_diff": abs(base - cli_prediction),
                "saliency_change": masked_change(model, bmg, base, saliency_atoms),
                "ig_change": masked_change(model, bmg, base, ig_atoms),
                "random_change_mean": float(np.mean(random_changes)),
            }
        )
        attribution_rows.append(
            {
                "test_index": index,
                "saliency": saliency_scores.tolist(),
                "integrated_gradients": ig_scores.tolist(),
            }
        )
    frame = pd.DataFrame(rows)
    frame["saliency_wins_random"] = frame["saliency_change"] > frame["random_change_mean"]
    frame["ig_wins_random"] = frame["ig_change"] > frame["random_change_mean"]
    saliency_difference = (frame["saliency_change"] - frame["random_change_mean"]).to_numpy()
    ig_difference = (frame["ig_change"] - frame["random_change_mean"]).to_numpy()
    bootstrap_rng = np.random.default_rng(20260901)
    frame.to_csv(args.out_dir / "molecule_results.csv", index=False)
    (args.out_dir / "attributions.json").write_text(
        json.dumps(attribution_rows), encoding="utf-8"
    )
    summary = {
        "status": "PASS",
        "surface": "polaris_hidden_test_attribution_compatibility",
        "model": str(args.model.resolve()),
        "model_sha256": sha256(args.model),
        "n_molecules": len(frame),
        "selection": "first molecules after SHA-256 ordering of test index and SMILES",
        "ig_steps": args.ig_steps,
        "random_repeats": args.random_repeats,
        "mask_fraction": 0.2,
        "mean_saliency_change": float(frame["saliency_change"].mean()),
        "mean_ig_change": float(frame["ig_change"].mean()),
        "mean_random_change": float(frame["random_change_mean"].mean()),
        "mean_saliency_minus_random": float(saliency_difference.mean()),
        "saliency_minus_random_bootstrap_95ci": bootstrap_interval(
            saliency_difference, bootstrap_rng
        ),
        "mean_ig_minus_random": float(ig_difference.mean()),
        "ig_minus_random_bootstrap_95ci": bootstrap_interval(
            ig_difference, bootstrap_rng
        ),
        "saliency_win_fraction": float(frame["saliency_wins_random"].mean()),
        "ig_win_fraction": float(frame["ig_wins_random"].mean()),
        "max_prediction_abs_diff": float(frame["prediction_abs_diff"].max()),
        "runtime_seconds": time.perf_counter() - started,
        "test_targets_accessed": False,
        "interpretation": "Computational compatibility and perturbation relevance only; Polaris HCLint has no atom rationale, so no missed-rationale CRC claim is made.",
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

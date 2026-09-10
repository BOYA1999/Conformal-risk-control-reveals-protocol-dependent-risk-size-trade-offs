import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from molxai_crc import calibrate_crc, loss_table
from run_bxaic_probe import gradient_scores
from run_gradient_grid import (
    BXAIC_TASKS,
    GOOGLE_TASKS,
    FRACTIONS,
    GraphClassifier,
    bxaic_partitions,
    google_partitions,
    set_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "artifacts/experiment/gradient_grid_main"
OUT = ROOT / "results/reviewer_crc_resampling"
ALPHA = 0.10
SIZE_MULTIPLIERS = [0.25, 0.50, 1.00]


def stable_seed(*values):
    digest = hashlib.sha256("|".join(map(str, values)).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def save_cache(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def compute_cell(family, task, model_kind, partitions):
    cell_id = f"{family}__{task}__{model_kind}__seed42"
    cache_path = OUT / "cache" / f"{cell_id}.npz"
    if cache_path.exists():
        print(f"cache_hit={cell_id}", flush=True)
        return
    device = torch.device("cuda")
    checkpoint = MAIN / "checkpoints" / f"{cell_id}.pt"
    model = GraphClassifier(model_kind, partitions["fit"][0].x.shape[1]).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["state_dict"])
    model.eval()
    started = time.perf_counter()
    cal_scores, cal_truths, cal_timing = gradient_scores(model, partitions["calibration"], device, "ig", 32)
    test_scores, test_truths, test_timing = gradient_scores(model, partitions["test"], device, "ig", 32)
    cal_nonempty = [(score, truth) for score, truth in zip(cal_scores, cal_truths) if truth]
    test_nonempty = [(score, truth) for score, truth in zip(test_scores, test_truths) if truth]
    cal_scores_ne, cal_truths_ne = map(list, zip(*cal_nonempty))
    test_scores_ne, test_truths_ne = map(list, zip(*test_nonempty))
    cal_losses = loss_table(cal_scores_ne, cal_truths_ne, FRACTIONS)
    test_losses = loss_table(test_scores_ne, test_truths_ne, FRACTIONS)
    crc = calibrate_crc(cal_losses, FRACTIONS, ALPHA)
    main_cell = json.loads((MAIN / "cells" / f"{cell_id}.json").read_text(encoding="utf-8"))
    expected = main_cell["explainers"]["ig"]["metrics"]["alpha"]["0.10"]["crc"]
    observed_test_risk = float(test_losses[:, crc["index"]].mean())
    if abs(crc["fraction"] - expected["fraction"]) > 1e-12 or abs(observed_test_risk - expected["risk"]) > 1e-10:
        raise ValueError(f"frozen main-result mismatch for {cell_id}")
    fixed20 = set_metrics(test_scores_ne, test_truths_ne, 0.20)
    save_cache(
        cache_path,
        cal_losses=cal_losses,
        test_losses=test_losses,
        test_n_atoms=np.asarray([len(score) for score in test_scores_ne], dtype=int),
        test_n_rationale=np.asarray([len(truth) for truth in test_truths_ne], dtype=int),
        main_fraction=np.asarray(crc["fraction"]),
        main_test_risk=np.asarray(observed_test_risk),
        fixed20_iou=np.asarray(fixed20["iou"]),
        fixed20_risk=np.asarray(fixed20["risk"]),
        cal_seconds=np.asarray(cal_timing["seconds"]),
        test_seconds=np.asarray(test_timing["seconds"]),
        peak_vram_mib=np.asarray(max(cal_timing["peak_vram_mib"], test_timing["peak_vram_mib"])),
        total_seconds=np.asarray(time.perf_counter() - started),
    )
    print(f"cache_complete={cell_id}", flush=True)
    del model
    torch.cuda.empty_cache()


def resample_cell(cell_id, bootstrap):
    cache = np.load(OUT / "cache" / f"{cell_id}.npz")
    cal_losses, test_losses = cache["cal_losses"], cache["test_losses"]
    n_full = len(cal_losses)
    test_realized = np.mean(
        np.where(FRACTIONS[None, :] == 0, 0, np.ceil(FRACTIONS[None, :] * cache["test_n_atoms"][:, None]))
        / cache["test_n_atoms"][:, None],
        axis=0,
    )
    rows = []
    for multiplier in SIZE_MULTIPLIERS:
        n = max(1, int(round(multiplier * n_full)))
        rng = np.random.default_rng(stable_seed(cell_id, multiplier, bootstrap))
        counts = rng.multinomial(n, np.full(n_full, 1 / n_full), size=bootstrap)
        empirical = counts @ cal_losses / n
        method_risks = {
            "corrected": (n * empirical + 1.0) / (n + 1),
            "naive": empirical,
        }
        for method, risks in method_risks.items():
            feasible = risks <= ALPHA
            if not feasible[:, -1].all():
                raise ValueError(f"no feasible full set for {cell_id}, n={n}, method={method}")
            indices = feasible.argmax(axis=1)
            for replicate, index in enumerate(indices):
                rows.append({
                    "cell_id": cell_id,
                    "calibration_size_multiplier": multiplier,
                    "n_calibration": n,
                    "replicate": replicate,
                    "method": method,
                    "fraction": float(FRACTIONS[index]),
                    "mean_realized_atom_fraction": float(test_realized[index]),
                    "test_risk": float(test_losses[:, index].mean()),
                    "risk_violation": bool(test_losses[:, index].mean() > ALPHA),
                })
    metadata = {
        "cell_id": cell_id,
        "n_calibration_full": n_full,
        "n_test": int(len(test_losses)),
        "main_fraction": float(cache["main_fraction"]),
        "main_test_risk": float(cache["main_test_risk"]),
        "fixed20_iou": float(cache["fixed20_iou"]),
        "fixed20_risk": float(cache["fixed20_risk"]),
        "mean_test_atoms": float(cache["test_n_atoms"].mean()),
        "median_test_atoms": float(np.median(cache["test_n_atoms"])),
        "mean_test_rationale_fraction": float((cache["test_n_rationale"] / cache["test_n_atoms"]).mean()),
        "median_test_rationale_fraction": float(np.median(cache["test_n_rationale"] / cache["test_n_atoms"])),
        "cal_seconds": float(cache["cal_seconds"]),
        "test_seconds": float(cache["test_seconds"]),
        "peak_vram_mib": float(cache["peak_vram_mib"]),
        "total_seconds": float(cache["total_seconds"]),
    }
    return rows, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    specs = [
        (family, task, model)
        for family, tasks in (("bxaic", BXAIC_TASKS), ("google", GOOGLE_TASKS))
        for task in tasks
        for model in ("gin", "gcn")
    ]
    if args.limit:
        specs = specs[: args.limit]
    for family, task in sorted({(family, task) for family, task, _ in specs}):
        pending = [(f, t, m) for f, t, m in specs if f == family and t == task and not (OUT / "cache" / f"{f}__{t}__{m}__seed42.npz").exists()]
        if not pending:
            continue
        print(f"load_task={family}/{task}", flush=True)
        partitions = (
            bxaic_partitions(ROOT / "data/raw/bxaic/data.csv", ROOT / "data/raw/bxaic/explanations.sdf", task)
            if family == "bxaic"
            else google_partitions(ROOT / "reference/graph-attribution/data", task)
        )
        for _, _, model in pending:
            compute_cell(family, task, model, partitions)

    all_rows, metadata = [], []
    for family, task, model in specs:
        cell_id = f"{family}__{task}__{model}__seed42"
        rows, meta = resample_cell(cell_id, args.bootstrap)
        all_rows.extend(rows)
        meta.update({"family": family, "task": task, "model": model})
        metadata.append(meta)
    samples = pd.DataFrame(all_rows)
    cells = pd.DataFrame(metadata)
    summary = samples.groupby(
        ["method", "calibration_size_multiplier"], as_index=False
    ).agg(
        cells=("cell_id", "nunique"),
        resamples=("replicate", "size"),
        mean_fraction=("fraction", "mean"),
        median_fraction=("fraction", "median"),
        q25_fraction=("fraction", lambda x: x.quantile(0.25)),
        q75_fraction=("fraction", lambda x: x.quantile(0.75)),
        mean_realized_atom_fraction=("mean_realized_atom_fraction", "mean"),
        mean_test_risk=("test_risk", "mean"),
        risk_violation_frequency=("risk_violation", "mean"),
    )
    samples.to_csv(OUT / "resamples.csv.gz", index=False, compression="gzip")
    cells.to_csv(OUT / "cells.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    complete = len(specs) == 22 and args.bootstrap == 500
    payload = {
        "status": "complete" if complete else "partial",
        "alpha": ALPHA,
        "bootstrap_repetitions": args.bootstrap,
        "calibration_size_multipliers": SIZE_MULTIPLIERS,
        "cells": len(specs),
        "resample_rows": len(samples),
        "checkpoint_reuse": True,
        "predictor_retraining": False,
        "interpretation": "Fixed observed-pool stability diagnostic; not a fresh-population coverage proof.",
        "summary": json.loads(summary.to_json(orient="records")),
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()

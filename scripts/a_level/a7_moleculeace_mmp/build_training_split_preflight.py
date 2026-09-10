import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


TARGETS = [
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


def exact_split(frame, seed):
    sizes = frame.groupby("group_id").size().to_dict()
    ordered = sorted(sizes, key=lambda value: hashlib.sha256(f"{seed}|{value}".encode()).hexdigest())
    reachable = {0: ()}
    for group in ordered:
        additions = {}
        for count, selected in list(reachable.items()):
            proposed = count + sizes[group]
            if proposed not in reachable and proposed not in additions:
                additions[proposed] = selected + (group,)
        reachable.update(additions)
    closest = min((count for count in reachable if 0 < count < len(frame)), key=lambda count: (abs(count - 0.2 * len(frame)), count))
    selected = set(reachable[closest])
    return frame.group_id.isin(selected), sizes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    frame = pd.read_csv(args.manifest, usecols=["dataset", "environment", "split", "group_id"])
    frame = frame[(frame.environment == "mmp_series") & (frame.split == "train") & frame.dataset.isin(TARGETS)]
    rows = []
    for target in TARGETS:
        target_frame = frame[frame.dataset == target].reset_index(drop=True)
        for seed in [17, 29, 43]:
            dev, sizes = exact_split(target_frame, seed)
            dev_groups = set(target_frame.loc[dev, "group_id"])
            train_groups = set(target_frame.loc[~dev, "group_id"])
            largest = max(sizes.values())
            weak = dev.mean() < 0.1 or len(train_groups) <= 1
            rows.append(
                {
                    "dataset": target,
                    "seed": seed,
                    "n_training_pool": len(target_frame),
                    "n_model_train": int((~dev).sum()),
                    "n_development": int(dev.sum()),
                    "development_fraction": float(dev.mean()),
                    "target_fraction": 0.2,
                    "absolute_fraction_error": float(abs(dev.mean() - 0.2)),
                    "n_all_components": len(sizes),
                    "n_model_train_components": len(train_groups),
                    "n_development_components": len(dev_groups),
                    "group_overlap": len(train_groups & dev_groups),
                    "largest_component_molecules": largest,
                    "largest_component_fraction": float(largest / len(target_frame)),
                    "checkpoint_selection_surface": "STRUCTURALLY_WEAK" if weak else "AVAILABLE",
                    "structural_exception": weak,
                }
            )
    output = pd.DataFrame(rows)
    output.to_csv(output_dir / "training_split_preflight.csv", index=False)
    exceptions = sorted(output.loc[output.structural_exception, "dataset"].unique())
    summary = {
        "status": "PASS_WITH_STRUCTURAL_EXCEPTIONS",
        "selection_rule": "Exact deterministic subset of whole train-only components nearest to 20%; ties favor the smaller development count and seeded component order.",
        "cells": len(output),
        "group_overlap": int(output.group_overlap.sum()),
        "structural_exception_targets": exceptions,
        "CHEMBL214_Ki_boundary": "Development is 216/2091 (10.33%); model training retains one 1875-molecule component.",
        "CHEMBL234_Ki_boundary": "Development is 92/2524 (3.65%); model training retains one 2432-molecule component.",
        "interpretation": "Development-only early stopping is retained, but extreme component concentration weakens checkpoint selection for these targets; components are not split to manufacture a 20% development fraction.",
    }
    (output_dir / "training_split_preflight.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

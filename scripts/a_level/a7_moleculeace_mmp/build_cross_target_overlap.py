import argparse
import itertools
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


def pairwise(scope, sets):
    return [
        {
            "scope": scope,
            "target_a": target_a,
            "target_b": target_b,
            "shared_molecules": len(sets[target_a] & sets[target_b]),
        }
        for target_a, target_b in itertools.combinations(TARGETS, 2)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    pairs = pd.read_csv(args.pairs)
    manifest = pd.read_csv(args.manifest, usecols=["dataset", "environment", "canonical_smiles"])
    manifest = manifest[(manifest.environment == "mmp_series") & manifest.dataset.isin(TARGETS)]
    selected_sets = {
        target: set(pairs.loc[pairs.dataset == target, "smiles_a"]) | set(pairs.loc[pairs.dataset == target, "smiles_b"])
        for target in TARGETS
    }
    manifest_sets = {target: set(manifest.loc[manifest.dataset == target, "canonical_smiles"]) for target in TARGETS}
    rows = pairwise("selected_mmp_panel", selected_sets) + pairwise("full_mmp_series_manifest", manifest_sets)
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "cross_target_overlap.csv", index=False)
    selected = table[(table.scope == "selected_mmp_panel") & (table.shared_molecules > 0)]
    full = table[(table.scope == "full_mmp_series_manifest") & (table.shared_molecules > 0)]
    selected_max = selected.sort_values("shared_molecules", ascending=False).iloc[0]
    full_max = full.sort_values("shared_molecules", ascending=False).iloc[0]
    summary = {
        "status": "DESCRIPTIVE_DEPENDENCE_BOUNDARY",
        "selected_panel_nonzero_target_pairs": len(selected),
        "selected_panel_pairwise_shared_molecule_sum": int(selected.shared_molecules.sum()),
        "selected_panel_max_overlap": {"target_a": selected_max.target_a, "target_b": selected_max.target_b, "shared_molecules": int(selected_max.shared_molecules)},
        "full_manifest_nonzero_target_pairs": len(full),
        "full_manifest_pairwise_shared_molecule_sum": int(full.shared_molecules.sum()),
        "full_manifest_max_overlap": {"target_a": full_max.target_a, "target_b": full_max.target_b, "shared_molecules": int(full_max.shared_molecules)},
        "boundary": "Targets share molecules and are not independent datasets. Target-resampling intervals describe this fixed overlapping 11-target panel; they are not population confidence intervals or evidence of independent-target generalization.",
    }
    (output_dir / "cross_target_overlap_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

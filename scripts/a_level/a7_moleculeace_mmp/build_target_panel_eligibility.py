import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from build_mmp_pairs import TARGETS, build_pairs, cut_records, disjoint_pairs, fingerprints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--included-counts", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    eligible_pool = [row["dataset"] for row in selection["selected"]]
    excluded_targets = sorted(set(eligible_pool) - set(TARGETS))
    manifest = pd.read_csv(
        args.manifest,
        usecols=["canonical_smiles", "activity", "exp_nm", "split", "group_id", "dataset", "environment"],
    )
    frame = manifest[
        (manifest.environment == "mmp_series")
        & manifest.dataset.isin(excluded_targets)
        & manifest.split.isin(["calibration", "test"])
    ].copy()
    audit = Counter()
    record_cache = {}
    atom_count_cache = {}
    fingerprint_cache = {}
    for position, smiles in enumerate(sorted(frame.canonical_smiles.unique()), start=1):
        records, atom_count, _ = cut_records(smiles, audit)
        record_cache[smiles] = records
        atom_count_cache[smiles] = atom_count
        fingerprint_cache[smiles] = fingerprints(smiles)
        if position % 500 == 0:
            print(f"eligibility fragmentation {position}/{frame.canonical_smiles.nunique()}", flush=True)
    rows = []
    for target in excluded_targets:
        for split in ["calibration", "test"]:
            subset = frame[(frame.dataset == target) & (frame.split == split)].reset_index(drop=True)
            local = Counter()
            candidates = build_pairs(subset, record_cache, atom_count_cache, fingerprint_cache, local)
            selected = disjoint_pairs(candidates)
            rows.append(
                {
                    "dataset": target,
                    "split": split,
                    "n_molecules": len(subset),
                    "n_candidate_pairs": len(candidates),
                    "n_disjoint_pairs": len(selected),
                }
            )
            print(f"{target} {split}: {len(selected)}", flush=True)
    included = pd.read_csv(args.included_counts)
    for row in included.itertuples(index=False):
        rows.append(
            {
                "dataset": row.dataset,
                "split": row.split,
                "n_molecules": int(row.n_molecules),
                "n_candidate_pairs": int(row.n_candidate_pairs),
                "n_disjoint_pairs": int(row.n_pairs),
            }
        )
    table = pd.DataFrame(rows).sort_values(["dataset", "split"]).reset_index(drop=True)
    target_minimum = table.groupby("dataset").n_disjoint_pairs.min()
    table["minimum_20_this_split"] = table.n_disjoint_pairs >= 20
    table["eligible_both_splits"] = table.dataset.map(target_minimum.ge(20))
    table["selected_for_a7"] = table.dataset.isin(TARGETS)
    table.to_csv(output_dir / "target_panel_eligibility.csv", index=False)
    observed_included = sorted(target_minimum[target_minimum >= 20].index)
    observed_excluded = {
        target: {
            split: int(table.loc[(table.dataset == target) & (table.split == split), "n_disjoint_pairs"].iloc[0])
            for split in ["calibration", "test"]
        }
        for target in sorted(target_minimum[target_minimum < 20].index)
    }
    summary = {
        "status": "PASS" if observed_included == sorted(TARGETS) and len(eligible_pool) == 15 else "FAIL",
        "source_eligible_targets": len(eligible_pool),
        "selection_rule": "Retain every frozen source-eligible target with at least 20 auditable molecule-disjoint exact-single-cut cliff pairs in both calibration and test.",
        "selected_targets": observed_included,
        "excluded_targets": observed_excluded,
        "audit_for_four_recomputed_targets": dict(audit),
        "boundary": "Selection is a pre-model feasibility filter. Exclusion counts are reported and are not relaxed after observing predictor or attribution results.",
    }
    (output_dir / "target_panel_eligibility_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

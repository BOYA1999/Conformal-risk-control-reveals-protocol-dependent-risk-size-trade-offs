# Data contract v0.1

Freeze date: 2026-08-09

## B-XAIC

- Source: https://huggingface.co/datasets/mproszewska/B-XAIC
- Observed revision: `c963b9ee34862b4115dccf7941ba55c9bee16ad0`
- License: CC BY-SA 4.0
- Rows: 50,000; split 0 has 40,000 train, 5,000 validation, and 5,000 test before the fixed parser exclusion. Within official train, derive a task-stratified 12.5% development set by SHA-256 of ChEMBL ID; the remainder is fit. Official validation is final calibration.
- SHA-256 `data.csv`: `14853568ECE75E5C5666C7190E6402CE8D24B3DD3DB171460E668C82C06344FC`
- SHA-256 `explanations.sdf`: `83C86A6366AFF1DB12A3E439CEAB9A4F275508FAF1BEB5E1B8ADB261465E70AA`
- Fixed exclusion: row 997, `CHEMBL3327018`, split-0 test. Reason: outcome-blind RDKit valence sanitization failure.
- Mapping audit: 49,999/50,000 sanitizable; no duplicate ChEMBL IDs or canonical rows; no cross-partition canonical duplicates; no out-of-range rationale indices. Repeated indices from overlapping SMARTS matches are reduced to a set.
- Authoritative graph: SDF atom order. The 22 canonical-SMILES differences are diagnostics and do not remap atoms.

## Google Research graph-attribution

- Source: https://github.com/google-research/graph-attribution
- Commit: `03e7495379df26a21395b25c6a14d92dc27fc3b0`
- License: Apache 2.0; `data/NOTICE` records permission to redistribute the ZINC subset.
- Tasks: Benzene (12,000), Logic7/Alkane-Carbonyl (4,326), Logic8/Fluoride-Carbonyl (8,671), Logic10/Amine-Ether-Benzene (8,687).
- Authoritative labels: `y_true.npz`. The CSV `label` field is not the task target for Logic7/8/10 and is prohibited for model training or evaluation.
- Authoritative rationale: last node-mask column in `true_raw_attribution_datadicts.npz`, the union of valid fragment matches.
- Splits: preserve official train/test. Within true-label strata ordered by SHA-256 of `mol_id`, assign the first 20% to calibration, the next 10% to development, and the remaining 70% to fitting.
- Audit: all four tasks have valid SMILES, complete/disjoint official splits, disjoint derived calibration, binary masks, exact node alignment, and exact positive-label/non-empty-rationale correspondence.
- File-level hashes: `artifacts/intake/graph_attribution_audit.json`.

## molucn

- Role: external activity-cliff proxy stress test only.
- It cannot satisfy the exact-rationale replication requirement and is excluded from conformal guarantee claims.

## Frozen prohibition

No test-driven row removal, label correction, split regeneration, or rationale remapping. Any newly discovered structural problem routes to a dated contract amendment before test metrics are computed, or to `STOP` if it changes the estimand.

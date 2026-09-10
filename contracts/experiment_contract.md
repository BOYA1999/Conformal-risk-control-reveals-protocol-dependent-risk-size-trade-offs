# Frozen experiment contract v0.1

Freeze date: 2026-08-09
Status: frozen for feasibility; full-grid fields marked `TBD-before-test` must be resolved before any final test evaluation.

## Data partitions

- Training-fit: predictor fitting only.
- Development: early stopping, code validation, and the runtime probe only.
- Calibration: conformal parameter selection and calibration-only baselines only; untouched during development.
- Test: one final evaluation after code, metrics, and gates are frozen.
- Duplicate canonical molecules, conflicting labels, and scaffold overlap are reported per split. No retrospective repair is allowed after test access.
- B-XAIC row 997 (`CHEMBL3327018`) is excluded for all tasks before modeling because both its SMILES and SDF record fail RDKit valence sanitization. This single, outcome-blind exclusion was frozen during schema audit; split-0 test size becomes 4,999.
- For B-XAIC, a deterministic label-stratified 12.5% SHA-256-by-ChEMBL-ID subset of official train is development; the remaining 87.5% is fit, official valid is calibration, and official test is final test.
- Google `graph-attribution` uses `y_true.npz` as the target, not the CSV `label` field. Official train is deterministically divided within true-label strata by SHA-256 of `mol_id`: 20% calibration, 10% development, and 70% fit. Official test remains untouched.

## Primary population

Molecules with a valid atom map and non-empty exact reference rationale. No-rationale cases form a separately reported negative population.

For Google `graph-attribution`, the last attribution-mask column is the documented union of all valid fragment matches and is the frozen rationale target. Earlier alternative mask columns are not used for threshold selection.

## Nested set

For molecule `x` with heavy-atom attribution scores, rank atoms from highest to lowest after a preregistered tie rule. Let `S_lambda(x)` contain the first `ceil(lambda * n_atoms)` atoms, where `lambda` ranges over a fixed grid from 0 to 1. Score-threshold nesting may be secondary.

## Primary loss

`L_miss(S, Y) = 1 - |S intersect Y| / |Y|`, bounded in `[0,1]` and non-increasing as `lambda` grows.

## Calibration

Use conformal risk control on the calibration sample to choose the smallest set family member whose finite-sample upper risk criterion is at most `alpha`. The exact formula and tie handling must match a synthetic oracle test before real test labels are loaded.

## Risk levels

`alpha = 0.05, 0.10, 0.20`.

## Outcomes

Primary:
- mean test missed-rationale loss;
- target-risk pass/fail with bootstrap uncertainty shown only as descriptive support;
- mean and median retained heavy-atom fraction.

Secondary:
- rationale precision and IoU;
- fidelity change after masking selected atoms;
- prediction AUROC/AUPRC;
- subgroup results by molecule size, rationale size, task, and scaffold novelty;
- no-rationale false-highlight size;
- cross-seed variability.

## Competence and validity gates

1. Valid canonical SMILES and exact atom-mask length agreement at least 99.5%; otherwise stop for schema investigation.
2. No identical canonical molecule across train/calibration/test after deduplication audit; conflicts are reported, not silently resolved.
3. Predictor AUROC greater than 0.65 and above a label-permutation control on each task entering XAI claims.
4. Randomized-model/label control must not match the trained model's rationale localization across the aggregate grid.
5. Mean retained atom fraction below 0.80 in at least two-thirds of primary cells at `alpha=0.10`; otherwise no practical-efficiency claim.
6. Exact-rationale replication on at least two benchmark families is mandatory.

## Statistics

- Unit of analysis: molecule, with task/model/explainer/seed cells preserved.
- Primary aggregation: macro-average across tasks after cell-level reporting.
- Paired bootstrap by molecule for size/quality differences; Holm correction within each declared comparison family.
- Do not use significance testing to replace the preregistered risk and efficiency gates.

## Compute gate

Run a measured probe on one B-XAIC task, one backbone, one seed, Integrated Gradients and GradInput, using only the derived development split and at most 1,000 explained molecules. Do not compute calibration or test metrics. Record training time, explanation time per molecule, peak VRAM, and disk footprint. Extrapolate the exact accepted grid with a 1.5 safety factor. If above 30 days, report before expanding hardware.

## Stop conditions

- Direct prior art with the same molecular target and guarantee.
- No legally usable second exact-rationale benchmark.
- Invalid or irreproducible rationale atom mapping.
- Synthetic CRC tests fail.
- Projected full runtime above 30 days without user approval.
- Test leakage or post-hoc gate changes.

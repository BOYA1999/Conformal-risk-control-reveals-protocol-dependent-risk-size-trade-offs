# Paired Liver calibration-rule comparison

The source dataset, six fixed models, original scores, source-order indices, two attribution targets, machine fraction grid and method definitions are unchanged. Only the corrected versus naive selector changes within a paired comparison. Index ties are the primary policy; exact-tie inclusion is a separately recalibrated sensitivity. The four methods, six models and two targets form 48 score combinations.

- `cell_metrics.csv`: 576 selector rows from two rules, two tie policies and three alpha values. Each row is a model/method/target/policy configuration, not a molecular record.
- `paired_cells.csv`: corrected-minus-naive differences within 288 policy/alpha configurations.
- `summary.csv`: shared-molecule summaries at alpha 0.10, with differences formed before resampling.
- `primary_table.csv`: eight method/target summaries under the original index policy.
- `validation.json`: input cache hashes, 288 original corrected rows reconstructed, 78,208 direct set checks and bootstrap specification.

Risk, size, precision, IoU and full-molecule fraction are unitless. The calibration set has 33 nonempty-reference molecules and the test set 31. The 2,000 bootstrap draws use the same 31 test molecules jointly across all six fixed models and methods, conditional on the calibration set and fitted checkpoints. Six models do not create six independent external datasets. Neither paired effects nor naive-rule results isolate predictor quality or biological reference validity.

Reconstruct with `scripts/revision/liver_calibration/analyze.py` and the external original score workspace. The generated molecule-level output remains external. MolRep's repository states CC BY-NC-ND 4.0; no distinct linked-data licence was identified. These source-derived aggregate summaries retain source attribution and conditions, are outside the root MIT scope and confer no additional reuse permission. See `DATA_SOURCES.md`.

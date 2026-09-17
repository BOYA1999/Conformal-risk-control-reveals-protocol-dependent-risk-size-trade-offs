# Complete-grid explanation comparison

The 66 cells comprise 11 tasks, two architectures and three seeds. Each row identified by `cell_id` is a model configuration, not a molecular record. Macro-task summaries first average six configurations within each task, then average the 11 observed tasks. These tasks belong to two benchmark families and are not 11 independent chemical-reference sources.

- `comparison_cells.csv`: four-method primary-policy results at alpha 0.10. `origin` distinguishes original GradInput/IG estimates from newly computed saliency/occlusion estimates.
- `method_summary.csv`, `family_summary.csv`: macro-task risk, realized retained fraction, precision and IoU; risk/efficiency/joint pass counts are descriptive finite-test counts.
- `same_run_subset_cells.csv`: the new scores restricted to the original matched identities. This comparison changes the evaluated/calibrated slice while holding cached rankings fixed.
- `historical_subset_cells.csv`: the original matched-subset estimates, retained as a separately identified comparison rather than claimed as fresh replay.
- `tie_sensitivity_cells.csv`: newly recalibrated exact-cutoff tie inclusion for saliency and occlusion.
- `fixed_budget_cells.csv`: index-policy metrics at nominal 0.20 and 0.50 budgets.
- `paired_comparisons.csv`: paired 2,000-resample intervals across the 11 observed task clusters, seed 20260912.
- `order_reversals.csv`, `policy_order_comparisons.csv`: strict cell-level ordering reversals, with the comparison direction and tied cells explicit.
- `seal.json`: source/checkpoint, score-cache, correspondence and metric-replay completion record.
- `cells.csv`, `summary.json`: saliency/occlusion results across both policies and all three risk levels.
- `validation_summary.json`, `import_lineage_audit.json`: aggregate validation and frozen source-code correspondence, with row-level replay records kept external.

Risk is mean missed-reference fraction; `mean_atom_fraction` is realized explanation size. All fractions are unitless. `saving_vs_random` is the retained-fraction difference from the original calibrated random baseline, not a same-size localization advantage. Exact-tie size can exceed the nominal fraction. Bootstrap intervals are descriptive for the observed task collection, and finite-test passes are not per-cell conformal guarantees. Zero-feature occlusion keeps topology fixed and is not chemical deletion.

The B-XAIC and Graph Attribution sources and their terms are listed in `DATA_SOURCES.md`. These source-derived aggregate results are outside the root MIT license. B-XAIC attribution and CC BY-SA 4.0 conditions remain applicable; Graph Attribution retains its Apache-2.0 and underlying source-data terms. No raw inputs, source-row identities, checkpoint, attribution arrays or third-party source tree is distributed.

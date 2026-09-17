# P22: paired grid sensitivity and four-method curves

This analysis freezes the original 66 checkpoints and complete eligible partitions. P20 occlusion and Saliency scores are reused; IG and GradInput were recomputed once with the original scoring functions and batching. The resulting P01 scores are a new fixed-checkpoint execution, not claimed to be the exact historical scores. The same score arrays are evaluated under the historical machine grid and exact integer hundredths, with the original atom-index tie policy.

- `curves.csv`: 106,656 rows of cell-level calibration/test mean risk, realized size, precision and IoU over 101 grid points, four methods and both grids. These are aggregate cell/grid records, not molecule rows.
- `operating_points.csv`: 1,584 independently calibrated frozen working points at alpha0.05/0.10/0.20.
- `historical_points.csv`, `historical_replay.csv`, `historical_replay_summary.csv`: original work points and separately measured score-reexecution differences.
- `grid_effect_cells.csv`, `grid_effect_summary.csv`: fixed-historical-q, fixed-current-machine-q and independently recalibrated contrasts.
- `cardinality_counts.csv`: affected occurrences and occurrence-by-grid-point counts. Repeated graphs across model/seed/task cells are not independent molecules.
- `method_summary.csv`, `family_summary.csv`, `task_summary.csv`, `curve_summary.csv`: continuous risk-size summaries with source origin explicit.
- `paired_method_comparisons.csv`, `method_order_grid.csv`, `threshold_counts.csv`: paired task-cluster comparisons and descriptive operating screens.
- `independent_QA.json`, `imported_source_audit.json`: recorded independent evaluator and source-hash checks; not an external publication or retraining claim.

Every cell has its own calibrated fraction. Its working point need not lie on a curve formed by averaging all cells at a common nominal fraction. Test curves are descriptive and are not used to choose a protocol. All metrics/fractions are unitless; positive differences use the named direction in each file. Task intervals describe the observed task set, with two source families.

For stored-score reconstruction, use the separate reviewer workspace, preserving its recorded `run_grid.py` beside the original `environment.json`. Set `MOLXAI_WORK_ROOT` to its extracted `molxai-work` and run `scripts/revision/grid_sensitivity/validate_and_summarize.py`. The packaged independent evaluator checks the recorded runner hash and the portable package's frozen scientific source hashes, then reconstructs every saved curve and working point. It writes generated checks into the external grid directory; use a fresh working copy after verifying the immutable bundle.

The `run_grid.py` entry point supports another fixed-checkpoint score run after raw source acquisition. Use a fresh external `revision/grid` output directory to avoid resume/skip behavior. It reuses the public source-audit implementation for current external paths; the recorded source identity is not overwritten to impersonate the historical run. A new full score run is not required to verify the stored-score results.

Source-derived B-XAIC aggregates retain CC BY-SA4.0 and Graph Attribution retains its stated source conditions. These outputs are outside the root MIT scope. Checkpoints, source-row identities and per-atom scores remain external.
# Figure 3 numerical reconstruction

`figure_sources/` contains 1,212 task-macro shared-q curve rows and 36 independently calibrated macro operating points. From the package root, `python -B scripts/revision/grid_sensitivity/rebuild_figure_sources.py --output ../figure-source-check` recomputes both CSVs from the included cell-level aggregates and checks byte identity. It does not render a manuscript figure or infer that macro operating points lie on the shared-q curve. The new output directory must not already exist.

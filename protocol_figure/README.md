# Figure 5 protocol comparisons

`figure_sources/` contains the three actual numerical tables used by Figure 5: 22 task-cell target/integration comparisons (P15/P16), four set-family comparisons (P12/P13), and four reference-objective comparisons (P14). Their fields contain aggregate measurements, not molecular identities or atom scores.

`inputs/` preserves the corresponding structure-free recorded source summaries. These inputs are not relabelled as the primary full-grid results. The target/integration source is the later same-run 11-cell execution; set-family and witness sources are the earlier cached-score slice. Panel-level values must not be pooled or differenced across these evidence surfaces.

From the package root, run `python -B scripts/reproducibility/rebuild_fig4_fig5_sources.py --output ../figure45-source-check` with a new output directory outside the package. It reconstructs all five Figure 4/5 numerical CSVs and verifies byte identity. Figure 4 sources are in `results/task_symmetry/figure_sources/` and use existing P01/P20 cell tables, the P02 reference-first table and P23 intervals. This entry point performs no image rendering or new model/attribution computation.

Source-derived B-XAIC summaries retain CC BY-SA 4.0; Graph Attribution summaries retain the applicable source conditions described in `DATA_SOURCES.md`. These source-derived tables are not relicensed under the root code MIT licence.

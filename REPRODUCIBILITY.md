# Reproducibility

This repository records the manuscript-matched analysis set frozen on 2026-09-13. See [the three-level matrix](metadata/reproducibility/README.md), [checkpoint inventory](metadata/reproducibility/checkpoints.csv) and [analysis dependencies](metadata/reproducibility/analysis_dependencies.csv). The original author-trained weights, score caches and molecule-level outputs listed there are not publicly distributed.

## Reconstruct recorded outputs with the reviewer workspace

If the non-public verification materials are made available through an author-approved access route, verify and reconstruct them from a fresh extraction. The following commands run from the repository root and write to a new external directory:

```bash
python -B scripts/reproducibility/verify_reviewer_materials.py --root ../reviewer_verification
python -B scripts/reproducibility/rebuild_recorded_outputs.py --work-root ../reviewer_verification/molxai-work --output ../molxai-recorded-check
```

The second command reconstructs P20 selectors and metrics from all 66 cached score files, then reconstructs P21's 288 original calibrated metric rows and the 31-molecule bootstrap summaries from its stored scores. It checks against the frozen estimates at absolute tolerance 1e-10. It preserves the original runner and contract hashes when interpreting the original P20 metadata; the packaged path adapter is not presented as the originally executed file. This reconstructs statistics from recorded scores, without retraining or rerunning model attribution. Source-row mapping against the raw inputs is a separate validation layer.

## Revision analyses

P22's complete four-method curves and paired grid results are in `results/grid_sensitivity/`. The external reviewer workspace contains `revision/grid/scores` for the fixed-checkpoint gradient re-execution and `full_grid/scores` for the original P20 scores. For an independent stored-score reconstruction, make a working copy of the reviewer workspace, set `MOLXAI_WORK_ROOT` to that copy's `molxai-work`, and run:

```bash
python -B scripts/revision/grid_sensitivity/validate_and_summarize.py
```

The evaluator preserves the recorded runner hash and separately checks the portable package's frozen scientific source hashes. It reconstructs all 106,656 curve rows and 1,584 calibrated working points without generating attribution or training models. It writes verification products to the external `revision/grid` folder; do not run it against the immutable archive copy. For another fixed-checkpoint attribution execution after raw source acquisition, set `MOLXAI_GRID_ROOT` to a new external directory and run `scripts/revision/grid_sensitivity/run_grid.py`; this is a new numerical run. The runner refuses to resume an output produced by a different runner identity.

The symmetric task analysis runs from the aggregate tables already included here:

```bash
python -B scripts/revision/task_symmetry/analyze.py
```

Its default output is `../molxai-check/task_symmetry`. Set `MOLXAI_OUTPUT_DIR` to another new external directory if needed. Its intervals use 5,000 paired resamples of the observed task clusters after averaging six model/seed cells per task. The two benchmark families are not sampled as an independent population.

The Liver correction analysis needs the 96 original score files and original metric/summary tables in the external workspace. Set `MOLXAI_WORK_ROOT` to the extracted `reviewer_verification/molxai-work`, then run:

```bash
python -B scripts/revision/liver_calibration/analyze.py
```

Its default output is `../molxai-check/liver_calibration`. It reconstructs the original corrected results before comparing corrected and naive selectors. The 48 score combinations cross four methods, six fixed models and two targets; two tie policies and three alpha values are reported separately. The 2,000 shared test-molecule bootstrap draws are conditional on the frozen calibration set and checkpoints. The generated molecule-level table stays in the external workspace and is excluded from this repository. Packaged scripts differ from the recorded scripts only in portable input/output setup; `metadata/reproducibility/adapter_lineage.json` records both hashes and the numerical comparison.

## Verification levels

`python -B scripts/verify_release.py` checks the released tables, configuration coverage, recorded replay status, source syntax and manifest without accessing molecular inputs. `python -B -m unittest discover -s src -p "test_*.py"` exercises the core risk-control implementation.

`python -B -m unittest discover -s scripts -p "test_current_*.py"` checks direct-versus-vectorized metrics under both policies, the frozen floating-point schedule and rejection of unsupported pickle globals. These tests require the scientific dependencies but no source data or weights.

Full analysis requires the external source files and model checkpoints. The current aggregate results were checked locally against the frozen inputs, scores and checkpoints; those excluded files are not recreated by the package verifier. Molecule-bootstrap confidence intervals cannot be reconstructed from the released cell means alone.

## External workspace

Use the versions in `ENVIRONMENT.md`. From the repository root, set an external workspace:

```powershell
$env:MOLXAI_WORK_ROOT = '../molxai-work'
```

On a POSIX shell, use `export MOLXAI_WORK_ROOT=../molxai-work`. Relative paths below assume this value and commands run from the repository root.

```text
molxai-work/
  data/raw/bxaic/{data.csv,explanations.sdf}
  reference/graph-attribution/data/<task>/...
  artifacts/intake/graph_attribution_audit.json
  artifacts/experiment/gradient_grid_main/{cells,checkpoints}/...
  experiments/established_subset/cells/...
  full_grid/...
  independent_reference/data/raw/{Liver.csv,attributions.npz}
  independent_reference/results/...
```

Acquire B-XAIC at the pinned revision and Graph Attribution at the pinned commit in `DATA_SOURCES.md`. Keep the upstream checkout outside this repository. Verify the two B-XAIC hashes before processing. Graph Attribution contains NumPy object archives: use only the verified official files in a trusted local environment, never an arbitrary substitute archive.

```bash
python -B src/audit_graph_attribution.py --root ../molxai-work/reference/graph-attribution/data --out ../molxai-work/artifacts/intake/graph_attribution_audit.json
```

## Original predictors and matched subset

The original 66 checkpoints are not bundled. They were trained by `src/run_gradient_grid.py` using the published partition functions, GIN/GCN architectures and seeds 42, 123 and 2026. To generate a fresh set in an empty external workspace:

```bash
python -B scripts/reproduce_primary.py
python -B scripts/reference/run_established_explainer_benchmark.py
```

The first command requires CUDA and runs up to 100 epochs with batch size 128. The second uses those checkpoints for the five-method, at-most-100-reference-bearing-molecule matched subsets, with batch size 64 and 100 GNNExplainer epochs. GNNExplainer is not a complete-grid method. The wrapper imports PyTorch first for the tested Windows runtime; scientific source files remain hash-pinned.

Fresh training is a regeneration of the protocol, not bitwise replay of the manuscript's original checkpoints. Numerical differences must be reported as a new run. Exact original-run replay requires the original weights and matched-subset records. Do not combine freshly trained weights with the published aggregate tables and label them the same run. Scripts may skip already completed cells, so use a clean external workspace for a fresh run.

## Complete 66-cell extension

With the original or consistently regenerated predictor/subset records in the external layout:

```bash
python -B scripts/full_grid/run_full_grid.py --mode pilot
python -B scripts/full_grid/run_full_grid.py --mode full
python -B scripts/full_grid/validate_and_compare.py
python -B scripts/full_grid/replay_scores.py
python -B scripts/full_grid/seal_run.py
```

The pilot checks saliency batching and atom-occlusion parity on development molecules. Full execution computes both methods on every eligible calibration/test molecule while retaining full source-ID correspondence. Independent validation recomputes risk, size and subset/protocol comparisons. Score replay checks source alignment for all 66 cells and independently recomputes first/middle/last eligible test scores. Sealing requires all 66 cells and caches, source hashes, score replay and metric replay to pass.

The primary selector uses the original index tie-breaker and `np.linspace(0,1,101)` with floating-point `ceil`. This preserves the executed schedule, including grid-rounding effects; substituting exact decimal fractions changes the protocol. Exact-cutoff ties are a separately recalibrated sensitivity analysis. GradInput and IG entries in the published comparison table come from the original full-grid analysis, not a fresh four-method attribution run.

## Independent Liver reference

Read the source conditions in `DATA_SOURCES.md` before acquisition. The helper downloads only the two official files, verifies their SHA-256 values and writes them outside this repository:

```bash
python -B scripts/independent_reference/fetch_inputs.py
python -B scripts/independent_reference/audit_liver_inputs.py
python -B scripts/independent_reference/run_liver.py
python -B scripts/independent_reference/independent_validation.py
```

The original B-XAIC and Graph Attribution tables are also required to reproduce the overlap exclusions. The mask reader verifies pinned hashes, the single NumPy archive member, array schema and permitted NumPy reconstruction globals. Released SMILES atom order is preserved. The run uses CPU execution, six fitted models, 33 nonempty-reference calibration molecules and 31 such test molecules. Prediction evaluation retains all 114 admitted official test molecules.

The validator independently reconstructs features, checkpoint predictions, selected scores and all 288 metric rows, then generates the shared-molecule bootstrap summaries and class strata. The 31 test molecules, not six models treated as independent datasets, are the bootstrap sampling units. Class-stratified results retain shared calibration and carry no separate subgroup risk guarantee.

## Supporting analyses

The remaining entry points under `scripts/` use the sources and contracts listed in `DATA_SOURCES.md`. The `scripts/v8/` namespace holds the pharmacophore/rings/provenance checks. Detailed P07–P17 artifacts remain external; the public provenance map reports their metrics and identifies this access boundary explicitly.

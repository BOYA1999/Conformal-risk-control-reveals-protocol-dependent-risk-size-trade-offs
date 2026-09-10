# Reproducibility

## Release checks

Run these commands from the package root:

```bash
python -m unittest discover -s src -p "test_*.py"
python scripts/v8/verify_version_matched_aggregates.py
python scripts/verify_release.py
```

These checks exercise the finite-sample correction, recompute released summaries, validate the release boundary, parse the included Python files, and verify the SHA-256 manifest.

## External inputs

Raw molecule tables, structures, identifiers, checkpoints, score caches, and row-level outputs are intentionally absent. Reconstruct them from the public sources and pinned revisions in `DATA_SOURCES.md`, verify the recorded hashes, and keep all downloaded and generated material in a separate working directory.

The supplied scripts accept external roots through command-line arguments or environment variables. Do not copy excluded inputs or outputs into this package. The included aggregate checks do not claim to retrain models or reproduce excluded row-level files.

## Current analysis interfaces

The main entry points are under `scripts/`. Inspect each command's `--help` output before a rerun. The current aggregate checks use the pharmacophore and rings-count interfaces under `scripts/v8/`; they write fresh outputs to an external directory when inputs are supplied.

The release contains no model weights. Author-trained models must be regenerated with the published scripts, seeds, split rules, and configurations; third-party checkpoints must be obtained from their cited public sources. Exact end-to-end replay therefore depends on those external inputs.

## Evidence boundaries

- Established-explainer comparisons are descriptive summaries over the stated tasks, architectures, seeds, and frozen splits.
- Polaris metrics evaluate the official hidden-test split; the explanation audit is a label-free perturbation analysis, not rationale validation.
- The activity-cliff analysis uses an SAR-linked proxy and a finite matched-pair stress test; it does not establish causal ground truth or distribution-free generalization.
- The pharmacophore and rings-count analyses are fixed-slice or frozen-identity descriptive audits. They do not establish subgroup conformal guarantees, chemical-edit validation, or new biological ground truth.

Public upload and archival deposit are separate actions. This local package does not itself create a remote repository, release tag, or DOI.

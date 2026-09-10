# MolXAI-CRC reproducibility package 
This package contains the code, frozen contracts, structure-free aggregate results, source records, and verification scripts used for the current manuscript analysis. It describes one release only. Manuscript files, figures, author information, local paths, raw molecular records, row-level predictions or attributions, checkpoints, logs, caches, and third-party source trees are not included.

## Included analyses

- Conformal risk-control implementation and finite-sample checks.
- Established-explainer comparisons, calibration checks, controls, and sensitivity analyses.
- Polaris predictivity and explainer-compatibility audit.
- MoleculeACE matched-molecular-pair activity-cliff audit.
- Pharmacophore reference-return and explanation-size audit.
- Rings-count stratified audit.
- Reproducibility contracts, source records, aggregate-result checks, and release verification.

The activity-cliff reference is an experimental SAR-linked transformation-site proxy. It is not causal, mechanistic, complete, expert annotated, or ground truth. The pharmacophore reference is an operational computational reference, not experimental, biological, mechanistic, causal, or expert-annotated atom truth. Both analyses are bounded descriptive audits.

## Quick checks

Run from the package root:

```bash
python -m unittest discover -s src -p "test_*.py"
python scripts/v8/verify_version_matched_aggregates.py
python scripts/verify_release.py
```

The release verifier checks the manifest, Python syntax, required analysis surfaces, privacy and secret patterns, forbidden paths and file types, structure-bearing schemas, license boundaries, and frozen summary invariants.

## Repository map

- `src/`: core implementation and tests.
- `scripts/`: analysis and validation entry points. Commands that need raw inputs, checkpoints, or score caches take them from an external working directory.
- `contracts/`: frozen analysis contracts and license notes.
- `data/`: public-source metadata only.
- `results/`: structure-free aggregate evidence and provenance records.
- `scripts/v8/` and `results/v8/`: current aggregate verification and the current release summaries.
- `DATA_SOURCES.md`: public sources, pinned revisions, hashes, licenses, and input boundaries.
- `REPRODUCIBILITY.md`: external-input requirements and rerun instructions.
- `SECURITY_AND_PRIVACY.md`: release exclusions and privacy checks.

## License boundary

The root MIT License covers repository-authored code and documentation. External datasets, pretrained models, software, and data-derived aggregate results retain their upstream terms; the relevant notices are kept beside the affected material. Raw records, structures, identifiers, weights, and third-party source trees are not redistributed.



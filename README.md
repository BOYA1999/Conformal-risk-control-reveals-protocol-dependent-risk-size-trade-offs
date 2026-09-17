# MolXAI-CRC reproducibility package

The current release contains the manuscript-matched analysis set frozen on 2026-09-13, with stable provenance identifiers P01–P24. For an immutable citation, use the commit or tagged archive associated with the release rather than the moving `main` branch.

Code, frozen protocols and aggregate results for *Conformal risk control reveals protocol-dependent risk-size trade-offs*.

The central distinction is between reference-coverage risk, explanation size and chemical interpretation. Controlling the first does not establish either of the other two. Evaluation choices can also change method comparisons.

## Analyses

- Four explanations across 66 task/model/seed cells: Gradient × Input, Integrated Gradients, saliency and atom-feature occlusion. The two added methods use the original checkpoints and full eligible partitions; the gradient baselines retain their original analysis provenance.
- Independent Liver chemical-reference analysis: six newly trained classifiers, four explanations, two logit targets and two tie policies. Source-provided alert masks are assessed after excluding structural overlap with the original benchmark families.
- Matched-subset, fixed-budget, exact-tie, calibration and random-control comparisons.
- Pharmacophore, activity-cliff, rings-count and Polaris prediction/compatibility audits.
- P23 symmetric eleven/eight-task summaries in `results/task_symmetry/` and P24 paired corrected/naive Liver calibration analyses in `results/liver_calibration/`.
- P22 same-score machine/exact-integer grid comparisons and complete four-method curves in `results/grid_sensitivity/`; P01 checkpoint re-execution is distinguished from historical points.

The atom references have different origins and limitations. Literature alerts, computational pharmacophores and transformation-site proxies are not interchangeable with experimental causal atom labels. Result provenance is recorded separately for each analysis.

## Verify the package

Run from this directory:

```bash
python -B -m unittest discover -s src -p "test_*.py"
python -B -m unittest discover -s scripts -p "test_current_*.py"
python -B scripts/verify_release.py
```

The verifier checks source-free aggregate means and coverage, recorded replay status, Python syntax, the SHA-256 manifest and release exclusions. It does not retrain models or replay molecule-level bootstrap intervals from aggregate means. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for those operations.

The three verification levels, actual input dependencies and checkpoint identities are listed in [metadata/reproducibility/README.md](metadata/reproducibility/README.md). Original author-trained checkpoints, score caches and molecule-level outputs are not publicly distributed; a file hash identifies an artifact but does not provide access to it.

## Access boundary

| Level | Publicly available here | Additional requirement |
|---|---|---|
| Package verification | Code, contracts, structure-free aggregate tables, provenance metadata and the SHA-256 manifest | None beyond the listed Python dependencies |
| Fresh analysis | Training and analysis entry points | Retrieve the pinned third-party source data under their original terms; new training produces a new run |
| Exact recorded replay | Reconstruction scripts and artifact identities | Original author-trained checkpoints, score caches and molecule-level records are not public, so exact fixed-checkpoint or recorded-score replay is not available from this repository alone |

## Contents

- `src/`: core methods, predictor code and unit tests.
- `scripts/full_grid/`, `scripts/independent_reference/`: current analysis and independent replay entry points.
- `scripts/`: supporting analyses and package verification.
- `contracts/`: frozen scientific protocols.
- `results/`: structure-free tables and analysis provenance.
- [DATA_SOURCES.md](DATA_SOURCES.md): official acquisition routes, checksums and source conditions.
- [ENVIRONMENT.md](ENVIRONMENT.md): executed software versions.

Inputs and generated molecule-level files belong in an external working directory selected by `MOLXAI_WORK_ROOT`. Manuscripts, figures, raw molecular records, masks, scores, weights, caches, logs and third-party source trees are not included.

## License

The root MIT License covers repository-authored code and documentation. Source data and source-derived aggregate results are not relicensed. Their attribution and reuse boundaries are documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the affected result directories.

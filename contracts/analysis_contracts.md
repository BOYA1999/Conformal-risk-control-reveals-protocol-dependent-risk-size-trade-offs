# Analysis contracts

The follow-up analyses are secondary, frozen analyses. They cannot change the primary tasks, models, seeds, attribution scores, calibration fractions, or test-set claims.

## Molecule-level follow-up

- Scope: seed-42, 22 task--backbone cells, official test partition.
- Explainer: Integrated Gradients with 20 integration steps.
- Analyses: rationale-fraction groups, molecule-size groups, null-rationale retention, fidelity against equal-size random masks, bootstrap summaries, and scaffold novelty.
- Boundary: descriptive failure analysis only; it cannot retune the primary experiment.

## Random-model control

- Control: an untrained architecture-matched network at the same initialization seed as each trained cell.
- Scope: 66 task--backbone--seed cells and both gradient explainers.
- Primary metric: paired fixed-20%-atom-budget IoU.
- Boundary: contradiction/control evidence only.

## GNNExplainer development gate

- Scope: 22 seed-42 task--backbone development cells, at most 200 hash-ordered non-empty-rationale molecules per cell, split 1:1 for calibration/evaluation.
- Configuration: 100 epochs, learning rate 0.01, phenomenon explanations, object node masks.
- Boundary: development-only model-selection screen; no final-test claim.

## Calibration-pool resampling

- Scope: 22 seed-42 Integrated Gradients cells, 500 resamples at each of 25%, 50%, and 100% calibration-pool sizes.
- Boundary: fixed-pool stability diagnostic, not a new conformal guarantee or a population error probability.

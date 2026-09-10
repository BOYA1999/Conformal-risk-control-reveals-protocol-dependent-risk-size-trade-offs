# MolXAI-CRC analysis-version and result-provenance audit

## Status and path convention

- Status: **PASS** after repair against the final SI Table S37.
- Stable IDs: `P01`--`P19`, one row per final S37 entry.
- Final S37 source snapshot: `_revision_v8_20260908/staging/source/supplementary.tex`.
- All artifact names below are repository-relative. No local drive, user name, or private host root is stored.
- `results/...` and `experiments/...` follow the science repository namespace used in S37; `analysis/...` follows the V8 revision artifact namespace.
- No Word or source TeX file was modified. The old `si_s37_compact_provenance_table.tex` remains an untouched pre-integration draft and is not an authoritative machine-readable output.

## Final-S37 provenance map

| ID | Display | Frozen surface | Target, selector, and oracle | Canonical artifact | Comparison gate |
|---|---|---|---|---|---|
| P01 | Main Table 2; Fig.3 | Original 66 GIN/GCN cells; 11-task full eligible partitions | True-class signed IG zero/20 and GradInput; `index_tiebreak_v1`; no Table 2 oracle | `results/gradient_grid_phase_a/alpha_summary.csv` | Canonical primary branch; not subset, seed42-only, retraining, or exact-tie rows |
| P02 | Main Fig.5 | P01 tasks/partitions; reference-only ideal order | Historical task-level rationale-first global-fraction oracle | `results/reviewer_oracle_crc/paired_cells_alpha010.csv` | Pair only with P01 or P10 original branch |
| P03 | Main Table 3; Fig.6 | Original 66 checkpoints; five methods; at-most-100 matched molecules per split | Method-specific true-class scores; IG zero/20; `index_tiebreak_v1`; no table oracle | `experiments/established_explainers_20260901/cells.csv` | Keep internally paired IG 0.073/0.603/0.528/0.502; not P05 |
| P04 | Table S8 A4 | 33 newly trained residual GINE cells; official partitions | True-class IG zero/20 with bonds fixed; exact ties; measured-schedule oracle | `experiments/jcim_a_level_20260904/a4_gine` | Architecture/representation branch, not a bond-only ablation |
| P05 | Table S8 A5 | Original 66 checkpoints and P03 identities; target scores recomputed together | True/positive/predicted targets; IG zero/20 and GradInput; exact ties | `experiments/jcim_a_level_20260904/a5_target_null` | Same identities as P03, but a tie-inclusive same-run replay |
| P06 | Table S8 gate | Complete test population plus fit-only eligible gate rows | Supervised null gate then predicted-class GradInput; exact ties; abstention reported separately | `experiments/jcim_a_level_20260904/a5_target_null` | Different denominator and estimand from nonnull CRC rows |
| P07 | Table S16 | Historical cached rankings; 11 seed42 GIN cells | Signed/positive-part/absolute transformations; exact ties; no oracle | `experiments/jcim_v4_pre_submission_20260905/score_semantics` | Cached version; signed IG risk 0.080708 |
| P08 | Table S17 | Original 11 seed42 GIN checkpoints; P07 identities; first re-execution | True-class IG; zero/fit mean; 20/50/100; exact ties; no oracle | `experiments/jcim_v4_pre_submission_20260905/score_semantics` | First rerun; zero-IG20 risk 0.079708 |
| P09 | Table S24 | 66 fresh predictors trained from public inputs | P01/P03 settings; historical index selector; paired original/fresh reporting | `experiments/jcim_v6_reproduction_20260905/primary/comparison/si_s24_retraining_comparison.csv` | Sensitivity only; originals remain canonical |
| P10 | Table S27 original | P01/P02 descriptively re-aggregated over 11 or 8 tasks | Original true-class IG/GradInput; index policy; P02 oracle | `experiments/jcim_v7_20260908/descriptive_audit` | Compare 11 versus 8 tasks only within the original branch |
| P11 | Table S27 GINE | P04 descriptively re-aggregated over 11 or 8 tasks | GINE IG zero/20; exact ties; P04 schedule oracle | `experiments/jcim_v7_20260908/descriptive_audit` | Do not pool with P10 |
| P12 | Table S28 global | P07 cached rankings; 11 seed42 GIN cells | Signed true-class scores; index-truncated global family; measured-count oracle | `experiments/jcim_v7_20260908/set_family` | Neither P03 nor re-executed P15 |
| P13 | Table S28 minmax | Same rankings and identities as P12 | Minmax threshold; exact cutoff ties; family-specific measured-count oracle | `experiments/jcim_v7_20260908/set_family` | Direct set-family sensitivity to P12 only |
| P14 | Table S29 | Four Graph Attribution tasks; 713 matched records | Global/minmax families; union versus legal-witness loss; loss-specific calibration/oracle | `experiments/jcim_v7_20260908/set_family` | Four-task estimand; use same-set union risk for cross-loss diagnosis |
| P15 | Table S30 | Original 11 seed42 GIN checkpoints; 1,066/1,047 IDs; V7 re-execution | Raw-logit IG; zero/fit mean; 20/50/100/200; exact ties; variant-specific oracle | `experiments/jcim_v7_20260908/target_integration` | Zero-IG20 risk 0.078799; not P07, P08, or P12 |
| P16 | Tables S31-S32 | P15 checkpoints, identities, and same-run scores | Raw versus margin; IG20/200 and GradInput; exact ties; target-specific oracle | `experiments/jcim_v7_20260908/target_integration` | Compare only defined paired contrasts; target scales differ |
| P17 | Table S33 | P15/P16 calibrated sets; 20 equal-count random masks | Zero/fit-mean feature replacement; inherited exact-tie set; no oracle | `experiments/jcim_v7_20260908/target_integration` | Predictor sensitivity, not chemical deletion |
| P18 | Tables S34-S35 | Released K4_2ar GCN; frozen 512/512 positive slices and attributions | Exact reference return versus independently calibrated method-specific selected-count-schedule capacity oracles; exact ties | `analysis/pharm_oracle` | Exact-return and schedule-capacity denominators are different estimands |
| P19 | Table S36 | Historical full rings-count GIN/GCN plus frozen A5 GIN identities/checkpoint for V8 rerun | Observed/positive-class IG20; historical index or matched exact-tie branch; no stratum recalibration | `analysis/rings_stratified` | Two surfaces, random multiplicities, and oracle definitions must remain separate |

The CSV is the complete machine-readable source for checkpoint/cache identity, molecules/tasks, target, baseline/steps, tie policy, oracle, high-precision metrics, artifact path, and comparison restriction.

## Collisions resolved

### P03 versus P05

| Version | Risk | Retained | Precision | IoU | Selector |
|---|---:|---:|---:|---:|---|
| P03 Main Table 3; Fig.6 | 0.0733357 | 0.6033943 | 0.5277467 | 0.5016920 | Historical index tie break |
| P05 S8/A5 true-class replay | 0.0731052 | 0.6031832 | 0.5523698 | 0.5264407 | Same-run exact-tie inclusion |

Matching rounded risk and retained values do not make the localization metrics interchangeable. P03 stays internally paired across five explainers; P05 remains the target/tie sensitivity.

### P07, P08, and P15

| Version | Raw zero-IG20 risk | Retained | Meaning |
|---|---:|---:|---|
| P07 / S16 | 0.0807079 | 0.6276938 | Historical cached score version |
| P08 / S17 | 0.0797079 | 0.6279583 | First score re-execution |
| P15 / S30 | 0.0787988 | 0.6279617 | V7 same-run target/integration re-execution |

These are replay diagnostics, not replacement estimates. Tiny score changes near exact ties can change calibrated sets.

### P12 versus P15

Both display IG risk/retained rounded to 0.079/0.628, but P12 uses cached rankings with an index-truncated global family and an oracle retained fraction near 0.285. P15 uses re-executed scores, exact ties, and a variant-specific oracle retained fraction near 0.281.

### P18 oracle definitions

- Exact reference return: binary computed-reference score, lambda 0.01, test risk 0, retained 0.582217, IoU 1.
- Grad-CAM schedule-matched capacity oracle: lambda 0.53, test risk 0.084297, retained 0.556714, capacity excess 0.363963 relative to actual Grad-CAM retained 0.920677.

The exact-return denominator and method-specific schedule-capacity denominator cannot share one oracle label.

### P19 oracle and sampling definitions

- Historical full branch: each GIN/GCN cell uses 4,935 eligible calibration molecules and a task-level rationale-first oracle; the fraction is not recalibrated within label strata.
- V8 matched branch: one GIN seed42 cell uses 100 calibration and 100 test identities and a method-target-specific selected-count-schedule oracle; 29 test molecules are positive and 71 negative.

The two branches also differ in tie policy and random comparator multiplicity. They cannot be pooled. The V8 rerun recovered all four A5 nominal fractions, while risk/retained replay drift up to 0.001056 remains explicit.

## Mandatory comparison guards

1. Never pool the 66-cell full grid, 66-cell matched subset, 11-cell seed42 slice, four-task witness slice, one-task pharmacophore slice, or one-task rings-count label strata.
2. Never pool the historical global-fraction oracle, GINE measured-schedule oracle, S28 family-specific oracle, S30 target/variant oracle, pharmacophore exact-reference-return benchmark, pharmacophore method-specific capacity oracle, historical rings-count task oracle, or matched rings-count method-target oracle.
3. Keep P09 original/fresh pairs visible; fresh retraining does not overwrite P01/P03.
4. Keep P18 exact return and schedule-matched capacity values under separate labels.
5. Keep P19 whole-population calibration explicit; label-stratified rows do not establish class-conditional CRC control.

## Hash-bound source state

| Repository-relative file | SHA-256 | Role |
|---|---|---|
| `_revision_v8_20260908/staging/source/main.tex` | `02FA4636DAD2D48538A2FDC819B788ACC595B7D3CEB2ABA5C98F2F72BF733A7F` | Final V8 main-source snapshot referencing S37 |
| `_revision_v8_20260908/staging/source/supplementary.tex` | `3D6ED9CC978710B39F3004787D56DF0C1F1675DDE088F634C4C662E1BBEBC0E2` | Final S34-S37/P01-P19 source snapshot |
| `source/main.tex` | `DF58E8080F4DF17C0144AF2F99978EF5A0BDA0534C7168A57A7952E4B4BF1276` | Untouched pre-staging source |
| `source/supplementary.tex` | `D0149C9AB397E0A31AB7190F26662FC2145115C98A5A7204029A62C48ACB5CE5` | Untouched pre-staging source; does not contain S34-S37 |

## Gate

**PASS for provenance synchronization.** P01--P19 now match final SI Table S37, P03 displays as `Main Table 3; Fig.6`, canonical artifacts are repository-relative, and no local absolute path is retained. This audit does not authorize submission or public release.

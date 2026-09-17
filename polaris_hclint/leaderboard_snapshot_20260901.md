# Polaris HCLint leaderboard snapshot

Captured: 2026-09-01

Source: <https://polarishub.io/benchmarks/polaris/adme-fang-hclint-1>

The official page lists 2,229 training and 575 hidden-test molecules and ranks entries by the benchmark's main metric, Pearson correlation. Visible reference rows relevant to this run were:

| Entry | MAE | MSE | R2 | Spearman | Pearson | Explained variance |
|---|---:|---:|---:|---:|---:|---:|
| seqera-gradient | 0.285 | 0.145 | 0.626 | 0.792 | 0.796 | 0.626 |
| 1B_MPNN_LargeMix-and-Phenomics | 0.296 | 0.160 | 0.589 | 0.774 | 0.778 | 0.600 |
| CheMeleon | 0.336 | 0.189 | 0.514 | 0.725 | 0.720 | 0.514 |
| This work, CheMeleon seed mean | 0.338 | 0.200 | 0.486 | 0.719 | 0.713 | 0.494 |
| This work, fixed three-seed ensemble | 0.325 | 0.184 | 0.526 | 0.738 | 0.733 | 0.530 |

The two “This work” rows are local evaluations through `polaris-lib` and were not submitted to the public leaderboard. They establish competitive modern foundation-model performance, not a new state-of-the-art record.

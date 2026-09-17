# Symmetric task and family summaries

These rows retain the original eleven-task main analysis and add the fixed exclusion of B, P and X, leaving eight tasks. Each task first averages its six backbone/seed cells; task means are then weighted equally. Original IG/GradInput and P20 occlusion/Saliency provenance remain identified in `origin`.

- `summary.csv`: eleven/eight-task and pooled/family means, observed risk/size/joint counts.
- `task_means.csv`: the 44 task/method means.
- `paired_task_differences.csv`: within-task differences from IG.
- `paired_summary.csv`: 5,000 paired resamples of the observed task set, seed 20260913, after within-task averaging.
- `validation.json`: recorded calculation and source hashes, not an assertion of public raw-data access.

Risk, retained atom fraction, precision and IoU are unitless. Counts of passing cells are descriptive. Families are only the two observed sources; their task summaries and intervals do not establish inference to a population of independent molecular-XAI benchmarks. The B/P/X exclusion was defined by task semantics, not selected according to new method results.

Reconstruct with `scripts/revision/task_symmetry/analyze.py`. The recorded and portable script hashes differ because of external path setup; numerical output comparison is in `metadata/reproducibility/adapter_lineage.json`. Source-derived aggregates retain B-XAIC CC BY-SA 4.0 and Graph Attribution source terms and are outside the root MIT scope.

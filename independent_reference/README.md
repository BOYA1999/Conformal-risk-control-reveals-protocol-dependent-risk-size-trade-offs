# Independent chemical-reference analysis

These tables use the MolRep Liver release described in [Rao et al., Patterns (2022)](https://doi.org/10.1016/j.patter.2022.100628), with clinical label lineage to [Liu et al., Journal of Cheminformatics (2015)](https://doi.org/10.1186/s13321-015-0053-y). The atom masks are source-provided literature-alert references. They are not experimental causal atom labels or the separate chemist-annotation experiment reported by Rao et al.

- `admission_summary.json`: source counts, exclusions, partitions and source hashes.
- `predictor_model_table.csv`: prediction metrics for the six trained models.
- `cell_metrics.csv`: 288 model/target/method/policy/alpha combinations.
- `summary.csv`: 16 combinations at alpha 0.10, averaged over six models. Confidence intervals resample the same 31 test molecules after averaging across models; they cannot be reconstructed from cell means alone.
- `class_strata.csv`: descriptive class-specific results under the shared calibration rule (4, 15 and 12 test references in classes 0, 1 and 2).
- `validation_summary.json`: structure-free record of the local source, score and metric replay.

Risk is mean missed-reference fraction; size is mean retained-atom fraction. Precision and IoU use the released reference mask. `equal_size_random_risk` is the exact expected missed-reference fraction for uniform selection with each realized set size. `tie_inflation` is the realized increase over the corresponding index-truncated set. All fractions are unitless. Empty references are excluded from localization evaluation, not from prediction evaluation. `cell_id` identifies a model configuration, not a molecule.

The [MolRep license](https://github.com/biomed-AI/MolRep/blob/main/LICENSE.txt) states CC BY-NC-ND 4.0. No separate license for the linked data files was identified. These source-derived aggregate tables are excluded from the root MIT license; attribution is retained and no broader reuse permission is asserted. Obtain the source files directly from the official release and assess the upstream conditions for the intended use. No raw records, structures, masks, molecule-level output, third-party source code or weights are distributed here.

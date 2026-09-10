# Current analysis interfaces

`pharmacophore_oracle/audit_pharmacophore_oracle.py` reconstructs the reference-return and schedule-matched capacity-oracle summaries from an external pharmacophore output directory supplied through `MOLXAI_PHARMACOPHORE_ROOT`.

`rings_stratified/run_analysis.py` and `validate_analysis.py` use external project, analysis, and score-cache roots supplied through `MOLXAI_PROJECT_ROOT`, `MOLXAI_ANALYSIS_ROOT`, and `MOLXAI_SCORE_CACHE`. Set `MOLXAI_OUTPUT_DIR` to an external folder for a fresh run.

`verify_version_matched_aggregates.py` is self-contained and checks the aggregate surfaces and provenance records included in this release. It does not claim to retrain models or validate excluded molecule-level inputs.

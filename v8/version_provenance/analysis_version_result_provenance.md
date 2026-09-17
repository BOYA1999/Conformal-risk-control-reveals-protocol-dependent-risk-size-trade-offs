# Analysis provenance

Each identifier denotes a distinct evidence surface. Checkpoint, population, score, set-policy and oracle differences are retained; rounded agreement does not establish same-run equivalence. P20/P21 additionally have their sealed source records in `results/extension_provenance.json`. P22/P23/P24 adapters and actual reconstruction checks are documented in `metadata/reproducibility/adapter_lineage.json`.

`public_aggregate_location` names included aggregates. P07-P17 detailed inputs remain outside this repository; the `external-workspace/` prefix is a logical external key, not bundled-file availability. Some original detailed records were used for local verification as identified in the reproducibility dependency map. No external access route is implied.

| ID | Analysis | Public aggregate location |
|---|---|---|
| P01 | primary_full_grid | `results/original_audits/gradient_grid_phase_a/alpha_summary.csv` |
| P02 | primary_reference_first_oracle | `results/original_audits/reviewer_oracle_crc/paired_cells_alpha010.csv` |
| P03 | matched_five_method_subset | `results/established_explainers/cells.csv` |
| P04 | bond_aware_gine_exact_ties | `results/a_level/a4_gine` |
| P05 | same_run_target_sensitivity | `results/a_level/a5_target_null` |
| P06 | supervised_mask_presence_gate | `results/a_level/a5_target_null` |
| P07 | cached_score_semantics | `results/v8/version_provenance/analysis_version_result_provenance.csv` |
| P08 | ig_path_recomputation | `results/v8/version_provenance/analysis_version_result_provenance.csv` |
| P09 | fresh_predictor_retraining | `results/v8/version_provenance/analysis_version_result_provenance.csv` |
| P10 | primary_task_composition | `results/v8/version_provenance/analysis_version_result_provenance.csv` |
| P11 | gine_task_composition | `results/v8/version_provenance/analysis_version_result_provenance.csv` |
| P12 | global_fraction_set_family | `results/protocol_figure/inputs/set_family_task_macro.csv` |
| P13 | minmax_set_family | `results/protocol_figure/inputs/set_family_task_macro.csv` |
| P14 | alternative_witness_loss | `results/protocol_figure/inputs/witness_cross_loss_family_macro.csv` |
| P15 | ig_target_integration_20_to_200_steps | `results/protocol_figure/inputs/target_comparisons.csv` |
| P16 | raw_logit_margin_comparison | `results/protocol_figure/inputs/target_comparisons.csv` |
| P17 | feature_replacement_sensitivity | `results/v8/version_provenance/analysis_version_result_provenance.csv` |
| P18 | pharmacophore_reference_return_and_capacity | `results/v8/pharmacophore_oracle` |
| P19 | rings_count_class_strata | `results/v8/rings_stratified` |
| P20 | complete_original_partition_established_explainer_extension | `results/full_grid/method_summary.csv` |
| P21 | independent_liver_chemical_reference | `results/independent_reference/summary.csv` |
| P22 | machine_vs_exact_integer_grid | `results/grid_sensitivity/method_summary.csv` |
| P23 | symmetric_four_method_task_composition | `results/task_symmetry/summary.csv` |
| P24 | liver_corrected_vs_naive_crc | `results/liver_calibration/primary_table.csv` |

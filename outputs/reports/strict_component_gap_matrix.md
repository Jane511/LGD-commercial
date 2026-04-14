# Strict Component Gap Matrix (5.1-5.8)

Rules applied:
- Evidence requires code + produced outputs + logic traceability.
- Strict-all-options: all documented options must pass for status=Implemented.

## cashflow_methodology

### 5.1 EAD at default and conversion — Proxy-only
- Documented options: observed_ead | segment_ccf_average | glm_or_logistic_bounded_conversion
- Strict option audit: 1:PASS:observed_ead | 2:PASS:segment_ccf_average | 3:FAIL:glm_or_logistic_bounded_conversion
- Evidence code paths: notebooks/03_commercial_cashflow_lgd.ipynb | notebooks/04_receivables_invoice_finance_lgd.ipynb | notebooks/05_trade_contingent_facilities_lgd.ipynb | notebooks/06_asset_equipment_finance_lgd.ipynb | src/demo_pipeline.py
- Evidence outputs: present:commercial_framework_ead_ccf_summary.csv,receivables_invoice_finance_ead_summary.csv,trade_contingent_ead_conversion_summary.csv,asset_equipment_finance_ead_summary.csv
- Logic trace: EAD/CCF proxies flow into segment/base-downturn summaries and final framework outputs in commercial notebooks.
- Gap reason: GLM/logistic conversion option is documented but not implemented end-to-end in production pipeline.
- Exact remediation tasks: Build dedicated EAD model package with option switch: observed, segmented-average, GLM/logit conversion. | Integrate selected EAD/CCF outputs into product pipelines before severity model execution. | Export component output table with loan-level EAD, CCF, model_option_used, and confidence diagnostics. | Add tests for option parity and bounds (ead>=drawn, ccf in [0,1]).
- Acceptance criteria: All 3 documented EAD options implemented and selectable; output table includes ead, ccf, option_used; tests pass.

### 5.2 Cure probability — Proxy-only
- Documented options: logistic_regression | segmented_cure_table | constrained_expert_overlay
- Strict option audit: 1:FAIL:logistic_regression | 2:PASS:segmented_cure_table | 3:PASS:constrained_expert_overlay
- Evidence code paths: src/lgd_calculation.py | notebooks/02_residential_mortgage_lgd.ipynb | notebooks/07_development_finance_lgd.ipynb
- Evidence outputs: present:cure_overlay_report.csv,proxy_flags_report.csv
- Logic trace: Cure proxies present in engines; explicit fitted logistic appears in mortgage notebook only; not standard across cashflow modules.
- Gap reason: Not all documented options are implemented for cashflow modules; logistic cure is not integrated portfolio-wide.
- Exact remediation tasks: Train cure probability models using logistic + segmented tables + constrained expert overlay fallback with explicit routing rules. | Persist cure outputs as loan-level `p_cure` and segment-level cure summaries for all applicable modules. | Wire cure split directly into economic LGD assembly for non-mortgage modules. | Add calibration and discrimination tests by segment and vintage.
- Acceptance criteria: All 3 cure options implemented with routing logic; cure outputs produced for applicable products; calibration tests pass.

### 5.3 Non-cure severity — Proxy-only
- Documented options: glm_regression | beta_regression | segment_weighted_average
- Strict option audit: 1:FAIL:glm_regression | 2:FAIL:beta_regression | 3:PASS:segment_weighted_average
- Evidence code paths: src/lgd_calculation.py | notebooks/03_commercial_cashflow_lgd.ipynb | notebooks/19_cashflow_lending_pd_alignment.ipynb
- Evidence outputs: present:commercial_framework_segment_summary.csv,commercial_framework_estimate_vs_realised_by_segment.csv
- Logic trace: Severity currently driven by rule-based formulas and weighted segment summaries; no integrated GLM/beta severity engines.
- Gap reason: GLM and beta severity options documented but absent from productionized cashflow pipeline.
- Exact remediation tasks: Implement explicit non-cure severity model set: GLM, beta regression, and thin-sample segment average fallback. | Add model selection config and persist `severity_model_option_used` per observation. | Export segment severity diagnostics and compare against realised liquidation outcomes. | Add regression fit and bounded-output validation tests.
- Acceptance criteria: All 3 severity options implemented; bounded severity outputs and diagnostics exported; regression tests pass.

### 5.4 Recovery timing — Proxy-only
- Documented options: survival_hazard | duration_regression | segment_timing_average
- Strict option audit: 1:FAIL:survival_hazard | 2:FAIL:duration_regression | 3:PASS:segment_timing_average
- Evidence code paths: src/lgd_calculation.py | notebooks/03_commercial_cashflow_lgd.ipynb | notebooks/04_receivables_invoice_finance_lgd.ipynb
- Evidence outputs: present:commercial_framework_loan_level_output.csv,receivables_invoice_finance_loan_level_output.csv
- Logic trace: Timing proxies (e.g., workout_months) influence downturn scalars; no survival/duration fitted component models.
- Gap reason: Only segment/proxy timing present; survival and duration model options missing.
- Exact remediation tasks: Implement time-to-recovery modelling module with survival/hazard, duration regression, and segment-average fallback. | Integrate predicted recovery timing into discounting step for economic LGD. | Export first/final recovery timing outputs and timing-distribution summaries. | Add monotonic and plausibility tests for timing outputs.
- Acceptance criteria: All 3 timing options implemented; timing tables produced and used in discounting; timing validation tests pass.

### 5.5 Recovery cost rate — Proxy-only
- Documented options: segment_historical_average_cost | regression_by_workout_type
- Strict option audit: 1:PASS:segment_historical_average_cost | 2:FAIL:regression_by_workout_type
- Evidence code paths: src/demo_pipeline.py | notebooks/03_commercial_cashflow_lgd.ipynb | notebooks/06_asset_equipment_finance_lgd.ipynb
- Evidence outputs: present:recovery_waterfall.csv,asset_equipment_finance_loan_level_output.csv
- Logic trace: Cost effects are represented via proxy rates/penalties; no regression cost model by workout complexity.
- Gap reason: Regression cost model option documented but not implemented.
- Exact remediation tasks: Implement recovery-cost model set with segment-average and regression-by-path options. | Integrate predicted cost rates into net recovery PV computation. | Export cost-by-path and cost-rate model diagnostics. | Add non-negativity and reconciliation tests against realised cost proxies.
- Acceptance criteria: Both cost options implemented; cost outputs exported and integrated; reconciliation tests pass.

### 5.6 Final realised economic LGD — Proxy-only
- Documented options: direct_realised_lgd | second_stage_glm_beta_segment_table
- Strict option audit: 1:PASS:direct_realised_lgd | 2:FAIL:second_stage_glm_beta_segment_table
- Evidence code paths: src/lgd_calculation.py | src/lgd_final.py | notebooks/03_commercial_cashflow_lgd.ipynb
- Evidence outputs: present:commercial_framework_loan_level_output.csv,lgd_final.csv
- Logic trace: Economic/final LGD is calculated through deterministic chain; optional second-stage generalisation model is not implemented.
- Gap reason: Second-stage model option is absent.
- Exact remediation tasks: Refactor economic LGD builder to enforce cashflow-level discounting path and model-option trace columns. | Add optional second-stage generalisation model registry (GLM/beta/segment table) with deterministic selection policy. | Export realised economic LGD + long-run segment baseline from unified component outputs. | Add reconciliation test from cashflows to loan-level economic LGD.
- Acceptance criteria: Direct realised and second-stage options both implemented and traceable; economic LGD tables fully reproducible.

### 5.7 Downturn LGD and conservatism — Proxy-only
- Documented options: stress_window_calibration | macro_linked_overlay | policy_floor_and_moc
- Strict option audit: 1:FAIL:stress_window_calibration | 2:PASS:macro_linked_overlay | 3:PASS:policy_floor_and_moc
- Evidence code paths: src/lgd_calculation.py | src/overlay_parameters.py | scripts/run_validation_sequence.py
- Evidence outputs: present:downturn_lgd_output.csv,overlay_trace_report.csv,parameter_version_report.csv
- Logic trace: Macro-linked overlay + MoC/floor chain exists and is governed; stress-window calibration mode not implemented.
- Gap reason: One documented option (stress-window calibration) missing.
- Exact remediation tasks: Add downturn calibration modes: stress-window calibration, macro-linked overlay, and policy floor/MoC layer as separate selectable options. | Persist applied downturn mode and parameters in overlay trace outputs. | Export segment-level downturn sensitivity under each mode. | Add tests ensuring deterministic application order and `lgd_downturn>=lgd_economic`.
- Acceptance criteria: All 3 downturn options implemented and selectable; overlay trace identifies mode + params; ordering and inequality checks pass.

### 5.8 End-to-end component assembly — Proxy-only
- Documented options: component_chain_with_option_coverage_across_all_5_1_to_5_7
- Strict option audit: 1:FAIL:all_upstream_component_options_implemented
- Evidence code paths: src/lgd_calculation.py | scripts/run_validation_sequence.py | outputs/tables/validation_sequence_report.csv
- Evidence outputs: present:lgd_segment_summary.csv,validation_sequence_report.csv
- Logic trace: Assembly exists, but strict-all-options fails due to component-level option gaps upstream.
- Gap reason: End-to-end chain is functional but not option-complete under strict-all-options rule.
- Exact remediation tasks: Implement full end-to-end component orchestrator with explicit intermediate artifacts (EAD, cure/path, severity, timing, costs, economic, downturn, final). | Add option-coverage validator that fails when any documented model option is unimplemented for a component. | Publish component lineage report linking every final LGD row to upstream component options and tables. | Add integration tests that rebuild final LGD from component artifacts and match published outputs.
- Acceptance criteria: Component chain artifacts are complete and reproducible; lineage report covers all final LGD rows; integration parity tests pass.

## property_backed_methodology

### 5.1 EAD at default — Proxy-only
- Documented options: observed_ead | ccf_utilisation_model | segmented_or_glm
- Strict option audit: 1:PASS:observed_ead | 2:PASS:ccf_utilisation_model | 3:FAIL:segmented_or_glm
- Evidence code paths: notebooks/07_development_finance_lgd.ipynb | notebooks/11_bridging_loan_lgd.ipynb | src/lgd_calculation.py
- Evidence outputs: present:bridging_loan_level_output.csv,cre_investment_loan_level_output.csv ; missing:development_lgd_results.csv
- Logic trace: Property modules use observed/proxy EAD patterns; development notebook exports expected outputs but files currently missing.
- Gap reason: Segmented/GLM option absent; required development output evidence currently missing in outputs/tables.
- Exact remediation tasks: Implement property EAD package with observed-EAD, utilisation/CCF model, and segmented/GLM options under one interface. | Integrate EAD option selector into mortgage, CRE, development, and bridging pipelines before cure/path branching. | Re-run development notebook export path so `development_lgd_results.csv` is produced and versioned with metadata. | Add EAD validation tests for ccf bounds, utilisation monotonicity, and option parity across property modules.
- Acceptance criteria: Observed, utilisation/CCF, and segmented/GLM EAD options implemented for property products; development output evidence is present; EAD tests pass.

### 5.2 Cure probability — Proxy-only
- Documented options: logistic_regression | segmented_cure_tables | tree_challenger
- Strict option audit: 1:PASS:logistic_regression | 2:PASS:segmented_cure_tables | 3:FAIL:tree_challenger
- Evidence code paths: notebooks/02_residential_mortgage_lgd.ipynb | notebooks/07_development_finance_lgd.ipynb | src/lgd_calculation.py
- Evidence outputs: present:cure_overlay_report.csv,proxy_flags_report.csv
- Logic trace: Logistic and segment/proxy cure logic evidenced (mortgage/development), but tree-based challenger not implemented.
- Gap reason: Tree-based challenger option missing.
- Exact remediation tasks: Add tree-based cure challenger model and model governance routing (champion/challenger) for property products. | Persist cure model outputs and challenger deltas in loan-level component table with `cure_model_option_used`. | Integrate challenger evaluation summary into validation sequence report for property segments. | Add calibration/discrimination tests comparing logistic, segmented, and tree options by segment.
- Acceptance criteria: Logistic, segmented, and tree-challenger cure options implemented with champion/challenger outputs; calibration tests pass.

### 5.3 Resolution path model — Proxy-only
- Documented options: multinomial_logistic | segmented_transition_matrix | tree_challenger
- Strict option audit: 1:FAIL:multinomial_logistic | 2:PASS:segmented_transition_matrix | 3:FAIL:tree_challenger
- Evidence code paths: notebooks/08_cre_investment_lgd.ipynb | notebooks/12_mezz_second_mortgage_lgd.ipynb | src/lgd_calculation.py
- Evidence outputs: present:cre_investment_resolution_path_summary.csv,mezz_second_mortgage_ranking_summary.csv
- Logic trace: Resolution-path segmentation exists in CRE outputs; explicit multinomial/tree path models absent.
- Gap reason: Only segmented/path heuristics evidenced; multinomial and tree options missing.
- Exact remediation tasks: Implement resolution-path model package supporting multinomial-logistic, segmented-transition, and tree-challenger options. | Call resolution-path package before severity stage and persist `resolution_path_option_used` + path probabilities. | Export resolution-path diagnostics table (confusion matrix, path share drift, feature importance for tree). | Add tests verifying probabilities sum to 1 and selected path feeds downstream severity selection.
- Acceptance criteria: Resolution-path options (multinomial, segmented matrix, tree challenger) implemented and traced; probability and linkage tests pass.

### 5.4 Non-cure severity conditional on path — Proxy-only
- Documented options: glm_linear | beta_regression | segmented_weighted_average
- Strict option audit: 1:FAIL:glm_linear | 2:FAIL:beta_regression | 3:PASS:segmented_weighted_average
- Evidence code paths: src/lgd_calculation.py | notebooks/08_cre_investment_lgd.ipynb | notebooks/12_mezz_second_mortgage_lgd.ipynb
- Evidence outputs: present:cre_investment_scenario_summary.csv,mezz_second_mortgage_waterfall_snapshot.csv
- Logic trace: Path-conditioned severity proxies/waterfall logic present; GLM and beta regressions not implemented.
- Gap reason: Two documented model options missing.
- Exact remediation tasks: Implement path-conditional severity models: GLM, beta regression, and segmented weighted fallback with shared interface. | Integrate selected severity model after resolution-path assignment and before timing/cost modules. | Export path-conditional severity diagnostics by property type and resolution path. | Add bounded-output and backtest tests against realised path-conditioned recoveries.
- Acceptance criteria: Path-conditional severity options implemented with bounded outputs and diagnostics; backtests pass.

### 5.5 Recovery timing — Proxy-only
- Documented options: survival_hazard | duration_regression | segmented_timing_average
- Strict option audit: 1:FAIL:survival_hazard | 2:FAIL:duration_regression | 3:PASS:segmented_timing_average
- Evidence code paths: src/lgd_calculation.py | notebooks/09_residual_stock_lgd.ipynb | notebooks/11_bridging_loan_lgd.ipynb
- Evidence outputs: present:bridging_delay_summary.csv,residual_stock_loan_level_output.csv,land_subdivision_loan_level_output.csv
- Logic trace: Timing metrics (time_to_sale/time_to_recovery proxies) are produced; survival/duration model options missing.
- Gap reason: Only segmented/proxy timing implemented.
- Exact remediation tasks: Implement property recovery timing models for survival/hazard and duration regression plus segmented fallback. | Integrate timing predictions into discounting cashflow engine for all property-backed modules. | Export timing distribution table with model option tags and path/property segmentation. | Add timing plausibility tests (non-negative, realistic percentile bounds, segment stability checks).
- Acceptance criteria: All timing options implemented and used in discounting with validated timing distributions.

### 5.6 Recovery costs — Proxy-only
- Documented options: segmented_average_cost | regression_by_path_or_property_type
- Strict option audit: 1:PASS:segmented_average_cost | 2:FAIL:regression_by_path_or_property_type
- Evidence code paths: notebooks/08_cre_investment_lgd.ipynb | notebooks/10_land_subdivision_lgd.ipynb | src/lgd_calculation.py
- Evidence outputs: present:cre_investment_loan_level_output.csv,land_subdivision_loan_level_output.csv
- Logic trace: Cost effects enter as segmented/additive proxies; no regression-by-path/property model.
- Gap reason: Regression option missing.
- Exact remediation tasks: Implement recovery-cost regression by path/property type as an explicit option alongside segmented averages. | Integrate selected cost model into net recovery computation and persist `cost_model_option_used`. | Export cost diagnostics table by workout path and property subtype with residual analysis. | Add non-negativity and reconciliation checks against realised/proxy cost components.
- Acceptance criteria: Segmented and regression-by-path/property cost options implemented, exported, and reconciled.

### 5.7 Final realised economic LGD — Proxy-only
- Documented options: direct_realised_lgd | regression_beta_segmentation_layer
- Strict option audit: 1:PASS:direct_realised_lgd | 2:FAIL:regression_beta_segmentation_layer
- Evidence code paths: src/lgd_calculation.py | notebooks/07_development_finance_lgd.ipynb | notebooks/08_cre_investment_lgd.ipynb
- Evidence outputs: present:cre_investment_loan_level_output.csv,bridging_loan_level_output.csv ; missing:development_lgd_results.csv
- Logic trace: Direct realised/proxy economic LGD path exists across property notebooks/engines; second-stage regression/beta layer not implemented.
- Gap reason: Second option absent and one expected development output currently missing.
- Exact remediation tasks: Implement second-stage economic-LGD layer (regression/beta/segmentation) as configurable option after direct realised LGD. | Ensure development exports are regenerated so all property products have produced economic-LGD evidence tables. | Export economic LGD option comparison table with drift checks by product and segment. | Add reconciliation tests from component cashflows to economic LGD for each property module.
- Acceptance criteria: Direct and second-stage economic LGD options implemented across property modules with reproducible outputs.

### 5.8 Downturn LGD and conservatism — Proxy-only
- Documented options: stress_window_calibration | macro_linked_overlay | policy_floor_moc
- Strict option audit: 1:FAIL:stress_window_calibration | 2:PASS:macro_linked_overlay | 3:PASS:policy_floor_moc
- Evidence code paths: src/lgd_calculation.py | src/overlay_parameters.py | scripts/run_stage9_cross_product_validation.py
- Evidence outputs: present:overlay_trace_report.csv,cre_investment_scenario_summary.csv,stage9_cross_product_validation_report.csv
- Logic trace: Macro-linked downturn and MoC/floor logic present across property outputs; stress-window calibration mode missing.
- Gap reason: Not all documented options implemented.
- Exact remediation tasks: Add stress-window downturn calibration option for property modules alongside existing macro-linked and floor/MoC logic. | Persist downturn option selection and parameter provenance in overlay trace per property loan. | Export property downturn sensitivity and conservatism decomposition table by segment and scenario. | Add deterministic precedence tests and invariant checks (`lgd_downturn>=lgd_economic`) across property outputs.
- Acceptance criteria: Stress-window, macro overlay, and policy floor/MoC downturn options all implemented with deterministic precedence and invariants validated.

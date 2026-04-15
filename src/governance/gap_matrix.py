from __future__ import annotations

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = PROJECT_ROOT / "outputs" / "portfolio"
REPORT_DIR = PROJECT_ROOT / "outputs" / "portfolio" / "reports"

# All output directories to search when checking evidence table existence.
_OUTPUT_SEARCH_DIRS = (
    "outputs/portfolio",
    "outputs/mortgage",
    "outputs/cashflow_lending",
    "outputs/property_backed_lending",
)


def _exists(path: str) -> bool:
    return (PROJECT_ROOT / path).exists()


def _fmt_paths(paths: list[str]) -> str:
    return " | ".join(paths)


def _fmt_tables(tables: list[str]) -> str:
    present = [t for t in tables if any(_exists(f"{d}/{t}") for d in _OUTPUT_SEARCH_DIRS)]
    missing = [t for t in tables if t not in present]
    parts = [f"present:{','.join(present)}" if present else "present:none"]
    if missing:
        parts.append(f"missing:{','.join(missing)}")
    return " ; ".join(parts)


def _audit(options: list[str], passed_idx: set[int]) -> str:
    out = []
    for i, opt in enumerate(options, 1):
        status = "PASS" if i in passed_idx else "FAIL"
        out.append(f"{i}:{status}:{opt}")
    return " | ".join(out)


def _tasks(doc: str, component: str) -> str:
    task_map = {
        ("cashflow_methodology", "5.1"): [
            "Build dedicated EAD model package with option switch: observed, segmented-average, GLM/logit conversion.",
            "Integrate selected EAD/CCF outputs into product pipelines before severity model execution.",
            "Export component output table with loan-level EAD, CCF, model_option_used, and confidence diagnostics.",
            "Add tests for option parity and bounds (ead>=drawn, ccf in [0,1]).",
        ],
        ("cashflow_methodology", "5.2"): [
            "Train cure probability models using logistic + segmented tables + constrained expert overlay fallback with explicit routing rules.",
            "Persist cure outputs as loan-level `p_cure` and segment-level cure summaries for all applicable modules.",
            "Wire cure split directly into economic LGD assembly for non-mortgage modules.",
            "Add calibration and discrimination tests by segment and vintage.",
        ],
        ("cashflow_methodology", "5.3"): [
            "Implement explicit non-cure severity model set: GLM, beta regression, and thin-sample segment average fallback.",
            "Add model selection config and persist `severity_model_option_used` per observation.",
            "Export segment severity diagnostics and compare against realised liquidation outcomes.",
            "Add regression fit and bounded-output validation tests.",
        ],
        ("cashflow_methodology", "5.4"): [
            "Implement time-to-recovery modelling module with survival/hazard, duration regression, and segment-average fallback.",
            "Integrate predicted recovery timing into discounting step for economic LGD.",
            "Export first/final recovery timing outputs and timing-distribution summaries.",
            "Add monotonic and plausibility tests for timing outputs.",
        ],
        ("cashflow_methodology", "5.5"): [
            "Implement recovery-cost model set with segment-average and regression-by-path options.",
            "Integrate predicted cost rates into net recovery PV computation.",
            "Export cost-by-path and cost-rate model diagnostics.",
            "Add non-negativity and reconciliation tests against realised cost proxies.",
        ],
        ("cashflow_methodology", "5.6"): [
            "Refactor economic LGD builder to enforce cashflow-level discounting path and model-option trace columns.",
            "Add optional second-stage generalisation model registry (GLM/beta/segment table) with deterministic selection policy.",
            "Export realised economic LGD + long-run segment baseline from unified component outputs.",
            "Add reconciliation test from cashflows to loan-level economic LGD.",
        ],
        ("cashflow_methodology", "5.7"): [
            "Add downturn calibration modes: stress-window calibration, macro-linked overlay, and policy floor/MoC layer as separate selectable options.",
            "Persist applied downturn mode and parameters in overlay trace outputs.",
            "Export segment-level downturn sensitivity under each mode.",
            "Add tests ensuring deterministic application order and `lgd_downturn>=lgd_economic`.",
        ],
        ("cashflow_methodology", "5.8"): [
            "Implement full end-to-end component orchestrator with explicit intermediate artifacts (EAD, cure/path, severity, timing, costs, economic, downturn, final).",
            "Add option-coverage validator that fails when any documented model option is unimplemented for a component.",
            "Publish component lineage report linking every final LGD row to upstream component options and tables.",
            "Add integration tests that rebuild final LGD from component artifacts and match published outputs.",
        ],
        ("property_backed_methodology", "5.1"): [
            "Implement property EAD package with observed-EAD, utilisation/CCF model, and segmented/GLM options under one interface.",
            "Integrate EAD option selector into mortgage, CRE, development, and bridging pipelines before cure/path branching.",
            "Re-run development notebook export path so `development_lgd_results.csv` is produced and versioned with metadata.",
            "Add EAD validation tests for ccf bounds, utilisation monotonicity, and option parity across property modules.",
        ],
        ("property_backed_methodology", "5.2"): [
            "Add tree-based cure challenger model and model governance routing (champion/challenger) for property products.",
            "Persist cure model outputs and challenger deltas in loan-level component table with `cure_model_option_used`.",
            "Integrate challenger evaluation summary into validation sequence report for property segments.",
            "Add calibration/discrimination tests comparing logistic, segmented, and tree options by segment.",
        ],
        ("property_backed_methodology", "5.3"): [
            "Implement resolution-path model package supporting multinomial-logistic, segmented-transition, and tree-challenger options.",
            "Call resolution-path package before severity stage and persist `resolution_path_option_used` + path probabilities.",
            "Export resolution-path diagnostics table (confusion matrix, path share drift, feature importance for tree).",
            "Add tests verifying probabilities sum to 1 and selected path feeds downstream severity selection.",
        ],
        ("property_backed_methodology", "5.4"): [
            "Implement path-conditional severity models: GLM, beta regression, and segmented weighted fallback with shared interface.",
            "Integrate selected severity model after resolution-path assignment and before timing/cost modules.",
            "Export path-conditional severity diagnostics by property type and resolution path.",
            "Add bounded-output and backtest tests against realised path-conditioned recoveries.",
        ],
        ("property_backed_methodology", "5.5"): [
            "Implement property recovery timing models for survival/hazard and duration regression plus segmented fallback.",
            "Integrate timing predictions into discounting cashflow engine for all property-backed modules.",
            "Export timing distribution table with model option tags and path/property segmentation.",
            "Add timing plausibility tests (non-negative, realistic percentile bounds, segment stability checks).",
        ],
        ("property_backed_methodology", "5.6"): [
            "Implement recovery-cost regression by path/property type as an explicit option alongside segmented averages.",
            "Integrate selected cost model into net recovery computation and persist `cost_model_option_used`.",
            "Export cost diagnostics table by workout path and property subtype with residual analysis.",
            "Add non-negativity and reconciliation checks against realised/proxy cost components.",
        ],
        ("property_backed_methodology", "5.7"): [
            "Implement second-stage economic-LGD layer (regression/beta/segmentation) as configurable option after direct realised LGD.",
            "Ensure development exports are regenerated so all property products have produced economic-LGD evidence tables.",
            "Export economic LGD option comparison table with drift checks by product and segment.",
            "Add reconciliation tests from component cashflows to economic LGD for each property module.",
        ],
        ("property_backed_methodology", "5.8"): [
            "Add stress-window downturn calibration option for property modules alongside existing macro-linked and floor/MoC logic.",
            "Persist downturn option selection and parameter provenance in overlay trace per property loan.",
            "Export property downturn sensitivity and conservatism decomposition table by segment and scenario.",
            "Add deterministic precedence tests and invariant checks (`lgd_downturn>=lgd_economic`) across property outputs.",
        ],
    }
    return " | ".join(task_map[(doc, component)])


def _acceptance(doc: str, component: str) -> str:
    checks = {
        ("cashflow_methodology", "5.1"): "All 3 documented EAD options implemented and selectable; output table includes ead, ccf, option_used; tests pass.",
        ("cashflow_methodology", "5.2"): "All 3 cure options implemented with routing logic; cure outputs produced for applicable products; calibration tests pass.",
        ("cashflow_methodology", "5.3"): "All 3 severity options implemented; bounded severity outputs and diagnostics exported; regression tests pass.",
        ("cashflow_methodology", "5.4"): "All 3 timing options implemented; timing tables produced and used in discounting; timing validation tests pass.",
        ("cashflow_methodology", "5.5"): "Both cost options implemented; cost outputs exported and integrated; reconciliation tests pass.",
        ("cashflow_methodology", "5.6"): "Direct realised and second-stage options both implemented and traceable; economic LGD tables fully reproducible.",
        ("cashflow_methodology", "5.7"): "All 3 downturn options implemented and selectable; overlay trace identifies mode + params; ordering and inequality checks pass.",
        ("cashflow_methodology", "5.8"): "Component chain artifacts are complete and reproducible; lineage report covers all final LGD rows; integration parity tests pass.",
        ("property_backed_methodology", "5.1"): "Observed, utilisation/CCF, and segmented/GLM EAD options implemented for property products; development output evidence is present; EAD tests pass.",
        ("property_backed_methodology", "5.2"): "Logistic, segmented, and tree-challenger cure options implemented with champion/challenger outputs; calibration tests pass.",
        ("property_backed_methodology", "5.3"): "Resolution-path options (multinomial, segmented matrix, tree challenger) implemented and traced; probability and linkage tests pass.",
        ("property_backed_methodology", "5.4"): "Path-conditional severity options implemented with bounded outputs and diagnostics; backtests pass.",
        ("property_backed_methodology", "5.5"): "All timing options implemented and used in discounting with validated timing distributions.",
        ("property_backed_methodology", "5.6"): "Segmented and regression-by-path/property cost options implemented, exported, and reconciled.",
        ("property_backed_methodology", "5.7"): "Direct and second-stage economic LGD options implemented across property modules with reproducible outputs.",
        ("property_backed_methodology", "5.8"): "Stress-window, macro overlay, and policy floor/MoC downturn options all implemented with deterministic precedence and invariants validated.",
    }
    return checks[(doc, component)]


def build_matrix() -> pd.DataFrame:
    rows = []

    cash_doc = "cashflow_methodology"
    prop_doc = "property_backed_methodology"

    # Cashflow 5.1-5.8
    rows.append({
        "doc": cash_doc,
        "component_id": "5.1",
        "component_name": "EAD at default and conversion",
        "documented_model_options": "observed_ead | segment_ccf_average | glm_or_logistic_bounded_conversion",
        "strict_option_audit": _audit([
            "observed_ead",
            "segment_ccf_average",
            "glm_or_logistic_bounded_conversion",
        ], {1, 2}),
        "evidence_code_paths": _fmt_paths([
            "notebooks/cashflow_lending/03_commercial_cashflow_lgd.ipynb",
            "notebooks/cashflow_lending/04_receivables_invoice_finance_lgd.ipynb",
            "notebooks/cashflow_lending/05_trade_contingent_facilities_lgd.ipynb",
            "notebooks/cashflow_lending/06_asset_equipment_finance_lgd.ipynb",
            "src/demo_pipeline.py",
        ]),
        "evidence_output_tables": _fmt_tables([
            "commercial_framework_ead_ccf_summary.csv",
            "receivables_invoice_finance_ead_summary.csv",
            "trade_contingent_ead_conversion_summary.csv",
            "asset_equipment_finance_ead_summary.csv",
        ]),
        "logic_trace": "EAD/CCF proxies flow into segment/base-downturn summaries and final framework outputs in commercial notebooks.",
        "status": "Proxy-only",
        "gap_reason": "GLM/logistic conversion option is documented but not implemented end-to-end in production pipeline.",
    })

    rows.append({
        "doc": cash_doc,
        "component_id": "5.2",
        "component_name": "Cure probability",
        "documented_model_options": "logistic_regression | segmented_cure_table | constrained_expert_overlay",
        "strict_option_audit": _audit([
            "logistic_regression",
            "segmented_cure_table",
            "constrained_expert_overlay",
        ], {2, 3}),
        "evidence_code_paths": _fmt_paths([
            "src/lgd_calculation.py",
            "notebooks/mortgage/02_residential_mortgage_lgd.ipynb",
            "notebooks/property_backed_lending/07_development_finance_lgd.ipynb",
        ]),
        "evidence_output_tables": _fmt_tables([
            "cure_overlay_report.csv",
            "proxy_flags_report.csv",
        ]),
        "logic_trace": "Cure proxies present in engines; explicit fitted logistic appears in mortgage notebook only; not standard across cashflow modules.",
        "status": "Proxy-only",
        "gap_reason": "Not all documented options are implemented for cashflow modules; logistic cure is not integrated portfolio-wide.",
    })

    rows.append({
        "doc": cash_doc,
        "component_id": "5.3",
        "component_name": "Non-cure severity",
        "documented_model_options": "glm_regression | beta_regression | segment_weighted_average",
        "strict_option_audit": _audit([
            "glm_regression",
            "beta_regression",
            "segment_weighted_average",
        ], {3}),
        "evidence_code_paths": _fmt_paths([
            "src/lgd_calculation.py",
            "notebooks/cashflow_lending/03_commercial_cashflow_lgd.ipynb",
            "notebooks/cashflow_lending/19_cashflow_lending_pd_alignment.ipynb",
        ]),
        "evidence_output_tables": _fmt_tables([
            "commercial_framework_segment_summary.csv",
            "commercial_framework_estimate_vs_realised_by_segment.csv",
        ]),
        "logic_trace": "Severity currently driven by rule-based formulas and weighted segment summaries; no integrated GLM/beta severity engines.",
        "status": "Proxy-only",
        "gap_reason": "GLM and beta severity options documented but absent from productionized cashflow pipeline.",
    })

    rows.append({
        "doc": cash_doc,
        "component_id": "5.4",
        "component_name": "Recovery timing",
        "documented_model_options": "survival_hazard | duration_regression | segment_timing_average",
        "strict_option_audit": _audit([
            "survival_hazard",
            "duration_regression",
            "segment_timing_average",
        ], {3}),
        "evidence_code_paths": _fmt_paths([
            "src/lgd_calculation.py",
            "notebooks/cashflow_lending/03_commercial_cashflow_lgd.ipynb",
            "notebooks/cashflow_lending/04_receivables_invoice_finance_lgd.ipynb",
        ]),
        "evidence_output_tables": _fmt_tables([
            "commercial_framework_loan_level_output.csv",
            "receivables_invoice_finance_loan_level_output.csv",
        ]),
        "logic_trace": "Timing proxies (e.g., workout_months) influence downturn scalars; no survival/duration fitted component models.",
        "status": "Proxy-only",
        "gap_reason": "Only segment/proxy timing present; survival and duration model options missing.",
    })

    rows.append({
        "doc": cash_doc,
        "component_id": "5.5",
        "component_name": "Recovery cost rate",
        "documented_model_options": "segment_historical_average_cost | regression_by_workout_type",
        "strict_option_audit": _audit([
            "segment_historical_average_cost",
            "regression_by_workout_type",
        ], {1}),
        "evidence_code_paths": _fmt_paths([
            "src/demo_pipeline.py",
            "notebooks/cashflow_lending/03_commercial_cashflow_lgd.ipynb",
            "notebooks/cashflow_lending/06_asset_equipment_finance_lgd.ipynb",
        ]),
        "evidence_output_tables": _fmt_tables([
            "recovery_waterfall.csv",
            "asset_equipment_finance_loan_level_output.csv",
        ]),
        "logic_trace": "Cost effects are represented via proxy rates/penalties; no regression cost model by workout complexity.",
        "status": "Proxy-only",
        "gap_reason": "Regression cost model option documented but not implemented.",
    })

    rows.append({
        "doc": cash_doc,
        "component_id": "5.6",
        "component_name": "Final realised economic LGD",
        "documented_model_options": "direct_realised_lgd | second_stage_glm_beta_segment_table",
        "strict_option_audit": _audit([
            "direct_realised_lgd",
            "second_stage_glm_beta_segment_table",
        ], {1}),
        "evidence_code_paths": _fmt_paths([
            "src/lgd_calculation.py",
            "src/lgd_final.py",
            "notebooks/cashflow_lending/03_commercial_cashflow_lgd.ipynb",
        ]),
        "evidence_output_tables": _fmt_tables([
            "commercial_framework_loan_level_output.csv",
            "lgd_final.csv",
        ]),
        "logic_trace": "Economic/final LGD is calculated through deterministic chain; optional second-stage generalisation model is not implemented.",
        "status": "Proxy-only",
        "gap_reason": "Second-stage model option is absent.",
    })

    rows.append({
        "doc": cash_doc,
        "component_id": "5.7",
        "component_name": "Downturn LGD and conservatism",
        "documented_model_options": "stress_window_calibration | macro_linked_overlay | policy_floor_and_moc",
        "strict_option_audit": _audit([
            "stress_window_calibration",
            "macro_linked_overlay",
            "policy_floor_and_moc",
        ], {2, 3}),
        "evidence_code_paths": _fmt_paths([
            "src/lgd_calculation.py",
            "src/overlay_parameters.py",
            "src/pipeline/validation_pipeline.py",
        ]),
        "evidence_output_tables": _fmt_tables([
            "downturn_lgd_output.csv",
            "overlay_trace_report.csv",
            "parameter_version_report.csv",
        ]),
        "logic_trace": "Macro-linked overlay + MoC/floor chain exists and is governed; stress-window calibration mode not implemented.",
        "status": "Proxy-only",
        "gap_reason": "One documented option (stress-window calibration) missing.",
    })

    rows.append({
        "doc": cash_doc,
        "component_id": "5.8",
        "component_name": "End-to-end component assembly",
        "documented_model_options": "component_chain_with_option_coverage_across_all_5_1_to_5_7",
        "strict_option_audit": _audit([
            "all_upstream_component_options_implemented",
        ], set()),
        "evidence_code_paths": _fmt_paths([
            "src/lgd_calculation.py",
            "src/pipeline/validation_pipeline.py",
            "outputs/portfolio/validation_sequence_report.csv",
        ]),
        "evidence_output_tables": _fmt_tables([
            "lgd_segment_summary.csv",
            "validation_sequence_report.csv",
        ]),
        "logic_trace": "Assembly exists, but strict-all-options fails due to component-level option gaps upstream.",
        "status": "Proxy-only",
        "gap_reason": "End-to-end chain is functional but not option-complete under strict-all-options rule.",
    })

    # Property-backed 5.1-5.8
    rows.append({
        "doc": prop_doc,
        "component_id": "5.1",
        "component_name": "EAD at default",
        "documented_model_options": "observed_ead | ccf_utilisation_model | segmented_or_glm",
        "strict_option_audit": _audit([
            "observed_ead",
            "ccf_utilisation_model",
            "segmented_or_glm",
        ], {1, 2}),
        "evidence_code_paths": _fmt_paths([
            "notebooks/property_backed_lending/07_development_finance_lgd.ipynb",
            "notebooks/property_backed_lending/11_bridging_loan_lgd.ipynb",
            "src/lgd_calculation.py",
        ]),
        "evidence_output_tables": _fmt_tables([
            "bridging_loan_level_output.csv",
            "cre_investment_loan_level_output.csv",
            "development_lgd_results.csv",
        ]),
        "logic_trace": "Property modules use observed/proxy EAD patterns; development notebook exports expected outputs but files currently missing.",
        "status": "Proxy-only",
        "gap_reason": "Segmented/GLM option absent; required development output evidence currently missing in outputs/property_backed_lending.",
    })

    rows.append({
        "doc": prop_doc,
        "component_id": "5.2",
        "component_name": "Cure probability",
        "documented_model_options": "logistic_regression | segmented_cure_tables | tree_challenger",
        "strict_option_audit": _audit([
            "logistic_regression",
            "segmented_cure_tables",
            "tree_challenger",
        ], {1, 2}),
        "evidence_code_paths": _fmt_paths([
            "notebooks/mortgage/02_residential_mortgage_lgd.ipynb",
            "notebooks/property_backed_lending/07_development_finance_lgd.ipynb",
            "src/lgd_calculation.py",
        ]),
        "evidence_output_tables": _fmt_tables([
            "cure_overlay_report.csv",
            "proxy_flags_report.csv",
        ]),
        "logic_trace": "Logistic and segment/proxy cure logic evidenced (mortgage/development), but tree-based challenger not implemented.",
        "status": "Proxy-only",
        "gap_reason": "Tree-based challenger option missing.",
    })

    rows.append({
        "doc": prop_doc,
        "component_id": "5.3",
        "component_name": "Resolution path model",
        "documented_model_options": "multinomial_logistic | segmented_transition_matrix | tree_challenger",
        "strict_option_audit": _audit([
            "multinomial_logistic",
            "segmented_transition_matrix",
            "tree_challenger",
        ], {2}),
        "evidence_code_paths": _fmt_paths([
            "notebooks/property_backed_lending/08_cre_investment_lgd.ipynb",
            "notebooks/property_backed_lending/12_mezz_second_mortgage_lgd.ipynb",
            "src/lgd_calculation.py",
        ]),
        "evidence_output_tables": _fmt_tables([
            "cre_investment_resolution_path_summary.csv",
            "mezz_second_mortgage_ranking_summary.csv",
        ]),
        "logic_trace": "Resolution-path segmentation exists in CRE outputs; explicit multinomial/tree path models absent.",
        "status": "Proxy-only",
        "gap_reason": "Only segmented/path heuristics evidenced; multinomial and tree options missing.",
    })

    rows.append({
        "doc": prop_doc,
        "component_id": "5.4",
        "component_name": "Non-cure severity conditional on path",
        "documented_model_options": "glm_linear | beta_regression | segmented_weighted_average",
        "strict_option_audit": _audit([
            "glm_linear",
            "beta_regression",
            "segmented_weighted_average",
        ], {3}),
        "evidence_code_paths": _fmt_paths([
            "src/lgd_calculation.py",
            "notebooks/property_backed_lending/08_cre_investment_lgd.ipynb",
            "notebooks/property_backed_lending/12_mezz_second_mortgage_lgd.ipynb",
        ]),
        "evidence_output_tables": _fmt_tables([
            "cre_investment_scenario_summary.csv",
            "mezz_second_mortgage_waterfall_snapshot.csv",
        ]),
        "logic_trace": "Path-conditioned severity proxies/waterfall logic present; GLM and beta regressions not implemented.",
        "status": "Proxy-only",
        "gap_reason": "Two documented model options missing.",
    })

    rows.append({
        "doc": prop_doc,
        "component_id": "5.5",
        "component_name": "Recovery timing",
        "documented_model_options": "survival_hazard | duration_regression | segmented_timing_average",
        "strict_option_audit": _audit([
            "survival_hazard",
            "duration_regression",
            "segmented_timing_average",
        ], {3}),
        "evidence_code_paths": _fmt_paths([
            "src/lgd_calculation.py",
            "notebooks/property_backed_lending/09_residual_stock_lgd.ipynb",
            "notebooks/property_backed_lending/11_bridging_loan_lgd.ipynb",
        ]),
        "evidence_output_tables": _fmt_tables([
            "bridging_delay_summary.csv",
            "residual_stock_loan_level_output.csv",
            "land_subdivision_loan_level_output.csv",
        ]),
        "logic_trace": "Timing metrics (time_to_sale/time_to_recovery proxies) are produced; survival/duration model options missing.",
        "status": "Proxy-only",
        "gap_reason": "Only segmented/proxy timing implemented.",
    })

    rows.append({
        "doc": prop_doc,
        "component_id": "5.6",
        "component_name": "Recovery costs",
        "documented_model_options": "segmented_average_cost | regression_by_path_or_property_type",
        "strict_option_audit": _audit([
            "segmented_average_cost",
            "regression_by_path_or_property_type",
        ], {1}),
        "evidence_code_paths": _fmt_paths([
            "notebooks/property_backed_lending/08_cre_investment_lgd.ipynb",
            "notebooks/property_backed_lending/10_land_subdivision_lgd.ipynb",
            "src/lgd_calculation.py",
        ]),
        "evidence_output_tables": _fmt_tables([
            "cre_investment_loan_level_output.csv",
            "land_subdivision_loan_level_output.csv",
        ]),
        "logic_trace": "Cost effects enter as segmented/additive proxies; no regression-by-path/property model.",
        "status": "Proxy-only",
        "gap_reason": "Regression option missing.",
    })

    rows.append({
        "doc": prop_doc,
        "component_id": "5.7",
        "component_name": "Final realised economic LGD",
        "documented_model_options": "direct_realised_lgd | regression_beta_segmentation_layer",
        "strict_option_audit": _audit([
            "direct_realised_lgd",
            "regression_beta_segmentation_layer",
        ], {1}),
        "evidence_code_paths": _fmt_paths([
            "src/lgd_calculation.py",
            "notebooks/property_backed_lending/07_development_finance_lgd.ipynb",
            "notebooks/property_backed_lending/08_cre_investment_lgd.ipynb",
        ]),
        "evidence_output_tables": _fmt_tables([
            "cre_investment_loan_level_output.csv",
            "bridging_loan_level_output.csv",
            "development_lgd_results.csv",
        ]),
        "logic_trace": "Direct realised/proxy economic LGD path exists across property notebooks/engines; second-stage regression/beta layer not implemented.",
        "status": "Proxy-only",
        "gap_reason": "Second option absent and one expected development output currently missing.",
    })

    rows.append({
        "doc": prop_doc,
        "component_id": "5.8",
        "component_name": "Downturn LGD and conservatism",
        "documented_model_options": "stress_window_calibration | macro_linked_overlay | policy_floor_moc",
        "strict_option_audit": _audit([
            "stress_window_calibration",
            "macro_linked_overlay",
            "policy_floor_moc",
        ], {2, 3}),
        "evidence_code_paths": _fmt_paths([
            "src/lgd_calculation.py",
            "src/overlay_parameters.py",
            "src/pipeline/validation_pipeline.py",
        ]),
        "evidence_output_tables": _fmt_tables([
            "overlay_trace_report.csv",
            "cre_investment_scenario_summary.csv",
            "stage9_cross_product_validation_report.csv",
        ]),
        "logic_trace": "Macro-linked downturn and MoC/floor logic present across property outputs; stress-window calibration mode missing.",
        "status": "Proxy-only",
        "gap_reason": "Not all documented options implemented.",
    })

    for row in rows:
        cid = row["component_id"]
        doc = row["doc"]
        row["exact_remediation_tasks"] = _tasks(doc, cid)
        row["acceptance_criteria"] = _acceptance(doc, cid)

    df = pd.DataFrame(rows)
    order = ["5.1", "5.2", "5.3", "5.4", "5.5", "5.6", "5.7", "5.8"]
    df["_o"] = df["component_id"].map({k: i for i, k in enumerate(order)})
    df = df.sort_values(["doc", "_o"]).drop(columns=["_o"]).reset_index(drop=True)
    return df


def to_markdown(df: pd.DataFrame) -> str:
    lines = [
        "# Strict Component Gap Matrix (5.1-5.8)",
        "",
        "Rules applied:",
        "- Evidence requires code + produced outputs + logic traceability.",
        "- Strict-all-options: all documented options must pass for status=Implemented.",
        "",
    ]

    for doc in ["cashflow_methodology", "property_backed_methodology"]:
        part = df[df["doc"] == doc].copy()
        lines.append(f"## {doc}")
        lines.append("")
        for _, r in part.iterrows():
            lines.append(f"### {r['component_id']} {r['component_name']} — {r['status']}")
            lines.append(f"- Documented options: {r['documented_model_options']}")
            lines.append(f"- Strict option audit: {r['strict_option_audit']}")
            lines.append(f"- Evidence code paths: {r['evidence_code_paths']}")
            lines.append(f"- Evidence outputs: {r['evidence_output_tables']}")
            lines.append(f"- Logic trace: {r['logic_trace']}")
            lines.append(f"- Gap reason: {r['gap_reason']}")
            lines.append(f"- Exact remediation tasks: {r['exact_remediation_tasks']}")
            lines.append(f"- Acceptance criteria: {r['acceptance_criteria']}")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = build_matrix()
    csv_path = TABLE_DIR / "strict_component_gap_matrix.csv"
    md_path = REPORT_DIR / "strict_component_gap_matrix.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(to_markdown(df), encoding="utf-8")

    print(f"rows={len(df)}")
    print(f"csv={csv_path}")
    print(f"md={md_path}")


if __name__ == "__main__":
    main()

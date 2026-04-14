# Validation Sequence Report

- Steps passed: 6/6

| Step | Status | Detail |
|---|---|---|
| demo_pipeline | PASS | source=generated; products=['cashflow_lending', 'commercial', 'development', 'mortgage', 'run_metadata']; missing_products=none; core_payload=True |
| core_governance_reports | PASS | rows={"cure_overlay_report.csv": 7, "fallback_usage_report.csv": 6, "overlay_trace_report.csv": 4, "parameter_version_report.csv": 3, "proxy_flags_report.csv": 2, "run_metadata_report.csv": 1, "segmentation_consistency_report.csv": 4, "unemployment_year_bucket_report.csv": 3}; missing=none; empty=none |
| validation_report_hooks | PASS | weighted_accuracy_all_products=True; governance_rows=34; vintage_rows=88; time_rows=21; oot_rows=3; ranking_rows=3; origination_source_rows=3 |
| final_lgd_layer | PASS | summary_rows=5; check_pass_rate=100.00% |
| reproducibility_determinism | PASS | checks=6; all_equal=True |
| notebook_reproducibility_scan | PASS | notebooks=20; reproducibility_proxy_rate=100.00% |
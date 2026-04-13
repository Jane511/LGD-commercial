# Validation Sequence Report

- Steps passed: 5/5

| Step | Status | Detail |
|---|---|---|
| demo_pipeline | PASS | outputs=6; validation_pass=True; missing=none |
| core_governance_reports | PASS | rows={"cure_overlay_report.csv": 7, "fallback_usage_report.csv": 6, "proxy_flags_report.csv": 2, "unemployment_year_bucket_report.csv": 3}; missing=none; empty=none |
| validation_report_hooks | PASS | weighted_accuracy_all_products=True; governance_rows=31; vintage_rows=88; time_rows=21; oot_rows=3; ranking_rows=3; origination_source_rows=3 |
| final_lgd_layer | PASS | summary_rows=5; check_pass_rate=100.00% |
| notebook_reproducibility_scan | PASS | notebooks=20; reproducibility_proxy_rate=100.00% |
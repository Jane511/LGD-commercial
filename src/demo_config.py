from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[1]
REPO_NAME='LGD-Cashflow-and-Property-Lending'
PIPELINE_KIND='lgd'
EXPECTED_OUTPUTS=['lgd_segment_summary.csv', 'recovery_waterfall.csv', 'downturn_lgd_output.csv', 'lgd_validation_report.csv']

# Commercial LGD & Recovery Project

This repository is the Loss Given Default and recovery analytics layer in the commercial credit-risk stack. It uses synthetic facility data, collateral and recovery assumptions, and supporting borrower or industry context to estimate downturn LGD outcomes and recovery views for a commercial lending portfolio. The main outputs feed downstream expected loss, stress testing, pricing, and capital workflows.

## What this repo is

This project demonstrates how a bank-style commercial LGD workflow can be presented in a clear, portfolio-ready format. It focuses on severity, recoveries, and downturn treatment using transparent assumptions so the repo is easy to review without needing internal workout data.

## Where it sits in the stack

Upstream inputs:
- `PD-and-scorecard-commercial`
- `industry-analysis`
- collateral and recovery assumptions

Downstream consumers:
- `expected-loss-engine-commercial`
- `stress-testing-commercial`
- `RAROC-pricing-and-return-hurdle`
- `RWA-capital-commercial`

## Key outputs

- `outputs/tables/lgd_segment_summary.csv`
- `outputs/tables/recovery_waterfall.csv`
- `outputs/tables/downturn_lgd_output.csv`
- `outputs/tables/lgd_validation_report.csv`
- `outputs/tables/pipeline_validation_report.csv`

## Repo structure

- `data/`: raw, interim, processed, and external demo inputs
- `src/`: reusable LGD, recovery, and pipeline logic
- `scripts/`: wrapper scripts for pipeline execution
- `docs/`: methodology, assumptions, data dictionary, and validation notes
- `notebooks/`: reviewer-facing walkthrough notebooks
- `outputs/`: exported tables, reports, and sample artifacts
- `tests/`: validation and regression checks

## How to run

```powershell
python -m src.codex_run_pipeline
```

Or:

```powershell
python scripts/run_codex_pipeline.py
```

## Limitations / Demo-Only Note

- All portfolio data is synthetic and included for demonstration only.
- Recovery timing, collateral treatment, and downturn overlays use simplified assumptions rather than governed workout datasets.
- The repo is intended for portfolio presentation and methodology discussion, not for production impairment or regulatory LGD use.

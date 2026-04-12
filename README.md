# Commercial Loss Given Default & Recovery Project

This repository is the LGD and recovery layer in the public commercial credit-risk stack. It uses borrower risk context, synthetic facility data, and collateral and recovery assumptions to produce downturn LGD tables, recovery waterfalls, and validation outputs for downstream loss, stress, pricing, and capital workflows.

## What this repo is

This project demonstrates how a bank-style commercial LGD workflow can be presented in a clear, portfolio-ready format. It focuses on severity, recoveries, and downturn treatment using transparent assumptions so the repo is easy to review without needing internal workout data.

## Where it sits in the stack

This repo sits after borrower risk assessment and industry overlay work, and before expected loss, stress testing, pricing, and capital reporting in the public commercial stack.

## Key inputs

- borrower risk outputs from `PD-and-scorecard-commercial`
- industry and macro overlay context from `industry-analysis`
- synthetic facility, collateral, workout, and recovery assumption tables staged under `data/`

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
- `docs/`: methodology, assumptions, data dictionary, validation notes, and the long-form LGD methodology guide
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

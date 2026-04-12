# Methodology - LGD-commercial

1. Load or generate synthetic demo data.
2. Standardise borrower, facility, exposure, collateral, and financial fields.
3. Build utilisation, margin, DSCR, leverage, liquidity, working-capital, and collateral coverage features.
4. Run the `lgd` engine.
5. Validate and export CSV outputs.

## Extended methodology guide

The detailed banking-style methodology guide for this repo lives in `docs/lgd_australian_bank_methodology_guide.md` so the public repo root stays aligned with the rest of the portfolio.

## Output contract

- `outputs/tables/lgd_segment_summary.csv`
- `outputs/tables/recovery_waterfall.csv`
- `outputs/tables/downturn_lgd_output.csv`
- `outputs/tables/lgd_validation_report.csv`

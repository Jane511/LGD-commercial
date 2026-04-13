# Data Dictionary - LGD-commercial

## Purpose

This dictionary captures the key fields used by the integrated LGD framework, with specific coverage for the commercial parent framework and sub-segments.

## Core portfolio/demo fields

| Field | Description |
| --- | --- |
| `borrower_id` | Synthetic borrower identifier (demo pipeline). |
| `facility_id` | Synthetic facility identifier (demo pipeline). |
| `segment` | Portfolio segment (demo pipeline). |
| `industry` | Industry grouping. |
| `product_type` | Facility or product type. |
| `limit` | Approved or committed limit (demo pipeline). |
| `drawn` | Current drawn balance (demo pipeline). |
| `pd` | Demonstration PD input. |
| `lgd` | Demonstration LGD input. |
| `ead` | Exposure at default input/output field. |

## Commercial framework core fields

| Field | Description |
| --- | --- |
| `loan_id` | Facility identifier used across commercial outputs. |
| `facility_type` | `Term Loan`, `Revolving Credit`, `Overdraft`. |
| `security_type` | Security package proxy (`Property`, `PPSR - P&E`, `PPSR - Receivables`, `PPSR - Mixed`, `GSR Only`). |
| `facility_limit` | Committed limit at default observation point. |
| `drawn_balance` | Drawn amount at default observation point. |
| `undrawn_amount` | Undrawn commitment (`facility_limit - drawn_balance`). |
| `ccf` | Baseline conversion factor for undrawn amount. |
| `default_date` | Default date for monitoring and OOT splits. |
| `realised_lgd` | Discounted cashflow-based realised severity proxy. |
| `contract_rate_proxy` | Contract-rate proxy. |
| `cost_of_funds_proxy` | Cost-of-funds proxy. |
| `discount_rate` | `max(contract_rate_proxy, cost_of_funds_proxy)`. |
| `framework_segment` | Standard parent-framework segment assignment. |
| `lgd_base_framework` | Base/economic LGD output. |
| `lgd_downturn_framework` | Downturn LGD output. |
| `lgd_final_framework` | Final regulatory-style LGD output. |

## Segment-specific commercial proxy fields

### Receivables / invoice finance
`eligible_receivables_balance`, `ineligible_receivables_pct`, `ageing_over_90d_pct`, `dilution_pct`, `collections_control_score`, `advance_rate`.

### Trade / contingent facilities
`is_contingent_flag`, `claim_probability_base`, `claim_probability_downturn`, `cash_security_pct`, `contingent_commitment_amount`, `ead_from_contingent_base`.

### Asset / equipment finance
`asset_type`, `asset_age_years`, `residual_balloon_pct`, `remarketing_discount_pct`, `secondary_market_liquidity`, `repossession_months`, `sale_timing_months`, `asset_haircut_proxy`.

## Output contracts

Core outputs:

- `outputs/tables/lgd_segment_summary.csv`
- `outputs/tables/recovery_waterfall.csv`
- `outputs/tables/downturn_lgd_output.csv`
- `outputs/tables/lgd_validation_report.csv`

Commercial parent outputs:

- `outputs/tables/commercial_framework_loan_level_output.csv`
- `outputs/tables/commercial_framework_weighted_lgd_by_segment.csv`
- `outputs/tables/commercial_framework_base_vs_downturn_comparison.csv`
- `outputs/tables/commercial_framework_weighted_lgd_by_vintage.csv`
- `outputs/tables/commercial_framework_estimate_vs_realised_by_segment.csv`
- `outputs/tables/commercial_framework_validation_checks.csv`

Commercial segment outputs:

- `outputs/tables/receivables_invoice_finance_*.csv`
- `outputs/tables/trade_contingent_*.csv`
- `outputs/tables/asset_equipment_finance_*.csv`

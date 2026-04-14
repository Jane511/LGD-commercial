# LGD-commercial: Integrated LGD Framework (AU Bank-Style)

This repository is an integrated Australian bank-style LGD portfolio project. It produces **base / downturn / final LGD** outputs and recovery metrics across mortgage, cash-flow lending, and property-backed lending products using **transparent proxy assumptions** (no internal workout tape is included).

## What this repo is

- A practical, auditable LGD framework designed to be interview-ready and easy to review.
- A multi-product structure with consistent output contracts so a cross-product view can be built.
- A repo-safe validation standard that is **script-first** (no reliance on interactive notebook plotting).

## LGD concepts used (foundation)

The modelling approach follows a simple but bank-standard LGD decomposition (as reflected in your LGD concept notes):

1. **Start from LGD as a net-recovery concept**

   `LGD = (EAD - NetRecoveryPV) / EAD`

   where `NetRecoveryPV = PV(Recoveries) - PV(Costs)`.

2. **Cure vs non-cure separation (especially important for mortgages)**

   `PV(Recoveries) = P(cure)*PV(Recovery|cure) + (1-P(cure))*PV(Recovery|non-cure)`

3. **Non-cure can resolve via multiple paths** (restructure, sale/liquidation, guarantor, etc.)

   `PV(Recovery|non-cure) = Sum_over_paths P(path|non-cure) * PV(Recovery|path)`

4. **Timing matters**

   Future cashflows are discounted to present value using a product-appropriate discount rate proxy.

5. **Portfolio aggregation is exposure-weighted**

   `WeightedLGD = Sum(LGD_i * EAD_i) / Sum(EAD_i)` (no simple mean LGD).

In this repo, where detailed workout cashflows are not available, the above concepts are implemented with transparent proxies (recovery haircuts, recovery time proxies, explicit costs, and discounting rules).

## Module map (notebooks)

- `02_residential_mortgage_lgd.ipynb`: mortgage LGD (LVR, LMI proxy, cure overlay, foreclosure/liquidation loss if non-cure)
- `03_commercial_cashflow_lgd.ipynb`: parent commercial cash-flow LGD framework
  - term lending (secured/partially secured/unsecured)
  - overdraft/revolver (EAD/CCF sensitivity)
  - integration hooks for receivables / trade / asset finance sub-segments
- `04_receivables_invoice_finance_lgd.ipynb`: receivables / invoice finance (eligible pool, ageing, dilution, collections control, advance rate / EAD headroom)
- `05_trade_contingent_facilities_lgd.ipynb`: trade / contingent (claim conversion to EAD, security / cash backing, tenor, recovery timing)
- `06_asset_equipment_finance_lgd.ipynb`: asset / equipment finance (asset type/age, residual exposure, repossession + remarketing)
- `07_development_finance_lgd.ipynb`: development finance (GRV, completion stage, cost-to-complete, sell-through / exit scenarios)
- `08_cre_investment_lgd.ipynb`: CRE investment (LVR, DSCR, WALE, vacancy, tenant concentration, refinance vs forced sale)
- `09_residual_stock_lgd.ipynb`: residual stock (absorption, discount-to-clear, holding cost, time-to-sale)
- `10_land_subdivision_lgd.ipynb`: land/subdivision (liquidity + market depth, haircut, longer recovery time)
- `11_bridging_loan_lgd.ipynb`: bridging (exit type/certainty, valuation risk, time to exit, failed-exit stress)
- `12_mezz_second_mortgage_lgd.ipynb`: mezz/2nd mortgage (recovery waterfall: collateral value → senior → residual for mezz)
- `13_cross_product_comparison.ipynb`: integrated cross-product comparison (weighted LGD, downturn sensitivity, recovery time, portfolio mix, risk ranking)

## Upstream industry contract (compact)

Industry integration now uses the compact `industry-analysis` export contract under `data/exports/`:

- `industry_risk_scores.parquet` (required)
- `macro_regime_flags.parquet` (required)
- `downturn_overlay_table.parquet` (required)
- `property_market_overlays.parquet` (optional; required only for property-market overlay logic)

Legacy broad CSV dependencies were removed from integration code:

- `industry_base_risk_scorecard.csv`
- `industry_generated_benchmarks.csv`
- `industry_working_capital_risk_metrics.csv`
- `industry_stress_test_matrix.csv`
- `industry_credit_appetite_strategy.csv`
- `industry_esg_sensitivity_overlay.csv`
- `concentration_limits.csv`

Current behaviour is intentionally compact and LGD-specific:

- risk score / risk level / WC overlay / debt-to-EBITDA / ICR / ESG fields are sourced from `industry_risk_scores.parquet` when present
- stress matrix is derived from `downturn_overlay_table.parquet` + `macro_regime_flags.parquet`
- credit appetite and concentration limits are derived internally as lightweight LGD helpers (not full strategy rebuilds)

Overlay/parameter governance (versioned):

- canonical parameter table: `data/config/overlay_parameters.csv`
- manifest guardrail: `data/config/overlay_parameters_manifest.json`
- every run emits parameter version/hash and overlay trace reports

## Output contract (what to expect)

`outputs/tables/` intentionally contains multiple tables per module (loan-level, segment summaries, sensitivities, validation checks, monitoring). Key canonical outputs:

- Cross-product comparison: `outputs/tables/cross_product_comparison.csv`
- Portfolio final layer: `outputs/tables/lgd_final.csv`, `outputs/tables/lgd_final_summary_by_product.csv`
- Governance/validation: `outputs/tables/policy_parameter_register.csv`, `outputs/tables/validation_sequence_report.csv`
- Stage-specific checks: `outputs/tables/bridging_stage7_validation_report.csv`, `outputs/tables/stage9_cross_product_validation_report.csv`
- Overlay/governance checks: `outputs/tables/overlay_trace_report.csv`, `outputs/tables/parameter_version_report.csv`, `outputs/tables/segmentation_consistency_report.csv`, `outputs/tables/reproducibility_determinism_report.csv`

Note: some historical duplicate CSV exports were removed so the repo exports one canonical filename per table.

## Repo-safe execution and validation

Run the end-to-end demo pipeline:

```powershell
python scripts/run_demo_pipeline.py
```

Repo-safe validation standard (script-first):

```powershell
python scripts/run_validation_sequence.py
```

Validation sequence with explicit source selection:

```powershell
python scripts/run_validation_sequence.py --source generated
python scripts/run_validation_sequence.py --source controlled --controlled-root data/controlled
```

Stage 7 (Bridging Loan) repo-safe validation (non-interactive plotting backend, `plt.show()` disabled):

```powershell
python scripts/run_stage7_bridging_validation.py
```

Stage 9 (Cross-product integration) repo-safe validation:

```powershell
python scripts/run_stage9_cross_product_validation.py
```

## New-loan scoring interface (API + CLI)

Python API (single loan):

```python
from src.lgd_scoring import score_single_loan

result = score_single_loan(
    payload={
        "loan_id": "L-1001",
        "ead": 175000,
        "realised_lgd": 0.31,
        "lmi_eligible": 1,
        "mortgage_class": "Standard",
    },
    product_type="mortgage",
    scenario_id="baseline",
)
```

CLI (single loan JSON):

```powershell
python scripts/score_new_loan.py --product-type mortgage --single-json data\sample_loan.json --output outputs\tables\single_loan_scored.json
```

CLI (batch CSV):

```powershell
python scripts/score_new_loan.py --product-type commercial --input-csv data\sample_commercial_loans.csv --output outputs\tables\commercial_scored.csv
```

CLI (score directly from generated/controlled source adapters):

```powershell
python scripts/score_new_loan.py --product-type development --batch-from-source --source-mode generated --output outputs\tables\development_scored.csv
python scripts/score_new_loan.py --product-type mortgage --single-json data\sample_loan.json --use-source-template --source-mode controlled --controlled-root data/controlled --output outputs\tables\mortgage_scored.json
```

Normalized scoring outputs include:
- `loan_id`, `product_type`, `lgd_base`, `lgd_downturn`, `lgd_final`
- overlay trace: `macro_downturn_scalar`, `industry_downturn_adjustment`, `combined_downturn_scalar`
- provenance: `overlay_source`, `parameter_version`, `parameter_hash`, `scenario_id`, `source_mode`

## Data-source swap checklist (generated -> controlled systems)

Goal: keep model structure fixed and only swap input source + recalibration assets.

1. Prepare canonical controlled input templates:

```powershell
python scripts/prepare_controlled_input_templates.py --output-root data/controlled/templates
```

2. Populate controlled-system files in `data/controlled/` with canonical names:
- `{product}_loans.csv` or `.parquet`
- `{product}_cashflows.csv` or `.parquet`
- products: `mortgage`, `commercial`, `development`, `cashflow_lending`

3. Run with controlled source using the same pipeline contract:

```powershell
python scripts/run_pipeline_with_source.py --source controlled --controlled-root data/controlled --include-reporting
```

4. Verify governance and determinism outputs are still produced:
- `overlay_trace_report.csv`
- `parameter_version_report.csv`
- `segmentation_consistency_report.csv`
- `run_metadata_report.csv`
- `reproducibility_determinism_report.csv`

5. Recalibrate parameters/models on controlled data; do not change core pipeline interfaces unless required by approved model governance.

Adapter entry points:
- `src/data_source_adapter.py::load_datasets(...)`
- `src/data_source_adapter.py::validate_dataset_contract(...)`

## Documentation and governance

Start here:

- `docs/methodology_cashflow_lending.md` (cash-flow lending training manual)
- `docs/methodology_property_backed_lending.md` (property-backed lending manual)
- `docs/data_dictionary.md` (key input/output fields and output table catalogue)

Documentation policy (portfolio repo):

- Keep documentation concise and implementation-linked.
- Prefer one canonical section per topic rather than multiple note files.

## Logging and diagnostics

The pipeline emits structured `logging` output at key decision points. To see it, configure logging before running scripts:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

Key log events include:

- **Overlay parameter load** — version, row count, and hash prefix on every run
- **File I/O** — file path and shape (rows × columns) on successful load; descriptive `FileNotFoundError` / `ValueError` if a file is missing or corrupt
- **Numeric coercion** — `WARNING` if any `pd.to_numeric(..., errors="coerce")` call silently converts values to NaN
- **Explicit defaults** — `WARNING` when a required column is absent or NaN rows are filled with a schema default
- **Segmentation** — `WARNING` when any segment column falls back to `"Unknown"` because the source column is missing
- **Vintage / OOT fallback** — `INFO` summary of how many rows used each origination-date tier (observed / seasoning / proxy); `WARNING` if the constant years-on-book fallback is used
- **Discount-rate fallback** — `WARNING` when tier-4 or tier-5 fallbacks are used (no rate data available)
- **Post-overlay validation** — hard `ValueError` if `lgd_final` is outside [0, 1]; `WARNING` if portfolio mean LGD is implausible or downturn scalar direction is contradicted by floor interactions

To run `lgd_final` as a standalone script:

```powershell
python -m src.lgd_final
```

## Limitations (portfolio project)

- All portfolio data is synthetic and included for demonstration only.
- Recovery timing, cure overlays, and downturn logic use transparent proxies; they are not calibrated to internal workout datasets.
- Vintage/out-of-time validation is simulated using proxy origination-year logic when observed origination dates are unavailable. Use `require_observed=True` in `add_vintage_columns()` to enforce observed origination dates in production contexts.
- This is not a production model approval pack; it is an integrated portfolio framework for discussion and demonstration.

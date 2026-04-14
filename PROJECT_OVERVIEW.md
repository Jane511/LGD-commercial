# Project Overview - LGD-commercial

`LGD-commercial` is the LGD, recoveries, and validation layer in an integrated public credit-risk portfolio stack.

It is designed to be:

- practical and auditable (bank-style, not academic)
- cross-product (consistent output contracts across products)
- portfolio-ready (clear assumptions, clear limitations)
- reproducible (script-first validation, non-interactive notebook handling)

## 1. Portfolio role

This repo converts borrower risk context, facility structure, collateral assumptions, cure / workout logic, and stress drivers into:

- loan-level and segment-level **base / downturn / final LGD** outputs
- recovery timing metrics (product-appropriate)
- governance and validation reporting (fallback usage, proxy flags, cure overlay reporting)
- cross-product comparison outputs (LGD, sensitivity, recovery time, portfolio mix)

## 2. LGD concepts used (from the concept notes)

This repo implements a simplified version of the standard LGD components, using transparent proxies where detailed workout cashflows are not available.

### 2.1 Core definition

LGD is defined as a loss fraction of exposure at default:

`LGD = (EAD - NetRecoveryPV) / EAD`

where:

- `EAD` is the exposure at default (including CCF effects where relevant)
- `NetRecoveryPV` is the present value of recoveries net of costs

### 2.2 Decomposing net recovery

`NetRecoveryPV = PV(Recoveries) - PV(Costs)`

This is why discounting (timing) and costs matter in bank LGD logic.

### 2.3 Cure vs non-cure (two-stage view)

For products where cure is meaningful (especially mortgages), recovery is split into cure and non-cure components:

`PV(Recoveries) = P(cure)*PV(Recovery|cure) + (1-P(cure))*PV(Recovery|non-cure)`

In this repo’s mortgage module, cure probability is modelled using proxy drivers (LVR, arrears stage, borrower type / behaviour proxies). Final LGD follows:

`LGD_final = (1 - P(cure)) * LGD_liquidation`

### 2.4 Non-cure resolution paths

If a loan does not cure, it can still resolve via multiple pathways (restructure, liquidation/forced sale, guarantor, etc.). Expected non-cure recovery is path-weighted:

`PV(Recovery|non-cure) = Sum_over_paths P(path|non-cure) * PV(Recovery|path)`

In the portfolio project, “paths” are implemented as simple scenario logic and overlays by product, with assumptions documented in the methodology manuals.

### 2.5 Timing and discounting

Recoveries received later are worth less. The repo uses product-appropriate discount-rate logic in line with the policy position:

`discount_rate = max(contract_rate_proxy, cost_of_funds_proxy)`

and applies discounting to recovery timing proxies (for example months-to-sale / months-to-recovery).

### 2.6 Exposure-weighted aggregation

All portfolio-level summaries are exposure-weighted:

`WeightedLGD = Sum(LGD_i * EAD_i) / Sum(EAD_i)`

This is the repo standard (simple average LGD is not used for portfolio reporting).

## 3. What is included (product modules)

Reviewer notebooks live under `notebooks/` and export canonical tables under `outputs/tables/`.

- Residential mortgage: cure + liquidation if non-cure
- Commercial cash-flow lending framework:
  - term lending (secured/partially secured/unsecured)
  - overdraft/revolver (explicit EAD/CCF uplift logic)
  - receivables / invoice finance sub-segment
  - trade / contingent facilities sub-segment
  - asset / equipment finance sub-segment
- Property-backed secured modules:
  - development finance
  - CRE investment
  - residual stock
  - land / subdivision
  - bridging loans
  - mezzanine / 2nd mortgage (recovery waterfall)
- Cross-product framework:
  - consistent comparison across products (weighted LGD, downturn sensitivity, recovery time, portfolio mix, risk ranking)

## 4. Upstream inputs and downstream consumers

Upstream inputs (context providers):

- `PD-and-scorecard-commercial`
- `industry-analysis` compact contract via `data/exports/` parquet files:
  - `industry_risk_scores.parquet`
  - `macro_regime_flags.parquet`
  - `downturn_overlay_table.parquet`
  - `property_market_overlays.parquet` (optional; property-backed overlay only)

Downstream consumers (typical users of LGD outputs):

- `expected-loss-engine-commercial`
- `stress-testing-commercial`
- `RAROC-pricing-and-return-hurdle`
- `RWA-capital-commercial`

## 5. Governance, validation, and reproducibility

This repo avoids “run all notebook cells” as a validation standard. The baseline is script-first execution.

Commands:

```powershell
python scripts/run_demo_pipeline.py
python scripts/run_validation_sequence.py
python scripts/run_stage7_bridging_validation.py
python scripts/run_stage9_cross_product_validation.py
```

Validation outputs include:

- fallback usage and counts (so governance reviewers can see where proxies/fallbacks were applied)
- proxy arrears / behavioural flags and cure overlay flags (where used)
- year-bucket macro fallback reporting
- structural checks (non-empty outputs, required columns present, plausible ranges)

## 6. Documentation set

- `docs/methodology_cashflow_lending.md` (training manual)
- `docs/methodology_property_backed_lending.md` (manual)
- `docs/data_dictionary.md` (fields and outputs)

## 7. Limitations and calibration status

- This is a portfolio/demo repo with synthetic data and proxy assumptions.
- Downturn logic is linked to simple stress drivers, but is not calibrated to internal macro models.
- The repo is intended to demonstrate an integrated LGD framework, not to replicate a bank’s production calibration pack.

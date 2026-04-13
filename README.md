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

## Output contract (what to expect)

`outputs/tables/` intentionally contains multiple tables per module (loan-level, segment summaries, sensitivities, validation checks, monitoring). Key canonical outputs:

- Cross-product comparison: `outputs/tables/cross_product_comparison.csv`
- Portfolio final layer: `outputs/tables/lgd_final.csv`, `outputs/tables/lgd_final_summary_by_product.csv`
- Governance/validation: `outputs/tables/policy_parameter_register.csv`, `outputs/tables/validation_sequence_report.csv`
- Stage-specific checks: `outputs/tables/bridging_stage7_validation_report.csv`, `outputs/tables/stage9_cross_product_validation_report.csv`

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

Stage 7 (Bridging Loan) repo-safe validation (non-interactive plotting backend, `plt.show()` disabled):

```powershell
python scripts/run_stage7_bridging_validation.py
```

Stage 9 (Cross-product integration) repo-safe validation:

```powershell
python scripts/run_stage9_cross_product_validation.py
```

## Documentation and governance

Start here:

- `docs/methodology_cashflow_lending.md` (cash-flow lending training manual)
- `docs/methodology_property_backed_lending.md` (property-backed lending manual)
- `docs/data_dictionary.md` (key input/output fields and output table catalogue)

Documentation policy (portfolio repo):

- Keep documentation concise and implementation-linked.
- Prefer one canonical section per topic rather than multiple note files.

## Limitations (portfolio project)

- All portfolio data is synthetic and included for demonstration only.
- Recovery timing, cure overlays, and downturn logic use transparent proxies; they are not calibrated to internal workout datasets.
- Vintage/out-of-time validation is simulated using proxy origination-year logic when observed origination dates are unavailable.
- This is not a production model approval pack; it is an integrated portfolio framework for discussion and demonstration.

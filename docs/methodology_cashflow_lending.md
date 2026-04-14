# Cash-Flow Lending Methodology Manual

This document explains the cash-flow lending LGD framework in plain language and links each method step to practical banking logic.

## Section 6-8 Delta Summary (Current Version)

| Section | Delta added | Where to read |
| --- | --- | --- |
| 6 | Current component-to-model implementation map (5.1-5.8), plus upstream industry contract and derived fields used | Section `5.9` |
| 7 | Additional generated governance and traceability datasets for overlays, parameters, segmentation, metadata, reproducibility, and strict gap audit | Section `7.1` |
| 8 | Explicit strict-all-options status linkage for Section-5 components with remediation source reference | Section `8.5` |

## 1. What is cash-flow lending LGD

Cash-flow lending LGD estimates the proportion of exposure likely to be lost after a borrower defaults, after considering:

- exposure at default (EAD)
- recoveries received over time
- recovery and legal costs
- discounting of delayed recoveries
- downturn conditions

Core formula concept:

`LGD = discounted_loss / EAD`

where discounted loss reflects net present value of recoveries and costs.

## 2. Products in scope

Cash-flow lending products covered in this portfolio:

1. SME / middle-market term lending
2. Overdraft / revolving working-capital lines
3. Receivables / invoice finance
4. Trade / contingent facilities (for example bank guarantees)
5. Asset / equipment finance

Why these are separated:

- repayment source differs
- EAD behaviour differs (drawn vs undrawn vs contingent)
- recovery path differs
- timing and cost profile differs

## 3. Policy baseline used in this repo

1. Exposure-weighted LGD aggregation: `Sum(LGD*EAD)/Sum(EAD)`

2. Discount-rate rule: `discount_rate = max(contract_rate_proxy, cost_of_funds_proxy)`

3. Downturn LGD linked to stress drivers (not flat unexplained overlays)

4. Transparent proxy assumptions where real workout data is unavailable

## 4. Data needed by component

### 4.1 Facility and product data

Needed for product segmentation and EAD behaviour.

Typical fields:

- facility/customer ID
- product type
- secured/unsecured flag
- security type
- facility limit, drawn, undrawn
- tenor
- origination date, default date

### 4.2 Borrower financial data

Needed for repayment capacity and distress signals.
Typical fields:

- revenue / turnover
- EBITDA proxy
- leverage
- DSCR / ICR
- working-capital indicators
- watchlist / covenant signals
- industry and size bands

### 4.3 Workout and recovery data (core LGD calibration tape)

Needed to estimate realised severity.
Typical fields:

- EAD at default
- recovery cashflows (amount and date)
- write-offs
- restructure / settlement outcomes
- resolution type (cure, restructure, liquidation, write-off)
- final resolution date

### 4.4 Recovery costs

Needed because LGD is based on net recoveries.
Typical fields:

- legal costs
- receiver/administrator costs
- enforcement/collection costs
- valuation and disposal costs
- holding costs

### 4.5 Time-to-resolution

Needed for discounting and delay modelling.
Typical fields:

- time to first recovery
- time to final recovery
- enforcement start
- total resolution duration

### 4.6 Collateral/support data

Still relevant as secondary support in cash-flow lending.
Typical fields:

- collateral type and coverage
- guarantee type and support strength
- PPSR/GSA flags
- receivables pool metrics (if invoice finance)
- asset class (if asset finance)

### 4.7 Macro and environment data

Used for downturn calibration/overlays.
Typical fields:

- default year/quarter
- unemployment
- rates/cash-rate context
- insolvency environment
- industry stress context

### 4.8 Validation and monitoring fields

Needed for backtesting and governance reporting.
Typical fields:

- origination vintage
- default vintage
- risk grade
- model-estimated LGD at default
- realised LGD at workout completion

## 5. Component models used

This section expands the component design in your provided cash-flow PDFs and maps each component to:

1. historical datasets needed
2. practical model choices
3. expected outputs

### 5.1 Component 1: EAD at default and conversion

Business question:

How much exposure is actually at risk when default occurs?

Historical datasets/drivers:

- facility/product data: product type, facility structure, secured/unsecured, tenor
- exposure data: limit, drawn balance, undrawn commitment, contingent amount
- utilisation behaviour: utilisation trend before default, drawing pattern near stress
- default event data: default date, default trigger

Model options:

- Term loans:
  - direct observed EAD at default
  - simple adjustment factors if accrual/add-ons are needed
- Revolvers/overdraft/contingent:
  - segment CCF averages
  - GLM/linear model on conversion drivers
  - logistic/bounded conversion model when behaviour is nonlinear

Outputs:

- EAD estimate
- CCF estimate for undrawn/contingent part
- segment-level EAD conversion summary

### 5.2 Component 2: Cure probability

Business question:

What is the probability the exposure returns to performing and avoids full liquidation workout?

Historical datasets/drivers:

- arrears status/progression
- borrower financial strength (DSCR/ICR, leverage, profitability, liquidity)
- watchlist/covenant and delinquency behaviour
- security support and guarantee support
- product type and industry stress context

Model options:

- logistic regression (standard bank approach when sample is sufficient)
- scorecard/table segmentation when sample is thin
- constrained expert overlay where hard data is limited

Outputs:

- loan-level cure probability
- segment-level cure rates
- cure vs non-cure split for downstream LGD build

### 5.3 Component 3: Non-cure severity (loss given liquidation)

Business question:
If no cure occurs, what loss severity is expected after net recoveries?

Historical datasets/drivers:

- EAD at default
- realised recovery cashflows and source mix (operations, collateral sale, guarantee, insurance, legal settlement)
- recovery costs (legal, enforcement, disposal, holding)
- security type and collateral coverage
- guarantee effectiveness
- borrower weakness and industry conditions

Model options:

- GLM/regression severity model
- beta regression (bounded LGD in [0,1])
- segment-level exposure-weighted averages where data is limited

Outputs:

- non-cure LGD estimate
- severity by product/security/borrower segment

### 5.4 Component 4: Recovery timing

Business question:

How long does it take to recover cash and close the case?

Historical datasets/drivers:

- default date
- recovery dates (first and final)
- enforcement/legal milestone dates
- workout path and resolution type
- collateral and product characteristics

Model options:

- survival/hazard model (time-to-event)
- duration regression
- segment-level time-to-resolution averages for small samples

Outputs:

- expected time to first recovery
- expected time to final resolution
- timing distribution used for discounting

### 5.5 Component 5: Recovery cost rate

Business question:

What costs reduce gross recoveries and increase LGD?
Historical datasets/drivers:

- legal costs
- receiver/administrator costs
- collection/enforcement costs
- valuation/disposal/remarketing costs
- workout path and product type

Model options:

- segment historical average cost rate
- regression by workout type and complexity when volume allows

Outputs:

- expected cost rate
- expected cost by workout path/segment

### 5.6 Component 6: Final realised economic LGD

Business question:
What is the realised discounted loss after timing and cost effects?

Historical datasets/drivers:

- EAD at default
- full recovery cashflow history (amount and timing)
- full recovery-cost history (amount and timing)
- discount-rate inputs

Calculation approach:

1. discount recovery and cost cashflows to default date
2. compute net present recoveries
3. compute discounted economic loss and LGD

Model options:

- direct realised LGD calculation first
- optional second-stage model (GLM/beta/segment tables) to generalise for estimation at default

Outputs:

- loan-level realised economic LGD
- segment long-run weighted LGD baseline

### 5.7 Component 7: Downturn LGD and conservatism

Business question:
What LGD should be used under stressed macro/credit conditions?

Historical datasets/drivers:

- realised LGD by period/vintage
- macro stress context (unemployment, rates, insolvency pressure, industry conditions)
- stress-period recovery delays and cost inflation
- segment-specific stress channels (conversion uplift, weaker recoveries, higher haircuts)

Model options:

- downturn window calibration (stress-period weighted severity)
- macro-linked overlay model by segment/product
- policy floors and margin of conservatism add-ons

Outputs:

- downturn LGD
- final regulatory-style LGD (downturn + conservatism/floor logic)
- downturn sensitivity reporting by segment

### 5.8 How the components form product LGD (end-to-end)

This is the assembly logic used for cash-flow lending LGD.

Step 1: Estimate exposure at risk

- Use Component 1 to estimate product-level `EAD` (including CCF/conversion for undrawn or contingent amounts).

Step 2: Split the default path

- Use Component 2 to estimate `P(cure)`.
- Define non-cure probability as `1 - P(cure)`.

Step 3: Estimate non-cure loss severity

- Use Component 3 to estimate liquidation/non-cure severity.
- Use Components 4 and 5 to adjust severity for timing and cost effects.

Step 4: Calculate economic LGD

- Use Component 6 to convert recoveries/costs into discounted realised/economic LGD.
- Practical formulation:
  - `LGD_economic = P(cure) * LGD_if_cure + (1 - P(cure)) * LGD_if_non_cure`
  
  - In simplified implementations where cure loss is near zero:
    - `LGD_economic ~= (1 - P(cure)) * LGD_if_non_cure`

Step 5: Apply downturn and conservatism

- Use Component 7 to produce downturn LGD and final regulatory-style LGD (including policy floors/MoC where required).

Detailed mechanics for Step 5:

1. Start from economic/base LGD
   - Begin with `LGD_economic` from Steps 1-4 (already reflecting timing and net-cost effects).

2. Translate stress drivers into a downturn adjustment
   - Build segment-specific stress drivers, for example:
     - term lending: weaker cashflow serviceability, weaker collateral support, longer recovery duration
     - revolver/overdraft: higher stressed CCF/utilisation, weaker borrower cash conversion, longer workout
     - receivables: lower eligible pool %, higher ageing/dilution, weaker collections control, slower collections
     - trade/contingent: higher claim conversion and weaker security support
     - asset/equipment: larger remarketing haircuts, weaker secondary liquidity, longer repossession/sale time
   - Convert these into a downturn uplift (scalar or additive overlay) with transparent policy parameters.

3. Compute downturn LGD
   - Typical scalar form:
     - `LGD_downturn = clip(LGD_economic * DownturnScalar, 0, 1)`
   - Typical additive form (where used):
     - `LGD_downturn = clip(LGD_economic + DownturnAddon, 0, 1)`
   - Use one standard approach per segment to keep the framework auditable and explainable.

4. Apply Margin of Conservatism (MoC)
   - Add a conservative adjustment to address model/data uncertainty:
     - `LGD_after_MoC = clip(LGD_downturn + MoC_addon, 0, 1)`
   - MoC can be segment-dependent (for example higher where data is thinner or behaviour is more volatile).

5. Apply policy/supervisory floor
   - Enforce a minimum LGD by segment/security hierarchy:
     - `LGD_final = max(LGD_after_MoC, LGD_floor)`
   - Floors are used to avoid implausibly low outcomes and maintain prudential consistency.

6. Preserve application order and governance
   - Recommended order:
     1) economic LGD
     2) downturn adjustment
     3) MoC
     4) floor
   - Keep this order stable across runs for reproducibility and governance review.

7. Validate directional behaviour
   - Confirm `LGD_downturn >= LGD_economic`.
   - Confirm `LGD_final >= LGD_downturn` when MoC/floor applies.
   - Monitor downturn sensitivity by segment in percentage points.

8. Report outputs for model review
   - For each segment/product, export:
     - `ead_weighted_lgd_base` (economic)
     - `ead_weighted_lgd_downturn`
     - `ead_weighted_lgd_final`
     - `downturn_sensitivity_pp` and `final_minus_base_pp`

Step 6: Aggregate to portfolio views

- Aggregate with exposure weighting (not simple averages):
  - `Weighted LGD = Sum(LGD_i * EAD_i) / Sum(EAD_i)`

How this maps by cash-flow product:

1. Term lending:
   - usually stable EAD, stronger focus on cure/non-cure severity and timing/cost.

2. Overdraft/revolver:
   - stronger EAD/CCF component because undrawn utilisation can increase near default.

3. Receivables finance:
   - stronger dependence on collateral quality drivers (ageing, dilution, eligibility, collections controls) in severity and timing.

4. Trade/contingent:
   - strongest dependence on contingent conversion to funded exposure before severity is applied.
   
5. Asset/equipment finance:
   - strongest dependence on repossession, remarketing haircut, liquidity, and disposal timing/cost.

This component chain is what turns product-level behaviour into consistent, auditable cash-flow lending LGD outputs.

### 5.9 Component Construction Logic

This section explains the practical construction path for each component using:
1. upstream `industry-analysis` outputs
2. repo-generated/internal datasets
3. model or rule method
4. base-to-downturn calculation flow

Common upstream industry inputs (current contract):
- `industry_risk_scores.parquet`
- `macro_regime_flags.parquet`
- `downturn_overlay_table.parquet`
- `property_market_overlays.parquet` (optional in cash-flow; mainly property use)

Common generated/internal datasets used by cash-flow modules:
- product loan panels from `src/data_generation.py` (`commercial`, `cashflow_lending`)
- cashflow/recovery assumptions (`*_cashflows`, `workout_months`, proxy cost and timing drivers)
- intermediate segment tags and scenario flags in `src/lgd_calculation.py`

#### 5.9.1 Component 5.1 EAD at default and conversion
- Upstream used: `industry_risk_scores` (industry risk level used as risk context), `downturn_overlay_table` (stress add-on/scalar context)
- Generated/internal used: limits, drawn/undrawn, contingent exposures, utilisation proxies, product flags
- Implemented in:
  - `notebooks/03_commercial_cashflow_lgd.ipynb`
  - `notebooks/04_receivables_invoice_finance_lgd.ipynb`
  - `notebooks/05_trade_contingent_facilities_lgd.ipynb`
  - `notebooks/06_asset_equipment_finance_lgd.ipynb`
  - `src/lgd_calculation.py` (`CommercialLGDEngine`, `CashFlowLendingLGDEngine`)
- Method: rule-based EAD + proxy CCF mapping by product/segment; weighted segment averages where required
- Calculation flow:
  1. Base EAD from observed drawn amount (or funded equivalent for contingent products)
  2. Conversion on undrawn/contingent via CCF proxy
  3. `EAD_total = Drawn + (Undrawn * CCF)`
  4. Downturn conversion uplift applied by stress scalar/add-on where configured

#### 5.9.2 Component 5.2 Cure probability
- Upstream used: `industry_risk_scores` and `macro_regime_flags` as stress context inputs to cure overlays
- Generated/internal used: arrears/behaviour proxy flags, DSCR/ICR proxies, segment-level cure assumptions
- Implemented in:
  - `src/lgd_calculation.py` (`CashFlowLendingLGDEngine.apply_overlays`, shared cure/overlay path)
  - `notebooks/03_commercial_cashflow_lgd.ipynb` (commercial proxy cure assumptions)
- Method: segmented cure mapping with proxy overlay (logistic-style option documented but not fully portfolio-wide)
- Calculation flow:
  1. Assign base cure rate by segment
  2. Apply risk adjustments (industry, behaviour, stress regime)
  3. Bound to `[0,1]`
  4. `P(non_cure) = 1 - P(cure)`

#### 5.9.3 Component 5.3 Non-cure severity
- Upstream used: `industry_risk_scores` (risk-score linked haircut influence)
- Generated/internal used: collateral proxy strength, guarantee support, recovery channel assumptions
- Implemented in:
  - `notebooks/03_commercial_cashflow_lgd.ipynb`
  - `notebooks/04_receivables_invoice_finance_lgd.ipynb`
  - `notebooks/05_trade_contingent_facilities_lgd.ipynb`
  - `notebooks/06_asset_equipment_finance_lgd.ipynb`
  - `src/lgd_calculation.py` (commercial and cashflow severity assembly)
- Method: deterministic proxy severity + segment weighted averaging
- Calculation flow:
  1. Estimate gross recovery rate by segment/security
  2. Apply industry-linked recovery haircut proxy
  3. Convert to severity: `LGD_non_cure = 1 - Recovery_rate_net`

#### 5.9.4 Component 5.4 Recovery timing
- Upstream used: `macro_regime_flags` and `downturn_overlay_table` for stress-delay context
- Generated/internal used: workout month proxies, product timing bands, recovery lag assumptions
- Implemented in:
  - `notebooks/03_commercial_cashflow_lgd.ipynb`
  - `notebooks/04_receivables_invoice_finance_lgd.ipynb`
  - `notebooks/05_trade_contingent_facilities_lgd.ipynb`
  - `notebooks/06_asset_equipment_finance_lgd.ipynb`
  - `src/lgd_calculation.py` (timing-adjusted overlay chain)
- Method: rule-based timing bands (no survival model in current productionized path)
- Calculation flow:
  1. Assign base months-to-recovery by segment
  2. Apply downturn delay scalar/add-on
  3. Use timing factor in economic discounting

#### 5.9.5 Component 5.5 Recovery cost rate
- Upstream used: stress context from `downturn_overlay_table` where cost pressure is reflected in stress settings
- Generated/internal used: legal/enforcement/administration cost proxies by workout path
- Implemented in:
  - `notebooks/03_commercial_cashflow_lgd.ipynb`
  - `notebooks/04_receivables_invoice_finance_lgd.ipynb`
  - `notebooks/05_trade_contingent_facilities_lgd.ipynb`
  - `notebooks/06_asset_equipment_finance_lgd.ipynb`
  - `src/lgd_calculation.py` (net-recovery severity assembly)
- Method: segment-level cost-rate mapping with conservative add-ons
- Calculation flow:
  1. Assign base cost rate per segment/path
  2. Apply stress uplift where required
  3. `Net_recovery = Gross_recovery - Recovery_costs`

#### 5.9.6 Component 5.6 Final realised economic LGD
- Upstream used: indirect via component-level stress context and industry effects
- Generated/internal used: EAD, recovery and cost assumptions, timing and discount rate proxies
- Implemented in:
  - `src/lgd_calculation.py` (`run_full_pipeline`, product `apply_overlays` outputs)
  - `src/lgd_final.py` (portfolio final-layer LGD assembly)
  - `notebooks/03_commercial_cashflow_lgd.ipynb` (commercial framework economic LGD views)
- Method: deterministic discounted-loss assembly
- Calculation flow:
  1. `Recovery_PV = Recovery / (1 + discount_rate)^t`
  2. `Cost_PV = Cost / (1 + discount_rate)^t`
  3. `LGD_base = (EAD - (Recovery_PV - Cost_PV)) / EAD`
  4. Bound output to `[0,1]`

#### 5.9.7 Component 5.7 Downturn LGD and conservatism
- Upstream used: all three core upstream files
  - `macro_regime_flags` -> downturn regime switch
  - `downturn_overlay_table` -> stress scalar/add-on
  - `industry_risk_scores` -> industry downturn adjustment
- Generated/internal used: versioned parameters (`overlay_parameters.csv`), segmentation outputs
- Implemented in:
  - `src/lgd_calculation.py` (`resolve_overlay_contract`, product overlay application)
  - `src/overlay_parameters.py` (parameter loading/validation)
  - `data/config/overlay_parameters.csv` (parameter values)
- Method: shared overlay resolver with deterministic order
- Calculation flow:
  1. Start `LGD_base` (economic)
  2. Apply macro overlay -> `macro_downturn_scalar`
  3. Apply industry adjustment -> `industry_downturn_adjustment`
  4. Combine -> `combined_downturn_scalar`
  5. Apply MoC and floor
  6. `LGD_downturn = clip(LGD_base * combined_downturn_scalar, 0, 1)`
  7. `LGD_final = max(clip(LGD_downturn + MoC, 0, 1), LGD_floor)`

#### 5.9.8 Component 5.8 End-to-end assembly and reporting
- Upstream used: compact industry contract files above
- Generated/internal used: product loan/cashflow panels, segment tags, governance reporting tables
- Implemented in:
  - `src/lgd_calculation.py` (`run_full_pipeline`, `build_governance_reporting_tables`)
  - `scripts/run_validation_sequence.py` (validation and reproducibility reporting)
  - `scripts/build_strict_component_gap_matrix.py` (strict component evidence matrix)
- Method: pipeline orchestration with additive governance outputs
- Calculation flow:
  1. Build base component outputs (EAD, cure, severity, timing, costs)
  2. Construct `LGD_base`
  3. Apply downturn/MoC/floor to get `LGD_downturn` and `LGD_final`
  4. Aggregate exposure-weighted metrics:
     - `Weighted LGD = Sum(LGD_i * EAD_i) / Sum(EAD_i)`
  5. Publish governance trace tables (`overlay_trace_report.csv`, `parameter_version_report.csv`, etc.)

## 6. How this repo implements cash-flow lending

Primary notebooks:

1. `notebooks/03_commercial_cashflow_lgd.ipynb` (parent framework)
2. `notebooks/04_receivables_invoice_finance_lgd.ipynb`
3. `notebooks/05_trade_contingent_facilities_lgd.ipynb`
4. `notebooks/06_asset_equipment_finance_lgd.ipynb`

What each notebook’s code does:

### 6.1 `03_commercial_cashflow_lgd.ipynb` (parent framework)

- loads synthetic commercial loan/cashflow data from shared generators
- engineers core commercial drivers (leverage, DSCR/ICR proxy, margin proxy, watchlist/covenant flags, security status)
- applies standard commercial segment mapping
- builds detailed term-loan and revolver/overdraft LGD logic (base, downturn, final)
- loads sub-segment override outputs (receivables/trade/asset) by `loan_id` when available
- produces integrated weighted LGD comparison across commercial segments
- creates monitoring, vintage, concentration, estimate-vs-realised, and validation tables
- exports parent framework outputs under `outputs/tables/commercial_framework_*.csv`

How this is achieved:
- Logic: engineered borrower/collateral features are mapped into segment rules, then base, downturn, and final LGD are calculated in sequence.
- Methods: ratio features (for example leverage, DSCR/ICR proxies), threshold banding (security/risk classes), and weighted aggregation.
- Formula examples:
  - `LGD_base = (EAD - NetRecoveryPV) / EAD`
  - `Weighted LGD = Sum(LGD_i * EAD_i) / Sum(EAD_i)`
  - `LGD_downturn = clip(LGD_base * DownturnScalar, 0, 1)`

### 6.2 `04_receivables_invoice_finance_lgd.ipynb`

- isolates receivables-led facilities from the base commercial dataset
- creates receivables pool-quality proxies (eligible/ineligible balance, ageing, dilution, debtor concentration, collections control)
- builds EAD logic linked to receivables eligibility and advance-rate/headroom usage
- builds LGD logic linked to pool deterioration, controls leakage risk, collection timing, and costs
- applies downturn stresses (slower collections, weaker eligibility, higher disputes/dilution proxies)
- runs segment-level weighted LGD views (concentration/ageing/advance-rate bands) plus sensitivity
- exports receivables-specific outputs under `outputs/tables/receivables_invoice_finance_*.csv`

How this is achieved:
- Logic: receivables quality is converted into EAD conversion and recovery strength through eligibility, ageing, and dilution proxies.
- Methods: ratio construction (`eligible_balance / total_balance`), risk banding (ageing and concentration buckets), stress multipliers for downturn.
- Formula examples:
  - `Recovery_rate_proxy = Eligible_pool_ratio * (1 - Dilution_rate) * Collection_factor`
  - `LGD_base = 1 - Recovery_rate_proxy`
  - `LGD_downturn = clip(LGD_base * stress_multiplier, 0, 1)`

### 6.3 `05_trade_contingent_facilities_lgd.ipynb`

- isolates trade/contingent proxy segment from the base commercial dataset
- assigns product/transaction proxies (bank guarantee, standby LC, performance bond, etc.)
- builds contingent-to-funded EAD conversion logic (base and downturn claim conversion factors)
- builds LGD logic from cash security, collateral support, customer risk, post-claim timing, and legal/processing costs
- applies downturn stress through higher claim conversion, weaker support effects, and longer resolution timing
- generates weighted LGD by product/security level and conversion sensitivity scenarios
- exports trade/contingent outputs under `outputs/tables/trade_contingent_*.csv`

How this is achieved:
- Logic: contingent obligations are first converted to funded exposure, then adjusted for support quality and expected recovery timing.
- Methods: conversion-factor mapping, rule-based haircut overlays, and scenario scaling.
- Formula examples:
  - `EAD_funded = Contingent_amount * Claim_conversion_factor`
  - `Recovery_rate = Support_strength * (1 - Haircut) * Timing_factor`
  - `LGD = (EAD_funded - Recovery_value) / EAD_funded`

### 6.4 `06_asset_equipment_finance_lgd.ipynb`

- isolates asset/equipment segment from base commercial loans
- assigns asset categories (vehicles, standard equipment, specialised machinery) and asset-risk proxies (age, residual/balloon, condition/location, liquidity)
- builds instalment-style EAD logic with residual exposure and stressed uplift
- builds recovery/LGD logic from repossession timing, remarketing discount, sale proceeds, enforcement/holding cost, and discounting
- applies downturn via larger haircuts, weaker liquidity, and longer repossession-to-sale timelines
- generates weighted LGD by asset type/liquidity and sensitivity scenarios
- exports asset/equipment outputs under `outputs/tables/asset_equipment_finance_*.csv`

How this is achieved:
- Logic: each asset is scored on liquidity, age, and disposal friction; this drives haircut, timing, and cost assumptions.
- Methods: rule-based segmentation, haircut mapping tables, and discounted recovery calculation.
- Formula examples:
  - `Sale_proceeds_proxy = Market_value * (1 - Haircut)`
  - `Recovery_PV = Sale_proceeds_proxy / (1 + discount_rate)^t`
  - `LGD = (EAD - (Recovery_PV - Cost_PV)) / EAD`

Parent segment mapping standard:

1. `PPSR - Receivables` -> Receivables / Invoice Finance
2. `PPSR - P&E` -> Asset / Equipment Finance
3. Overdraft/Revolver + `GSR Only` -> Trade / Contingent Facilities (Proxy)
4. Other Overdraft/Revolver -> Overdraft / Revolver
5. Else -> SME / Middle-Market Term Lending

### 6.5 Relocated Note

The former Section 6.5 component addendum has been moved and expanded to Section `5.9 Component Construction Logic` so component design and component implementation logic are kept together in one place.

## 7. Key outputs

Main cash-flow outputs:

- `outputs/tables/commercial_framework_weighted_lgd_by_segment.csv`
- `outputs/tables/commercial_framework_base_vs_downturn_comparison.csv`
- `outputs/tables/commercial_framework_weighted_lgd_by_vintage.csv`
- `outputs/tables/commercial_framework_estimate_vs_realised_by_segment.csv`
- `outputs/tables/commercial_framework_validation_checks.csv`

Segment-level outputs:

- `outputs/tables/receivables_invoice_finance_*.csv`
- `outputs/tables/trade_contingent_*.csv`
- `outputs/tables/asset_equipment_finance_*.csv`

### 7.1 Additional current-version generated datasets (governance and traceability)

The current version additionally generates:

1. `outputs/tables/overlay_trace_report.csv`
2. `outputs/tables/parameter_version_report.csv`
3. `outputs/tables/segmentation_consistency_report.csv`
4. `outputs/tables/run_metadata_report.csv`
5. `outputs/tables/reproducibility_determinism_report.csv`
6. `outputs/tables/strict_component_gap_matrix.csv`
7. `outputs/reports/strict_component_gap_matrix.md`

## 8. Gaps for this portfolio (explicit)
### 8.1 Data gaps

1. No internal workout tape (historical default and recovery records) is included.
2. No governed internal cure history by product.
3. No real legal timeline and enforcement system extracts.
4. No guaranteed linkage to source-system product processors for trade, receivables, and asset platforms.
5. No full disposal-level asset finance records (sale channel, condition-adjusted realised price, enforcement trail).

### 8.2 Model calibration gaps

Closed in this repo (proxy-level):
1. Structured macro/industry overlay logic is now standardised through a shared overlay resolver and versioned parameter tables.
2. Cross-module parameterisation is now consistent and reproducible (single parameter version/hash across products).
3. Proxy segmentation consistency is now centrally enforced across modules via shared standardized segment tags.

Still open (production):
4. Segment models remain proxy-based and are not calibrated to internal realised default/recovery history.
5. Validation still relies on synthetic proxy outcomes rather than governed production OOT cohorts.

### 8.3 Governance and validation maturity gaps

Closed in this repo (proxy-level):
1. Governance outputs now include `overlay_trace_report.csv`, `parameter_version_report.csv`, `segmentation_consistency_report.csv`, and `reproducibility_determinism_report.csv`.

Still open (production):
2. Independent model validation pack is not included.
3. Model risk approval workflow and challenger model evidence are not included.
4. Data lineage reconciliation to GL and source systems is not included.
5. Recalibration policy thresholds and periodic governance cycles are not included.

### 8.4 Documentation structure gap addressed

central limitations and use considerations should live in one primary methodology location with concise references elsewhere. This split (cash-flow vs property-backed) addresses that structure requirement.

### 8.5 Current strict component status for Section 5

Under strict-all-options assessment (every documented option required for full implementation), components `5.1` to `5.8` in this cash-flow manual are currently `Proxy-only` in `outputs/tables/strict_component_gap_matrix.csv`.

Meaning in current version:
1. The component stages are implemented and feed final LGD outputs.
2. At least one documented option per component remains absent or approximated.
3. Exact remediation tasks and acceptance criteria are maintained in the strict matrix output.

## 9. How inputs are obtained in realities

source these inputs from controlled systems, not proxies:

- loan servicing systems
- collections/workout systems
- legal/enforcement records
- asset disposal and valuation systems
- receivables ledger and dilution controls
- trade/contingent exposure systems

This repo documents that target state but remains a portfolio-grade proxy implementation.

## 10. Use-test statement

This manual is suitable for onboarding, HR review, and interview demonstration.

It is not suitable for production impairment, regulatory capital, or formal model approval without internal data integration, calibration, and independent validation.

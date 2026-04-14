# Property-Backed Lending Methodology Manual

Audience: HR, new team members, analysts, and interview reviewers.

This document explains the property-backed LGD framework in practical bank-style language and links the methodology to this repo's notebook implementation.

## Section 6-8 Delta Summary (Current Version)

| Section | Delta added | Where to read |
| --- | --- | --- |
| 6 | Current component-to-model implementation map (5.1-5.8), plus upstream industry contract and derived fields used | Section `5.9` |
| 7 | Additional generated governance and traceability datasets for overlays, parameters, segmentation, metadata, reproducibility, and strict gap audit | Section `7.1` |
| 8 | Explicit strict-all-options status linkage for Section-5 components with remediation source reference | Section `8.6` |

## 1. What is property-backed lending LGD

Property-backed LGD estimates expected loss severity after default for facilities where collateral value and collateral realisation path are primary recovery drivers.

Core concept:

- start with EAD at default
- model cure and non-cure outcomes
- model resolution path and non-cure severity
- include recovery timing and recovery costs
- discount recoveries/costs to default date
- apply downturn and conservatism overlays

## 2. Property-backed products in scope

1. Residential mortgage
2. Development finance
3. CRE investment
4. Residual stock
5. Land/subdivision
6. Bridging loan
7. Mezzanine/second mortgage

Why product separation matters:

- repayment source differs (salary/rent/sales/refinance/completion)
- collateral behaviour differs (stabilised asset vs project vs land vs subordinated claim)
- workout path differs (cure, refinance, voluntary sale, forced sale, waterfall recovery)
- timing and cost dynamics differ materially

## 3. Policy baseline used in this repo

1. Exposure-weighted LGD aggregation: `Sum(LGD*EAD)/Sum(EAD)`
2. Discount-rate rule: `discount_rate = max(contract_rate_proxy, cost_of_funds_proxy)`
3. Downturn LGD linked to stress drivers, not flat unexplained scalars
4. Cure modelling where relevant (especially mortgage)
5. Transparent proxy assumptions where internal workout data is not available

## 4. Historical data required (property-backed focus)

This section is aligned to the property-backed PDF guidance.

### 4.1 Facility and exposure data

Used for exposure structure and EAD definition.

Typical fields:

- loan/facility ID, borrower ID
- product type and ranking (first mortgage, second mortgage, mezzanine)
- committed limit, drawn, undrawn
- interest rate proxies
- origination date, maturity date, default date
- EAD at default

### 4.2 Borrower and repayment-source data

Used to identify cure/restructure/refinance capacity.

Typical fields:

- borrower type
- income/salary proxy (mortgage)
- rent roll/NOI and DSCR/ICR (CRE)
- leverage
- guarantee support
- arrears stage and hardship/restructure flags
- prior delinquency behaviour

### 4.3 Collateral and valuation data

Core for property-backed severity modelling.

Typical fields:

- collateral type and location
- valuation at origination/default/workout
- LVR and updated LVR bands
- occupancy/vacancy
- WALE/tenant concentration (CRE)
- completion stage, GRV, cost-to-complete, presales (development)
- unsold stock/absorption metrics (residual stock)
- zoning/stage/market depth (land)

### 4.4 Workout and recovery data

Core calibration tape for realised LGD.

Typical fields:

- recovery amounts by date and source
- resolution type (cure, refinance, voluntary sale, forced sale, restructure, write-off)
- write-offs and settlements
- sale dates and realised proceeds
- legal and enforcement milestones

### 4.5 Recovery cost data

Needed for net-recovery LGD.

Typical fields:

- legal costs
- receiver/administrator costs
- valuation and selling costs
- holding costs
- project completion costs (development)
- asset management/disposal costs during enforcement

### 4.6 Timing data

Needed for discounting and duration modelling.

Typical fields:

- default date
- first recovery date
- final recovery date
- sale/refinance date
- enforcement/receiver appointment date

### 4.7 Macro and environment data

Used for downturn calibration and stress overlays.

Typical fields:

- default vintage and period flags
- unemployment and rates context
- property price/valuation stress
- vacancy and cap-rate stress (CRE)
- construction and sell-through stress (development/residual)

### 4.8 Validation and monitoring fields

Used for backtesting and stability governance.

Typical fields:

- origination vintage
- default vintage
- risk grade and segment tags
- model-estimated LGD at default
- realised LGD at workout completion

## 5. Component models used (detailed bank-style)

### 5.1 Component 1: EAD at default

Business question:
How much exposure was at risk when default occurred?

Historical datasets/drivers:

- drawn/undrawn balances
- facility type and redraw/utilisation pattern
- staged draw profile (development)
- contingent/revolving features where present

Model options:

- observed EAD for plain amortising mortgage facilities
- CCF/utilisation model where redraw or staged drawings matter
- segmented averages or GLM/regression when required

Outputs:

- EAD estimate
- CCF estimate where applicable

### 5.2 Component 2: Cure probability

Business question:
What is probability of returning to performing without full liquidation?

Historical datasets/drivers:

- arrears progression and repayment behaviour
- hardship/restructure history
- serviceability metrics and borrower strength
- LVR and collateral support
- borrower/product/property segment

Model options:

- logistic regression (standard practical model)
- segmented cure tables where sample is small
- tree-based challenger models as secondary checks only

Outputs:

- loan-level cure probability
- segment cure-rate table

### 5.3 Component 3: Resolution path model

Business question:
If non-cure, which path is most likely (refinance, voluntary sale, forced sale, restructure)?

Historical datasets/drivers:

- observed resolution type history
- LVR at default and valuation stress
- property-market and refinance conditions
- borrower strength
- completion stage (development)
- debt ranking (mezzanine/second)

Model options:

- multinomial logistic regression
- segmented transition matrices when sample is limited
- tree-based challengers when enough history exists

Outputs:

- probability of each resolution path
- path probability matrix by segment

### 5.4 Component 4: Non-cure severity conditional on path

Business question:
Given non-cure/path, what LGD severity is expected?

Historical datasets/drivers:

- EAD at default
- realised sale proceeds and collateral value
- LVR and guarantee support
- debt ranking/waterfall position
- recovery source mix and net operating strength (CRE)
- completion/cost-to-complete/GRV (development)
- sale discount and cost metrics

Model options:

- GLM/linear regression
- beta regression (bounded LGD)
- segmented weighted averages for thin samples

Outputs:

- conditional non-cure/path severity
- severity by segment and driver band

### 5.5 Component 5: Recovery timing

Business question:
How long does recovery take?

Historical datasets/drivers:

- default date, first/final recovery dates
- sale/refinance dates
- legal milestones and enforcement start
- property type, geography, workout path

Model options:

- survival/hazard models
- duration regression
- segmented timing averages for smaller samples

Outputs:

- expected months to first recovery
- expected months to final resolution
- expected sale/refinance lag

### 5.6 Component 6: Recovery costs

Business question:
What cost rates reduce gross recovery?

Historical datasets/drivers:

- legal, receiver, valuation, selling, holding, completion, and enforcement costs
- workout path and product segment

Model options:

- segmented average cost rates
- regression by path/property type where volume supports

Outputs:

- expected recovery cost rate
- expected cost by segment/path

### 5.7 Component 7: Final realised economic LGD

Business question:
What is final discounted LGD after timing and costs?

Historical datasets/drivers:

- EAD at default
- all recovery cashflows and dates
- all cost cashflows and dates
- discount-rate inputs

Calculation logic:

1. discount recoveries/costs to default date
2. derive net present recoveries
3. compute economic loss and LGD

Component assembly examples:

- `LGD = (1 - P(cure)) * LGD_non_cure`
- or path-weighted: `LGD = Sum(P(path_j) * LGD_path_j)`

Model options:

- direct realised LGD calculation from history
- regression/beta/segmentation as explanatory or predictive layer

Outputs:

- loan-level economic LGD
- long-run weighted LGD by segment

### 5.8 Component 8: Downturn LGD and conservatism

Business question:
What LGD is appropriate in stressed conditions?

Historical datasets/drivers:

- realised LGD by vintage/period
- stress-period recovery timing and cost behaviour
- property price declines
- vacancy and cap-rate stress (CRE)
- development sell-through stress and cost escalation
- widening sale discounts and weaker refinance conditions

Model options:

- stress-window calibration by segment
- macro-linked downturn scalar or additive overlay
- policy floors and Margin of Conservatism (MoC)

Output sequence:

1. `LGD_economic`
2. `LGD_downturn`
3. `LGD_final` after MoC/floor

Outputs:

- downturn LGD
- final regulatory-style LGD
- downturn sensitivity by product/segment

### 5.9 Component Construction Logic (moved from Section 6.9 and expanded)

This section links each property-backed component to the practical implementation flow:
data inputs -> assumptions -> method -> output.

Common upstream industry inputs (current contract):
- `industry_risk_scores.parquet`
- `macro_regime_flags.parquet`
- `downturn_overlay_table.parquet`
- `property_market_overlays.parquet` (property-specific overlay input)

Common generated/internal datasets:
- product loan and cashflow panels from `src/data_generation.py`
- collateral/LVR/workout proxy fields from product notebooks
- shared overlay parameters from `data/config/overlay_parameters.csv`
- governance traces from `src/lgd_calculation.py`

#### 5.9.1 Component 5.1 EAD at default
- Upstream used: `industry_risk_scores` (risk context), `macro_regime_flags` (stress regime context)
- Generated/internal used: drawn/undrawn balance proxies, staged draw assumptions (development), redraw/utilisation flags
- Implemented in:
  - `notebooks/02_residential_mortgage_lgd.ipynb`
  - `notebooks/07_development_finance_lgd.ipynb`
  - `notebooks/11_bridging_loan_lgd.ipynb`
  - `src/lgd_calculation.py` (`MortgageLGDEngine`, `DevelopmentLGDEngine`)
- Method: observed EAD where available; utilisation/CCF proxy mapping for redraw/staged exposures
- Calculation flow:
  1. Base observed EAD from default snapshot
  2. Add conversion for undrawn/staged part
  3. `EAD_total = Drawn + (Undrawn * CCF_proxy)`
  4. Apply stress uplift if downturn regime is active

#### 5.9.2 Component 5.2 Cure probability
- Upstream used: `industry_risk_scores` and `macro_regime_flags`
- Generated/internal used: arrears behaviour, LVR bands, borrower strength proxies, hardship flags
- Implemented in:
  - `notebooks/02_residential_mortgage_lgd.ipynb` (two-stage cure structure)
  - `notebooks/07_development_finance_lgd.ipynb` (development cure proxies)
  - `src/lgd_calculation.py` (cure overlays in mortgage/development pipelines)
- Method: segmented cure mapping with proxy/logistic style treatment in current implementation
- Calculation flow:
  1. Base cure rate by segment
  2. Adjust for stress and risk factors
  3. Clamp to `[0,1]`
  4. `P(non_cure) = 1 - P(cure)`

#### 5.9.3 Component 5.3 Resolution path
- Upstream used: `property_market_overlays`, `macro_regime_flags`, `downturn_overlay_table`
- Generated/internal used: LVR, completion stage, exit-type proxies, debt ranking
- Implemented in:
  - `notebooks/08_cre_investment_lgd.ipynb`
  - `notebooks/11_bridging_loan_lgd.ipynb`
  - `notebooks/12_mezz_second_mortgage_lgd.ipynb`
  - `src/lgd_calculation.py` (property path and overlay integration)
- Method: segmented path allocation (refinance/voluntary sale/forced sale/restructure) with rule-based probabilities
- Calculation flow:
  1. Assign path probabilities by segment
  2. Apply stress overlay to shift probabilities toward adverse paths
  3. Normalize probabilities so they sum to 1

#### 5.9.4 Component 5.4 Non-cure severity conditional on path
- Upstream used: `industry_risk_scores`, `property_market_overlays`
- Generated/internal used: collateral value proxies, haircut settings, waterfall position, path-specific costs
- Implemented in:
  - `notebooks/08_cre_investment_lgd.ipynb`
  - `notebooks/09_residual_stock_lgd.ipynb`
  - `notebooks/10_land_subdivision_lgd.ipynb`
  - `notebooks/12_mezz_second_mortgage_lgd.ipynb`
  - `src/lgd_calculation.py` (property severity assembly)
- Method: path-conditional proxy severity/waterfall logic
- Calculation flow:
  1. Estimate recoverable value by path
  2. Apply collateral haircut and ranking effects
  3. `LGD_path = (EAD - NetRecovery_path) / EAD`
  4. Use weighted path severity in component assembly

#### 5.9.5 Component 5.5 Recovery timing
- Upstream used: `macro_regime_flags`, `downturn_overlay_table`, `property_market_overlays`
- Generated/internal used: sale delay, absorption, enforcement lag proxies
- Implemented in:
  - `notebooks/09_residual_stock_lgd.ipynb`
  - `notebooks/10_land_subdivision_lgd.ipynb`
  - `notebooks/11_bridging_loan_lgd.ipynb`
  - `notebooks/08_cre_investment_lgd.ipynb`
  - `src/lgd_calculation.py` (timing-linked overlay impacts)
- Method: segmented timing assumptions with stress delay factors
- Calculation flow:
  1. Base months-to-recovery by property/workout segment
  2. Stress delay adjustment under downturn regime
  3. Timing factor used for discounting

#### 5.9.6 Component 5.6 Recovery costs
- Upstream used: `downturn_overlay_table` (stress cost pressure context)
- Generated/internal used: legal, holding, disposal and completion-cost proxies by path
- Implemented in:
  - `notebooks/07_development_finance_lgd.ipynb`
  - `notebooks/08_cre_investment_lgd.ipynb`
  - `notebooks/10_land_subdivision_lgd.ipynb`
  - `notebooks/12_mezz_second_mortgage_lgd.ipynb`
  - `src/lgd_calculation.py` (net-recovery cost integration)
- Method: segmented average cost-rate mapping with stress uplift
- Calculation flow:
  1. Base cost rate by segment/path
  2. Downturn uplift on cost rates
  3. `NetRecovery = GrossRecovery - RecoveryCosts`

#### 5.9.7 Component 5.7 Final realised economic LGD
- Upstream used: indirect through component stress inputs above
- Generated/internal used: EAD, path probabilities, severity, timing, and cost components
- Implemented in:
  - `src/lgd_calculation.py` (`run_full_pipeline`, mortgage/development/commercial property outputs)
  - `src/lgd_final.py` (final-layer portfolio LGD outputs)
  - `notebooks/02_residential_mortgage_lgd.ipynb`
  - `notebooks/07_development_finance_lgd.ipynb`
  - `notebooks/08_cre_investment_lgd.ipynb`
- Method: deterministic economic LGD assembly
- Calculation flow:
  1. Compute path-level recoveries/costs and discount to default date
  2. Combine path results:
     - `LGD_economic = Sum(P(path_j) * LGD_path_j)`
  3. Bound to `[0,1]`

#### 5.9.8 Component 5.8 Downturn LGD and conservatism
- Upstream used: all compact upstream files, especially property overlays
- Generated/internal used: overlay parameter tables and product segmentation
- Implemented in:
  - `src/lgd_calculation.py` (`resolve_overlay_contract`, product overlay application)
  - `src/overlay_parameters.py`
  - `data/config/overlay_parameters.csv`
  - `scripts/run_validation_sequence.py` (governance and determinism checks)
- Method: shared resolver with deterministic order (`base -> macro -> industry/property -> MoC -> floor`)
- Calculation flow:
  1. Start `LGD_base` / `LGD_economic`
  2. Apply macro and property/industry downturn adjustments
  3. `LGD_downturn = clip(LGD_base * combined_downturn_scalar, 0, 1)`
  4. Add MoC and enforce floor:
     - `LGD_final = max(clip(LGD_downturn + MoC, 0, 1), LGD_floor)`

## 6. How this repo implements property-backed lending

What each property-backed notebook's code does:

### 6.1 `notebooks/02_residential_mortgage_lgd.ipynb`

- generates/loads mortgage default and workout proxy data
- constructs mortgage drivers (LVR, LMI, arrears/behaviour proxies, borrower type)
- implements two-stage cure framework and non-cure liquidation loss
- applies macro-linked downturn overlays (house-price, unemployment, rate shock channels)
- produces weighted base/downturn/final outputs and governance/validation checks

How this is achieved:
- Logic: split defaults into cure vs non-cure, estimate non-cure loss, then apply downturn overlays.
- Methods: borrower/collateral ratio features (for example LVR), risk banding, weighted averaging, and stress multipliers.
- Formula examples:
  - `LGD_base = (EAD - NetRecoveryPV) / EAD`
  - `LGD_economic = P(cure)*LGD_cure + (1-P(cure))*LGD_non_cure`
  - `LGD_downturn = clip(LGD_base * DownturnScalar, 0, 1)`

### 6.2 `notebooks/07_development_finance_lgd.ipynb`

- builds development-specific drivers (GRV, completion %, cost-to-complete, presale/sell-through proxies)
- models scenario exits (as-is vs complete-and-sell style logic)
- applies stronger downturn impact for development stress channels
- exports segment summaries, scenario summaries, loan-level results, validation checks

How this is achieved:
- Logic: compare exit pathways and select severity based on completion stage, GRV pressure, and cost-to-complete burden.
- Methods: scenario scoring, proxy transition logic, and stressed cost/timing adjustments.
- Formula examples:
  - `LVR_as_if_complete = EAD / GRV`
  - `NetRecovery = SaleProceeds - CostToComplete - WorkoutCosts`
  - `LGD = (EAD - NetRecoveryPV) / EAD`

### 6.3 `notebooks/08_cre_investment_lgd.ipynb`

- builds CRE segments (office/retail/industrial/mixed)
- uses drivers: LVR, DSCR, WALE, vacancy, tenant concentration, cap-rate expansion
- models refinance vs forced-sale resolution path effects
- exports base/downturn/final weighted outputs, resolution-path and sensitivity tables, validation checks

How this is achieved:
- Logic: assign each loan a likely resolution path, then apply path-specific severity and timing assumptions.
- Methods: ratio analysis (LVR, DSCR), thresholding/banding (vacancy, WALE), weighted path combination.
- Formula examples:
  - `P(forced_sale)` increases when LVR is high and DSCR is weak
  - `LGD_economic = Sum(P(path_j) * LGD_path_j)`
  - `Weighted LGD = Sum(LGD_i * EAD_i) / Sum(EAD_i)`

### 6.4 `notebooks/09_residual_stock_lgd.ipynb`

- models completed-but-unsold stock risk
- uses drivers: unsold units, absorption speed, discount-to-clear, holding cost, time to sale
- applies base and stress scenarios with weighted LGD outputs
- exports segment/scenario/loan-level outputs and validation checks

How this is achieved:
- Logic: translate stock overhang and sale friction into slower recovery and deeper effective discounts.
- Methods: proxy scaling (absorption and discount factors), timing adjustments, scenario multipliers.
- Formula examples:
  - `Recovery_rate_proxy = (1 - Discount_to_clear) * Absorption_factor`
  - `Recovery_PV = Recovery / (1 + discount_rate)^t`
  - `LGD = 1 - Recovery_rate_adjusted`

### 6.5 `notebooks/10_land_subdivision_lgd.ipynb`

- models no-income land/subdivision recovery path
- uses zoning/stage, liquidity depth, time-to-sell, value haircut and market depth proxies
- applies longer recovery duration and stronger downturn response
- exports segment/scenario/loan-level outputs and validation checks

How this is achieved:
- Logic: land liquidity depth and planning stage determine haircut intensity and time-to-sale assumptions.
- Methods: rule-based segmentation, haircut ladders, stress scalar uplift.
- Formula examples:
  - `Sale_value_proxy = Land_value * (1 - Haircut)`
  - `Timing_factor = 1 / (1 + discount_rate)^months`
  - `LGD = (EAD - (Sale_value_proxy * Timing_factor - Costs)) / EAD`

### 6.6 `notebooks/11_bridging_loan_lgd.ipynb`

- models exit-risk-driven bridging LGD
- uses exit type, exit certainty, valuation risk, and time-to-exit drivers
- includes delay and failed-exit stress scenarios
- exports bridging delay and scenario outputs with validation checks

How this is achieved:
- Logic: assign an exit-success profile, then apply delay/failure scenarios to recovery and timing.
- Methods: probability-weighted scenario estimation and threshold-based exit certainty bands.
- Formula examples:
  - `Expected_recovery = P(success)*Recovery_success + (1-P(success))*Recovery_failure`
  - `LGD = (EAD - Expected_recovery_PV) / EAD`
  - `Downturn_LGD = LGD_base * stress_multiplier`

### 6.7 `notebooks/12_mezz_second_mortgage_lgd.ipynb`

- implements recovery waterfall logic (collateral -> senior debt -> residual to mezzanine)
- uses total LVR, attachment point, subordination, and value decline drivers
- produces base/downturn/final outputs plus mezz-vs-senior ranking view
- exports waterfall snapshot, segment/scenario outputs, and validation checks

How this is achieved:
- Logic: recoveries are allocated through a ranking waterfall (senior first, mezz residual second).
- Methods: waterfall arithmetic, attachment-point thresholds, stressed collateral decline overlays.
- Formula examples:
  - `Residual_for_mezz = max(Collateral_net - Senior_claim, 0)`
  - `Mezz_recovery_rate = Residual_for_mezz / Mezz_EAD`
  - `Mezz_LGD = 1 - Mezz_recovery_rate`

### 6.8 `notebooks/13_cross_product_comparison.ipynb` (integration layer)

- ingests product outputs and standardises comparison definitions
- compares weighted LGD, downturn sensitivity, recovery time, and portfolio mix
- builds cross-product risk ranking and integrated comparison tables

How this is achieved:
- Logic: convert module outputs to common fields, then compute portfolio-comparable metrics.
- Methods: normalization of labels/segments, EAD-weighted aggregation, ranking by weighted LGD and sensitivity.
- Formula examples:
  - `Weighted_metric = Sum(metric_i * EAD_i) / Sum(EAD_i)`
  - `Downturn_sensitivity_pp = (LGD_downturn - LGD_base) * 100`
  - rank products by weighted final LGD and stress uplift

### 6.9 Relocated Note

The former Section 6.9 component addendum has been moved and expanded to Section `5.9 Component Construction Logic` to keep component design and implementation logic in one place.

## 7. Key outputs (property-backed)

- `outputs/tables/cre_investment_*.csv`
- `outputs/tables/residual_stock_*.csv`
- `outputs/tables/land_subdivision_*.csv`
- `outputs/tables/bridging_*.csv`
- `outputs/tables/mezz_second_mortgage_*.csv`
- `outputs/tables/cross_product_*.csv`

### 7.1 Additional current-version generated datasets (governance and traceability)

1. `outputs/tables/overlay_trace_report.csv`
2. `outputs/tables/parameter_version_report.csv`
3. `outputs/tables/segmentation_consistency_report.csv`
4. `outputs/tables/run_metadata_report.csv`
5. `outputs/tables/reproducibility_determinism_report.csv`
6. `outputs/tables/strict_component_gap_matrix.csv`
7. `outputs/reports/strict_component_gap_matrix.md`

## 8. Gaps and future calibration (property-backed portfolio)

This section integrates the provided `gaps_propetybacked.pdf` guidance.

### 8.1 Data and calibration limitations

1. Stress factors, recovery assumptions, and cure settings are still proxy/demo calibrations.
2. Observed internal workout recoveries, liquidation timelines, and cure outcomes are not used.
3. Some origination/vintage fields are proxy-derived for non-mortgage modules.
4. OOT validation is indicative and should be replaced with real default cohorts.

### 8.2 Methodology gaps

Closed in this repo (proxy-level):
1. Macro/industry overlay logic is now structured via a shared overlay resolver with deterministic precedence.
2. Cross-module parameterisation is now versioned and consistent across mortgage/commercial/development/cashflow engines.
3. Segmentation consistency is now centrally enforced with shared standardized segment tags.

Still open (production):
4. Behavioural detail remains simplified in some modules (for example refinance-vs-forced-sale granularity).
5. Development segment stability still requires recalibration to internal outcomes.
6. Macro sensitivities are not calibrated to internal realised outcomes.
7. Geographic/planning-regime segmentation is not yet fully embedded.
8. Cross-product comparison remains simplified for transparency.

### 8.3 Governance and consistency gaps

Closed in this repo (proxy-level):
1. Standardisation control is now supported by shared overlay traces and parameter-version governance outputs.
2. Code-to-doc consistency checks are reinforced by deterministic reproducibility reporting.

Still open (production):
3. Parameter governance completeness and calibration-status tracking still need production hardening.
4. Formal model authority hierarchy can be strengthened further.

### 8.4 Implementation and technical gaps

1. Full end-to-end reproducibility checks should continue across all modules after each major change.
2. Some tests remain functional/project-grade rather than production-grade validation depth.
3. Environment artefacts/permission issues can still affect local runs.
4. Portfolio-level integration quality should continue to be tightened as modules mature.

### 8.5 Next-step roadmap

1. Replace proxy assumptions with internal workout, cure, and enforcement data.
2. Recalibrate component models (EAD/CCF, cure/path, severity, timing, costs, downturn).
3. Expand governance with stronger parameter control and authority hierarchy.
4. Upgrade validation to fuller OOT and independent model-validation standards.

### 8.6 Current strict component status for Section 5

Under strict-all-options assessment (every documented option required for full implementation), property-backed components `5.1` to `5.8` are currently `Proxy-only` in `outputs/tables/strict_component_gap_matrix.csv`.

Meaning in current version:
1. Component stages exist and are connected to final LGD outputs.
2. One or more documented model options per component remain missing or approximated.
3. Exact remediation tasks and acceptance criteria are maintained per row in the strict matrix output.

## 9. How to obtain these inputs in production

In production environments, these inputs are sourced from controlled systems and governed pipelines:

- loan servicing and collateral systems
- collections/workout platforms
- legal/enforcement systems
- valuation/disposal systems
- model risk governance and independent validation channels

This repo documents that target state, but current implementation is portfolio/project grade.

## 10. Use-test statement

This manual is suitable for onboarding, HR review, and interview demonstration.

It is not suitable for production impairment, capital, or formal regulatory model use without internal data integration, calibration, and independent validation.

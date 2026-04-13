# Property-Backed Lending Methodology Manual

Audience: HR, new team members, analysts, and interview reviewers.

This document explains the property-backed LGD framework in practical bank-style language and links the methodology to this repo's notebook implementation.

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

## 6. How this repo implements property-backed lending

What each property-backed notebook's code does:

### 6.1 `notebooks/02_residential_mortgage_lgd.ipynb`

- generates/loads mortgage default and workout proxy data
- constructs mortgage drivers (LVR, LMI, arrears/behaviour proxies, borrower type)
- implements two-stage cure framework and non-cure liquidation loss
- applies macro-linked downturn overlays (house-price, unemployment, rate shock channels)
- produces weighted base/downturn/final outputs and governance/validation checks

### 6.2 `notebooks/07_development_finance_lgd.ipynb`

- builds development-specific drivers (GRV, completion %, cost-to-complete, presale/sell-through proxies)
- models scenario exits (as-is vs complete-and-sell style logic)
- applies stronger downturn impact for development stress channels
- exports segment summaries, scenario summaries, loan-level results, validation checks

### 6.3 `notebooks/08_cre_investment_lgd.ipynb`

- builds CRE segments (office/retail/industrial/mixed)
- uses drivers: LVR, DSCR, WALE, vacancy, tenant concentration, cap-rate expansion
- models refinance vs forced-sale resolution path effects
- exports base/downturn/final weighted outputs, resolution-path and sensitivity tables, validation checks

### 6.4 `notebooks/09_residual_stock_lgd.ipynb`

- models completed-but-unsold stock risk
- uses drivers: unsold units, absorption speed, discount-to-clear, holding cost, time to sale
- applies base and stress scenarios with weighted LGD outputs
- exports segment/scenario/loan-level outputs and validation checks

### 6.5 `notebooks/10_land_subdivision_lgd.ipynb`

- models no-income land/subdivision recovery path
- uses zoning/stage, liquidity depth, time-to-sell, value haircut and market depth proxies
- applies longer recovery duration and stronger downturn response
- exports segment/scenario/loan-level outputs and validation checks

### 6.6 `notebooks/11_bridging_loan_lgd.ipynb`

- models exit-risk-driven bridging LGD
- uses exit type, exit certainty, valuation risk, and time-to-exit drivers
- includes delay and failed-exit stress scenarios
- exports bridging delay and scenario outputs with validation checks

### 6.7 `notebooks/12_mezz_second_mortgage_lgd.ipynb`

- implements recovery waterfall logic (collateral -> senior debt -> residual to mezzanine)
- uses total LVR, attachment point, subordination, and value decline drivers
- produces base/downturn/final outputs plus mezz-vs-senior ranking view
- exports waterfall snapshot, segment/scenario outputs, and validation checks

### 6.8 `notebooks/13_cross_product_comparison.ipynb` (integration layer)

- ingests product outputs and standardises comparison definitions
- compares weighted LGD, downturn sensitivity, recovery time, and portfolio mix
- builds cross-product risk ranking and integrated comparison tables

## 7. Key outputs (property-backed)

- `outputs/tables/cre_investment_*.csv`
- `outputs/tables/residual_stock_*.csv`
- `outputs/tables/land_subdivision_*.csv`
- `outputs/tables/bridging_*.csv`
- `outputs/tables/mezz_second_mortgage_*.csv`
- `outputs/tables/cross_product_*.csv`

## 8. Gaps and future calibration (property-backed portfolio)

This section integrates the provided `gaps_propetybacked.pdf` guidance.

### 8.1 Data and calibration limitations

1. Stress factors, recovery assumptions, and cure settings are still proxy/demo calibrations.
2. Observed internal workout recoveries, liquidation timelines, and cure outcomes are not used.
3. Some origination/vintage fields are proxy-derived for non-mortgage modules.
4. OOT validation is indicative and should be replaced with real default cohorts.

### 8.2 Methodology gaps

1. Some behaviours remain simplified (for example cure detail and refinance-vs-forced-sale dynamics in some modules).
2. Development segment stability requires targeted recalibration.
3. Macro sensitivities are not calibrated to internal realised outcomes.
4. Geographic/planning-regime segmentation is not yet fully embedded.
5. Cross-product comparison is intentionally simplified for transparency.

### 8.3 Governance and consistency gaps

1. Full methodology standardisation across modules remains a continuing control task.
2. Some policy rules/floors may require tighter consistency checks in code-to-doc implementation.
3. Parameter governance completeness and calibration-status tracking need continued hardening.
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

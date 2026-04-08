# LGD Methodology Guide for Australian Bank Lending Products

## Comprehensive Review & Step-by-Step Build Process

---

# Part A: Review of Current Implementations

## A.1 Implementation Overview

This project contains two LGD implementations:

| Dimension | Implementation 1: APRA Mortgage LGD | Implementation 2: Multi-Product LGD |
|-----------|--------------------------------------|--------------------------------------|
| **Location** | `2. Australian APRA-Style Mortgage LGD Model/` | `LGD/lgd_project_repo/` |
| **Products** | Residential mortgage only | Mortgage, Development, Commercial |
| **Default definition** | Synthetic risk screen (high LTV, low FICO, high DTI) | All loans assumed defaulted |
| **EAD** | `current_actual_upb` with timing adjustment | `principal` directly |
| **Recovery types** | 4 recovery + 4 cost types at cashflow level | Single aggregate recovery stream |
| **Discount rate** | 8% fixed | Random 3-7% per loan |
| **Realised LGD formula** | `(EAD + PV(Costs) - PV(Recoveries)) / EAD` | Same formula |
| **Segmentation** | LTV bucket x FICO bucket, standard/non-standard | Product type only |
| **Long-run average** | Segment average (simple mean) | Segment average (simple mean) |
| **Downturn** | 1.10x scalar | 1.10x scalar |
| **Margin of conservatism** | +2pp | +5pp |
| **APRA overlays** | LMI (80% factor), 10% floor, standard/non-standard, 1.1x scalar | None |
| **Statistical model** | OLS regression with LOOCV + segmented benchmark | None |
| **Validation** | MAE, RMSE, R-squared | None |
| **Sample size** | ~8 defaulted loans | 200 synthetic loans (all defaulted) |

## A.2 Alignment with Australian Bank Practice

### What aligns well (Implementation 1)

- Correct economic loss formula with cashflow-level discounting
- APRA overlay structure (standard/non-standard, LMI, 10% floor, 1.1x scalar)
- Recovery and cost ledger with multiple cashflow types
- Downturn adjustment and margin of conservatism framework
- Capital linkage via illustrative RWA

### Critical gaps across both implementations

1. **No real default/workout data** -- synthetic data limits credibility
2. **No exposure-weighted averaging** -- both use simple means, not `Sum(LGD_i * EAD_i) / Sum(EAD_i)`
3. **No macro-linked downturn** -- flat 1.10x scalar with no analytical justification
4. **No cure modelling** -- mortgages have high cure rates (30-50%); two-stage models are standard
5. **No vintage or cohort analysis** -- no time-series dimension
6. **Discount rate issues** -- 8% is too high for Australian mortgages; should be contract rate or cost of funds (3-5%)
7. **No out-of-time validation** -- no temporal holdout testing
8. **Implementation 2 has zero product differentiation** -- all three products share identical parameter distributions and logic
9. **No commercial-specific framework** -- missing PPSR/GSR security, borrower financials, industry segmentation
10. **No development-specific framework** -- missing completion stage, pre-sales, cost-to-complete, scenario modelling

---

# Part B: Step-by-Step LGD Build Guide

---

# Product 1: Residential Mortgage LGD

## 1.1 Data Requirements

### Origination data
| Field | Description | Source |
|-------|-------------|--------|
| `original_ltv` | Loan-to-value ratio at origination | Loan system |
| `original_cltv` | Combined LTV (if multiple liens) | Loan system |
| `property_value_orig` | Original property valuation | Valuation system |
| `credit_score` | Borrower credit score (e.g., Equifax/Illion) | Bureau |
| `dti` | Debt-to-income ratio | Serviceability assessment |
| `loan_purpose` | Purchase, refinance, equity release | Application |
| `property_type` | House, unit, townhouse, rural | Valuation |
| `property_state` | NSW, VIC, QLD, etc. | Address |
| `occupancy` | Owner-occupier vs investor | Application |
| `loan_type` | P&I, interest-only, offset, redraw | Loan system |
| `lmi_flag` | Whether LMI was taken out | Insurance system |
| `lmi_insurer` | Insurer name and policy details | Insurance system |

### Servicing / performance data
| Field | Description | Source |
|-------|-------------|--------|
| `current_balance` | Outstanding principal balance | Servicing system |
| `accrued_interest` | Unpaid interest at default | Servicing system |
| `delinquency_status` | Days past due | Collections system |
| `months_on_book` | Loan seasoning | Loan system |

### Default and workout data
| Field | Description | Source |
|-------|-------------|--------|
| `default_date` | Date default was triggered | Default management |
| `default_trigger` | 90 DPD, bankruptcy, hardship, etc. | Collections |
| `resolution_type` | Cure, property sale, short sale, write-off | Workout |
| `resolution_date` | Date workout completed | Workout |
| `property_value_at_default` | Indexed or re-appraised value at default | Valuation |
| `sale_price` | Actual property sale proceeds | Settlement |
| `lmi_claim_amount` | Amount recovered from LMI insurer | Insurance |

### Recovery and cost cashflows (one row per cashflow)
| Field | Description |
|-------|-------------|
| `cashflow_date` | Date of recovery or cost |
| `cashflow_type` | Property sale, borrower cure, LMI claim, legal cost, etc. |
| `cashflow_amount` | Dollar amount (positive for recoveries, negative for costs) |

## 1.2 Default Definition

Per **APRA APS 220** and **Basel III**, a mortgage is in default when either:

1. **90 days past due (DPD)** on any material obligation, OR
2. **Unlikely to pay** -- the bank considers the borrower unlikely to fulfil obligations in full without recourse to actions such as realising security

### Cure definition
A defaulted mortgage is "cured" when the borrower:
- Returns to current status (0 DPD) for **at least 3-6 consecutive months** (bank-specific policy)
- All arrears are fully cleared
- No further hardship arrangement is in place

### Re-default treatment
If a cured loan subsequently re-defaults:
- It is treated as a **new default event** for LGD purposes
- **Chain-weighted LGD** may be applied: the recovery from the first default is partially credited, but the second workout's loss is calculated on the re-default EAD

### Implementation note
The current Implementation 1 uses a synthetic risk screen (high LTV + low FICO + high DTI) as a proxy for default identification. In production, defaults should come from the bank's **internal default history** tagged by the collections/workout system.

## 1.3 EAD Measurement

For a **fully-drawn amortising mortgage**:

```
EAD = Outstanding Principal Balance + Accrued Interest + Fees
```

For mortgages with **offset or redraw facilities**:

```
EAD = Drawn Balance + CCF x Undrawn Limit + Accrued Interest
```

Where:
- **CCF (Credit Conversion Factor)** = proportion of undrawn facility expected to be drawn at default
- APRA typically prescribes CCF = 40-100% depending on facility type
- For offset accounts, the net balance (loan minus offset) is the effective exposure

### Australian-specific considerations
- **Redraw facilities**: Borrowers may redraw ahead-of-schedule repayments; the bank must estimate how much redraw will be utilised at default
- **Construction loans** (new builds): Progressive drawdown means EAD may be less than approved limit
- **Interest-only loans**: Higher outstanding balance relative to original principal

### Gap in current implementation
Neither implementation includes accrued interest or fees in EAD. Implementation 1 uses `current_actual_upb` (reasonable proxy). Implementation 2 uses `principal` (origination amount, not balance at default).

## 1.4 Recovery & Cost Framework

### Recovery types (in order of typical magnitude for Australian mortgages)

| Recovery Type | Description | Typical % of Total Recovery |
|---------------|-------------|-----------------------------|
| **Property sale (mortgagee-in-possession)** | Bank exercises power of sale under mortgage deed | 70-90% |
| **Borrower cure** | Borrower returns to current, clears arrears | 30-50% of defaults cure |
| **LMI claim** | Insurer covers shortfall on high-LVR loans | Varies; up to 20% LGD reduction |
| **Short sale** | Borrower sells below debt balance with bank approval | Less common in Australia |
| **Guarantor recovery** | Recovery from guarantor (e.g., parent guarantee) | Case-specific |
| **Rental income during workout** | Rental from investment property during workout | Small, 1-3% of EAD |
| **Other** | Note sale, residual collections | Rare |

### Cost types

| Cost Type | Description | Typical Range (% of EAD) |
|-----------|-------------|--------------------------|
| **Legal costs** | Solicitor fees, court/tribunal costs | 1-3% |
| **Agent commissions** | Real estate agent fees on sale | 2-3% of sale price |
| **Property preservation** | Maintenance, security, insurance during workout | 0.5-2% |
| **Valuation fees** | Updated valuations during workout | 0.1-0.3% |
| **Tax and insurance** | Council rates, strata, insurance advanced by bank | 0.5-1.5% |
| **Internal workout costs** | Bank staff time, systems, overhead | 0.5-1% (often excluded) |

### LMI recovery modelling

In Australian practice, **Lenders Mortgage Insurance (LMI)** is a significant recovery source for high-LVR loans (typically >80% LTV at origination):

**Option A -- LMI as a separate cashflow (preferred)**:
```
LMI_Recovery = min(Economic_Loss_Before_LMI, Policy_Limit) x Claim_Success_Rate
```
Where `Claim_Success_Rate` reflects insurer eligibility, exclusions, and claim settlement history (typically 80-95% for qualifying claims).

**Option B -- LMI as LGD reduction factor (simplified, as in Implementation 1)**:
```
LGD_after_LMI = LGD_before_overlays x (1 - LMI_benefit_factor)
```
Implementation 1 uses `LMI_benefit_factor = 0.20` (i.e., 20% reduction).

**Best practice**: Model LMI as a separate recovery cashflow with its own timing and uncertainty, rather than a flat factor.

## 1.5 Discount Rate Selection

### Theory
Realised LGD is based on **economic loss**, meaning all recoveries and costs must be discounted back to the default date to reflect the time value of money and opportunity cost.

### Australian bank practice

| Approach | Rate | When Used |
|----------|------|-----------|
| **Contract rate** | The loan's own interest rate | Most common in Australian Big 4 |
| **Cost of funds** | Bank's internal transfer pricing rate | Alternative approach |
| **Risk-free + spread** | RBA cash rate + credit spread | Less common |

**Recommended range for Australian mortgages: 3-5% per annum** (reflecting current Australian mortgage rates and bank cost of funds).

### Discount formula

For each cashflow at time *t* (days after default):

```
PV(CF_t) = CF_t / (1 + r)^(t / 365)
```

Where:
- `CF_t` = cashflow amount at time *t*
- `r` = annual discount rate
- `t` = number of days between default date and cashflow date

### Gap in current implementation
- Implementation 1 uses **8%** -- too high for Australian mortgages (overstates economic loss)
- Implementation 2 uses **random 3-7%** per loan -- no basis for rate selection

## 1.6 Realised LGD Calculation

### Core formula

```
Economic_Loss = EAD + PV(Costs) - PV(Recoveries)

Realised_LGD = max(Economic_Loss / EAD, 0)
```

### Expanded form

```
Realised_LGD = max( [EAD + Sum(Cost_j / (1+r)^(t_j/365)) - Sum(Recovery_i / (1+r)^(t_i/365))] / EAD, 0 )
```

Where:
- `Recovery_i` = each recovery cashflow (property sale, cure payment, LMI claim, etc.)
- `Cost_j` = each cost cashflow (legal, agent, preservation, etc.)
- `t_i`, `t_j` = days from default date to each cashflow
- `r` = discount rate
- Floor at 0 prevents negative LGD (over-recovery cases are floored, not excluded)

### Treatment of cures
For cured loans where borrower returns to performing:
- **Recovery = EAD** (full recovery of outstanding balance)
- **Costs = workout costs incurred before cure** (typically small)
- **Realised LGD is typically near zero** (but not exactly zero due to costs and discounting)

This creates a **bimodal LGD distribution** for mortgages: a mass point near 0 (cures) and a spread distribution for loss cases. This is why **two-stage modelling** is critical (see Section 1.12).

### Worked example

```
Loan: $500,000 mortgage, default date 1-Jan-2024
Recovery: Property sold for $420,000 on 1-Jul-2024 (182 days)
Costs: Legal $8,000 (paid 1-Jun-2024, 152 days), Agent $12,600 (paid 1-Jul-2024)
Discount rate: 4.5%

PV(Property sale) = $420,000 / (1.045)^(182/365) = $420,000 / 1.0223 = $410,844
PV(Legal) = $8,000 / (1.045)^(152/365) = $8,000 / 1.0186 = $7,854
PV(Agent) = $12,600 / (1.045)^(182/365) = $12,600 / 1.0223 = $12,325

Economic_Loss = $500,000 + $7,854 + $12,325 - $410,844 = $109,335
Realised_LGD = $109,335 / $500,000 = 21.87%
```

## 1.7 Segmentation Strategy

### Primary segmentation drivers for Australian mortgages

| Driver | Segments | Rationale |
|--------|----------|-----------|
| **Standard vs non-standard** | Standard residential, Non-standard | APRA requires different treatment |
| **LTV at default** | <60%, 60-70%, 70-80%, 80-90%, 90%+ | Primary risk driver; higher LTV = higher LGD |
| **LMI status** | LMI eligible, No LMI | LMI materially reduces loss |
| **Property type** | House, Unit/Apartment, Rural | Units have higher LGD in downturns |
| **Occupancy** | Owner-occupier, Investor | Investors may default more strategically |
| **State/Region** | NSW, VIC, QLD, WA, etc. | Property markets differ significantly |
| **Resolution type** | Cure, Sale, Short sale, Write-off | Post-hoc segmentation for analysis |

### Recommended segmentation hierarchy

```
Level 1: Standard vs Non-standard (APRA requirement)
  Level 2: LTV band at default
    Level 3: LMI eligible vs not
      Level 4: Property type (if sufficient data)
```

### Minimum segment size
Each segment should have **at least 20-30 defaulted observations** for statistical reliability. If segments are too small, merge adjacent categories.

### Gap in current implementation
- Implementation 1: LTV bucket x FICO bucket (reasonable but missing LMI, property type, state)
- Implementation 2: Product type only (insufficient granularity)

## 1.8 Long-Run Average LGD

### Theory
The long-run average LGD should represent the **expected loss severity across a full economic cycle**, including both benign and stressed periods.

### Formula (exposure-weighted)

```
LR_LGD_s = Sum(LGD_i * EAD_i) / Sum(EAD_i)    for all loans i in segment s
```

Where the sum covers **at least one full economic cycle** (minimum 7 years of default history in Australia, ideally covering both the early 1990s recession and GFC periods).

### Why exposure-weighted, not simple average
Large loans should have proportionally more influence on the portfolio LGD estimate. A $5M default with 40% LGD should matter more than a $200K default with 40% LGD when estimating portfolio-level capital.

### Australian cycle considerations
- **1990-1992 recession**: Property prices fell 10-15%, unemployment reached 11%
- **2008-2009 GFC**: Moderate impact on Australian property; prices fell 5-8%
- **2017-2019**: Property correction in Sydney/Melbourne (10-15% peak-to-trough)
- **2020 COVID**: Brief impact, rapid recovery due to policy support

If the bank's default history does not cover a full cycle, **adjustment factors** must be applied to scale up benign-period observations.

### Gap in current implementation
Both implementations use **simple means** (`df.groupby('segment')['lgd_realized'].mean()`), not exposure-weighted averages.

## 1.9 Downturn LGD

### Regulatory requirement
APRA APS 113 requires that LGD estimates reflect **economic downturn conditions** -- the LGD parameter used for capital must not understate losses during stressed periods.

### Approaches (from simplest to most robust)

**Approach 1: Scalar method**
```
Downturn_LGD = LR_LGD x Scalar
```
Typical scalar: **1.10 - 1.20** for Australian mortgages.

**Approach 2: Additive method**
```
Downturn_LGD = LR_LGD + Downturn_Add-on
```
Add-on calibrated to difference between downturn-period and through-the-cycle LGD.

**Approach 3: Macro-economic regression (best practice)**
```
LGD_t = alpha + beta_1 * HPI_change_t + beta_2 * Unemployment_t + epsilon_t
```
Then forecast LGD under a downturn macro scenario (e.g., HPI -20%, unemployment 8%).

### Australian downturn calibration
For residential mortgages, key macro drivers are:
- **House Price Index (HPI) change**: CoreLogic national index
- **Unemployment rate**: ABS labour force statistics
- **Interest rate environment**: RBA cash rate
- **Consumer confidence**: Westpac-Melbourne Institute index

A credible Australian downturn scenario might be:
- National HPI decline: -15% to -20%
- Unemployment: 7-9%
- This typically increases mortgage LGD by 5-15pp above long-run average

### Gap in current implementation
Both implementations use a **flat 1.10x scalar** with no macro linkage, no analytical justification, and no sensitivity testing.

## 1.10 Margin of Conservatism (MoC)

### Regulatory requirement
APRA requires a margin of conservatism to be added to LGD estimates to account for:
- **Data limitations** (small sample sizes, incomplete history)
- **Model uncertainty** (estimation error, model risk)
- **Process limitations** (data quality issues, operational gaps)

### Calibration approach

| Factor | Low MoC | Medium MoC | High MoC |
|--------|---------|------------|----------|
| Sample size | >500 defaults | 100-500 defaults | <100 defaults |
| Data history | Full cycle (10+ years) | Partial cycle (5-10 years) | Short history (<5 years) |
| Model performance | High R-squared, low MAE | Moderate performance | Poor discrimination |
| Data quality | Complete, audited | Some gaps | Significant issues |

### Recommended MoC for Australian mortgages
- **Well-established portfolio**: +1 to 2 percentage points
- **Moderate data**: +2 to 4 percentage points
- **Limited data**: +4 to 6 percentage points

### Formula
```
LGD_with_MoC = Downturn_LGD + MoC
```

### Gap in current implementation
- Implementation 1: +2pp (reasonable for a well-established portfolio)
- Implementation 2: +5pp (appropriate for limited data, but not calibrated)

## 1.11 Regulatory Overlays (APRA-Specific)

### Overlay 1: Standard vs non-standard mortgage classification

**Standard residential mortgage** (APRA APS 113):
- Loan to a natural person, secured by residential property
- Borrower occupies or rents the property
- LVR within bank's standard lending criteria
- Not a specialised property type

**Non-standard residential mortgage**:
- High-LVR without LMI
- Non-conforming borrower profile
- Unusual property types
- Development-linked residential loans

APRA treatment differs: standard mortgages may use internal LGD models with a **10% floor**; non-standard may face higher floors or supervisory LGD.

### Overlay 2: LMI recognition

APRA allows banks to recognise LMI as a credit risk mitigant, subject to:
- Insurer is APRA-authorised
- Policy covers the specific loan and loss scenario
- No material exclusions that would void the claim
- Bank has a demonstrated claim history with the insurer

```
LGD_after_LMI = LGD_before_overlays x (1 - LMI_benefit)
```

Where `LMI_benefit` is calibrated to the insurer's historical claim success rate and coverage terms. Typical range: **15-25% LGD reduction** for eligible loans.

### Overlay 3: Residential mortgage LGD floor

```
Final_Mortgage_LGD = max(LGD_after_all_adjustments, Floor)
```

- **Standard residential mortgage floor**: **10%** (APRA requirement under AIRB)
- **Non-standard**: Typically higher, may be **15%** or supervisory LGD

### Overlay 4: APRA scalar (applied at RWA level, not LGD level)

```
RWA_after_scalar = RWA_from_IRB_formula x 1.10
```

The **1.1x APRA scalar** is a post-model capital conservatism adjustment applied to Risk-Weighted Assets, not to LGD directly. It reflects APRA's view that internal models may underestimate risk.

**Important**: This scalar does not change the LGD estimate itself -- it is applied downstream in the capital calculation.

## 1.12 Model Selection

### Why two-stage modelling is critical for mortgages

Australian mortgage portfolios have **high cure rates** (30-50% of defaults return to performing). This creates a **bimodal LGD distribution**:
- **Mass at ~0**: Cured loans with near-zero loss
- **Spread from 10-60%**: Loss cases where property is sold

A single-model approach (e.g., OLS regression) struggles with this bimodal distribution.

### Recommended: Two-stage model

**Stage 1: Probability of Cure (Classification)**
```
P(Cure) = f(LTV, seasoning, DPD_at_default, arrears_history, borrower_income_change)
```
- Model: Logistic regression or gradient boosting classifier
- Output: Probability that the default resolves as a cure (LGD ~ 0)

**Stage 2: LGD given Loss (Regression, conditional on non-cure)**
```
LGD | Loss = g(LTV_at_default, property_type, state, workout_duration, sale_discount)
```
- Model: Beta regression (naturally bounded 0-1), or OLS on logit-transformed LGD
- Only fitted on loss cases (non-cures)

**Combined LGD estimate:**
```
Expected_LGD = P(Cure) x LGD_cure + (1 - P(Cure)) x E[LGD | Loss]
```
Where `LGD_cure` is typically a small fixed value (e.g., 1-2% to cover workout costs).

### Alternative models

| Model | Pros | Cons | When to Use |
|-------|------|------|-------------|
| **Segmented average** | Simple, transparent, auditable | No continuous drivers, coarse | Small data, regulatory floor |
| **OLS regression** | Easy to interpret | Unbounded predictions (can exceed 0-1) | Benchmark/challenger only |
| **Beta regression** | Naturally bounded (0,1) | Less familiar to validators | Best single-stage option |
| **Tobit regression** | Handles censoring at 0 and 1 | Complex to interpret | When many zero-loss observations |
| **Gradient boosting** | High predictive power | Black box, harder to validate | Challenger model |
| **Two-stage (logistic + beta)** | Handles bimodal distribution | More complex | **Recommended for production** |

### Key predictor variables for Australian mortgages

| Variable | Expected Relationship | Importance |
|----------|----------------------|------------|
| LTV at default | Higher LTV → Higher LGD | **Primary driver** |
| Property type | Units > Houses for LGD | High |
| State/region | Mining states more volatile | Medium |
| Loan seasoning | Older loans → lower LGD (more equity) | Medium |
| Occupancy type | Investor slightly higher LGD | Medium |
| LMI status | LMI → lower LGD | High |
| Interest rate type | IO → higher LGD (less amortisation) | Medium |

### Gap in current implementation
- Implementation 1: OLS on raw LGD (unbounded, inappropriate for 0-1 response) + segmented benchmark
- Implementation 2: Segment averages only, no statistical model

## 1.13 Validation & Backtesting

### Validation framework

| Test | Description | Metric |
|------|-------------|--------|
| **Discriminatory power** | Does the model rank-order losses correctly? | Spearman correlation, AUC on loss bands |
| **Calibration** | Are predicted LGDs close to actual? | Mean predicted vs mean actual by segment |
| **Accuracy** | Point estimate accuracy | MAE, RMSE, R-squared |
| **Stability** | Is the model stable over time? | PSI (Population Stability Index) on predicted LGD |
| **Conservatism** | Does the model meet regulatory requirements? | `Mean(Predicted) >= Mean(Actual)` at portfolio and segment level |
| **Out-of-time** | Performance on future (unseen) data | MAE/RMSE on hold-out vintage |
| **Sensitivity** | Impact of input changes | Stress tests on key drivers (LTV +10pp, HPI -20%) |
| **Benchmarking** | Comparison to external references | Compare to APRA loss benchmarks, peer banks |

### Backtesting procedure

1. **Annual backtesting**: Compare predicted LGD (from model vintage) to actual realised LGD for defaults resolved in the past year
2. **Traffic light system**:
   - Green: Predicted within 10% of actual
   - Amber: Predicted within 20% of actual
   - Red: Predicted deviates >20% from actual → trigger model review
3. **Vintage analysis**: Track prediction accuracy by origination cohort and default cohort

### Gap in current implementation
- Implementation 1: Basic MAE/RMSE/R-squared on LOOCV (minimal)
- Implementation 2: No validation whatsoever

## 1.14 Current Implementation Gaps (Mortgage)

| Gap | Severity | Description |
|-----|----------|-------------|
| No real data | Critical | Synthetic data limits all conclusions |
| No cure modelling | High | 30-50% of mortgage defaults cure; bimodal distribution ignored |
| No two-stage model | High | Single OLS is inappropriate for bounded, bimodal LGD |
| 8% discount rate | Medium | Too high; overstates economic loss by 3-5pp |
| No exposure-weighted average | Medium | Simple mean biases long-run LGD |
| No vintage analysis | Medium | No time-series dimension for stability |
| No macro-linked downturn | Medium | Flat scalar has no analytical basis |
| LMI as factor not cashflow | Low | Simplified but conceptually reasonable |
| No state-specific analysis | Low | Property markets differ materially by state |
| Small default sample | High | Only 8 defaults in Implementation 1 |

---

# Product 2: Commercial Cash Flow Lending (PPSR + GSR Secured)

## 2.1 Data Requirements

### Borrower data
| Field | Description | Source |
|-------|-------------|--------|
| `borrower_id` | Unique entity identifier | CRM / Loan system |
| `entity_type` | Company, trust, partnership, sole trader | Legal records |
| `industry_anzsic` | ANZSIC industry classification | Application |
| `annual_revenue` | Most recent annual revenue | Financial statements |
| `ebitda` | Earnings before interest, tax, depreciation, amortisation | Financial statements |
| `total_debt` | Total borrower indebtedness | Bureau + internal |
| `leverage_ratio` | Total Debt / EBITDA | Derived |
| `interest_coverage_ratio` | EBITDA / Interest Expense | Derived |
| `dscr` | Debt Service Coverage Ratio | Derived |
| `years_in_business` | Operating history length | Application |
| `director_guarantees` | Whether directors have provided personal guarantees | Facility docs |

### Facility data
| Field | Description | Source |
|-------|-------------|--------|
| `facility_type` | Term loan, revolving, overdraft, trade finance | Loan system |
| `facility_limit` | Approved facility limit | Loan system |
| `drawn_balance` | Current drawn amount | Loan system |
| `undrawn_amount` | Limit minus drawn | Derived |
| `interest_rate` | Contract rate | Loan system |
| `maturity_date` | Facility expiry | Loan system |
| `seniority` | Senior secured, subordinated, unsecured | Facility docs |
| `covenant_compliance` | Current covenant status | Credit monitoring |

### Security data
| Field | Description | Source |
|-------|-------------|--------|
| `security_type` | Property, PPSR (equipment, vehicles, inventory, receivables), GSR, cash | Security register |
| `ppsr_registration` | PPSR registration number and status | PPSR register |
| `gsr_flag` | Whether General Security Agreement is in place | Facility docs |
| `collateral_value` | Most recent valuation of each security item | Valuation system |
| `collateral_type_detail` | Real property, plant & equipment, motor vehicles, livestock, crops, receivables, inventory | Security register |
| `valuation_date` | Date of most recent valuation | Valuation system |
| `security_ranking` | First ranking, second ranking | PPSR / Title search |

### Default and workout data
| Field | Description | Source |
|-------|-------------|--------|
| `default_date` | Date default triggered | Credit management |
| `default_trigger` | 90 DPD, covenant breach, insolvency event, voluntary administration | Collections |
| `resolution_strategy` | Receivership, voluntary administration, DOCA, workout, write-off | Credit management |
| `resolution_date` | Date workout completed | Credit management |

### Recovery cashflows
| Field | Description |
|-------|-------------|
| `cashflow_date` | Date of each recovery or cost |
| `cashflow_type` | Property sale, PPSR asset realisation, receivables collection, trade debtor payment, going-concern sale, GSR enforcement, director guarantee, receiver distribution, legal cost, receiver fee, etc. |
| `cashflow_amount` | Dollar amount |
| `security_item_ref` | Which security item the cashflow relates to |

## 2.2 Default Definition

Per **APRA APS 220** and **Basel III**, commercial default is triggered when:

1. **90 days past due** on any material obligation, OR
2. **Unlikely to pay**, evidenced by:
   - Borrower placed in receivership, voluntary administration, or liquidation
   - Facility classified as impaired or non-accrual
   - Material covenant breach with no waiver
   - Significant adverse change in borrower's financial condition
   - Bank initiates enforcement action

### Commercial-specific considerations
- **Cross-default clauses**: Default on one facility may trigger default on all facilities with the same borrower
- **Group default**: If one entity in a borrower group defaults, related entities may be flagged
- **Covenant-triggered default**: Financial covenant breaches (e.g., ICR < 1.5x, leverage > 4x) may constitute unlikely-to-pay even before DPD triggers

## 2.3 EAD Measurement

### For term loans (fully drawn)
```
EAD = Outstanding Principal + Accrued Interest + Fees
```

### For revolving facilities / overdrafts (partially drawn)
```
EAD = Drawn Balance + CCF x Undrawn Amount + Accrued Interest
```

Where **CCF (Credit Conversion Factor)** reflects the tendency for borrowers to draw down undrawn facilities as they approach default.

| Facility Type | Typical CCF (APRA) | Rationale |
|---------------|-------------------|-----------|
| Committed revolving credit | 50-75% | Borrowers draw down as stress increases |
| Overdraft | 75-100% | Typically fully utilised before default |
| Trade finance / LC | 20-50% | Contingent, may not crystallise |
| Bank guarantees | 50-100% | Depending on beneficiary claim likelihood |

### APRA supervisory CCF (for standardised approach)
- Unconditionally cancellable: 0-10%
- Other commitments: 40-50% (maturity-dependent)
- Under IRB, banks estimate their own CCFs

## 2.4 Recovery & Cost Framework

### Recovery types for commercial lending

| Recovery Type | Description | Typical Recovery Rate |
|---------------|-------------|----------------------|
| **Real property sale** | Sale of mortgaged commercial/industrial property | 50-80% of valuation |
| **PPSR - Plant & equipment** | Realisation of registered P&E | 20-40% of book value |
| **PPSR - Motor vehicles** | Sale of registered vehicles | 40-60% of book value |
| **PPSR - Receivables** | Collection of assigned trade debtors | 30-60% of face value |
| **PPSR - Inventory** | Liquidation of registered inventory | 10-30% of book value |
| **PPSR - Livestock/crops** | Agricultural asset realisation | 30-50% of value |
| **GSR - All assets** | General security over all present and after-acquired property | Recovers from any unencumbered assets |
| **Going-concern sale** | Business sold as operating entity | Variable; 40-80% of enterprise value |
| **Director guarantee** | Personal guarantee enforcement | 10-30% of guarantee amount (collection difficulty) |
| **Insurance claim** | Key person, trade credit, or asset insurance | Case-specific |
| **Receiver distributions** | Periodic distributions from appointed receiver | Timing unpredictable |

### Cost types for commercial workout

| Cost Type | Description | Typical Range (% of EAD) |
|-----------|-------------|--------------------------|
| **Receiver/administrator fees** | Court-appointed receiver or voluntary administrator | 3-8% |
| **Legal costs** | Complex commercial litigation, enforcement | 2-5% |
| **Valuation and QS fees** | Multiple asset valuations | 0.5-1.5% |
| **Asset management** | Operating costs during receivership | 1-5% |
| **Environmental remediation** | Clean-up costs for contaminated sites | 0-10%+ (tail risk) |
| **GST/tax obligations** | Priority tax claims | Case-specific |
| **Internal workout** | Bank staff, specialist teams | 1-2% |

### PPSR recovery hierarchy

Under Australian law, PPSR-registered security interests are prioritised:
1. **Purchase Money Security Interest (PMSI)** -- highest priority for specific assets
2. **First-registered perfected security interest** -- priority by registration date
3. **Unperfected / unregistered interests** -- lowest priority, may lose to other creditors

Banks must ensure PPSR registrations are **perfected** (correctly registered before grantor's insolvency) to protect recovery rights.

### GSR (General Security Agreement) mechanics

A GSR gives the bank a security interest over **all present and after-acquired property** of the borrower. In practice:
- Acts as a "sweep" over residual assets not specifically pledged
- Priority depends on PPSR registration timing
- In administration/liquidation, circulating assets (receivables, inventory) are subject to statutory priority claims (employee entitlements, tax)

## 2.5 Discount Rate Selection

### Commercial lending discount rates

| Approach | Typical Rate | Rationale |
|----------|-------------|-----------|
| **Contract rate** | 5-8% | Reflects the borrower's cost of borrowing |
| **Bank's cost of funds + spread** | 4-6% | Internal transfer pricing |
| **Risk-adjusted rate** | 6-10% | Reflects commercial credit risk premium |

**Recommended: 5-7%** for Australian commercial lending, reflecting higher risk and longer workout periods than residential mortgage.

### Important: Workout periods are longer
Commercial workouts typically take **12-36 months** (vs 6-18 months for mortgages), making the discount rate more impactful on realised LGD.

## 2.6 Realised LGD Calculation

### Same core formula as mortgage
```
Economic_Loss = EAD + PV(Costs) - PV(Recoveries)
Realised_LGD = max(Economic_Loss / EAD, 0)
```

### Commercial-specific considerations

1. **Multiple security items**: Recoveries come from diverse sources (property + PPSR assets + guarantees). Each has different timing and certainty.

2. **Receiver distributions**: Unlike a single property sale, commercial recoveries arrive as periodic distributions over months/years.

3. **Going-concern vs liquidation**: The resolution strategy materially affects LGD:
   - Going-concern sale: LGD typically 20-50%
   - Orderly liquidation: LGD typically 40-70%
   - Fire sale / forced liquidation: LGD typically 60-90%

4. **Partially secured exposure**: If security covers only part of the EAD:
   ```
   LGD = LGD_secured_portion x (Secured_EAD / Total_EAD) + LGD_unsecured_portion x (Unsecured_EAD / Total_EAD)
   ```

### Worked example

```
Facility: $2,000,000 commercial term loan, default date 1-Jan-2024
Security: Commercial property (valued $1.5M) + PPSR over P&E ($300K book) + GSR
Discount rate: 6%

Recoveries:
  Property sold: $1,200,000 at month 18 (548 days)
  P&E realised: $90,000 at month 12 (365 days)
  GSR sweep: $50,000 at month 24 (730 days)
  Director guarantee: $30,000 at month 20 (608 days)

Costs:
  Receiver fees: $120,000 at month 24
  Legal: $45,000 at month 6 (182 days)
  Valuation: $15,000 at month 1 (30 days)

PV(Recoveries):
  Property: $1,200,000 / (1.06)^(548/365) = $1,200,000 / 1.0912 = $1,099,706
  P&E: $90,000 / (1.06)^(365/365) = $90,000 / 1.06 = $84,906
  GSR: $50,000 / (1.06)^(730/365) = $50,000 / 1.1236 = $44,501
  Guarantee: $30,000 / (1.06)^(608/365) = $30,000 / 1.1008 = $27,254
  Total PV(Recoveries) = $1,256,367

PV(Costs):
  Receiver: $120,000 / (1.06)^(730/365) = $120,000 / 1.1236 = $106,803
  Legal: $45,000 / (1.06)^(182/365) = $45,000 / 1.0296 = $43,706
  Valuation: $15,000 / (1.06)^(30/365) = $15,000 / 1.0048 = $14,928
  Total PV(Costs) = $165,437

Economic_Loss = $2,000,000 + $165,437 - $1,256,367 = $909,070
Realised_LGD = $909,070 / $2,000,000 = 45.45%
```

## 2.7 Segmentation Strategy

### Primary segmentation drivers for commercial lending

| Driver | Segments | Rationale |
|--------|----------|-----------|
| **Security type** | Property-backed, PPSR-only, Unsecured, Mixed | Primary LGD determinant |
| **Security coverage** | Fully secured (>100%), Partially secured (50-100%), Under-secured (<50%) | Higher coverage = lower LGD |
| **Industry sector** | Agriculture, Manufacturing, Retail, Services, Construction, etc. | Industry-specific recovery dynamics |
| **Facility type** | Term loan, Revolving, Overdraft, Trade finance | Affects EAD and recovery |
| **Borrower size** | SME (<$50M revenue), Mid-market, Large corporate | Larger firms have more recoverable assets |
| **Seniority** | Senior secured, Senior unsecured, Subordinated | Priority of claims |

### Recommended segmentation hierarchy

```
Level 1: Security type (Property-backed vs PPSR-only vs Unsecured)
  Level 2: Security coverage ratio band
    Level 3: Industry sector (if sufficient data)
      Level 4: Borrower size band
```

## 2.8 Long-Run Average LGD

### Formula (same as mortgage, exposure-weighted)
```
LR_LGD_s = Sum(LGD_i * EAD_i) / Sum(EAD_i)    for all loans i in segment s
```

### Commercial-specific considerations
- **Data scarcity**: Commercial portfolios have fewer defaults than mortgage books. A mid-tier bank may have only 50-200 commercial defaults over 10 years.
- **Heterogeneity**: Commercial loans are far more heterogeneous than mortgages, making long-run averages less stable.
- **Expert judgement**: Where data is insufficient, APRA allows banks to supplement quantitative estimates with **expert judgement**, provided it is documented and challenged.

### Typical Australian commercial LGD ranges

| Security Type | Typical LR LGD Range |
|---------------|---------------------|
| Property-backed (first mortgage) | 20-35% |
| PPSR - Plant & equipment | 40-55% |
| PPSR - Receivables / inventory | 45-65% |
| GSR only (no specific security) | 50-70% |
| Unsecured | 60-80% |

## 2.9 Downturn LGD

### Commercial downturn drivers
- **GDP growth**: Recession reduces enterprise values and asset prices
- **Industry-specific shocks**: Mining downturn (WA), drought (agriculture), retail disruption
- **Commercial property prices**: Office/industrial/retail vacancy rates
- **Credit conditions**: Tighter lending reduces buyer pool for distressed assets
- **Unemployment**: Reduces consumer demand, affecting commercial viability

### Downturn scalars for commercial lending

| Security Type | Benign Scalar | Moderate Downturn | Severe Downturn |
|---------------|---------------|-------------------|-----------------|
| Property-backed | 1.00x | 1.10-1.15x | 1.20-1.30x |
| PPSR assets | 1.00x | 1.15-1.20x | 1.25-1.40x |
| Unsecured | 1.00x | 1.10-1.15x | 1.15-1.25x |

### Macro-regression approach
```
LGD_t = alpha + beta_1 * GDP_growth_t + beta_2 * CommercialPropertyIndex_t + beta_3 * CreditSpread_t + epsilon_t
```

## 2.10 Margin of Conservatism

### Commercial MoC calibration

| Factor | Rationale | Typical Add-on |
|--------|-----------|----------------|
| Small default sample | <100 defaults | +2-3pp |
| High heterogeneity | Diverse industries and structures | +1-2pp |
| Data quality issues | Incomplete workout records | +1-2pp |
| Model limitations | Segment-level only, no regression | +1-2pp |
| **Total typical MoC** | | **+3-5pp** |

## 2.11 Regulatory Overlays (APRA-Specific)

### APS 112 - Collateral recognition
APRA's **APS 112 (Capital Adequacy: Standardised Approach to Credit Risk)** and **APS 113 (IRB)** specify rules for recognising collateral:

- **Eligible financial collateral**: Cash, government securities, listed equities (with haircuts)
- **Eligible real estate**: Must be valued independently, regularly re-valued, legally enforceable
- **PPSR-registered assets**: Recognised if perfected, valued conservatively, and realisable
- **Haircuts**: APRA mandates haircuts on collateral values (e.g., 30-40% for commercial property, 50-70% for moveable assets)

### Supervisory LGD (if not using internal model)

Under the standardised approach, APRA prescribes:
| Seniority | Supervisory LGD |
|-----------|----------------|
| Senior secured (eligible collateral) | 25-35% |
| Senior unsecured | 40-45% |
| Subordinated | 75% |

### SME firm-size adjustment
For SME exposures (annual revenue < $75M AUD), the Basel/APRA IRB formula includes a **firm-size adjustment** that reduces the correlation parameter:
```
Correlation = 0.12 x (1 - EXP(-50 x PD)) / (1 - EXP(-50))
            + 0.24 x (1 - (1 - EXP(-50 x PD)) / (1 - EXP(-50)))
            - 0.04 x (1 - (Revenue - 5) / 45)
```
This reduces RWA for smaller borrowers, indirectly affecting the capital impact of LGD.

### Specialised lending
Some commercial exposures may be classified as **specialised lending** (income-producing real estate, project finance, etc.) and require:
- **Slotting approach**: Assign to supervisory categories (Strong, Good, Satisfactory, Weak, Default) with prescribed LGDs
- Or internal model with enhanced data requirements

## 2.12 Model Selection

### Preferred approaches for commercial LGD

**Approach 1: Segmented average with expert overlay (most common in Australian banks)**
- Calculate exposure-weighted average LGD by segment (security type x coverage band x industry)
- Apply expert judgement overlay for segments with insufficient data
- Advantage: Transparent, auditable, regulator-friendly
- Disadvantage: Coarse, no continuous variable relationships

**Approach 2: Regression model**
```
LGD = beta_0 + beta_1 * SecurityCoverage + beta_2 * IndustryDummy + beta_3 * FacilityType + beta_4 * BorrowerSize + epsilon
```
- Use beta regression for bounded outcomes
- Include interaction terms (e.g., SecurityType x IndustryDownturnIndicator)
- Requires >200 defaults for reliable estimation

**Approach 3: Decision tree / random forest**
- Useful when relationships are non-linear and interactive
- Good for identifying threshold effects (e.g., security coverage below 80% → sharp LGD increase)
- Use as challenger model alongside segmented approach

### Key predictor variables

| Variable | Expected Relationship |
|----------|----------------------|
| Security coverage ratio | Higher coverage → lower LGD (primary driver) |
| Security type | Property > PPSR > Unsecured |
| Seniority | Senior < Subordinated |
| Industry sector | Stable industries < Cyclical industries |
| Borrower size | Larger firms → more recoverable assets |
| Facility type | Term loan < Revolving (revolving has higher EAD uncertainty) |
| Resolution type | Going-concern < Orderly liquidation < Fire sale |

## 2.13 Validation & Backtesting

Same framework as mortgage (Section 1.13) with additional considerations:
- **Segment stability is critical** due to small samples per segment
- **External benchmarking** against Moody's/S&P corporate recovery studies
- **Sensitivity testing** on collateral haircuts (±10-20pp)
- **Coverage ratio monitoring**: Track whether collateral values are deteriorating relative to exposures

## 2.14 Current Implementation Gaps (Commercial)

| Gap | Severity | Description |
|-----|----------|-------------|
| No commercial data fields | Critical | No borrower financials, no PPSR/GSR data, no industry, no facility structure |
| No security type differentiation | Critical | All commercial loans treated identically |
| No facility structure | High | No drawn/undrawn, no CCF, no seniority |
| No PPSR/GSR recovery modelling | High | No asset-level recovery by security type |
| No industry segmentation | High | Commercial LGD varies dramatically by industry |
| No receiver/administrator cost modelling | Medium | Commercial costs are much higher than mortgage |
| Identical parameters to mortgage | High | Same 30-90% recovery, 2-5% cost -- not realistic for commercial |
| No expert judgement framework | Medium | Insufficient data requires documented expert overlay |

---

# Product 3: Development Finance LGD

## 3.1 Data Requirements

### Project data
| Field | Description | Source |
|-------|-------------|--------|
| `development_type` | Residential (units/houses), Commercial, Mixed-use, Industrial | Application |
| `lot_count` / `unit_count` | Number of lots or units in the project | DA / Plans |
| `total_development_cost` | TDC including land, construction, fees, interest | Cost plan |
| `gross_realisation_value` | GRV -- total expected sales revenue on completion | Valuation |
| `as_is_value` | Current market value in present state | Valuation |
| `as_if_complete_value` | Value assuming project completed | Valuation |
| `land_value` | Value of underlying land | Valuation |
| `completion_percentage` | Current % of construction completed | QS report |
| `project_timeline_months` | Total planned project duration | Project plan |
| `pre_sale_count` | Number of units with binding pre-sale contracts | Sales report |
| `pre_sale_value` | Dollar value of pre-sales | Sales report |
| `pre_sale_coverage` | Pre-sale value / Facility limit | Derived |
| `builder_name` | Head contractor details | Contract |
| `builder_financial_health` | Builder solvency indicators | Due diligence |
| `qs_last_report_date` | Date of last Quantity Surveyor progress report | QS system |
| `cost_to_complete` | Remaining construction cost at assessment date | QS report |
| `da_approval_status` | Development Approval status and conditions | Council |
| `sunset_date` | Contract sunset clause date for pre-sales | Pre-sale contracts |

### Facility data
| Field | Description | Source |
|-------|-------------|--------|
| `facility_limit` | Approved development facility limit | Loan system |
| `drawn_balance` | Current drawn amount (progressive drawdown) | Loan system |
| `capitalised_interest` | Interest capitalised to facility balance | Loan system |
| `interest_reserve` | Pre-funded interest reserve balance | Loan system |
| `lvr_as_is` | Drawn / As-is value | Derived |
| `lvr_as_if_complete` | Facility limit / As-if-complete value | Derived |
| `ltc` | Facility limit / Total Development Cost | Derived |

### Default and workout data
Same structure as commercial, plus:
| Field | Description |
|-------|-------------|
| `completion_stage_at_default` | Pre-construction, Early construction, Mid-construction, Near-complete, Complete-unsold |
| `cost_to_complete_at_default` | Remaining construction cost at default |
| `pre_sales_at_default` | Pre-sale coverage at time of default |
| `fund_to_complete_decision` | Whether bank decided to fund project to completion |

## 3.2 Default Definition

Same Basel/APRA framework as commercial (90 DPD / unlikely to pay), plus development-specific triggers:

1. **Pre-sale hurdle failure**: Borrower fails to achieve minimum pre-sale requirements by agreed date
2. **Builder insolvency**: Head contractor enters administration or liquidation
3. **Cost overrun**: Project costs exceed budget by material margin (e.g., >15%) without additional equity
4. **Planning revocation**: Development Approval cancelled or materially amended
5. **Interest reserve exhaustion**: Pre-funded interest reserve depleted before project completion
6. **Environmental/contamination discovery**: Unforeseen site issues halting construction
7. **Sunset clause exercise**: Purchasers exercise sunset clauses to rescind pre-sale contracts (particularly relevant in downturn)

## 3.3 EAD Measurement

### Progressive drawdown
Unlike term loans, development facilities are drawn progressively as construction milestones are met:

```
EAD = Current Drawn Balance + Capitalised Interest + Undrawn_CCF x Undrawn Amount
```

### Critical consideration: Fund-to-complete risk
If the bank decides to **fund the project to completion** after default (common strategy), the EAD **increases post-default**:

```
EAD_if_fund_to_complete = Current Drawn + Cost_to_Complete + Additional_Interest + Holding_Costs
```

This is unique to development finance -- the bank may intentionally increase its exposure post-default to maximise the recovery value of the completed project.

### EAD scenarios by completion stage

| Completion Stage | Typical Drawn % of Limit | Fund-to-Complete Risk |
|------------------|--------------------------|----------------------|
| Pre-construction (land only) | 30-40% | High (full build cost ahead) |
| Early construction (0-30%) | 40-60% | High |
| Mid-construction (30-70%) | 60-80% | Medium |
| Near-complete (70-95%) | 80-95% | Low (small cost to complete) |
| Complete, unsold | 95-100% | None (holding costs only) |

## 3.4 Recovery & Cost Framework

### Recovery scenarios by completion stage

This is the **most critical dimension** for development finance LGD. The recovery profile depends fundamentally on where the project is at default:

#### Scenario A: Pre-construction default (land only)
```
Recovery = Land_Sale_Price - Selling_Costs
```
- Land is sold "as-is"
- Recovery typically 60-80% of original land value (discount for distressed sale)
- Relatively quick resolution (6-12 months)

#### Scenario B: Mid-construction default (partially built)
```
Recovery = min(Complete_and_Sell, Sell_As_Is)
```
**This is the worst-case scenario for LGD:**
- Partially built structures have **negative value** (cost to complete exceeds incremental value in distressed market)
- Finding a new builder is difficult and costly
- "As-is" value may be **below land value** (demolition costs)
- If bank funds to completion:
  ```
  Recovery = GRV_at_completion x Distressed_Discount - Cost_to_Complete - Additional_Holding_Costs
  ```

#### Scenario C: Near-complete default
```
Recovery = Completion_Cost + Sale_of_Completed_Units
```
- Bank typically funds to completion (small remaining cost)
- Sale at completed value, potentially discounted 5-15%
- Best recovery scenario for construction-stage defaults

#### Scenario D: Complete but unsold
```
Recovery = Sum(Unit_Sale_Prices) - Marketing_Costs - Holding_Costs
```
- No construction risk, but market risk
- Extended holding period if market is weak
- Pre-sales provide some certainty (if not rescinded via sunset clauses)

### Recovery types

| Recovery Type | Description |
|---------------|-------------|
| **Unit/lot sales** | Individual sales of completed product |
| **Bulk sale** | Sell remaining inventory to single buyer (typically at 15-25% discount) |
| **Land sale (pre-construction)** | Sell underlying land |
| **Builder/contractor bond claims** | Recovery from builder's performance bonds or insurance |
| **Pre-sale deposits** | Forfeit deposits from defaulting purchasers |
| **GSR / PPSR recovery** | Additional assets of the borrower entity |
| **Guarantor recovery** | Personal guarantees from directors/sponsors |

### Cost types

| Cost Type | Description | Typical Range |
|-----------|-------------|---------------|
| **Cost to complete** | Remaining construction cost (DOMINANT COST) | 20-60% of TDC |
| **Receiver/administrator fees** | Typically % of realisations | 3-5% of recoveries |
| **QS and project management** | Ongoing supervision during completion | 1-3% of cost-to-complete |
| **Legal costs** | Complex, often multi-party disputes | 2-5% of EAD |
| **Holding costs** | Rates, insurance, site security, interest carry | 1-3% per annum on EAD |
| **Marketing and sales** | Agent commissions and marketing for completed product | 3-5% of sale price |
| **Defect rectification** | Remediation of construction defects | 1-5% of construction cost |
| **GST/stamp duty** | Tax obligations on transactions | Varies |

## 3.5 Discount Rate Selection

### Development finance discount rates

| Approach | Rate | Rationale |
|----------|------|-----------|
| **Contract rate** | 7-10% | Development facilities carry premium rates |
| **Risk-adjusted rate** | 8-12% | Reflects higher project risk |

**Recommended: 7-9%** for Australian development finance, reflecting higher risk and potentially long workout periods (12-36 months).

### Important: Workout period varies dramatically by completion stage

| Completion at Default | Typical Workout Period |
|----------------------|----------------------|
| Pre-construction | 6-12 months (land sale) |
| Mid-construction | 18-36 months (complete + sell) |
| Near-complete | 12-18 months (finish + sell) |
| Complete, unsold | 6-18 months (marketing + sell) |

## 3.6 Realised LGD Calculation

### Same core formula, but with fund-to-complete adjustment

**If bank does NOT fund to completion:**
```
Economic_Loss = EAD_at_default + PV(Costs) - PV(Recoveries)
Realised_LGD = max(Economic_Loss / EAD_at_default, 0)
```

**If bank DOES fund to completion (common):**
```
Total_EAD = EAD_at_default + Cost_to_Complete + Additional_Interest_Carry
Economic_Loss = Total_EAD + PV(Other_Costs) - PV(Completed_Sales)
Realised_LGD = max(Economic_Loss / EAD_at_default, 0)
```

Note: The denominator is still **EAD at the point of default** (not total funding including post-default), per Basel convention. This means development finance LGD can **exceed 100%** if the bank funds to completion and the market has declined.

### Worked example: Mid-construction default

```
Development: 20-unit residential project
Facility limit: $8,000,000
Drawn at default: $5,500,000 (including $400K capitalised interest)
Completion: 45%
Cost to complete: $3,200,000
Pre-sales: 12 units at $550,000 each = $6,600,000
As-if-complete GRV: $11,000,000
Discount rate: 8%

Decision: Bank funds to completion

Post-default funding: $3,200,000 (construction) + $320,000 (interest 6 months) = $3,520,000
Total bank outlay: $5,500,000 + $3,520,000 = $9,020,000

Recovery timeline:
  Month 12: Construction completes
  Month 14-24: Units sold (some pre-sales settle, some open market)

Recoveries (undiscounted): $9,800,000 (12 pre-sales at $550K, 8 open market at $500K avg)
Costs: Receiver $200K, Legal $150K, Marketing $300K, Defects $100K = $750K

PV(Recoveries) at month 18 average: $9,800,000 / (1.08)^(18/12) = $8,724,558
PV(Costs) at month 18 average: $750,000 / (1.08)^(18/12) = $667,759
PV(Post-default funding) at month 6 average: $3,520,000 / (1.08)^(6/12) = $3,384,615

Economic_Loss = $5,500,000 + $3,384,615 + $667,759 - $8,724,558 = $827,816
Realised_LGD = $827,816 / $5,500,000 = 15.05%

(Note: If market declined 20% during workout, GRV would drop, pushing LGD to 30-40%+)
```

## 3.7 Segmentation Strategy

### Primary segmentation drivers for development finance

| Driver | Segments | Rationale |
|--------|----------|-----------|
| **Completion stage at default** | Pre-construction, Early (0-30%), Mid (30-70%), Late (70-95%), Complete-unsold | **Primary driver** -- determines recovery scenario |
| **Development type** | Residential apartments, Residential houses/lots, Commercial, Industrial, Mixed-use | Different market dynamics and cost structures |
| **Pre-sale coverage** | >100%, 80-100%, 50-80%, <50% | Higher pre-sales = more certain recovery |
| **LVR (as-if-complete)** | <60%, 60-70%, 70-80%, >80% | Higher LVR = more exposed to value decline |
| **Project scale** | Small (<10 units), Medium (10-50), Large (50+) | Larger projects harder to sell in bulk |
| **Location** | Capital city CBD, Suburban, Regional | Market depth affects recovery speed and discount |

### Recommended segmentation hierarchy

```
Level 1: Completion stage at default (5 bands)
  Level 2: Development type
    Level 3: Pre-sale coverage band
      Level 4: LVR band (if sufficient data)
```

### Critical insight
**Completion stage is the single most important LGD driver** in development finance. A mid-construction default on a 50-unit apartment project can have LGD of 40-60%, while the same project defaulting near completion might have LGD of 10-20%.

## 3.8 Long-Run Average LGD

### Exposure-weighted formula (same as other products)
```
LR_LGD_s = Sum(LGD_i * EAD_i) / Sum(EAD_i)
```

### Development-specific challenges

1. **Very limited data**: Even large Australian banks may have only 10-50 development defaults in a decade
2. **High heterogeneity**: Each project is unique -- location, type, scale, market conditions
3. **Survivorship bias**: Banks that managed development portfolios conservatively may have fewer defaults but higher LGD (the defaults that did occur were severe)
4. **Expert judgement is essential**: APRA expects banks with limited data to document expert overlays

### Typical Australian development finance LGD ranges

| Completion Stage | Typical LGD Range (benign) | Typical LGD Range (downturn) |
|------------------|---------------------------|------------------------------|
| Pre-construction | 15-30% | 25-45% |
| Mid-construction | 30-50% | 45-70% |
| Near-complete | 10-25% | 20-40% |
| Complete, unsold | 10-20% | 20-40% |

## 3.9 Downturn LGD

### Development finance is the most cyclically sensitive product

Key downturn dynamics:
1. **Property price decline**: GRV drops 15-30% in severe downturn, directly increasing LGD
2. **Pre-sale rescission**: Purchasers exercise **sunset clauses** to walk away from contracts, eliminating "locked-in" recovery
3. **Construction cost inflation**: Builder costs may rise during stress (supply chain disruption), increasing cost-to-complete
4. **Extended sales period**: Takes longer to sell completed stock in weak market, increasing holding costs
5. **Builder insolvency**: Stressed builders go under, causing delays and additional costs
6. **Fire sale discount**: Distressed sales of partially-built projects attract much larger discounts

### Downturn scalars for development finance

| Completion Stage | Moderate Downturn Scalar | Severe Downturn Scalar |
|------------------|--------------------------|----------------------|
| Pre-construction | 1.10-1.15x | 1.20-1.30x |
| Mid-construction | 1.15-1.25x | 1.30-1.50x |
| Near-complete | 1.10-1.15x | 1.20-1.35x |
| Complete, unsold | 1.15-1.20x | 1.25-1.40x |

### Scenario-based downturn approach (recommended)

Rather than a simple scalar, use scenario analysis:
```
Downturn_LGD = LGD_under_scenario(
  GRV_decline = -20%,
  Pre_sale_rescission_rate = 30%,
  Construction_cost_increase = +10%,
  Sales_period_extension = +6 months,
  Fire_sale_discount = +15%
)
```

## 3.10 Margin of Conservatism

### Development MoC calibration

| Factor | Add-on | Rationale |
|--------|--------|-----------|
| Very small default sample (<30) | +3-4pp | Extreme data limitation |
| High project heterogeneity | +2-3pp | Each project is unique |
| Scenario model uncertainty | +1-2pp | Assumptions drive results |
| Construction/completion risk | +1-2pp | Non-financial risk factors |
| **Total typical MoC** | **+5-8pp** | |

Development finance warrants the **highest MoC** of all three products due to extreme data limitations and project-level variability.

## 3.11 Regulatory Overlays (APRA-Specific)

### Specialised lending classification
APRA may classify development finance as **specialised lending** under APS 113, specifically:
- **High-Volatility Commercial Real Estate (HVCRE)** for speculative development
- **Income-Producing Real Estate (IPRE)** for investment-grade development

### Slotting approach
If the bank cannot demonstrate sufficient internal modelling capability, APRA may require the **slotting approach**:

| Slot | Description | Supervisory LGD (indicative) |
|------|-------------|------------------------------|
| Strong | Well-advanced, strong pre-sales, experienced developer | 20-25% |
| Good | Good progress, adequate pre-sales, sound sponsor | 30-35% |
| Satisfactory | On-track but some concerns | 40-45% |
| Weak | Material issues (delays, cost overruns, weak sales) | 55-65% |
| Default | In default | Realised LGD |

### Higher correlation parameter
APRA/Basel applies a **higher asset correlation** for HVCRE (1.25x the standard corporate correlation), which increases RWA even for the same LGD. This is applied in the IRB risk-weight formula, not in the LGD estimate itself.

### APRA scalar
Same **1.1x scalar** on RWA as for other AIRB portfolios.

## 3.12 Model Selection

### Preferred approaches for development finance

**Approach 1: Scenario-based expert model (most appropriate)**
- Define recovery scenarios by completion stage and market conditions
- Estimate LGD under each scenario using engineering-style cost/value analysis
- Weight scenarios by probability
- Advantage: Captures project-level complexity, works with very limited data
- Used by most Australian banks for development portfolios

**Approach 2: Decision tree**
```
If completion < 30%:
  If pre-sales > 80%: LGD = base + adjustment
  Else: LGD = base + higher_adjustment
Elif completion < 70%:
  ...
```
- Simple threshold-based rules
- Transparent and auditable
- Can be calibrated to limited data + expert judgement

**Approach 3: Monte Carlo simulation (advanced)**
- Simulate key variables: GRV change, cost-to-complete variance, sales timeline
- Run 10,000+ scenarios
- LGD = average loss across scenarios
- Advantage: Captures uncertainty and tail risk
- Disadvantage: Complex, requires careful parameter calibration

### Statistical modelling is generally NOT feasible
Due to very small samples (10-50 defaults at most Australian banks) and extreme heterogeneity, traditional regression-based LGD models are **not reliable** for development finance. Expert-judgement-based approaches, supported by scenario analysis, are the standard.

## 3.13 Validation & Backtesting

Same framework as commercial (Section 2.13) with additional considerations:
- **Scenario plausibility testing**: Are the assumed GRV declines, cost overruns, and rescission rates realistic?
- **External benchmarking**: Compare to developer insolvency recovery data, ASIC receiver reports
- **Sensitivity analysis is critical**: Test LGD sensitivity to ±10% GRV, ±20% cost-to-complete, ±6 months sales period
- **Annual recalibration**: Development markets change rapidly; parameters should be refreshed annually

## 3.14 Current Implementation Gaps (Development Finance)

| Gap | Severity | Description |
|-----|----------|-------------|
| No development-specific data | Critical | No completion stage, no pre-sales, no TDC, no GRV, no QS data |
| No completion-stage segmentation | Critical | This is THE primary LGD driver for development |
| No scenario modelling | Critical | Fund-to-complete vs sell-as-is decision not modelled |
| No cost-to-complete framework | High | Dominant cost category for development entirely absent |
| Same parameters as mortgage | High | 30-90% recovery fraction is unrealistic for mid-construction defaults |
| No pre-sale risk modelling | High | Sunset clause rescission risk ignored |
| No specialised lending / slotting | Medium | APRA may require slotting approach |
| No expert judgement framework | Medium | Essential given extreme data limitations |
| No HVCRE treatment | Medium | Higher correlation parameter not considered |

---

# Part C: Cross-Product Comparison Matrix

| Dimension | Residential Mortgage | Commercial Cash Flow | Development Finance |
|-----------|---------------------|---------------------|---------------------|
| **Primary security** | Residential property | Mixed (property, PPSR, GSR) | Development site + WIP |
| **Typical LGD range** | 10-25% (with cures) | 30-55% | 15-50% (stage dependent) |
| **Key LGD driver** | LTV at default | Security coverage ratio | Completion stage at default |
| **Typical workout period** | 6-18 months | 12-36 months | 12-36 months |
| **Cure rate** | 30-50% | 10-20% | 5-10% |
| **Data availability** | Good (large portfolios) | Moderate (fewer defaults) | Poor (very few defaults) |
| **Downturn sensitivity** | Medium | Medium-High | Very High |
| **Model complexity** | Two-stage (cure + loss) | Segmented + regression | Scenario-based / expert |
| **APRA floors** | 10% (standard), 15% (non-standard) | Supervisory LGD if no model | Slotting may be required |
| **LMI applicable** | Yes | No | No |
| **Fund-to-complete risk** | No | No | Yes (unique feature) |
| **Discount rate** | 3-5% | 5-7% | 7-9% |
| **Recommended MoC** | +1-3pp | +3-5pp | +5-8pp |
| **APRA scalar** | 1.1x on RWA | 1.1x on RWA | 1.1x on RWA (+ higher correlation if HVCRE) |

---

# Part D: Key Formulas Reference

## D.1 Core LGD Formulas

### Realised LGD
```
Realised_LGD = max( (EAD + PV(Costs) - PV(Recoveries)) / EAD, 0 )
```

### Present Value of Cashflows
```
PV(CF_t) = CF_t / (1 + r)^(t / 365)
```
Where `t` = days from default date, `r` = annual discount rate.

### Exposure-Weighted Long-Run Average
```
LR_LGD_s = Sum_i(LGD_i * EAD_i) / Sum_i(EAD_i)     for all i in segment s
```

## D.2 Downturn Adjustment Formulas

### Scalar method
```
Downturn_LGD = LR_LGD * Scalar
```

### Additive method
```
Downturn_LGD = LR_LGD + Add_on
```

### Macro-regression method
```
LGD_t = alpha + beta_1 * MacroVar1_t + beta_2 * MacroVar2_t + ... + epsilon_t
Downturn_LGD = Predicted LGD under downturn macro scenario
```

## D.3 Regulatory Formulas

### Final Mortgage LGD (with APRA overlays)
```
Step 1: Downturn_LGD = Realised_LGD * Downturn_Scalar
Step 2: LGD_with_MoC = Downturn_LGD + MoC
Step 3: LGD_after_LMI = LGD_with_MoC * (1 - LMI_benefit)   [if eligible]
Step 4: Final_LGD = max(LGD_after_LMI, Floor)               [Floor = 10% standard, 15% non-standard]
```

### EAD with CCF (for revolving facilities)
```
EAD = Drawn + CCF * Undrawn + Accrued_Interest
```

### IRB Risk-Weight Function (for reference)
```
K = LGD * [N((1-R)^(-0.5) * G(PD) + (R/(1-R))^0.5 * G(0.999)) - PD] * (1-1.5*b)^(-1) * (1+(M-2.5)*b)
RWA = K * 12.5 * EAD
RWA_after_APRA = RWA * 1.10
```
Where:
- `N()` = standard normal CDF
- `G()` = standard normal inverse CDF
- `R` = asset correlation
- `b` = maturity adjustment = (0.11852 - 0.05478 * ln(PD))^2
- `M` = effective maturity

## D.4 Two-Stage Mortgage LGD Model
```
Stage 1: P(Cure) = Logistic(beta_0 + beta_1*LTV + beta_2*Seasoning + ...)
Stage 2: E[LGD|Loss] = BetaRegression(alpha + gamma_1*LTV + gamma_2*PropertyType + ...)
Combined: E[LGD] = P(Cure) * LGD_cure + (1 - P(Cure)) * E[LGD|Loss]
```

## D.5 Development Finance Fund-to-Complete LGD
```
Total_Outlay = EAD_at_default + Cost_to_Complete + Interest_Carry + Holding_Costs
Total_Recovery = Sum(Unit_Sales) + Other_Recoveries
Economic_Loss = Total_Outlay + PV(Workout_Costs) - PV(Total_Recovery)
Realised_LGD = max(Economic_Loss / EAD_at_default, 0)
```
Note: LGD can exceed 100% when post-default funding (cost-to-complete) exceeds incremental recovery.

---

# Part E: Consolidated Gap Analysis & Recommendations

## E.1 Gap Summary by Category

### Data Gaps
| Gap | Affects | Priority |
|-----|---------|----------|
| No real default/workout data | All products | Critical |
| No borrower financials for commercial | Commercial | Critical |
| No PPSR/GSR security register data | Commercial | Critical |
| No completion stage / QS data | Development | Critical |
| No pre-sale data | Development | Critical |
| No cost-to-complete data | Development | High |
| No indexed property values at default | Mortgage | Medium |
| No LMI claim history | Mortgage | Medium |

### Methodology Gaps
| Gap | Affects | Priority |
|-----|---------|----------|
| No two-stage cure model | Mortgage | High |
| No exposure-weighted averaging | All products | High |
| No macro-linked downturn | All products | High |
| No scenario modelling | Development | High |
| No fund-to-complete decision logic | Development | High |
| OLS on raw LGD (unbounded) | Mortgage | Medium |
| No vintage/cohort analysis | All products | Medium |
| Unjustified discount rates | All products | Medium |

### Product-Specific Gaps
| Gap | Affects | Priority |
|-----|---------|----------|
| All 3 products share identical generation logic | Multi-product framework | Critical |
| No security type differentiation | Commercial | Critical |
| No completion stage segmentation | Development | Critical |
| No industry segmentation | Commercial | High |
| No facility structure (drawn/undrawn/CCF) | Commercial | High |
| No PPSR recovery hierarchy | Commercial | High |
| No pre-sale rescission modelling | Development | High |
| No state-specific property analysis | Mortgage | Medium |

### Regulatory Gaps
| Gap | Affects | Priority |
|-----|---------|----------|
| Implementation 2 has zero APRA overlays | All products | Critical |
| No specialised lending / slotting | Development | High |
| No APS 112 collateral recognition | Commercial | High |
| No HVCRE treatment | Development | Medium |
| No SME firm-size adjustment | Commercial | Medium |
| No supervisory LGD fallback | Commercial | Medium |

### Validation Gaps
| Gap | Affects | Priority |
|-----|---------|----------|
| No out-of-time validation | All products | High |
| No stability monitoring (PSI) | All products | High |
| No external benchmarking | All products | Medium |
| No conservatism testing | All products | Medium |
| No sensitivity analysis | All products | Medium |

## E.2 Recommended Enhancement Roadmap

### Phase 1: Data Foundation
- Source or simulate realistic default/workout data for each product
- Build product-specific data schemas with all required fields
- Implement data quality checks and completeness monitoring

### Phase 2: Product-Specific LGD Engines
- **Mortgage**: Implement two-stage model (cure probability + LGD|loss), add exposure-weighted averaging
- **Commercial**: Build security-type segmentation, add PPSR/GSR recovery framework, implement industry-based analysis
- **Development**: Build completion-stage segmentation, implement fund-to-complete scenario model, add pre-sale risk

### Phase 3: Regulatory Overlay Layer
- Implement APRA overlays for all products (floors, LMI, standard/non-standard)
- Add specialised lending / slotting framework for development
- Implement APS 112 collateral recognition for commercial
- Add macro-linked downturn methodology with scenario calibration

### Phase 4: Validation Framework
- Build out-of-time testing capability
- Implement PSI monitoring for model stability
- Add conservatism testing (predicted vs actual at segment level)
- Create sensitivity analysis toolkit

### Phase 5: Capital Integration
- Link LGD to PD and EAD models
- Implement IRB risk-weight function
- Apply APRA scalar and produce RWA outputs
- Connect to pricing / hurdle rate framework

---

# Part F: Industry Risk Integration

## F.1 Overview

Industry risk analysis outputs have been integrated into the LGD framework to replace flat downturn scalars with industry-sensitive adjustments. The integration draws on a nine-layer industry analysis pipeline that produces risk scores, working capital metrics, stress scenarios, and benchmarks for 9 ANZSIC industry divisions, sourced from ABS and RBA public data.

**Source project:** Industry Risk Analysis (ANZSIC-aligned, ABS/RBA data-driven)

**Products affected:** Commercial Cash Flow, Development Finance

**Module:** `src/industry_risk_integration.py`

## F.2 Industry Risk Score Framework

Each industry receives a composite risk score (1–5 scale):

$$\text{Industry Base Risk Score} = 0.55 \times \text{Classification Risk} + 0.45 \times \text{Macro Risk}$$

| Risk Level | Score Range | Industries |
|:-----------|:-----------|:-----------|
| Elevated | > 3.0 | Agriculture (3.50), Manufacturing (3.50), Wholesale Trade (3.23), Retail Trade (3.23) |
| Medium | 2.5 – 3.0 | Accommodation & Food (2.68), Construction (2.68) |
| Medium | ≤ 2.5 | Health Care (2.22), Professional Services (2.18), Transport (2.14) |

**Mining** has no match in the industry analysis and is assigned a conservative default (score = 3.50, Elevated).

## F.3 Integration Formulas

### F.3.1 Industry-Adjusted Downturn Scalar

$$\text{adjusted\_scalar} = \text{base\_scalar} \times \left(1 + \alpha \times \frac{\text{risk\_score} - 2.5}{2.0}\right)$$

- $\alpha = 0.15$ (sensitivity parameter, 15% max swing)
- Base scalar comes from security type (Commercial) or completion stage (Development)
- Adjustments bounded to ±7–10% of base scalar

**Calibration rationale:** The midpoint of 2.5 is set slightly below the exposure-weighted average industry risk score across the portfolio, ensuring that industry adjustments are net LGD-increasing at the portfolio level. The 15% sensitivity was calibrated to produce material but bounded differentiation — sufficient to distinguish Elevated from Medium industries in validation, but constrained enough that no single industry adjustment dominates the LGD estimate.

### F.3.2 Industry-Adjusted Margin of Conservatism

$$\text{adjusted\_moc} = \text{base\_moc} \times \left(1 + \beta \times \frac{\text{risk\_score} - 2.5}{2.5}\right)$$

- $\beta = 0.20$ (sensitivity parameter)
- Base MoC: 3pp (Commercial), 5pp (Development)
- Higher-risk industries receive larger MoC, reflecting greater data uncertainty

### F.3.3 Industry Recovery Haircut

$$\text{haircut} = 0.02 \times \max(\text{risk\_score} - 2.0, 0)$$

- Returns 0% for score ≤ 2.0, up to 6% for score = 5.0
- Applied additively to realised LGD before downturn scaling
- Reflects deeper distressed-sale discounts in higher-risk industries

### F.3.4 Working Capital LGD Adjustment

$$\text{adjustment} = 0.01 \times \max(\text{wc\_overlay\_score} - 2.0, 0)$$

- WC overlay score ranges from 1.89 (Accommodation & Food) to 3.89 (Manufacturing)
- Adds 0–1.9 percentage points to LGD
- Reflects lower administration recoveries for industries with poor working capital

## F.4 Enhanced LGD Pipeline

### Commercial Cash Flow

```
Realised LGD
  → + Industry recovery haircut
  → × Industry-adjusted downturn scalar (base from security type)
  → + Industry-adjusted MoC (base 3pp)
  → + Working capital LGD adjustment
  → Supervisory LGD floor check (APS 112)
  → Cap at 100%
```

### Development Finance

```
Realised LGD
  → + Industry recovery haircut
  → × Industry-adjusted downturn scalar (base from completion stage)
  → + Industry-adjusted MoC (base 5pp)
  → Cap at 100%
```

Industry risk also contributes 0–2 points to the **APRA slotting score**, which can shift projects between slotting categories (Strong/Good/Satisfactory/Weak).

## F.5 Segmentation Enhancement

Industry risk band (`Low` / `Medium` / `Elevated`) is added as a segmentation dimension:

| Product | Original Segmentation | Enhanced Segmentation |
|:--------|:---------------------|:---------------------|
| Commercial | Security type × Coverage band | Security type × Coverage band × **Industry risk band** |
| Development | Completion stage × Dev type × Presale band × LVR band | + **Industry risk band** |

## F.6 APRA Compliance

| Requirement | How Addressed |
|:-----------|:-------------|
| LGD floors preserved | 10%/15% mortgage floors and supervisory LGD (APS 112) remain hard floors applied after all industry adjustments |
| Downturn LGD (APS 113) | Industry-specific scalars are more defensible than flat scalars — reflect differential cyclicality |
| MoC reflects data limitations | Industry-varying MoC appropriate — data depth and model uncertainty vary by sector |
| No double-counting with PD | LGD adjustment targets recovery/workout risk channels, not default probability |
| Slotting framework preserved | Four-category structure and supervisory risk weights unchanged |
| Conservatism direction | Net portfolio-level LGD impact is increasing or neutral |

## F.7 Industry Name Mapping

| LGD Model Name | ANZSIC Name |
|:---------------|:-----------|
| Agriculture | Agriculture, Forestry And Fishing |
| Manufacturing | Manufacturing |
| Retail Trade | Retail Trade |
| Construction | Construction |
| Transport | Transport, Postal And Warehousing |
| Professional Services | Professional, Scientific And Technical Services |
| Accommodation & Food | Accommodation And Food Services |
| Health Care | Health Care and Social Assistance |
| Wholesale Trade | Wholesale Trade |
| Mining | *(No match — conservative default: Elevated, score 3.50)* |

## F.8 Development Type to Industry Mapping

| Development Type | Mapped Industry | Rationale |
|:----------------|:---------------|:----------|
| Residential Apartments | Construction | Builder/developer risk |
| Residential Houses/Lots | Construction | Builder/developer risk |
| Commercial Office | Professional Services | Tenant demand risk |
| Mixed-Use | Construction | Builder/developer risk |
| Industrial | Manufacturing | End-user demand risk |

## F.9 Data Sources

All industry risk scores derive from publicly available Australian data:

- **Australian Bureau of Statistics (ABS):** Industry classifications (ANZSIC), business indicators, labour force, building approvals, inventory ratios, profit margins
- **Reserve Bank of Australia (RBA):** Cash rates, economic tables
- **Payment Times Reporting Scheme (PTRS):** Official AR/AP payment days

## F.10 Key Files

| File | Purpose |
|:-----|:--------|
| `src/industry_risk_integration.py` | Core integration module — loader, formulas, enrichment |
| `src/data_generation.py` | Enhanced with `INDUSTRY_RISK_PROFILES` and industry-sensitive recovery rates |
| `src/lgd_calculation.py` | `CommercialLGDEngine` and `DevelopmentLGDEngine` enhanced with industry overlays |
| `src/validation.py` | `industry_attribution_analysis()` and `compare_models()` added |
| `data/external/industry_risk/` | Industry analysis CSV outputs (7 files) |
| `notebooks/05_industry_risk_integration.ipynb` | Full integration demonstration and validation |

---

# Appendix: Glossary of Australian Banking Terms

| Term | Definition |
|------|------------|
| **APRA** | Australian Prudential Regulation Authority -- prudential regulator of ADIs |
| **APS 113** | Prudential Standard on Capital Adequacy: Internal Ratings-Based Approach |
| **APS 112** | Prudential Standard on Capital Adequacy: Standardised Approach |
| **APS 220** | Prudential Standard on Credit Risk Management |
| **AIRB** | Advanced Internal Ratings-Based approach |
| **ADI** | Authorised Deposit-Taking Institution |
| **CCF** | Credit Conversion Factor -- converts off-balance-sheet to on-balance-sheet equivalent |
| **CLTV** | Combined Loan-to-Value ratio (multiple liens) |
| **DA** | Development Approval -- council planning permission |
| **DOCA** | Deed of Company Arrangement -- insolvency restructure mechanism |
| **DSCR** | Debt Service Coverage Ratio |
| **DTI** | Debt-to-Income ratio |
| **EAD** | Exposure at Default |
| **GRV** | Gross Realisation Value -- total expected sales revenue of a development |
| **GSR** | General Security Agreement -- security over all present and after-acquired property |
| **HVCRE** | High-Volatility Commercial Real Estate |
| **ICR** | Interest Coverage Ratio |
| **IO** | Interest-Only (loan repayment type) |
| **IRB** | Internal Ratings-Based approach to credit risk capital |
| **LGD** | Loss Given Default |
| **LMI** | Lenders Mortgage Insurance -- protects the lender, not borrower |
| **LTC** | Loan-to-Cost ratio |
| **LTV / LVR** | Loan-to-Value Ratio |
| **MoC** | Margin of Conservatism |
| **P&I** | Principal and Interest (loan repayment type) |
| **PD** | Probability of Default |
| **PPSR** | Personal Property Securities Register -- national register of security interests in personal property |
| **PMSI** | Purchase Money Security Interest -- priority interest under PPSR |
| **QS** | Quantity Surveyor -- independent construction cost assessor |
| **RWA** | Risk-Weighted Assets |
| **TDC** | Total Development Cost |

# LGD-Commercial: Learning Manual for New Staff

**Audience:** New joiners with credit risk or model governance background  
**Purpose:** Understand the integrated LGD framework, from proxy baseline to APS 113 calibration layer  
**How to use:** Start with Part 1 (foundations). Then read Part 2 for the dual-framework approach (proxy baseline + calibration). Work through Part 3 module by module.

---

## Contents

- [Part 1: Credit Risk Foundations](#part-1-credit-risk-foundations)
  - [1.1 Why Banks Worry About Losses](#11-why-banks-worry-about-losses)
  - [1.2 The Three Key Risk Numbers: PD, LGD, EAD](#12-the-three-key-risk-numbers-pd-lgd-ead)
  - [1.3 What is LGD?](#13-what-is-lgd)
  - [1.4 How Recoveries Work](#14-how-recoveries-work)
  - [1.5 Why Timing Matters: Discounting](#15-why-timing-matters-discounting)
  - [1.6 Downturn LGD: The Regulatory Requirement](#16-downturn-lgd-the-regulatory-requirement)
  - [1.7 Key Terms Glossary](#17-key-terms-glossary)
- [Part 2: The Dual-Framework Approach](#part-2-the-dual-framework-approach)
  - [2.1 Proxy Baseline (Portfolio Demo)](#21-proxy-baseline-portfolio-demo)
  - [2.2 APS 113 Calibration Layer (New)](#22-aps-113-calibration-layer-new)
  - [2.3 Pipeline Integration](#23-pipeline-integration)
  - [2.4 When to Use Each Framework](#24-when-to-use-each-framework)
- [Part 3: Module-by-Module Walkthrough](#part-3-module-by-module-walkthrough)
  - [Module 01: Feature Engineering](#module-01-feature-engineering)
  - [Module 02: Residential Mortgage LGD](#module-02-residential-mortgage-lgd)
  - [Module 03: Commercial Cash-Flow Lending LGD](#module-03-commercial-cash-flow-lending-lgd)
  - [Module 04: Receivables and Invoice Finance LGD](#module-04-receivables-and-invoice-finance-lgd)
  - [Module 05: Trade and Contingent Facilities LGD](#module-05-trade-and-contingent-facilities-lgd)
  - [Module 06: Asset and Equipment Finance LGD](#module-06-asset-and-equipment-finance-lgd)
  - [Module 07: Development Finance LGD](#module-07-development-finance-lgd)
  - [Module 08: CRE Investment LGD](#module-08-cre-investment-lgd)
  - [Module 09: Residual Stock LGD](#module-09-residual-stock-lgd)
  - [Module 10: Land and Subdivision LGD](#module-10-land-and-subdivision-lgd)
  - [Module 11: Bridging Loan LGD](#module-11-bridging-loan-lgd)
  - [Module 12: Mezzanine and Second Mortgage LGD](#module-12-mezzanine-and-second-mortgage-lgd)
  - [Module 13: Cross-Product Comparison](#module-13-cross-product-comparison)
- [Part 4: The Calibration Pipeline](#part-4-the-calibration-pipeline)
  - [4.1 Workout Data Generation](#41-workout-data-generation)
  - [4.2 Long-Run LGD Calculation](#42-long-run-lgd-calculation)
  - [4.3 Downturn Overlay](#43-downturn-overlay)
  - [4.4 LGD-PD Correlation Adjustment](#44-lgd-pd-correlation-adjustment)
  - [4.5 Margin of Conservatism (MoC)](#45-margin-of-conservatism-moc)
  - [4.6 Regulatory Floor](#46-regulatory-floor)
  - [4.7 Validation and Backtesting](#47-validation-and-backtesting)
- [Part 5: How the Pipeline Fits Together](#part-5-how-the-pipeline-fits-together)
- [Part 6: Running the Project](#part-6-running-the-project)

---

## Part 1: Credit Risk Foundations

### 1.1 Why Banks Worry About Losses

When a bank lends money, there is always a chance the borrower cannot repay. This is called **credit risk**. If enough borrowers default at the same time — say, during a recession — the bank could lose so much money that it fails.

Regulators (in Australia, APRA — the Australian Prudential Regulation Authority) require banks to:

1. **Measure** how much they could lose on every loan.
2. **Hold capital** (their own money, not depositors') as a buffer against those losses.
3. **Report** those estimates transparently.

This project builds the tools that estimate those losses — specifically, the **LGD** component. APRA's IRB framework (APS 113) specifies exactly how to do this.

---

### 1.2 The Three Key Risk Numbers: PD, LGD, EAD

Banks use three numbers together to estimate how much money they might lose on a loan:

| Symbol | Name | Plain English | Example |
|--------|------|---------------|---------|
| **PD** | Probability of Default | How likely is the borrower to stop paying? | 2% chance in the next year |
| **LGD** | Loss Given Default | If they do default, what fraction of the loan is lost? | 35% of the loan is unrecovered |
| **EAD** | Exposure at Default | How much is actually owed when they default? | $500,000 outstanding |

The expected loss on a loan is:

```
Expected Loss = PD × LGD × EAD
             = 2% × 35% × $500,000
             = $3,500
```

This project builds the **LGD** models. The PD models live in a separate repo (PD-and-scorecard-commercial).

---

### 1.3 What is LGD?

**LGD (Loss Given Default)** is the percentage of the exposure you do *not* get back after a borrower defaults.

```
LGD = (EAD - NetRecoveryPV) / EAD
    = 1 - Recovery Rate
```

where `NetRecoveryPV = PV(Recoveries) - PV(Costs)` — the present value of what you recover, minus what it costs to recover it.

**Example:**
- Bank lends $200,000 secured against a house.
- Borrower defaults. Bank sells the house for $160,000 after paying $15,000 in legal and selling costs.
- Net recovery = $160,000 − $15,000 = $145,000.
- LGD = ($200,000 − $145,000) / $200,000 = **27.5%**

LGD is never exactly zero (there are always costs) and can exceed 100% if recovery is impossible and costs are high.

---

### 1.4 How Recoveries Work

Not all loans are the same. The amount recovered depends on:

**1. Security (collateral)**
- A loan backed by a house, equipment, or business assets can be recovered by selling that asset.
- An unsecured loan (e.g., overdraft) is recovered only if the borrower has cash or can restructure.

**2. Cure vs Non-Cure (especially mortgages)**
- **Cure:** Borrower fixes the problem (catches up on arrears, refinances).
- **Non-Cure:** Borrower cannot fix the problem; asset must be sold or restructured.

For mortgages, cure probability depends on LVR (loan-to-value ratio), arrears stage, and borrower type.

**3. Resolution path**
Non-cure resolution can happen via:
- Liquidation / forced sale (get whatever the asset sells for)
- Restructure / work-out (reduce the loan, extend the term, change repayment)
- Guarantor / third-party recovery
- Write-off (acknowledge the loss and move on)

Each path has a different recovery amount and timing.

**4. Time to recover (recovery lag)**
- Money received in 2 years is worth less than money received today (discounting).
- Property sales take time; restructures take time; workouts take time.
- Longer recovery lag = lower present value of recovery = higher LGD.

---

### 1.5 Why Timing Matters: Discounting

Money today is worth more than money tomorrow, because you can invest it. This is captured in **discounting**:

```
Present Value = Future Amount / (1 + Discount Rate) ^ Years
```

For example, if the discount rate is 5% and you recover $100,000 in 2 years:

```
PV = $100,000 / (1.05 ^ 2) = $100,000 / 1.1025 = $90,703
```

In this repo, the discount rate is typically the bank's **cost of funds** (what the bank pays for deposits) or the **contract rate** on the loan, whichever is higher.

**Why this matters for LGD:**
- If the borrower defaults and you recover $100,000 in 3 years instead of 1 year, the present value drops significantly.
- Assets that take longer to sell (land, development property) have **higher LGD** because recovery is delayed.
- Faster-moving collateral (equipment, receivables) have **lower LGD** because recovery is quicker.

---

### 1.6 Downturn LGD: The Regulatory Requirement

APRA requires banks to estimate LGD not just in normal times, but also in a **downturn** (recession, market stress).

In a downturn:
- Asset values fall (e.g., house prices drop 20%)
- Recovery takes longer (fewer buyers, lower urgency)
- Costs increase (forced sales, more legal intervention)
- Cure rates fall (borrowers in financial distress are less likely to cure)

**Regulatory expectation:**

```
LGD_downturn >= LGD_base (always)
```

The ratio is called the **downturn scalar**:

```
Downturn Scalar = LGD_downturn / LGD_base
```

For mortgages, a downturn scalar of 1.2–1.5 is typical (LGD goes up 20–50%). For unsecured/junior lending, it can be higher (2.0+).

---

### 1.7 Key Terms Glossary

| Term | Definition |
|------|-----------|
| **APS 113** | APRA standard for internal rating-based (IRB) credit risk models. Covers PD, LGD, EAD |
| **Base LGD** | LGD in normal economic conditions |
| **Downturn LGD** | LGD during a stress / recession scenario |
| **Final LGD** | LGD after all adjustments: downturn, MoC, regulatory floor |
| **EAD** | Exposure at Default — how much is owed when default happens |
| **LVR** | Loan-to-Value — the loan divided by the asset value |
| **MoC** | Margin of Conservatism — additional buffer required by regulators (s.65 of APS 113) |
| **LIP** | Loss in Liquidation / Loss in Portfolio — the loss from forced sale vs normal market value |
| **Cure Rate** | Probability that a defaulting borrower cures the default before full loss |
| **Workout LGD** | LGD calculated from observed workout data (historical defaults and their recovery outcomes) |
| **Realized LGD** | LGD actually observed after default (the historical ground truth) |
| **Proxy LGD** | LGD estimated using transparent proxies (recovery haircuts, recovery time, costs) without full workout data |
| **Vintage-EWA** | Vintage-Exposure-Weighted Average — adjusts for seasoning and portfolio mix changes over time |
| **Frye-Jacobs** | Model for LGD-PD correlation based on systematic economic factors |

---

## Part 2: The Dual-Framework Approach

This repo uses **two frameworks side-by-side**:

1. **Proxy Baseline** (for portfolio demonstration and new lending)
2. **APS 113 Calibration Layer** (for IRB model governance and stress testing)

Both produce LGD outputs, but they use different input sources and methodologies. Understanding when to use each is critical.

---

### 2.1 Proxy Baseline (Portfolio Demo)

The **proxy baseline** is a transparent, auditable way to estimate LGD when you do not have detailed workout history.

**Inputs:**
- Loan terms (amount, rate, tenor)
- Collateral assumptions (type, value, haircut on sale)
- Recovery timing (months to sale, months to restructure)
- Costs (legal, auction, holding)
- Macro overlays (downturn adjustments)

**Logic:**
```
LGD_base = 1 - Net Recovery Rate

Net Recovery Rate = (Collateral Value × (1 - Haircut) - Costs) 
                   × Discount Factor
                   / EAD
```

**Example (mortgage):**
- EAD: $250,000
- House value: $300,000 (LVR = 83%)
- Sale haircut: 8% + 3% costs = 11%
- Recovery: $300,000 × (1 - 0.11) = $267,000
- Discounting (2 years @ 5%): $267,000 / 1.05^2 = $241,932
- LGD_base = (250,000 - 241,932) / 250,000 = **3.2%**

**Downturn overlay (proxy):**
- House value falls 20% (stress scenario)
- Sale takes longer (12 months instead of 6)
- Cure rate falls 30%
- LGD_downturn = **12%** (e.g., 3.5× base)

**Advantages:**
- Transparent, easy to audit and explain
- Works for new lending (no historical data)
- Consistent across products
- Can be adjusted for market conditions

**Limitations:**
- Relies on assumed recovery haircuts and timing (not calibrated to actual workout data)
- Downturn overlays are scenario-based, not calibrated to historical stress periods
- Does not produce formal APS 113 compliance evidence

---

### 2.2 APS 113 Calibration Layer (New)

The **calibration layer** uses historical workout data to estimate LGD using APS 113-compliant methods.

**Data source:**
- Synthetic historical workout tape (2014–2024, 10 years of simulated defaults)
- Each record: loan characteristics, default date, resolution path, recovery amount, recovery timing, costs

**Pipeline (correct APS 113 order):**

```
Realized LGD (observed losses)
    ↓
Long-Run LGD (vintage-weighted, seasoning adjusted) — APS 113 s.43
    ↓
Downturn Overlay (recession loss factor) — APS 113 s.46–50
    ↓
LGD-PD Correlation Adjustment (Frye-Jacobs) — APS 113 s.55–57
    ↓
Margin of Conservatism (MoC) — APS 113 s.65
    Applied to downturn LGD, NOT long-run LGD
    ↓
Regulatory Floor (s.58, product-specific)
    ↓
Final Calibrated LGD
```

**Key modules:**

| Module | Purpose |
|--------|---------|
| `src/lgd_calculations.py` | Compute realized LGD from workout records; apply vintage-EWA |
| `src/regime_classifier.py` | Identify economic regimes (expansion, downturn, stress) from macro indicators |
| `src/rba_rates_loader.py` | Load real RBA B6 indicator rates for discount rate calculation |
| `src/lgd_pd_correlation.py` | Estimate LGD-PD systematic correlation (Frye-Jacobs model) |
| `src/moc_framework.py` | Calculate MoC from 5 APS 113 s.65 sources (data, model, estimation, coverage, uncertainty) |
| `src/apra_benchmarks.py` | Benchmark internal LGD vs APRA peer ADI impairment ratios (directional) |
| `src/validation_suite.py` | Extended validation: Gini coefficient, Hosmer-Lemeshow, PSI, out-of-time (OOT) testing |
| `src/aps113_compliance.py` | Generate APS 113 compliance map (which standard sections are satisfied) |

**Example calibration output:**

```
Product: Residential Mortgage
Long-Run LGD:        15.2%  (realized LGD from 2014-2024 defaults)
Downturn Overlay:    +8.5pp (recession makes it worse)
LGD Downturn:        23.7%  
LGD-PD Correlation:  -2.1pp (negative correlation adjustment)
LGD Before MoC:      21.6%  
MoC (+5 sources):    +3.2pp (data quality, model uncertainty, etc.)
Before Floor:        24.8%  
Regulatory Floor:    10.0%  (regulatory minimum)
Final LGD:           24.8%  (ceiling: max(before floor, floor))
```

**Advantages:**
- Based on observed historical loss outcomes
- Meets APS 113 regulatory requirements for IRB models
- Produces evidence of model governance (compliance map, MoC register, backtests)
- Supports stress testing and RAROC pricing

**Limitations:**
- Requires workout data (synthetic in this repo; real data needed for production)
- Synthetic workout data does not replicate real default and resolution behaviour
- Downturn overlay is estimated from synthetic defaults, not calibrated to real stress periods
- MoC values are illustrative and require Model Risk Committee approval in production

---

### 2.3 Pipeline Integration

Both frameworks run in parallel and feed into the same output contract.

**Proxy baseline used for:**
- Portfolio-level LGD demonstration
- New lending scoring (API)
- Cross-product comparison
- Sensitivity analysis

**Calibration layer used for:**
- IRB model validation
- Stress testing and CCAR submissions
- RWA capital calculations
- Model governance reporting
- Regulator evidence

**Output tables:** Both paths produce the same canonical output columns:
- `lgd_base`, `lgd_downturn`, `lgd_final`
- `source_path` (flag for which framework was used)
- Governance trace: `overlay_source`, `parameter_version`, `scenario_id`

---

### 2.4 When to Use Each Framework

| Use Case | Framework | Reason |
|----------|-----------|--------|
| New lending decision (single loan) | Proxy Baseline | Fast, works without historical data, transparent |
| Portfolio demo or presentation | Proxy Baseline | Simple, consistent across all products |
| IRB model validation (audit) | Calibration Layer | Required by APS 113, evidence-based |
| Stress testing / CCAR | Calibration Layer | Regulatory requirement, calibrated downturn |
| Sensitivity / what-if analysis | Proxy Baseline | Easy to adjust assumptions |
| RWA capital calculation | Calibration Layer (preferred) or Proxy Baseline | Capital rules specify which approach |

---

## Part 3: Module-by-Module Walkthrough

### Module 01: Feature Engineering

**Purpose:** Transform raw loan/borrower/collateral data into model-ready features.

**Inputs:**
- Loan-level raw data (term, rate, product)
- Borrower data (age, income, behaviour flags)
- Collateral data (type, value, location, condition)
- Macro data (interest rates, property price index, economic regime flags)

**Key operations:**
- **Bucketization:** Convert continuous variables (LVR, DSCR) into bands (e.g., 80-90%, 90-100%)
- **Vintage/seasoning:** Calculate years-on-book, apply seasoning curves
- **Segmentation:** Assign product code, geography, risk level
- **Macro mapping:** Map loan origination date to economic regime (expansion, downturn, stress)
- **Fallback handling:** If a required field is missing, apply documented default (log a warning)

**Output:** Feature-ready CSV with all required columns for product modules.

---

### Module 02: Residential Mortgage LGD

**What it does:** Calculates LGD for home loans, treating cure and non-cure outcomes separately.

**Proxy baseline approach:**

```
LGD_final = (1 - P(cure)) × LGD_liquidation
```

- **Cure probability:** Function of LVR, LMI flag, arrears stage, borrower type
- **LGD_liquidation:** Estimated from property value, sale haircut (8%), costs (3%), recovery timing (12 months), discounting

**Calibration layer approach:**
- Workout LGD from observed defaults (2014–2024)
- Vintage-EWA (adjust for portfolio composition over time)
- Downturn overlay based on property price index stress
- MoC from data quality and model uncertainty
- Regulatory floor (10% per APS 113)

**Key segments:**
- LVR buckets (≤60%, 60-80%, 80-90%, >90%)
- LMI flag (insured vs not)
- Borrower type (investor, owner-occupied, first-time buyer)

**Expected LGD range:** 5–35% (base), 8–50% (downturn), depending on LVR and LMI.

---

### Module 03: Commercial Cash-Flow Lending LGD

**What it does:** Parent framework for all commercial lending products (term, overdraft, receivables, trade, asset finance).

**Three sub-components:**

**A. Term Lending (secured, partially secured, unsecured)**
- Secured: estimated recovery from business asset sales, EBITDA-based covenant haircuts
- Unsecured: recovery from restructure / guarantor, lower recovery rates
- LGD drivers: security type, loan tenure, industry risk, DSCR

**B. Overdraft / Revolver**
- EAD not constant; depends on utilization and Credit Conversion Factor (CCF)
- LGD estimated for the drawn portion
- Typical LGD: 30–50%

**C. Integration hooks**
- Receivables/invoice finance
- Trade/contingent facilities
- Asset/equipment finance
- Sub-segments nested under cash-flow lending parent

**Key segments:**
- Security type (secured, partially secured, unsecured)
- Industry (primary driver of downturn behaviour)
- Loan tenure (<2yr, 2-5yr, 5-10yr, >10yr)
- Facility type (term, revolver, overdraft)

**Expected LGD range:** 25–65% (base), 40–80% (downturn).

---

### Module 04: Receivables and Invoice Finance LGD

**What it does:** LGD for lending against business receivables and invoices.

**Proxy approach:**
- Advance rate: typically 70–85% of eligible receivables
- Dilution: accounts receivable may be disputed or uncollectable (5–15% estimate)
- Collections control: bank can hold cash from customer payments (security)
- LGD depends on advance rate, dilution, and receivables quality

**Segments:**
- Customer concentration (single-customer vs multi-customer)
- Industry concentration (construction, manufacturing, services)
- Receivables ageing (0-30 days, 30-60 days, >60 days)

**Expected LGD:** 15–40% (typically lower than unsecured lending because security is strong).

---

### Module 05: Trade and Contingent Facilities LGD

**What it does:** LGD for trade finance (letters of credit, guarantees, bonds) and contingent liabilities.

**Proxy approach:**
- Contingent facility may not convert to drawn exposure (claim probability)
- If it does convert (e.g., guarantee is called), recovery depends on security (cash deposit, counter-guarantee)
- LGD = Expected loss on converted amount

**Segments:**
- Facility type (LC, guarantee, bond, standby credit)
- Tenor (<1 year, 1-5 years, >5 years)
- Security (cash-backed, guaranteed, unsecured)

**Expected LGD:** 10–50% (depends on claim probability and security type).

---

### Module 06: Asset and Equipment Finance LGD

**What it does:** LGD for loans secured against equipment, vehicles, machinery.

**Proxy approach:**
- Equipment value depreciates over time (age curve)
- Recovery from repossession and sale (auctioneering, remarketing)
- Residual exposure (loan balance may exceed asset value) is typically unsecured
- LGD = (Residual unsecured + Losses on asset sale) / EAD

**Segments:**
- Asset type (vehicles, plant & machinery, IT equipment)
- Equipment age (<2yr, 2-5yr, 5-10yr, >10yr)
- Tenure to maturity

**Expected LGD:** 20–45% (higher than mortgages because assets depreciate faster).

---

### Module 07: Development Finance LGD

**What it does:** LGD for loans to developers (land development, construction loans, project finance).

**Risk drivers:**
- Stage of completion (pre-commencement, early stage, completion, sell-through)
- Gross Revenue Value (GRV) relative to loan (loan coverage)
- Cost-to-complete risk (budget overruns, delays)
- Market risk (sell-through uncertainty, price risk)

**Proxy approach:**
- Base LGD low for completed projects with strong sell-through (e.g., 15%)
- Rises sharply for early-stage or high loan-to-value deals (e.g., 40–60%)

**Downturn:**
- Demand falls (sell-through delays)
- Prices fall (forced selling into weak market)
- Developers may walk away (option exercise on negative equity)
- Downturn scalar can be 2.0+ (development is cyclical)

**Expected LGD:** 20–50% (base), 40–75% (downturn).

---

### Module 08: CRE Investment LGD

**What it does:** LGD for lending against commercial real estate (office, retail, industrial, multi-residential).

**Risk drivers:**
- Loan-to-Value (LVR) of the property portfolio
- Debt Service Coverage Ratio (DSCR) — can the rent pay the interest + principal?
- Weighted Average Lease Expiry (WALE) — when do tenants' leases expire?
- Vacancy rate
- Tenant concentration (is it reliant on 1–2 big tenants?)

**Proxy approach:**
- If DSCR is weak (e.g., <1.2), recovery risk is high because rents may not cover costs
- Longer WALE is better (stable income for longer)
- Higher vacancy = weaker collateral

**Downturn:**
- Rents fall; DSCR deteriorates
- Some tenants cannot pay or leave (higher vacancy)
- Refinance risk: loan maturity falls due; difficult to refinance
- Forced sales into weak market

**Expected LGD:** 20–45% (base), 35–65% (downturn).

---

### Module 09: Residual Stock LGD

**What it does:** LGD for loans to real estate businesses holding unsold inventory (residual stock, vacant properties).

**Risk drivers:**
- Absorption rate (months to sell the stock)
- Discount-to-clear (how much to cut prices to sell)
- Holding costs (interest, rates, maintenance, security)
- Market conditions (buyer demand, inventory levels)

**Proxy approach:**
- LGD driven by time-to-sale and markdown
- Example: $10M stock, 24-month absorption, 12% discount-to-clear, 2% holding costs
- Recovery = $10M × (1 - 0.12 - 0.02) = $8.6M (86%)
- LGD = 14%

**Expected LGD:** 15–35%.

---

### Module 10: Land and Subdivision LGD

**What it does:** LGD for loans against raw land or land under subdivision.

**Risk drivers:**
- Land liquidity (market depth; how quickly can it be sold?)
- Zoning/planning approval status
- Development feasibility
- Market conditions

**Challenges:**
- Land has low liquidity (may take 12+ months to sell)
- No income stream (unlike investment real estate)
- Market risk (land values volatile, especially at cycle turns)
- Subdivision risk (development costs, off-the-plan sales uncertainty)

**Proxy approach:**
- Higher haircut on sale (e.g., 15–20% vs 8% for mortgages)
- Longer recovery time (12–18 months vs 6–12 months)
- LGD typically 25–40% (base), 45–60% (downturn)

**Expected LGD:** 25–40% (base), 45–60% (downturn).

---

### Module 11: Bridging Loan LGD

**What it does:** LGD for short-term loans to bridge a funding gap (typically 6–12 months).

**Risk drivers:**
- Exit certainty (is the exit event assured?)
- Valuation risk (is the property value secure?)
- Time to exit (how long until the main loan settles or property sells?)
- Failed-exit scenario (what if the planned sale or refinance doesn't happen?)

**Exit types:**
- Developer exit (sell-through of units)
- Investor exit (refinance into long-term mortgage)
- Sale/contract completion

**Proxy approach:**
- If exit is certain and near term (e.g., 3 months), LGD is low (5–10%)
- If exit is uncertain or distant (e.g., 12+ months), LGD rises (20–35%)
- Failed-exit scenario (loan rolls over or property sold at discount) drives downturn (LGD 35–50%)

**Expected LGD:** 5–20% (base, clear exit), 20–50% (base, uncertain exit), 35–60% (downturn).

---

### Module 12: Mezzanine and Second Mortgage LGD

**What it does:** LGD for junior lien mortgages and subordinated financing (mezzanine debt).

**Key feature: Recovery waterfall**

```
Collateral Value
    ↓ (pay 1st mortgage)
Senior Lender Recovery
    ↓ (pay 2nd mortgage / mezz debt)
Mezzanine Recovery
    ↓ (pay costs)
Residual Equity
```

**Example:**
- Property value: $500,000
- 1st mortgage (senior): $300,000
- 2nd mortgage (mezz): $100,000
- Costs: $20,000
- Waterfall: Senior gets $300,000; Mezz gets $180,000 (=$500,000 - $300,000 - $20,000); Equity holders get $0.

**Proxy approach:**
- Mezz LGD depends on how far down the waterfall it sits
- Typical mezz LGD: 40–70% (much higher than senior mortgages)

**Expected LGD:** 40–70% (base), 60–90% (downturn, because collateral value falls).

---

### Module 13: Cross-Product Comparison

**What it does:** Rolls up all 12 product modules into integrated portfolio view.

**Outputs:**
- **Weighted average LGD** by product and overall portfolio (exposure-weighted)
- **Downturn sensitivity** (ratio of downturn to base)
- **Recovery timing** (months to recover, segmented)
- **Portfolio mix** (contribution of each product to total exposure)
- **Risk ranking** (which products are riskiest?)

**Example output:**

| Product | Exposure | LGD Base | LGD Down | Downturn Scalar | Rank |
|---------|----------|----------|----------|-----------------|------|
| Mortgage | $2.5B | 12% | 18% | 1.50 | 1 (lowest risk) |
| Commercial | $1.2B | 38% | 55% | 1.45 | 3 |
| Development | $0.8B | 35% | 60% | 1.71 | 4 |
| Bridging | $0.3B | 15% | 42% | 2.80 | 5 (highest risk) |
| CRE Investment | $1.5B | 30% | 48% | 1.60 | 2 |
| **Portfolio** | **$6.3B** | **24%** | **39%** | **1.63** | |

---

## Part 4: The Calibration Pipeline

This section details the APS 113 calibration layer step-by-step.

---

### 4.1 Workout Data Generation

**Source:** `src/generators/` folder (11 product-specific generators)

**Purpose:** Create synthetic historical default and recovery data (2014–2024, 10 years).

**Process:**
1. For each product, sample loan characteristics (LVR, tenure, industry, etc.) from a distribution
2. Apply a default probability model (parameterized by macroeconomic regime)
3. For each default, simulate recovery: resolution path, recovery amount, timing, costs
4. Output: Parquet file with 1000–2000 simulated defaults per product

**Example (mortgage):**
```
loan_id, origination_date, ead, lgd_base, resolution_path, recovery_amount, recovery_months, costs
M0001, 2014-01-15, 280000, 0.18, cure, 275000, 2, 3500
M0002, 2014-02-10, 450000, 0.25, liquidation, 320000, 10, 18000
M0003, 2014-03-05, 200000, 0.12, restructure, 189000, 12, 4000
...
```

**Key assumption:** Synthetic defaults are generated to replicate realistic LGD distributions and macro sensitivity, but do not represent actual borrower behaviour.

---

### 4.2 Long-Run LGD Calculation

**Purpose:** Compute realized LGD from workout data, adjusted for seasoning and vintage mix.

**Formula:**

```
Long-Run LGD = Vintage-Weighted Exposure-Weighted Average of Realized LGD
             = Sum(Realized LGD_i × EAD_i × Vintage Weight_i) 
               / Sum(EAD_i × Vintage Weight_i)
```

**Vintage weighting:** Adjusts for the fact that older defaults (2014–2016) may not represent the current portfolio because:
- Lending practices have changed
- Borrower quality (PD) has changed
- Collateral values and volatility have changed

**Seasoning adjustment:** Newer defaults (within 5 years) are weighted differently to reflect portfolio maturity.

**Output:** Product-level long-run LGD (one number), optionally segmented by risk level.

**Example:**
```
Product: Residential Mortgage
Vintage 2014-2016: realized LGD 18%, weight 30%
Vintage 2017-2019: realized LGD 14%, weight 35%
Vintage 2020-2024: realized LGD 16%, weight 35%
Long-run LGD = (18% × 30% + 14% × 35% + 16% × 35%) = 15.9%
```

---

### 4.3 Downturn Overlay

**Purpose:** Estimate how LGD changes during a recession or stress period.

**Approach:**
1. Classify economic regimes (using RBA/ABS data or synthetic proxy)
2. Calculate LGD separately for each regime: normal, expansion, downturn, stress
3. Downturn overlay = LGD_downturn - LGD_normal

**Regime classification** (`src/regime_classifier.py`):
- Uses RBA Official Cash Rate (OCR) and economic data to identify cycles
- Downturn = 2 consecutive quarters of negative GDP growth (recession) or 10%+ unemployment
- Stress = Extreme event (GFC-like scenario)

**Example:**
```
Regime    | LGD Average | # Defaults | Notes
Normal    | 14%        | 120        | 2017-2019, 2021-2023
Expansion | 12%        | 85         | 2010-2012
Downturn  | 22%        | 45         | 2008 GFC, 2020 COVID
Stress    | 28%        | 12         | 2008 peak stress
```

**Downturn scalar:** (LGD_downturn / LGD_normal) = (22% / 14%) = **1.57**

**Regulatory floor (s.46–50):** Downturn LGD must be at least 1.1× LR-LGD under normal cycling.

---

### 4.4 LGD-PD Correlation Adjustment

**Purpose:** Adjust LGD for correlation with PD (systematic risk).

**Concept:** In a downturn, both defaults (PD) and losses (LGD) increase together:
- Borrowers are stressed → more default → harder to recover → both PD and LGD up
- This co-movement is **LGD-PD correlation**
- APS 113 requires adjustment for this systematic risk

**Method: Frye-Jacobs model** (`src/lgd_pd_correlation.py`)

```
LGD_adjusted = LGD_base + ρ × (PD_stressed - PD_normal) × β
```

where:
- ρ = LGD-PD correlation coefficient (typically 0.1–0.3 for mortgages, 0.2–0.5 for unsecured)
- PD_stressed - PD_normal = change in default probability under stress
- β = sensitivity of LGD to systematic shocks

**Output:** Adjustment to LGD (usually 1–5 percentage points, varies by product).

**Example (mortgage):**
```
Base LGD:           15%
PD effect on LGD:   +1.2pp (correlation adjustment)
Adjusted LGD:       16.2%
```

---

### 4.5 Margin of Conservatism (MoC)

**Purpose:** Add a buffer to account for model uncertainty, data limitations, and estimation risk.

**APS 113 s.65 requires MoC to reflect:**

1. **Data quality risk** — how complete/reliable is the workout data?
2. **Model risk** — are the assumptions sound?
3. **Estimation risk** — sample size, parameter uncertainty
4. **Coverage risk** — does the model cover all relevant products/segments?
5. **Uncertainty in forecasts** — how stable are future downturn assumptions?

**Framework** (`src/moc_framework.py`):

Each source is scored (low/medium/high risk):

| Source | Risk Level | MoC Contribution |
|--------|-----------|-----------------|
| Data Quality | High (synthetic) | +2.0pp |
| Model Risk | Medium | +1.0pp |
| Estimation Risk | Medium | +0.8pp |
| Coverage Risk | Low | +0.2pp |
| Forecast Uncertainty | High | +1.5pp |
| **Total MoC** | | **+5.5pp** |

**Application:**
```
LGD downturn:     22%
MoC:              +5.5pp
LGD before floor: 27.5%
```

**Note:** MoC is applied to **downturn LGD**, not long-run LGD, per APS 113 s.65.

---

### 4.6 Regulatory Floor

**Purpose:** Ensure LGD does not fall below regulatory minimums.

**APS 113 s.58 specifies minimum floors by product:**

| Product | Minimum Floor |
|---------|--------------|
| Mortgage | 5–10% |
| Commercial unsecured | 15–25% |
| Commercial secured | 25–35% |
| Development | 25–40% |
| CRE Investment | 20–25% |
| Equipment / Receivables | 15–20% |

**Application:**

```
LGD calculated:     5.8%
Regulatory floor:   10%
Final LGD:          max(5.8%, 10%) = 10%
```

**Note:** Floors ensure minimum capital is held even if estimated LGD is optimistic.

---

### 4.7 Validation and Backtesting

**Purpose:** Test that the calibration model works in practice.

**Methods:**

**1. In-Sample Validation:**
- Gini coefficient: measure of model discrimination (ability to separate low-loss from high-loss loans)
  - Target: Gini > 0.4 for adequate discrimination
- Hosmer-Lemeshow test: do predicted LGD quantiles match observed?
  - p-value > 0.05 indicates model fit is acceptable

**2. Out-of-Time (OOT) Validation:**
- Train model on 2014–2019 data
- Test on 2020–2024 data
- Measure whether predictions remain stable across different time periods

**3. Segment Validation:**
- Validate model separately for each product and risk segment
- Flag segments where model performs poorly (high error, poor fit)

**4. Stability Testing:**
- Parameter Stability Index (PSI): measure how much the population distribution has shifted
  - PSI < 0.1: stable
  - 0.1 < PSI < 0.25: moderate shift
  - PSI > 0.25: large shift (model may need retraining)

**Output:** Validation report with pass/fail on each test, flagged issues for Model Risk Committee review.

---

## Part 5: How the Pipeline Fits Together

### Execution Flow

```
1. Raw Input Data
   ├── Load loan data (CSV/Parquet)
   ├── Load macro regime flags
   └── Load discount rates (RBA B6)
        ↓
2. Feature Engineering (Module 01)
   └── Create features, vintage/seasoning, segmentation
        ↓
3. Product Modules (Modules 02–12)
   ├─ Proxy Baseline Path:
   │  ├── Calculate LGD_base (transparent proxy)
   │  ├── Apply downturn overlay (macro-driven)
   │  └── Apply regulatory floor
   │
   └─ Calibration Layer Path (parallel):
      ├── Load synthetic workout data
      ├── Calculate realized LGD, vintage-EWA
      ├── Apply downturn overlay (regime-based)
      ├── Apply LGD-PD correlation adjustment
      ├── Apply MoC (5 sources)
      └── Apply regulatory floor
        ↓
4. Cross-Product Aggregation (Module 13)
   ├── Merge all products
   ├── Compute exposure-weighted averages
   └── Sensitivity analysis
        ↓
5. Output Tables
   ├── Loan-level: lgd_base, lgd_downturn, lgd_final
   ├── Segment summaries
   ├── Governance reports (overlay trace, MoC register, compliance map)
   └── Validation reports (backtests, stability, OOT)
        ↓
6. Downstream Consumers
   ├── Expected-Loss-Engine (PD × LGD × EAD)
   ├── Stress Testing (scenarios with different downturn scalars)
   ├── RWA Capital (IRB formula with final LGD)
   └── RAROC Pricing (LGD in return hurdle calculations)
```

### Key Integration Points

**1. Overlay Parameter Governance**
- All overlay adjustments (downturn, MoC) sourced from `data/config/overlay_parameters.csv`
- Parameter version and hash logged on every run
- Versioned manifest at `data/config/overlay_parameters_manifest.json`

**2. Discount Rate Management** (`src/rba_rates_loader.py`)
- Real RBA B6 indicator rates used for PV calculations
- Fallback to tier-4/5 proxies (cost of funds, contract rate) if data unavailable
- All rate fallbacks logged with warnings

**3. Segmentation Consistency**
- All products use same segment codes (industry, geography, risk level, vintage)
- Consistency checked in validation report
- Missing segments flagged for review

**4. Reproducibility & Determinism**
- Synthetic workout data seed fixed (reproducibility)
- All random operations logged (transparency)
- Determinism check: run twice, outputs must be identical

---

## Part 6: Running the Project

### Quick Start

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Run demo pipeline (proxy baseline):**
```bash
python -m src.pipeline.demo_pipeline
```

**3. Run full calibration pipeline:**
```bash
python -m src.pipeline.calibration_pipeline --products all
```

**4. Run validation sequence:**
```bash
python -m src.pipeline.validation_pipeline
```

---

### Common Tasks

**Generate synthetic workout data:**
```bash
python -m src.data.generator --seed 42
```

**Score a single new loan (API):**
```python
from src.lgd_scoring import score_single_loan

result = score_single_loan(
    payload={"loan_id": "L-001", "ead": 250000, "mortgage_class": "Standard"},
    product_type="mortgage",
    scenario_id="baseline"
)
print(result)  # {'lgd_base': 0.12, 'lgd_downturn': 0.18, 'lgd_final': 0.18}
```

**Score a batch of loans (CLI):**
```bash
python -m src.scoring.scoring --product-type mortgage --input-csv data/sample_loans.csv --output outputs/scored.csv
```

**Generate APS 113 compliance map:**
```bash
python -m src.pipeline.calibration_pipeline --products mortgage --compliance-report-only
```

---

### Output Tables

Key outputs in `outputs/tables/`:

| File | Audience | Content |
|------|----------|---------|
| `lgd_final.csv` | Portfolio managers | Loan-level LGD (base, downturn, final) |
| `lgd_final_summary_by_product.csv` | Risk committee | Exposure-weighted summary |
| `policy_parameter_register.csv` | Governance | All parameters used (version, values, hash) |
| `validation_sequence_report.csv` | Model risk | Validation test results (Gini, HosL, PSI) |
| `moc_summary_all_products.csv` | Governance | MoC breakdown (5 sources, per product) |
| `aps113_compliance_map.csv` | Regulator | Which APS 113 standards are met (yes/partial/no) |
| `lgd_pd_correlation_report.csv` | Modellers | LGD-PD correlation estimates (by product) |
| `rba_discount_rate_register.csv` | Audit | Discount rates used (source, tier, fallback) |

---

### Troubleshooting

**Q: Output files have NaN values**
A: Check logs for numeric coercion warnings. A required column may be missing or misnamed. Verify input CSV against `docs/data_dictionary.md`.

**Q: Downturn scalar is unrealistic (too high or too low)**
A: Check `overlay_parameters.csv` for the downturn overlay entry for your product. Verify macro regime classification is correct. `src.regime_classifier` can be imported directly to inspect or regenerate regime flags.

**Q: MoC is very high**
A: Check `src/moc_framework.py` for the MoC scoring. Data quality may be flagged as high-risk because workout data is synthetic. In production, update with real workout data to lower MoC.

**Q: Validation report shows poor model fit**
A: Run OOT test on 2020–2024 data to check for structural breaks (COVID pandemic, rate changes). May need to recalibrate parameters. Escalate to Model Risk Committee.

---

## Key References

- **APS 113:** APRA's Internal Ratings-Based Approach to credit risk (www.apra.gov.au)
- **Methodology manuals:** `docs/methodology_cashflow_lending.md`, `docs/methodology_property_backed_lending.md`
- **Data dictionary:** `docs/data_dictionary.md`
- **README:** [README.md](../README.md)
- **Project overview:** [PROJECT_OVERVIEW.md](../PROJECT_OVERVIEW.md)

---

**Document version:** 1.0  
**Last updated:** April 2026  
**Maintained by:** Portfolio Credit Risk team


# LGD-Commercial: Learning Manual for New Staff

**Audience:** New joiners with no prior credit risk experience  
**Purpose:** Understand what this project does, why it matters, and how each module works  
**How to use:** Read Part 1 before anything else. Then work through Part 2 module by module in order.

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
- [Part 2: Module-by-Module Walkthrough](#part-2-module-by-module-walkthrough)
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
- [Part 3: How the Pipeline Fits Together](#part-3-how-the-pipeline-fits-together)
- [Part 4: Running the Project](#part-4-running-the-project)

---

## Part 1: Credit Risk Foundations

### 1.1 Why Banks Worry About Losses

When a bank lends money, there is always a chance the borrower cannot repay. This is called **credit risk**. If enough borrowers default at the same time — say, during a recession — the bank could lose so much money that it fails.

Regulators (in Australia, APRA — the Australian Prudential Regulation Authority) require banks to:

1. **Measure** how much they could lose on every loan.
2. **Hold capital** (their own money, not depositors') as a buffer against those losses.
3. **Report** those estimates transparently.

This project builds the tools that estimate those losses — specifically, the **LGD** component.

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

This project builds the **LGD** models. The PD models live in a separate repo.

---

### 1.3 What is LGD?

**LGD (Loss Given Default)** is the percentage of the exposure you do *not* get back after a borrower defaults.

```
LGD = (What you lent - What you recovered) / What you lent
    = 1 - Recovery Rate
```

**Example:**
- Bank lends $200,000 secured against a house.
- Borrower defaults. Bank sells the house for $160,000 after paying $15,000 in legal and selling costs.
- Net recovery = $160,000 − $15,000 = $145,000.
- LGD = ($200,000 − $145,000) / $200,000 = **27.5%**

LGD is never exactly zero (there are always costs) and can exceed 100% if costs are very high relative to the loan balance.

---

### 1.4 How Recoveries Work

Not all loans are the same. The amount recovered depends on:

**1. Security (collateral)**
- A loan backed by a house or equipment can be recovered by selling that asset.
- An unsecured loan (no asset backing) relies only on the borrower's future earnings.
- More and better security → lower LGD.

**2. Cure probability**
- Some borrowers in default eventually catch up on payments and return to normal — this is called a **cure**.
- If a borrower cures, the bank loses nothing (LGD ≈ 0 for that loan).
- If the loan does not cure, it enters a **workout** (recovery) process.

**3. Recovery costs**
- Every recovery incurs costs: legal fees, property valuation, agent commissions, maintenance of the asset during sale.
- Typical costs range from 5% to 15% of the property value.

**4. Recovery timing**
- A property sale might take 12–24 months from default to settlement.
- A receivables book might be collected in 3–6 months.
- Money recovered later is worth less than money recovered today (time value of money).

---

### 1.5 Why Timing Matters: Discounting

A dollar received today is worth more than a dollar received in 12 months, because today's dollar can be invested.

Banks use a **discount rate** (typically the contract rate or cost of funds) to convert future recoveries back to today's value:

```
Present Value = Future Recovery / (1 + discount_rate)^years
```

**Example:**
- Expected recovery: $100,000 in 18 months.
- Discount rate: 6%.
- Present value = $100,000 / (1.06)^1.5 = **$91,480**

So even though the bank nominally recovers $100,000, its *real* recovery in today's money is only $91,480. This increases the effective LGD.

---

### 1.6 Downturn LGD: The Regulatory Requirement

APRA requires banks to estimate not just the *average* LGD across good and bad times, but the LGD that would occur during a **downturn** — a severe economic stress, like the GFC or COVID-19 period.

```
Downturn LGD = Long-Run LGD × Downturn Scalar
```

The **downturn scalar** is a multiplier (typically 1.08–1.30) that increases LGD to reflect:
- Falling property prices reduce collateral values.
- Rising unemployment means fewer borrowers cure.
- Higher interest rates raise the effective discount cost.
- Longer workout times because courts and agents are busier.

On top of this, banks add a **Margin of Conservatism (MoC)** — a small additional buffer (e.g., 2pp) to account for model uncertainty:

```
Final LGD = max(Downturn LGD + MoC, Regulatory Floor)
```

Every product in this project has its own downturn scalar logic because different assets behave differently in a downturn (property vs. equipment vs. receivables).

---

### 1.7 Key Terms Glossary

| Term | Definition |
|------|-----------|
| **APRA** | Australian Prudential Regulation Authority — the bank regulator |
| **IRB** | Internal Ratings-Based approach — banks build their own risk models (vs. using regulator's standard tables) |
| **EAD** | Exposure at Default — total amount owed when the borrower defaults |
| **CCF** | Credit Conversion Factor — portion of undrawn credit that is added to EAD (e.g., a 75% CCF means 75% of the undrawn limit is counted) |
| **LTV** | Loan-to-Value ratio — loan amount ÷ property value. LTV 80% means the loan is 80% of the property's value. Higher LTV = more risk |
| **ICR** | Interest Coverage Ratio — borrower's annual income (EBITDA) ÷ annual interest cost. ICR < 1 means the borrower cannot cover interest from income |
| **DSCR** | Debt Service Coverage Ratio — similar to ICR but includes principal repayments |
| **LMI** | Lenders Mortgage Insurance — insurance that pays out if a mortgage borrower defaults. Typically required for LTV > 80% |
| **GRV** | Gross Realisable Value — expected value of a completed development property |
| **Cure** | When a borrower in default catches up on payments and returns to performing |
| **Workout** | The bank's process of recovering money from a defaulted loan |
| **Seniority** | Where a lender sits in the repayment queue. Senior lenders get paid first |
| **Mezzanine** | A second, junior loan behind the senior lender — gets paid only after senior is repaid in full |
| **MoC** | Margin of Conservatism — buffer added to model estimates to account for uncertainty |
| **OOT** | Out-of-Time test — testing a model on data from a future period it wasn't trained on |
| **PSI** | Population Stability Index — measures how much the distribution of a variable has shifted over time |
| **WALE** | Weighted Average Lease Expiry — for commercial property, the average time until leases expire |
| **Vintage** | The year a loan was originated (taken out) |

---

## Part 2: Module-by-Module Walkthrough

> **Before you start:** Every module follows the same three-step pattern:
> 1. **Build base LGD** — what is the average long-run loss on this type of loan?
> 2. **Apply downturn adjustment** — how much worse does it get in a recession?
> 3. **Apply regulatory overlays** — floors, margins of conservatism, APRA rules.

---

### Module 01: Feature Engineering

**File:** `notebooks/01_feature_engineering.ipynb`  
**Source code:** `src/data_generation.py`, `src/segmentation.py`

#### What is feature engineering?

Raw loan data (customer records from a bank's systems) rarely comes in the exact form models need. **Feature engineering** is the process of cleaning, combining, and transforming that raw data into the variables (features) the LGD model uses.

Think of it like preparing vegetables before cooking — you wash them, chop them to the right size, and separate them by type before they go into the recipe.

#### What this module does

1. **Generates synthetic loan data** — since this is a portfolio/demo project, it creates realistic fake datasets for each product type (mortgage, commercial, development, cashflow lending). Each fake dataset has the same columns and distributions a real bank dataset would have.

2. **Engineers key variables:**

   | Variable | How it's built | Why it matters |
   |----------|----------------|----------------|
   | `ltv_at_default` | Loan balance ÷ property value at default time | Higher LTV = less property cushion = higher LGD |
   | `seasoning_months` | Number of months from loan start to default | Older loans are often better collateral |
   | `coverage_band` | Security coverage ratio bucketed into Low/Medium/High | Used for segmentation |
   | `ltv_band` | LTV grouped into ranges (e.g., 0–70%, 70–80%, 80%+) | Segments mortgages |
   | `origination_year` | Year the loan was taken out | Used for vintage analysis |
   | `pd_score_band` | PD score grouped into A/B/C/D/E | Risk ranking for cashflow lending |

3. **Assigns standard segment tags** — every loan gets labelled with `std_module`, `std_product_segment`, and `std_security_or_stage_band` so all products can be compared consistently in Module 13.

4. **Validates data contracts** — checks that required columns are present and values are in plausible ranges before any model runs.

#### Key outputs

- Cleaned and enriched loan DataFrames for each product.
- Segmentation labels that flow through every downstream module.

---

### Module 02: Residential Mortgage LGD

**File:** `notebooks/02_residential_mortgage_lgd.ipynb`  
**Source code:** `src/lgd_calculation.py` → `MortgageLGDEngine`

#### What is a residential mortgage?

A mortgage is a loan to buy a house or apartment. The property is the **security** (collateral). If the borrower stops paying, the bank can sell the property to recover the loan.

#### Why mortgages have lower LGD than unsecured loans

- The property provides a concrete asset to sell.
- Australia's housing market has historically maintained prices.
- LMI (Lenders Mortgage Insurance) can pay out on high-LTV loans, further reducing the bank's loss.

#### How LGD is built (step by step)

**Step 1 — Arrears stage**

Before foreclosure, most banks track how many months behind the borrower is:

- 30–59 days past due → mild stress, cure is still likely
- 60–89 days → serious, but cures still occur
- 90+ days → very likely to proceed to workout

This module estimates the arrears stage using a proxy (LTV, DTI, credit score) when the observed stage is missing.

**Step 2 — Repayment behaviour proxy**

The model estimates how likely the borrower is to cure, based on:
- Loan type (P&I = principal and interest is safer than interest-only)
- Seasoning (loans in force longer tend to have more equity)
- DTI (Debt-to-Income ratio — how much of income goes to debt)
- LTV (less equity = less incentive to keep paying)
- Credit score

A **behaviour score** (+1 for good factors, −1 for bad) maps to a cure rate:
- Score ≥ 2: cure rate ~40%
- Score = 0: cure rate ~25%
- Score ≤ −2: cure rate ~12%

**Step 3 — Cure-adjusted base LGD**

```
LGD_base = (1 − cure_rate) × LGD_liquidation
```

Where `LGD_liquidation` is the expected loss if the property must be sold:
- Applies haircuts for selling costs (~8%), legal fees, time to sell.
- Higher LTV at default → smaller property cushion → higher liquidation LGD.

**Step 4 — Downturn scalar**

During a housing downturn:
- House prices fall (reduces collateral value).
- Unemployment rises (fewer cures).
- Interest rates rise (higher discount cost).

Each driver is combined into a multiplier applied to the base scalar (typically 1.05–1.25 for mortgages).

**Step 5 — APRA overlays**

- **MoC** (Margin of Conservatism): +2pp added to downturn LGD.
- **LMI benefit**: If LMI applies, LGD is reduced by ~20%.
- **Regulatory floor**: Standard loans must have LGD ≥ 10%; non-standard ≥ 15%.
- **APRA scalar**: Multiplied by 1.10 to reflect regulatory conservatism.

#### Key inputs

| Input | What it is |
|-------|-----------|
| `ltv_at_default` | LTV at the time of default (0–1 scale) |
| `lmi_eligible` | 1 if LMI insurance applies, 0 if not |
| `mortgage_class` | "Standard" or "Non-Standard" |
| `realised_lgd` | Actual observed loss (used to calibrate long-run LGD) |
| `ead` | Exposure at default ($) |

#### Key outputs

- `lgd_base`, `lgd_downturn`, `lgd_final` at loan level.
- Segment summaries by LTV band and mortgage class.
- Cure overlay flags and arrears stage sources.

---

### Module 03: Commercial Cash-Flow Lending LGD

**File:** `notebooks/03_commercial_cashflow_lgd.ipynb`  
**Source code:** `src/lgd_calculation.py` → `CommercialLGDEngine`

#### What is commercial cash-flow lending?

This is lending to businesses — not backed by a specific property but by the business's ability to generate cash and, sometimes, by business assets (equipment, receivables, real property).

Examples:
- A $2M term loan to a manufacturing company, secured against factory equipment.
- A $500k overdraft for a retailer, unsecured.
- A $1M loan to a law firm, partially secured by office fit-out.

#### Security types and why they matter

| Security Type | Description | Typical LGD |
|---------------|-------------|-------------|
| Property | Commercial real estate | 25–40% |
| PPSR – Plant & Equipment | Machinery, vehicles | 35–50% |
| PPSR – Receivables | Debts owed to the business | 40–55% |
| PPSR – Mixed | Combination of assets | 45–55% |
| Unsecured | No specific collateral | 55–75% |

PPSR = Personal Property Securities Register (Australia's register of secured interests in moveable assets).

#### How LGD is built

**Step 1 — Long-run LGD by security type**

Historical workout data shows that secured loans recover more. The model uses exposure-weighted averages of realised LGD within each security type and seniority band.

**Step 2 — Industry risk adjustment**

Some industries are riskier — a hospitality business's assets (kitchen equipment, fit-out) are harder to sell than a professional services firm's receivables. The model applies an **industry recovery haircut** sourced from the upstream `industry_risk_scores.parquet` file.

**Step 3 — Cure overlay**

Commercial borrowers can also cure (restructure debt, sell assets, find new equity). The model estimates:

```
cure_proxy = base_rate
           + uplift for high security coverage
           + uplift for strong ICR
           - penalty for long workout time
```

Secured borrowers have a higher cure rate (up to 30%), unsecured borrowers much less (~10%).

**Step 4 — Downturn scalar**

Three commercial downturn drivers:
- **Value decline**: fall in collateral value (security coverage ratio as proxy)
- **Cashflow weakness**: deterioration in ICR (how well income covers interest)
- **Recovery delay**: longer workout time in a downturn (more months to realise assets)

**Step 5 — Supervisory LGD floor by seniority**

APRA imposes minimum floors:
- Senior secured: 25%
- Senior unsecured: 45%
- Subordinated: 75%

#### Key inputs

| Input | What it is |
|-------|-----------|
| `security_type` | Type of collateral (Property, PPSR P&E, Unsecured, etc.) |
| `seniority` | Loan rank (Senior Secured, Senior Unsecured, Subordinated) |
| `security_coverage_ratio` | Collateral value ÷ loan balance |
| `icr` | Interest Coverage Ratio |
| `workout_months` | Estimated months to resolve the defaulted loan |
| `industry` | Business industry sector |

---

### Module 04: Receivables and Invoice Finance LGD

**File:** `notebooks/04_receivables_invoice_finance_lgd.ipynb`

#### What is invoice finance?

Businesses often have customers who owe them money but haven't paid yet (these are **receivables**). A bank can lend money against those receivables — effectively advancing cash to the business before its customers pay.

**Example:** A supplier sells $1M of goods to Woolworths. Woolworths pays in 60 days. The bank lends the supplier $800,000 now and gets repaid when Woolworths pays.

#### Why receivables LGD is different

The collateral is a pool of customer debts, not a physical asset. Recovery depends on:

1. **Eligible receivables** — not all debts qualify. Overdue invoices, disputed amounts, and debts from related parties are typically excluded.
2. **Pool quality** — the age of invoices matters. An invoice that is 0–30 days old is worth more than one that is 90+ days old (less likely to be paid). Ageing analysis tells you what fraction is current vs. overdue.
3. **Dilutions** — some invoices are never fully paid (returns, credits, disputes). Dilution rate = % of invoices that are reduced after issuance.
4. **Advance rate** — the bank only lends a fraction of eligible receivables (typically 70–85%). The gap is the "headroom" cushion.

#### How LGD is built

```
Eligible pool = Total receivables × (1 − ineligible fraction)
Adjusted pool = Eligible pool × (1 − dilution rate)
Net recovery = Adjusted pool × collection efficiency × discount factor
LGD = (EAD − Net recovery) / EAD
```

**Ageing discount:** Older invoices get a haircut applied — for example, 90+ day invoices may only be worth 40% of face value.

**Advance rate headroom:** If the bank lent $800k against $1M of receivables (80% advance rate), there is a $200k cushion. In a stress scenario, this cushion absorbs the first losses.

#### Downturn considerations

In a recession, customers slow down payments, dispute more invoices, and dilution rates rise. The downturn scalar reflects the compressed collection window and increased ineligible fraction.

---

### Module 05: Trade and Contingent Facilities LGD

**File:** `notebooks/05_trade_contingent_facilities_lgd.ipynb`

#### What are trade and contingent facilities?

These are off-balance-sheet commitments — the bank promises to pay if the customer cannot, rather than lending money directly.

Examples:
- **Documentary letters of credit (LC):** Bank guarantees payment when goods are shipped internationally.
- **Trade finance:** Short-term financing tied to specific import/export transactions.
- **Performance bonds/guarantees:** Bank promises to pay if a contractor fails to complete a project.

#### The EAD challenge: Credit Conversion Factor (CCF)

These facilities are not fully drawn — they are contingent on specific events. The **CCF** converts the undrawn/contingent amount into an equivalent loan amount:

```
EAD = Drawn amount + (Undrawn limit × CCF)
```

If a $1M guarantee facility is unused and the CCF is 50%, EAD = $500,000 even though no cash has been paid out yet.

In a default/claim event, the **claim conversion** is how much of the contingent exposure becomes a real payment obligation.

#### LGD drivers

- **Cash backing:** If the facility is fully cash-collateralised, LGD ≈ 0. Partial cash backing proportionally reduces LGD.
- **Security/guarantee:** Third-party guarantee or asset security reduces loss.
- **Tenor:** Longer trade facilities mean more time for things to go wrong.
- **Recovery timing:** After a claim is paid, the bank steps into the shoes of the beneficiary — recovery depends on the underlying business assets.

---

### Module 06: Asset and Equipment Finance LGD

**File:** `notebooks/06_asset_equipment_finance_lgd.ipynb`

#### What is asset finance?

The bank lends money to buy a specific asset — a truck, medical equipment, agricultural machinery — and the asset itself is the collateral. If the borrower defaults, the bank repossesses and sells the asset.

This is conceptually simpler than commercial lending because the collateral is a physical thing with a market value.

#### What makes asset finance LGD tricky

1. **Asset depreciation:** Unlike property, most equipment loses value over time. A truck worth $200,000 when purchased may be worth $80,000 three years later. The loan balance may still be $120,000 — creating negative equity.

2. **Residual exposure:** `Residual = Loan balance − Asset value`. If residual > 0, the asset alone cannot repay the loan.

3. **Asset type and age:** Specialised equipment (mining machinery, medical devices) is harder to sell than generic equipment (cars, forklifts). A niche market = longer time to sell = lower net recovery.

4. **Repossession and remarketing costs:** Physically recovering the asset, transporting it, inspecting it, and finding a buyer all cost money (~5–15% of asset value).

#### LGD formula

```
Residual LGD = max(0, Residual Exposure / EAD)
Asset recovery = Asset value × (1 − selling costs) / (1 + discount_rate)^months_to_sale
LGD_base = 1 − Asset recovery / EAD
```

Asset type adjustments:

| Asset Category | Typical Haircut | Reasoning |
|----------------|----------------|-----------|
| Passenger vehicles | 15–25% | Liquid market, easy to sell |
| Commercial vehicles | 20–30% | Good market, some specialisation |
| Agricultural equipment | 25–40% | Seasonal market, remote locations |
| Industrial/specialised | 35–55% | Thin market, buyer may not exist locally |
| Medical/dental equipment | 30–45% | Regulatory constraints on resale |

---

### Module 07: Development Finance LGD

**File:** `notebooks/07_development_finance_lgd.ipynb`  
**Source code:** `src/lgd_calculation.py` → `DevelopmentLGDEngine`

#### What is development finance?

Development finance funds the construction of new properties — apartment buildings, commercial centres, residential subdivisions. The bank lends money as the build progresses, and the loan is repaid when the completed properties are sold.

This is one of the **highest-risk** lending products because:
- The collateral (the building) may not exist yet when the loan is drawn.
- Construction can be delayed, go over budget, or the market can turn before completion.
- If the developer defaults mid-build, the bank inherits a partially built property.

#### Key concepts

**GRV (Gross Realisable Value):** The total expected revenue from selling all completed properties. This is the "ceiling" on recovery.

**Completion stage:** Where in the build process is the project?

| Stage | Description | Risk |
|-------|-------------|------|
| Pre-Construction | Plans approved, build not started | Very high — nothing tangible yet |
| Early Construction | Foundations, structure started | High — asset is not sellable |
| Mid-Construction | Building taking shape | Medium — has some "as is" value |
| Near-Complete | Almost finished | Lower — can sell near market value |
| Complete Unsold | Finished but not sold | Low — just a marketing/timing risk |

**Presale coverage:** Percentage of units already under contract before or during construction. High presale = lower risk (buyers committed before market turns).

**LVR as-if-complete:** Loan balance ÷ GRV. Measures how much of the completed value the loan represents.

#### How LGD is built

**Base recovery path:** If the bank needs to recover, options are:
1. Fund to completion → sell finished units at market value.
2. Sell the unfinished site "as is" at a significant discount.
3. Appoint a receiver who completes and sells (most costly).

```
LGD_base = stage_component
          − benefit_from_presales
          − benefit_from_low_LVR
```

**Scenario stress testing:** The module runs explicit stress scenarios:
- GRV decline (market falls 10–30%)
- Cost overrun (construction costs 15% over budget)
- Rescission rate (buyers cancel presale contracts — 10–30%)
- Extended sales period (takes 6–12 months longer to sell)

**Downturn scalar drivers:**
- GRV decline coefficient (how much property prices fall)
- Cost overrun coefficient (construction inflation)
- Sell-through delay (slower absorption in a weak market)

---

### Module 08: CRE Investment LGD

**File:** `notebooks/08_cre_investment_lgd.ipynb`

#### What is CRE investment lending?

CRE = Commercial Real Estate. This is lending to investors who own income-producing properties — office towers, shopping centres, industrial warehouses, hotels. The property generates rental income which pays the loan interest.

Unlike development finance, the building already exists and already earns income.

#### Key concepts

**LVR (Loan-to-Value Ratio):** Loan balance ÷ Property value. Lower LVR = more equity cushion.

**DSCR (Debt Service Coverage Ratio):** Annual net rental income ÷ Annual loan repayments. DSCR > 1.25 is generally considered safe; DSCR < 1.0 means the property cannot pay its own loan.

**WALE (Weighted Average Lease Expiry):** Average time until leases expire (in years). A building full of 10-year leases is far safer than one where all leases expire next year.

**Vacancy rate:** Percentage of space not currently leased. High vacancy = lower income = harder to repay the loan.

**Tenant concentration:** If 80% of rental income comes from one tenant, losing that tenant destroys the property's income and value.

#### How LGD is built

Recovery depends on selling the property. Sale proceeds are reduced by:
- Vacancy (empty space = lower income = lower valuation)
- Tenant risk (major tenant leaving just before sale)
- Market depth (illiquid property markets in a downturn take much longer to sell)
- Selling costs (agent fees, stamp duty, legal)

```
Stressed property value = Current value × (1 − vacancy haircut) × (1 − market discount)
LGD_base = max(0, (Loan balance − Stressed property value × (1 − selling costs)) / EAD)
```

**Refinance risk:** CRE loans often mature before they are fully paid off — the borrower needs to refinance (get a new loan). In a credit crunch, refinancing may be impossible, forcing a distressed sale.

---

### Module 09: Residual Stock LGD

**File:** `notebooks/09_residual_stock_lgd.ipynb`

#### What is residual stock?

Residual stock is completed development properties (apartments, houses) that have been built but not yet sold. This is essentially a development loan that has run to completion but the exit (selling the units) has not been achieved.

The developer is holding finished properties and the bank's loan is still outstanding.

#### Why it's risky

- The asset is now fully built — there is no "construction risk" left.
- But the market may be weak and buyers are scarce.
- Every month the units remain unsold, the developer pays holding costs (loan interest, rates, insurance, strata fees).
- In a downturn, prices may need to be cut significantly to achieve a fast sale.

#### Key LGD drivers

**Absorption rate:** How quickly are units selling? (Units per month)

**Discount-to-clear:** The percentage price reduction needed to sell quickly (vs. holding for full market price).

```
Net recovery per unit = Market price × (1 − discount_to_clear) − Selling costs − Holding costs per month × months_remaining
LGD = (Loan balance − Total net recovery) / EAD
```

**Time-to-clear:** Remaining units ÷ monthly absorption rate. More months to clear = more holding cost = higher LGD.

**Holding cost rate:** Typically 0.5–1.0% of loan value per month (interest + maintenance + outgoings).

The downturn scenario applies a further price decline on top of the already-discounted price, and slows the absorption rate.

---

### Module 10: Land and Subdivision LGD

**File:** `notebooks/10_land_subdivision_lgd.ipynb`

#### What is land and subdivision lending?

A developer buys raw land, obtains council approval to subdivide it into lots, and sells the lots to home builders or directly to buyers. The bank funds the purchase and development of the land.

This is considered the **highest risk** property lending because:
- Raw land produces no income (unlike a CRE investment property).
- The collateral value is highly uncertain — it depends entirely on what can be built and the market's appetite.
- Entitlement risk: council may deny or delay development approval.
- Land is the most illiquid property type — there is a thin buyer pool, especially in a downturn.

#### Key LGD drivers

**Market depth:** How many buyers exist for this type of land in this location? Rural lots in a weak market may have no buyers for months.

**Zoning and entitlements:** Land with approved development plans is worth much more than raw unzoned land. Losing entitlements destroys value.

**Infrastructure requirements:** Land needing significant infrastructure (roads, sewerage, power) before it can be sold carries high development costs that reduce net proceeds.

**Liquidity haircut:** In a forced sale, land prices typically fall 20–40% below market value because the seller cannot wait for the right buyer.

```
Forced sale value = Assessed value × (1 − liquidity haircut)
Net recovery = Forced sale value − Infrastructure costs − Legal/selling costs
LGD = (Loan balance − Net recovery) / EAD
```

---

### Module 11: Bridging Loan LGD

**File:** `notebooks/11_bridging_loan_lgd.ipynb`

#### What is a bridging loan?

A bridging loan is a short-term loan (typically 6–24 months) that "bridges" a gap — for example:
- A developer needs to buy a new site before selling the old one.
- A business needs cash before a capital raise completes.
- A homeowner buys a new house before selling the old one.

The loan is meant to be repaid from a specific, near-term event (property sale, refinancing, capital raise).

#### The exit risk

The entire LGD framework for bridging loans is built around **exit certainty**: how confident is the bank that the intended repayment event will actually happen?

| Exit Type | Example | Certainty |
|-----------|---------|-----------|
| Contracted sale | Exchange of contracts already signed | High |
| Refinance committed | New lender approved in principle | Medium-high |
| Refinance speculative | Borrower intends to refinance | Medium |
| Asset sale pending | Asset on market, not under contract | Medium-low |
| Business sale | Business sale process started | Low |

**Stage-based LGD:** The module calculates LGD for each exit stage, with higher certainty exits producing much lower LGD.

#### Valuation risk

Bridging loans are often short-term financings of assets whose value is uncertain:
- Property may be valued at peak market and decline before sale.
- Valuation uncertainty increases with unusual property types.

```
LGD_bridge = f(exit_certainty, time_to_exit, LVR, valuation_uncertainty)
```

**Failed exit scenario:** If the exit fails (sale falls through, refinance cannot be arranged), the bank must take possession and sell the asset — often at a forced-sale discount. This is the worst-case LGD.

---

### Module 12: Mezzanine and Second Mortgage LGD

**File:** `notebooks/12_mezz_second_mortgage_lgd.ipynb`

#### What is mezzanine lending?

Mezzanine lending (mezz) is lending that sits **behind** the senior (first) lender but **in front of** the equity. It is essentially a second mortgage.

**Recovery waterfall:** In a default, assets are sold and money flows in a strict order:

```
Sale proceeds
    │
    ▼
Senior lender repaid first  ← gets paid before anyone else
    │
    ▼
Mezzanine lender repaid    ← gets what's left after senior
    │
    ▼
Equity investor (developer) ← gets whatever remains (often zero)
```

#### Why mezzanine has very high LGD

Because the senior lender takes the first slice of recovery, the mezzanine lender only gets the *residual*. If the property sells for less than the senior loan balance, the mezzanine lender gets **nothing**.

**Example:**
- Property sells for $800,000.
- Senior loan: $700,000. Senior lender is repaid in full.
- Mezzanine loan: $150,000. Only $100,000 remains → mezzanine LGD = 33%.
- If property only sold for $650,000, senior lender takes $650k → mezzanine LGD = **100%**.

#### How LGD is built

```
Senior claim = Senior loan balance + Senior default interest + Senior costs
Residual for mezz = max(0, Property net proceeds − Senior claim)
LGD_mezz = max(0, (Mezzanine balance − Residual for mezz) / Mezzanine balance)
```

Key variables:
- **Combined LVR:** (Senior + Mezz) ÷ Property value. >85% is very high risk for mezz.
- **Mezz LVR:** Mezz balance ÷ Property value. Shows how exposed the mezz is.
- **Valuation haircut:** Forced-sale discount on the property.
- **Senior terms:** A senior lender with a high default interest rate accumulates a larger claim quickly.

---

### Module 13: Cross-Product Comparison

**File:** `notebooks/13_cross_product_comparison.ipynb`  
**Source code:** `src/lgd_calculation.py` → `build_cross_product_comparison_report()`

#### What is the cross-product comparison?

This module brings all the individual product LGD outputs together into a single portfolio view. It allows leadership, risk teams, and regulators to compare:

- Which products have the highest/lowest LGD?
- Where is the most capital required?
- How does the portfolio behave in a downturn vs. normal times?
- Are there concentrations of risk in any product or segment?

#### What it produces

**1. Portfolio summary table**

| Product | Loan Count | Total EAD ($M) | Avg LGD Base | Avg LGD Downturn | Avg LGD Final |
|---------|-----------|----------------|--------------|------------------|---------------|
| Mortgage | 500 | 120 | 18% | 22% | 24% |
| Commercial | 200 | 80 | 38% | 47% | 49% |
| Development | 50 | 45 | 42% | 54% | 56% |
| Cashflow Lending | 150 | 35 | 45% | 55% | 57% |

**2. Downturn sensitivity analysis**

How much does LGD increase when macro conditions deteriorate?

```
Downturn uplift = (LGD_downturn − LGD_base) / LGD_base × 100%
```

Development finance typically shows the highest uplift (market turn + construction risk) while residential mortgages show the lowest (LMI, liquid property markets).

**3. Recovery time comparison**

Different products take different amounts of time to resolve:
- Receivables: 3–6 months
- Residential mortgage: 12–18 months
- CRE investment: 12–24 months
- Development finance: 18–36 months

Longer resolution = more discounting = higher effective LGD.

**4. Portfolio mix and risk ranking**

The module identifies which products contribute the most to **Expected Loss (EL)**:

```
Product EL contribution = Product EAD share × Product LGD × PD assumption
```

**5. Segmentation consistency check**

Validates that all products use the same standard segment columns (`std_module`, `std_product_segment`, `std_security_or_stage_band`) so comparisons are apples-to-apples.

---

## Part 3: How the Pipeline Fits Together

Here is the full data flow from raw inputs to final outputs:

```
Raw loan data (synthetic or controlled)
        │
        ▼
  01: Feature Engineering
  (clean, engineer variables, segment)
        │
        ▼
  02–12: Product-specific LGD engines
  ┌─────────────────────────────────┐
  │ For each product:               │
  │ 1. Load data                    │
  │ 2. Validate inputs              │
  │ 3. Compute long-run LGD         │
  │ 4. Apply downturn scalar        │
  │ 5. Apply MoC and APRA overlays  │
  │ 6. Validate final LGD [0,1]     │
  └─────────────────────────────────┘
        │
        ▼
  13: Cross-product comparison
  (combine, compare, portfolio view)
        │
        ▼
  outputs/tables/
  ├── lgd_final.csv                   (loan-level final LGD)
  ├── cross_product_comparison.csv    (portfolio summary)
  ├── overlay_trace_report.csv        (audit trail)
  ├── parameter_version_report.csv    (governance)
  └── segmentation_consistency_report.csv
```

### The governance layer

Every run records:

- **Parameter version and hash** — which set of overlay parameters was used, with a cryptographic fingerprint so it cannot be quietly changed.
- **Overlay trace** — for every loan, what overlays were applied and from what source.
- **Fallback counts** — how many loans used proxy values instead of observed values (e.g., origination date estimated from seasoning months).
- **Validation results** — automated checks that LGD is in [0,1], mean LGD is plausible, and downturn LGD ≥ base LGD.

---

## Part 4: Running the Project

### Prerequisites

- Python 3.10 or later
- Install dependencies:

```bash
pip install -r requirements.txt
```

### Step 1 — Run the end-to-end demo pipeline

```bash
python scripts/run_demo_pipeline.py
```

This generates synthetic data, runs all product LGD engines, and writes outputs to `outputs/tables/`.

### Step 2 — Run the validation sequence

```bash
python scripts/run_validation_sequence.py
```

Runs all automated checks: backtesting, calibration, PSI monitoring, out-of-time validation, sensitivity analysis.

### Step 3 — Score a single new loan

```bash
python scripts/score_new_loan.py \
  --product-type mortgage \
  --single-json data/sample_loan.json \
  --output outputs/tables/single_loan_scored.json
```

### Step 4 — Run product-specific validation

```bash
python scripts/run_stage7_bridging_validation.py
python scripts/run_stage9_cross_product_validation.py
```

### Step 5 — Run lgd_final layer as a standalone module

```bash
python -m src.lgd_final
```

### Checking logs

Enable logging to see what the pipeline is doing:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

You will see messages like:
- File loaded successfully (path, rows, columns).
- Which overlay parameters were used.
- Any data quality warnings (missing columns filled with defaults, coercion to NaN, etc.).
- Final LGD summary statistics per product.

### Where to look when something goes wrong

| Symptom | Where to check |
|---------|---------------|
| FileNotFoundError with path | `data/config/overlay_parameters.csv` or `data/exports/*.parquet` missing |
| ValueError about column missing | Check the input data has the required columns listed in `src/lgd_scoring.py` `SCHEMAS` dict |
| Warning about many "Unknown" segments | Source column (e.g., `mortgage_class`, `security_type`) may have unexpected values |
| LGD = NaN for many loans | Likely a coercion issue — check WARNING logs for column name |
| LGD > 1 raises ValueError | `_validate_final_lgd` found a clipping failure — check overlay parameter magnitudes |

---

## Summary: What You Should Take Away

After reading this manual, you should understand:

1. **LGD = percentage of a loan you do not recover when a borrower defaults.**

2. **Every product has different collateral**, different cure probabilities, and different recovery times — which is why each module is different.

3. **Downturn LGD is what regulators care about** — how bad does it get in a recession, not just on average.

4. **The framework is transparent** — every proxy, fallback, and overlay is logged and reported so reviewers can trace any number back to its assumption.

5. **The pipeline is reproducible** — same inputs + same parameters → same outputs, every time. This is not optional in banking; it is a regulatory requirement.

6. **Module 13 ties it all together** — individual product LGDs combine into a portfolio view that shows where capital should be held.

---

*For methodology detail, see:*
- `docs/methodology_cashflow_lending.md`
- `docs/methodology_property_backed_lending.md`
- `docs/data_dictionary.md`

*For code structure, see:*
- `PROJECT_OVERVIEW.md`
- `README.md`

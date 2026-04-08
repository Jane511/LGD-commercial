# LGD Methodology Documentation

## 1. Regulatory Context

### APRA APS 113 — Internal Ratings-Based Approach

Australian ADIs using the Advanced IRB (AIRB) approach must estimate LGD at the facility level. Key requirements:

- LGD estimates must reflect **economic downturn conditions**
- Estimates must include a **margin of conservatism** for data/model limitations
- **Residential mortgages**: minimum 10% LGD floor (standard), 15% (non-standard)
- **LMI recognition** subject to insurer eligibility and claim history
- **1.1x APRA scalar** applied to Risk-Weighted Assets (not LGD directly)
- **Specialised lending** (incl. development finance) may require slotting approach

### Basel III IRB Framework

The IRB risk-weight function uses LGD as a key input:

```
K = LGD × [N(√(1/(1-R)) × G(PD) + √(R/(1-R)) × G(0.999)) - PD] × (1+(M-2.5)×b) / (1-1.5×b)
RWA = K × 12.5 × EAD
```

Where: N() = standard normal CDF, G() = inverse CDF, R = asset correlation, b = maturity adjustment.

---

## 2. Common Methodology

### 2.1 Default Definition (APS 220)

A facility is in default when either:
1. **90 days past due** on any material obligation, OR
2. **Unlikely to pay** in full without recourse to security

### 2.2 Realised LGD — Economic Loss

```
Economic Loss = EAD + PV(Costs) - PV(Recoveries)
Realised LGD = max(Economic Loss / EAD, 0)
```

All cashflows discounted to the default date:
```
PV(CF_t) = CF_t / (1 + r)^(days_from_default / 365)
```

### 2.3 Exposure-Weighted Long-Run Average

```
LR_LGD_s = Σ(LGD_i × EAD_i) / Σ(EAD_i)    for all i in segment s
```

Must cover at least one full economic cycle (minimum 7 years in Australia).

### 2.4 Downturn Adjustment

Three methods:
- **Scalar**: Downturn_LGD = LR_LGD × scalar
- **Additive**: Downturn_LGD = LR_LGD + add-on
- **Macro-regression**: LGD_t = α + β₁×HPI_change + β₂×Unemployment + ε

### 2.5 Margin of Conservatism

Additive adjustment reflecting data limitations, model uncertainty, and process gaps. Calibrated by sample size, history length, and model performance.

---

## 3. Residential Mortgage

### Data Inputs
Origination: LTV, credit score, DTI, property type, state, occupancy, LMI.
Default: EAD (balance + accrued interest), default date, resolution type.
Workout: Property sale proceeds, LMI claims, legal/agent/preservation costs.

### Discount Rate
Contract rate or cost of funds: **3.5–5.5%** per annum.

### Segmentation
```
Level 1: Standard vs Non-Standard (APRA requirement)
  Level 2: LTV band at default (<60%, 60-70%, 70-80%, 80-90%, 90%+)
    Level 3: LMI eligible (yes/no)
```

### Two-Stage Model

Mortgage defaults exhibit a **bimodal LGD distribution** (30-50% cure rate), requiring:

**Stage 1 — P(Cure)**: Logistic regression
```
P(Cure) = σ(β₀ + β₁×LTV + β₂×CreditScore + β₃×DTI + β₄×Seasoning + ...)
```

**Stage 2 — E[LGD|Loss]**: OLS on logit-transformed LGD (non-cure cases only)
```
logit(LGD) = γ₀ + γ₁×LTV + γ₂×PropertyType + γ₃×State + ...
E[LGD|Loss] = logit⁻¹(predicted)
```

**Combined**:
```
E[LGD] = P(Cure) × LGD_cure + (1 - P(Cure)) × E[LGD|Loss]
```
Where LGD_cure ≈ 1% (workout costs on cured loans).

### APRA Overlays
1. Downturn scalar: **1.15×**
2. Margin of conservatism: **+2pp**
3. LMI adjustment: LGD × (1 - 0.20) for eligible loans
4. Floor: max(LGD, 10%) for standard, max(LGD, 15%) for non-standard
5. APRA scalar: RWA × 1.10 (applied at capital stage)

---

## 4. Commercial Cash Flow Lending

### Data Inputs
Borrower: Revenue, EBITDA, leverage, ICR, industry, years in business.
Facility: Type (term/revolving/overdraft), limit, drawn, undrawn, seniority.
Security: Type (property/PPSR/GSR), PPSR registration, coverage ratio.
Workout: Recovery by security type, receiver fees, legal costs.

### EAD with CCF
```
EAD = Drawn + CCF × Undrawn + Accrued Interest
```
CCF: Term loan 100%, Revolving 50-75%, Overdraft 75-100%.

### Discount Rate
**5–7%** per annum (higher than mortgage, reflecting commercial risk premium).

### Segmentation
```
Level 1: Security type (Property / PPSR / GSR Only)
  Level 2: Coverage band (<50%, 50-80%, 80-100%, 100-120%, 120%+)
    Level 3: Industry sector
```

### Recovery by Security Type
| Security | Typical Recovery | Timing |
|----------|-----------------|--------|
| Property (first mortgage) | 50-80% of valuation | 10-24 months |
| PPSR - Plant & equipment | 20-40% of book value | 3-18 months |
| PPSR - Receivables | 30-60% of face value | 3-12 months |
| PPSR - Inventory | 10-30% of book value | 3-12 months |
| GSR (all-assets sweep) | 25-55% of registered value | 12-30 months |
| Director guarantee | 10-30% of guarantee amount | 6-24 months |

### Regulatory Overlays
1. Downturn scalars by security type: Property 1.15×, PPSR 1.20×, GSR 1.15×
2. Margin of conservatism: **+3pp**
3. Supervisory LGD floor: Senior secured 35%, Senior unsecured 45%
4. SME firm-size adjustment: correlation reduction for revenue < $75M AUD

---

## 5. Development Finance

### Data Inputs
Project: Development type, TDC, GRV, land value, completion %, pre-sale coverage, cost-to-complete, QS reports.
Facility: Limit, drawn (progressive), capitalised interest, LTC, LVR (as-if-complete).
Default: Completion stage, default trigger, fund-to-complete decision.

### Discount Rate
**7–9%** per annum (highest of all products, reflecting project risk).

### Segmentation
```
Level 1: Completion stage at default (primary driver)
  Level 2: Development type
    Level 3: Pre-sale coverage band
```

### Completion-Stage LGD Profiles
| Stage | Typical LGD (benign) | Typical LGD (downturn) |
|-------|---------------------|----------------------|
| Pre-Construction | 15-30% | 25-45% |
| Mid-Construction | 30-50% | 45-70% |
| Near-Complete | 10-25% | 20-40% |
| Complete Unsold | 10-20% | 20-40% |

Mid-construction defaults have the **highest LGD** because partially-built structures can have negative incremental value.

### Fund-to-Complete Model
When the bank funds to completion:
```
Total Outlay = EAD_at_default + Cost_to_Complete + Interest_Carry + Holding_Costs
Recovery = Σ(Unit Sales) + Other Recoveries
LGD = max((Total Outlay + PV(Costs) - PV(Recovery)) / EAD_at_default, 0)
```
Note: LGD can **exceed 100%** if post-default funding exceeds incremental recovery.

### Scenario Analysis
Key stress parameters:
- **GRV decline**: -10% to -40% (property market downturn)
- **Cost overrun**: +5% to +20% (construction cost inflation)
- **Pre-sale rescission**: 10-40% (sunset clause exercise)
- **Sales extension**: +3 to +12 months (weak market)

### Regulatory Overlays
1. Downturn scalars by completion stage: 1.15× to 1.30×
2. Margin of conservatism: **+5pp** (highest, due to data limitations)
3. APRA slotting: Strong (70% RW), Good (90%), Satisfactory (115%), Weak (250%)
4. HVCRE correlation multiplier: 1.25×

---

## 6. Validation Framework

| Test | Metric | Pass Criteria |
|------|--------|---------------|
| Accuracy | MAE, RMSE | MAE < 10pp |
| Discriminatory power | Spearman correlation | > 0.30 |
| Calibration | Predicted vs actual by segment | Within 20% |
| Conservatism | Mean predicted >= mean actual | Yes |
| Stability | PSI | < 0.25 |
| Out-of-time | Holdout MAE/RMSE | Within 20% of in-sample |
| Sensitivity | Parameter perturbation | No single driver moves LGD > 15pp |

---

## 6b. Industry Risk Integration

Industry risk scores from the Industry Risk Analysis project are integrated into Commercial and Development LGD engines. Scores are on a 1–5 scale derived from ABS/RBA public data across 9 ANZSIC divisions.

### Integration Points

| Component | Formula | Impact |
|:----------|:--------|:-------|
| Downturn scalar | `base × (1 + 0.15 × (score - 2.5) / 2.0)` | ±7-10% of base scalar |
| Margin of conservatism | `base × (1 + 0.20 × (score - 2.5) / 2.5)` | ±8% of base MoC |
| Recovery haircut | `0.02 × max(score - 2.0, 0)` | 0-6% additive to LGD |
| Working capital overlay | `0.01 × max(wc_score - 2.0, 0)` | 0-1.9pp additive to LGD |
| Slotting (development) | 0-2 points added to slotting score | Can shift one category |

### Validation Enhancement

| Test | Description |
|:-----|:-----------|
| Industry attribution | R-squared from industry alone vs combined with security type |
| Model comparison | Side-by-side baseline vs enhanced: accuracy, discriminatory power, conservatism, PSI |
| Industry calibration | Predicted vs actual LGD by industry risk band |

See `LGD_Australian_Bank_Methodology_Guide.md` Part F for full documentation.

---

## 7. Glossary

| Term | Definition |
|------|-----------|
| APRA | Australian Prudential Regulation Authority |
| APS 113 | Prudential Standard: Internal Ratings-Based Approach |
| CCF | Credit Conversion Factor |
| EAD | Exposure at Default |
| GRV | Gross Realisation Value |
| GSR | General Security Agreement |
| HVCRE | High-Volatility Commercial Real Estate |
| IRB | Internal Ratings-Based approach |
| LGD | Loss Given Default |
| LMI | Lenders Mortgage Insurance |
| LTV/LVR | Loan-to-Value Ratio |
| MoC | Margin of Conservatism |
| PPSR | Personal Property Securities Register |
| PSI | Population Stability Index |
| RWA | Risk-Weighted Assets |
| TDC | Total Development Cost |

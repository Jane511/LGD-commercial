# Australian Bank LGD Modelling Framework

> **Loss Given Default estimation for three lending products — aligned with APRA APS 113 / Basel III IRB standards.**

This project demonstrates a production-quality LGD modelling workflow covering the three core Australian bank lending products, with product-specific data generation, calculation engines, APRA regulatory overlays, and a full validation framework.

---

## Products Covered

| Product | Model Approach | Key LGD Driver | APRA Treatment |
|---------|---------------|-----------------|----------------|
| **Residential Mortgage** | Two-stage: P(Cure) logistic + LGD\|Loss regression | LTV at default | 10% floor, LMI recognition, standard/non-standard |
| **Commercial Cash Flow** | Segmented average + security-type overlay | Security coverage ratio | Supervisory LGD floor, SME firm-size adjustment |
| **Development Finance** | Scenario-based + slotting framework | Completion stage at default | HVCRE treatment, APRA slotting categories |

---

## Project Structure

```
lgd_project_repo/
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/                          # Core Python modules
│   ├── __init__.py
│   ├── data_generation.py        # Product-specific synthetic data generators
│   ├── lgd_calculation.py        # LGD engines with APRA overlays
│   ├── lgd_final.py              # EL-ready final LGD layer builder
│   ├── validation.py             # Backtesting & validation framework
│   └── industry_risk_integration.py  # Industry risk loader & LGD adjustments
│
├── notebooks/                    # Analysis notebooks (run in order)
│   ├── 01_residential_mortgage_lgd.ipynb
│   ├── 02_commercial_cashflow_lgd.ipynb
│   ├── 03_development_finance_lgd.ipynb
│   ├── 04_cross_product_comparison.ipynb
│   ├── 05_industry_risk_integration.ipynb
│   └── 06_lgd_final_layer.ipynb
│
├── data/
│   ├── raw/                      # Generated CSV datasets
│   ├── external/industry_risk/   # Industry analysis outputs (9 ANZSIC sectors)
│   └── processed/                # Intermediate outputs
│
├── outputs/
│   ├── figures/                  # Charts and visualisations
│   ├── tables/                   # Summary CSVs
│   └── models/                   # Serialised model objects
│
└── docs/
    └── methodology.md            # Full methodology documentation
```

---

## Methodology Overview

### Core LGD Formula

```
Realised LGD = max( (EAD + PV(Costs) - PV(Recoveries)) / EAD, 0 )
```

Where cashflows are discounted to the default date:

```
PV(CF_t) = CF_t / (1 + r)^(t / 365)
```

### Regulatory Pipeline (Enhanced with Industry Risk)

```
Realised LGD
  → + Industry recovery haircut (Commercial/Development)
    → Exposure-weighted long-run average (by segment)
      → Industry-adjusted downturn overlay (±7-10% of base scalar)
        → Industry-adjusted margin of conservatism
          → + Working capital LGD adjustment (Commercial)
            → APRA overlays (floors, LMI, standard/non-standard)
              → Final regulatory LGD
```

### Product-Specific Highlights

**Residential Mortgage**
- Two-stage model handles bimodal distribution (30-50% cure rate)
- Stage 1: Logistic regression for P(Cure)
- Stage 2: OLS on logit-transformed LGD for loss cases
- APRA overlays: LMI recognition (20% LGD reduction), 10% standard floor, 15% non-standard floor
- Illustrative IRB risk-weight function and 1.1x APRA scalar

**Commercial Cash Flow (PPSR + GSR)**
- Security-type segmentation: Property / PPSR (P&E, Receivables, Mixed) / GSR
- Coverage-ratio-band analysis
- CCF for revolving and overdraft facilities
- Downturn scalars by security type (1.15x property, 1.20x PPSR)
- SME firm-size adjustment to correlation parameter
- Supervisory LGD fallback (APS 112)

**Development Finance**
- Completion-stage segmentation (primary driver)
- Fund-to-complete vs sell-as-is decision modelling
- Cost-to-complete as dominant cost category
- Scenario analysis: GRV decline, cost overrun, pre-sale rescission, sales extension
- APRA slotting framework (Strong / Good / Satisfactory / Weak)
- HVCRE higher correlation multiplier

### Industry Risk Integration

Industry risk scores (1-5 scale) from the [Industry Risk Analysis](https://github.com/Jane511/industry_analysis) project are integrated into the Commercial and Development LGD engines:

- **Industry-sensitive downturn scalars** replace flat scalars (±7-10% adjustment based on sector cyclicality)
- **Industry-adjusted MoC** reflects differential data uncertainty across sectors
- **Recovery haircuts** for higher-risk industries (up to 6pp additive)
- **Working capital overlay** adds 0-1.9pp for industries with poor liquidity profiles (e.g. Manufacturing)
- **Industry risk band** added as a segmentation dimension (Low / Medium / Elevated)
- **Slotting input** for development finance (0-2 points based on sector risk)

9 ANZSIC industries covered, sourced from ABS and RBA public data. See `notebooks/05_industry_risk_integration.ipynb` for the full analysis.

### LGD Final Layer

The repository now includes a simplified final LGD layer for downstream Expected Loss usage. It converts the detailed product datasets into one facility-level output with:

- **Base LGD** by product/security bucket
- **LVR adjustment** for higher leverage exposures
- **Development stage adjustment** for earlier-stage projects
- **Industry risk adjustment** for higher-risk sectors
- **Downturn scalar** to convert adjusted LGD into stressed LGD

The final output is a single `lgd_final` number per loan, saved to `outputs/tables/lgd_final.csv`, ready for use in an EL engine:

```python
EL = PD * LGD_final * EAD
```

---

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Generate synthetic data

```bash
python -m src.data_generation
```

### Build the final LGD layer

```bash
python -m src.lgd_final
```

### Run notebooks

```bash
jupyter notebook notebooks/
```

Run notebooks 01-06 in order. `06_lgd_final_layer.ipynb` demonstrates the EL-ready final layer and reproduces the output CSV and validation checks.

---

## Validation Framework

| Test | Description |
|------|-------------|
| **Accuracy** | MAE, RMSE, R-squared |
| **Discriminatory power** | Spearman rank correlation |
| **Calibration** | Predicted vs actual by segment |
| **Conservatism** | One-sided t-test: predicted >= actual |
| **Stability** | Population Stability Index (PSI) |
| **Out-of-time** | Vintage-based holdout backtest |
| **Sensitivity** | Parameter perturbation analysis |

---

## Key Regulatory References

| Standard | Description |
|----------|-------------|
| **APRA APS 113** | Capital Adequacy: Internal Ratings-Based Approach |
| **APRA APS 112** | Capital Adequacy: Standardised Approach to Credit Risk |
| **APRA APS 220** | Credit Risk Management |
| **Basel III** | International framework for bank capital adequacy |

---

## Disclaimer

This project uses **synthetic data** for demonstration purposes. It is designed to show credit risk modelling concepts aligned with Australian banking practices and is **not** a regulatory-approved internal model, a full APS 113 implementation, or suitable for production use.

---

## Tech Stack

Python 3.10+ | pandas | NumPy | scikit-learn | statsmodels | SciPy | matplotlib | seaborn | Jupyter

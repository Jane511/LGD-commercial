# Project Overview — LGD Commercial Framework

## What this project demonstrates

This repo builds a **production-style LGD model** for 11 commercial and mortgage products,
following the APRA APS 113 IRB methodology. It is part of a broader credit risk portfolio
that covers the full capital and pricing chain — from PD scoring through LGD, EAD, RWA,
expected loss, stress testing, and RAROC pricing.

The goal is to show how a bank's model development team would approach LGD end-to-end:
rigorous methodology, governed parameters, proper validation, and code a colleague can
actually maintain and run.

---

## Portfolio stack

This is repo **1.2** in the commercial credit risk series:

| # | Repo | Role |
| - | ---- | ---- |
| 1.1 | PD-and-Scorecard-Commercial-SME-Lending | PD models, scorecards |
| **1.2** | **LGD-Framework-Commercial-SME-Lending** | **LGD, recoveries, MoC (this repo)** |
| 1.3 | EAD-and-CCF-Commercial-SME-Lending | Exposure at default, CCF |
| 1.4 | RWA-Capital-Modules | Risk-weighted assets |
| 1.5 | Expected-Loss-Engine-Commercial-SME-Lending | EL = PD × LGD × EAD |
| 1.6 | RAROC-Pricing-and-Return-Hurdle | Risk-adjusted pricing |
| 1.7 | Stress-Testing-Commercial-SME-Lending | Macro stress scenarios |
| 1.9 | industry_analysis | Industry risk scores, macro regime flags |

LGD outputs from this repo feed directly into the Expected Loss Engine (1.5), RWA Capital (1.4),
RAROC Pricing (1.6), and Stress Testing (1.7) modules.

---

## Two-layer architecture

The framework separates two concerns deliberately kept in parallel files:

```text
Layer 1 — Proxy Baseline (src/lgd_calculation.py)
    Transparent, rule-based LGD using collateral haircuts, cure overlays,
    recovery timing, and discounting. Fast to run, no historical data needed.
    Used for: new originations, portfolios without workout history.

Layer 2 — APS 113 Calibration (src/lgd_calculations.py)
    Full IRB pipeline on 10-year workout histories (2014–2024).
    Used for: regulatory capital, model validation, APRA submissions.
```

Both layers produce the same output contract (`lgd_base`, `lgd_downturn`, `lgd_final`)
so downstream consumers don't need to know which layer ran.

---

## APS 113 calibration pipeline

```text
Realised LGD (workout tape)
    ↓  Vintage exposure-weighted average  (APS 113 s.43)
Long-run LGD
    ↓  Downturn overlay — macro regime × product scalar  (s.46–50)
Downturn LGD
    ↓  Frye-Jacobs LGD-PD correlation adjustment  (s.55–57)
Correlation-adjusted LGD
    ↓  Margin of Conservatism — 5 APS 113 s.65 sources  (s.63–65)
MoC-loaded LGD
    ↓  Regulatory floor by product  (s.58)
Final calibrated LGD
```

The MoC is applied to downturn LGD, not long-run LGD — a requirement of APS 113 s.63
that is explicitly enforced and tested.

---

## Products covered

| Product | LGD drivers | Key risk factors |
| ------- | ----------- | ---------------- |
| Residential mortgage | LVR, LMI, cure probability, liquidation path | Arrears stage, borrower behaviour, property market |
| Commercial cashflow | Collateral coverage, DSCR, seniority | Security type, workout duration |
| Receivables / invoice finance | Eligible pool, advance rate, dilution | Ageing, concentration |
| Trade / contingent | Claim conversion, cash backing | Tenor, settlement risk |
| Asset & equipment finance | Asset type, residual value, repossession | Age, remarketing haircut |
| Development finance | GRV, completion stage, cost-to-complete | Market absorption, exit scenario |
| CRE investment | LVR, DSCR, WALE, vacancy | Refinance vs forced sale |
| Residual stock | Absorption rate, discount-to-clear | Holding cost, time-to-sale |
| Land subdivision | Liquidity, market depth | Haircut, longer recovery time |
| Bridging | Exit type/certainty, valuation risk | Failed-exit stress |
| Mezz / 2nd mortgage | Recovery waterfall position | Senior recovery residual |

---

## Technical highlights

**Parameter governance**
All LGD floors, downturn scalars, and MoC inputs live in a single versioned CSV
(`data/config/overlay_parameters.csv`) with a SHA-256 hash manifest. Any unauthorised
edit causes the pipeline to refuse to run, giving the same protection as a database-controlled
parameter table.

**Economic regime integration**
The downturn overlay is driven by economic regime classification (expansion / mild downturn /
severe downturn) derived from RBA macro series. When real upstream data is available
(`macro_regime_flags.parquet` from repo 1.9), it is used directly; otherwise the module
degrades gracefully to synthetic regime flags.

**Validation suite**
Every model run produces a full validation report covering:

- Accuracy: weighted MAE, RMSE, bias
- Rank-ordering: Gini coefficient and AUROC
- Calibration: Hosmer-Lemeshow goodness-of-fit
- Stability: Population Stability Index (PSI)
- Out-of-time backtest: vintage-based holdout performance
- Conservatism: model ≥ actual at portfolio and segment level

**Scoring API**
A loan-level scoring interface (`src/lgd_scoring.py`) takes a single JSON payload or
batch CSV and returns `lgd_base`, `lgd_downturn`, `lgd_final` with full overlay trace
and parameter provenance, suitable for integration into a pricing or decisioning system.

**Data source adapter**
The pipeline can run against three data sources without any code change:

- Built-in demo data (instant, no setup)
- Synthetic generated workout histories (10-year, all products)
- Controlled/real input tables (drop-in replacement via `--source controlled`)

---

## Key design decisions

**Why two files named `lgd_calculation.py` and `lgd_calculations.py`?**
Intentional. The proxy engine (`lgd_calculation.py`) is the stable Layer 1 baseline that
the calibration layer imports from. The `s` signals Layer 2. Both can be imported in the
same process without conflict.

**Why is validation merged into one file?**
`src/validation.py` contains both the proxy validation metrics and the IRB-grade extended
metrics (Gini, Hosmer-Lemeshow). One import path for all validation, no wrapper indirection.

**Why script-first, not notebook-first?**
Model validation that only works by running notebook cells interactively is not auditable.
All validation logic is importable Python. Notebooks are presentation layers over the same
underlying functions.

---

## Repo structure (top level)

```text
src/
├── lgd_calculation.py      Proxy engine — Layer 1
├── lgd_calculations.py     Calibration engine — Layer 2
├── lgd_scoring.py          Loan scoring API
├── validation.py           Full validation suite
├── moc_framework.py        MoC register (APS 113 s.65)
├── overlay_parameters.py   SHA-256-governed parameter manager
├── data/                   Data loaders (RBA rates, regime, source adapter)
├── generators/             11 synthetic workout data generators
├── pipeline/               CLI entry points
├── scoring/                Scoring CLI wrapper
└── governance/             APS 113 gap matrix

data/
├── raw/                    Demo input CSVs
├── config/                 Overlay parameters + hash manifest
├── external/               RBA B6 rates, APRA ADI statistics
└── generated/historical/   Synthetic Parquet workout histories

notebooks/02–13             Per-product notebooks + cross-product comparison
docs/                       Methodology manuals, data dictionary
```

---

## Limitations (honest assessment)

This is a demonstration framework, not a signed-off production model.

- Workout data is **synthetic** — calibrated outputs are illustrative only
- MoC values require Model Risk Committee sign-off in production
- APRA benchmarks use impairment ratios as a directional proxy, not institution-specific LGD
- APS 113 compliance map reports `partial` for s.60 and s.66 because data is not from
  a real internal workout tape

A real bank deployment would replace `data/generated/historical/` with an actual workout
tape and run the same calibration pipeline with no code changes required.

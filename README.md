# LGD Commercial — AU Bank-Style LGD Framework

> **SYNTHETIC DATA — DEMONSTRATION ONLY.**
> All workout histories in `data/generated/historical/` are synthetic.
> A real production deployment requires an internal workout tape and APRA Model Risk Committee sign-off.

Calculates **base, downturn, and final LGD** for 11 commercial and mortgage products across two layers:

- **Proxy baseline** — transparent rule-based LGD using collateral haircuts, cure overlays, and recovery timing
- **APS 113 calibration layer** — full IRB pipeline: realised LGD → long-run → downturn → Frye-Jacobs → MoC → floor

---

## 1. Setup

```bash
pip install -r requirements.txt
```

---

## 2. Score a new loan (most common task)

This is what you do when credit brings in a new facility for LGD assessment.

### Single loan — Python API

```python
from src.lgd_scoring import score_single_loan

# Mortgage (family: mortgage)
result = score_single_loan(
    payload={
        "loan_id":        "L-1001",
        "ead":            750000,
        "realised_lgd":   0.28,
        "lmi_eligible":   1,
        "mortgage_class": "Standard",   # or "Non-Standard"
    },
    product_type="mortgage",
)
print(result)
# {'lgd_base': 0.18, 'lgd_downturn': 0.24, 'lgd_final': 0.24, ...}

# Commercial cashflow lending (family: cashflow_lending)
result = score_single_loan(
    payload={
        "loan_id":       "L-2001",
        "ead":           2_500_000,
        "realised_lgd":  0.42,
        "security_type": "real_property",
        "seniority":     "senior_secured",
    },
    product_type="commercial_cashflow",   # NOT "commercial" — ambiguous and rejected
)
```

> **Product-type rules:**
> - `commercial` is **ambiguous and rejected** — use a specific sub-type (e.g. `commercial_cashflow`, `cre_investment`)
> - `commercial_lending` is **ambiguous and rejected** — same reason
> - `development` is **deprecated and rejected** — use `development_finance` instead

**Required fields by product type:**

| Family | Product type | Required fields |
| ------ | ------------ | --------------- |
| `mortgage` | `mortgage` | `loan_id`, `ead`, `realised_lgd`, `lmi_eligible` (0/1), `mortgage_class` ("Standard"/"Non-Standard") |
| `cashflow_lending` | `commercial_cashflow` | `loan_id`, `ead`, `realised_lgd`, `security_type`, `seniority` |
| `cashflow_lending` | `cashflow_lending` | `loan_id`, `ead`, `realised_lgd`, `pd_score_band`, `cashflow_product`, `seniority` |
| `property_backed_lending` | `development_finance` | `loan_id`, `ead`, `realised_lgd`, `completion_stage` |

All other fields default to conservative values if omitted.

### Single loan — CLI (from a JSON file)

Create `data/my_loan.json`:

```json
{
  "loan_id": "L-1001",
  "ead": 750000,
  "realised_lgd": 0.28,
  "lmi_eligible": 1,
  "mortgage_class": "Standard"
}
```

Then run:

```bash
python -m src.scoring.scoring \
    --product-type mortgage \
    --single-json data/my_loan.json \
    --output outputs/mortgage/my_loan_scored.json
```

### Batch of loans — CLI (from a CSV file)

Prepare a CSV with one row per loan (same column names as the JSON fields above):

```bash
python -m src.scoring.scoring \
    --product-type commercial_cashflow \
    --input-csv data/my_loans.csv \
    --output outputs/cashflow_lending/my_loans_scored.csv
```

### Scoring output fields

| Field | Meaning |
| ----- | ------- |
| `lgd_base` | Point-in-time base LGD |
| `lgd_downturn` | Downturn-adjusted LGD (use for capital) |
| `lgd_final` | Final LGD after all overlays and floor |
| `macro_downturn_scalar` | Macro regime multiplier applied |
| `overlay_source` | Where overlay parameters came from |
| `parameter_version` | Overlay parameter version used |

---

## 3. Run the demo pipeline (end-to-end check)

Runs the full proxy pipeline on built-in demo data and writes outputs to family-scoped folders under `outputs/`.
Use this to verify the system is working or to see example outputs.

```bash
python -m src.pipeline.demo_pipeline
```

Key output files produced:

Product-level outputs (one folder per family):

- `outputs/mortgage/mortgage_loan_level_output.csv`
- `outputs/mortgage/mortgage_segment_summary.csv`
- `outputs/cashflow_lending/commercial_loan_level_output.csv`
- `outputs/cashflow_lending/cashflow_lending_segment_summary.csv`
- `outputs/property_backed_lending/development_loan_level_output.csv`

Portfolio-level and governance outputs:

- `outputs/portfolio/lgd_segment_summary.csv` — EAD-weighted LGD across all products
- `outputs/portfolio/recovery_waterfall.csv` — collateral, recovery, and workout cost breakdown
- `outputs/portfolio/policy_parameter_register.csv` — governance register

---

## 4. Run the calibration pipeline (APS 113)

The calibration pipeline fits the full IRB model on 10-year synthetic workout histories.
Run this after generating historical data or when updating model parameters.

```bash
# Step 1 — generate 10-year synthetic workout histories (2014–2024)
python -m src.data.generator --seed 42

# Step 2 — run full calibration for all products
python -m src.pipeline.calibration_pipeline --products all

# Or calibrate a single product
python -m src.pipeline.calibration_pipeline --module mortgage
```

Calibration pipeline order (APS 113 s.43–65):

```text
Realised LGD → Long-run LGD (vintage EWA) → Downturn overlay
  → Frye-Jacobs LGD-PD correlation → MoC (5 sources) → Regulatory floor
```

---

## 5. Run validation

```bash
# Full validation sequence
python -m src.pipeline.validation_pipeline

# With explicit data source
python -m src.pipeline.validation_pipeline --source generated
```

Produces `outputs/portfolio/validation_sequence_report.csv` with PSI, OOT backtest, Gini, Hosmer-Lemeshow, and conservatism checks.

---

## 6. Run tests

```bash
pytest tests/ -v
```

---

## 7. Repository layout

```text
src/
├── lgd_calculation.py          Proxy engine — Layer 1 (never modify)
├── lgd_calculations.py         Calibration engine — Layer 2
├── lgd_scoring.py              Scoring logic (API)
├── lgd_final.py                Final LGD layer for EL outputs
├── moc_framework.py            Margin of Conservatism register (APS 113 s.65)
├── overlay_parameters.py       Overlay parameter manager (SHA-256 governed)
├── calibration_utils.py        Re-export wrapper used by notebooks
├── validation.py               Full validation suite (PSI, OOT, Gini, HL)
├── aps113_compliance.py        APS 113 compliance map
├── apra_benchmarks.py          APRA peer benchmarking
├── lgd_pd_correlation.py       Frye-Jacobs LGD-PD model
├── segmentation.py             Segment classification
├── commercial_data_controls.py Commercial data quality controls
├── industry_risk_integration.py Industry risk overlay integration
├── reproducibility.py          Seed management
│
├── data/                       Data loading and generation
│   ├── data_generation.py      Proxy generators (3 original products) — shared helpers here
│   ├── data_source_adapter.py  Switches between generated / controlled data sources
│   ├── rba_rates_loader.py     RBA B6 lending rates → discount rates
│   ├── regime_classifier.py    Economic regime classification
│   └── generator.py            CLI: python -m src.data.generator
│
├── generators/                 11 per-product synthetic workout generators (3-family structure)
│   ├── base_generator.py
│   ├── mortgage/               Residential mortgage generator
│   ├── cashflow_lending/       Commercial cashflow, receivables, trade, asset/equipment generators
│   └── property_backed_lending/ Development finance, CRE, residual stock, land, bridging, mezz generators
│
├── pipeline/                   CLI entry points
│   ├── demo_pipeline.py        python -m src.pipeline.demo_pipeline
│   ├── calibration_pipeline.py python -m src.pipeline.calibration_pipeline
│   └── validation_pipeline.py  python -m src.pipeline.validation_pipeline
│
├── scoring/                    Scoring CLI
│   └── scoring.py              python -m src.scoring.scoring
│
└── governance/
    └── gap_matrix.py           APS 113 component gap analysis

data/
├── raw/                        Proxy demo CSVs — do not modify
├── config/
│   ├── overlay_parameters.csv  Versioned overlay/floor parameters (all 11 products)
│   └── overlay_parameters_manifest.json  SHA-256 hash guard
├── external/
│   ├── rba_b6_rates.csv        RBA B6 lending rates 2014–2024
│   └── apra_adi_statistics.csv APRA quarterly impairment ratios
└── generated/
    └── historical/             Synthetic Parquet workout histories (created by generator)

notebooks/
├── 02–12                       Per-product LGD notebooks (proxy + APS 113 calibration)
└── 13                          Cross-product comparison

docs/
├── methodology_cashflow_lending.md       Cashflow lending methodology
├── methodology_property_backed_lending.md Property-backed methodology
└── data_dictionary.md                    Field definitions and output catalogue
```

---

## 8. Products covered

Products are organised into three families. Use the exact `product_type` strings listed below when scoring.

**Family: `mortgage`**

| # | Product | `product_type` | Notebook |
| - | ------- | -------------- | -------- |
| 1 | Residential Mortgage | `mortgage` | `02_residential_mortgage_lgd.ipynb` |

**Family: `cashflow_lending`**

| # | Product | `product_type` | Notebook |
| - | ------- | -------------- | -------- |
| 2 | Commercial Cashflow | `commercial_cashflow` | `03_commercial_cashflow_lgd.ipynb` |
| 3 | Receivables / Invoice Finance | `receivables` | `04_receivables_invoice_finance_lgd.ipynb` |
| 4 | Trade / Contingent | `trade_contingent` | `05_trade_contingent_facilities_lgd.ipynb` |
| 5 | Asset & Equipment Finance | `asset_equipment` | `06_asset_equipment_finance_lgd.ipynb` |

**Family: `property_backed_lending`**

| # | Product | `product_type` | Notebook |
| - | ------- | -------------- | -------- |
| 6 | Development Finance | `development_finance` | `07_development_finance_lgd.ipynb` |
| 7 | CRE Investment | `cre_investment` | `08_cre_investment_lgd.ipynb` |
| 8 | Residual Stock | `residual_stock` | `09_residual_stock_lgd.ipynb` |
| 9 | Land Subdivision | `land_subdivision` | `10_land_subdivision_lgd.ipynb` |
| 10 | Bridging | `bridging` | `11_bridging_loan_lgd.ipynb` |
| 11 | Mezz / 2nd Mortgage | `mezz_second_mortgage` | `12_mezz_second_mortgage_lgd.ipynb` |

> Rejected aliases: `commercial` and `commercial_lending` (ambiguous), `development` (deprecated — use `development_finance`).

Cross-product comparison: `13_cross_product_comparison.ipynb`

---

## 9. Overlay parameters and governance

All LGD floors, downturn scalars, and MoC inputs live in `data/config/overlay_parameters.csv`.
The file is hash-guarded — if you edit it you must regenerate the manifest or the pipeline will refuse to run:

```bash
# After editing overlay_parameters.csv, regenerate the hash manifest
python -m src.overlay_parameters --regenerate-manifest
```

Every pipeline run writes four audit files to `outputs/portfolio/`:

- `overlay_trace_report.csv` — which overlay was applied to which loan
- `parameter_version_report.csv` — version and hash of parameters used
- `segmentation_consistency_report.csv` — segment assignment checks
- `run_metadata_report.csv` — timestamp, seed, source mode

---

## 10. Switching to real data (controlled source)

When you have real loan data, place files in `data/controlled/` organised by family, or flat in the root (the adapter checks both locations):

```text
data/controlled/
├── mortgage/
│   ├── mortgage_loans.csv / mortgage_loans.parquet
│   └── mortgage_cashflows.csv / mortgage_cashflows.parquet
├── cashflow_lending/
│   ├── commercial_loans.csv          (SME/cashflow proxy)
│   ├── cashflow_lending_loans.csv
│   └── ...cashflows variants
└── property_backed_lending/
    ├── development_finance_loans.csv  (NOT "development_loans" — deprecated name)
    └── ...cashflows variants
```

Then run with the controlled source flag:

```bash
python -m src.pipeline.demo_pipeline --source controlled --controlled-root data/controlled
```

---

## 11. Key limitations

- All workout data is **synthetic** — calibration outputs are for demonstration only
- APRA ADI benchmarks use impairment ratios as a **directional proxy**, not a direct LGD benchmark
- MoC values are **illustrative** — require Model Risk Committee sign-off in production
- APS 113 compliance map shows `partial` for s.60 and s.66 because data is not from a real workout tape

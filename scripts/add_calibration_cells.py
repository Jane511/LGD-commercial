"""
One-shot script: append the APS 113 calibration section to notebooks 02-12.

Run once from the repo root:
    python scripts/add_calibration_cells.py

Each notebook gets four new cells appended after its existing content:
  1. Markdown divider (calibration section header)
  2. Import + config cell
  3. All 11 calibration steps (Steps 1-11)
  4. Summary display + APS 113 compliance output

Idempotent: skips any notebook that already contains the calibration marker.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
NB_DIR = REPO_ROOT / "notebooks"

# ── product configuration per notebook ──────────────────────────────────────
PRODUCT_CONFIG: list[dict] = [
    {
        "notebook": "02_residential_mortgage_lgd.ipynb",
        "product": "mortgage",
        "product_title": "Residential Mortgage",
        "segment_keys": ["mortgage_class", "lvr_band"],
        "model_lgd_col": "lgd_final",
        "aps113_floor_note": "10% (Standard+LMI) / 15% (Non-Standard) per APS 113 s.58",
    },
    {
        "notebook": "03_commercial_cashflow_lgd.ipynb",
        "product": "commercial_cashflow",
        "product_title": "Commercial Cashflow",
        "segment_keys": ["security_type", "facility_type"],
        "model_lgd_col": "lgd_final",
        "aps113_floor_note": "25% (Property) / 30% (PPSR) per APS 113 s.58",
    },
    {
        "notebook": "04_receivables_invoice_finance_lgd.ipynb",
        "product": "receivables",
        "product_title": "Receivables / Invoice Finance",
        "segment_keys": ["recourse_flag", "collections_control_flag"],
        "model_lgd_col": "lgd_final",
        "aps113_floor_note": "15% per APS 113 s.58",
    },
    {
        "notebook": "05_trade_contingent_facilities_lgd.ipynb",
        "product": "trade_contingent",
        "product_title": "Trade Contingent Facilities",
        "segment_keys": ["facility_type", "cash_collateral_band"],
        "model_lgd_col": "lgd_final",
        "aps113_floor_note": "15% per APS 113 s.58",
    },
    {
        "notebook": "06_asset_equipment_finance_lgd.ipynb",
        "product": "asset_equipment",
        "product_title": "Asset & Equipment Finance",
        "segment_keys": ["asset_class", "secondary_market_liquidity"],
        "model_lgd_col": "lgd_final",
        "aps113_floor_note": "20% per APS 113 s.58 — depreciating collateral",
    },
    {
        "notebook": "07_development_finance_lgd.ipynb",
        "product": "development_finance",
        "product_title": "Development Finance",
        "segment_keys": ["completion_stage_at_default", "presale_cover_band"],
        "model_lgd_col": "lgd_final",
        "aps113_floor_note": "25-40% depending on stage per APS 113 s.58",
    },
    {
        "notebook": "08_cre_investment_lgd.ipynb",
        "product": "cre_investment",
        "product_title": "CRE Investment",
        "segment_keys": ["asset_class_cre", "lvr_band"],
        "model_lgd_col": "lgd_final",
        "aps113_floor_note": "25% per APS 113 s.58",
    },
    {
        "notebook": "09_residual_stock_lgd.ipynb",
        "product": "residual_stock",
        "product_title": "Residual Stock",
        "segment_keys": ["market_depth_proxy", "stock_age_band"],
        "model_lgd_col": "lgd_final",
        "aps113_floor_note": "30% per APS 113 s.58",
    },
    {
        "notebook": "10_land_subdivision_lgd.ipynb",
        "product": "land_subdivision",
        "product_title": "Land Subdivision",
        "segment_keys": ["zoning_stage", "market_depth_proxy"],
        "model_lgd_col": "lgd_final",
        "aps113_floor_note": "35% per APS 113 s.58 — raw land highest floor",
    },
    {
        "notebook": "11_bridging_loan_lgd.ipynb",
        "product": "bridging",
        "product_title": "Bridging Loans",
        "segment_keys": ["exit_certainty_band", "exit_type"],
        "model_lgd_col": "lgd_final",
        "aps113_floor_note": "25% per APS 113 s.58",
    },
    {
        "notebook": "12_mezz_second_mortgage_lgd.ipynb",
        "product": "mezz_second_mortgage",
        "product_title": "Mezzanine / Second Mortgage",
        "segment_keys": ["seniority", "attachment_point_band"],
        "model_lgd_col": "lgd_final",
        "aps113_floor_note": "40% per APS 113 s.58 — junior ranking",
    },
]

# ── calibration marker (used for idempotency check) ─────────────────────────
CALIBRATION_MARKER = "## APS 113 Calibration Layer"


def _make_cells(cfg: dict) -> list[dict]:
    """Return the four new notebook cells for this product."""
    product = cfg["product"]
    title = cfg["product_title"]
    seg_keys = cfg["segment_keys"]
    model_lgd_col = cfg["model_lgd_col"]
    floor_note = cfg["aps113_floor_note"]
    seg_keys_repr = repr(seg_keys)

    # ── Cell 1: Markdown divider ─────────────────────────────────────────────
    md_header = f"""\
---

## APS 113 Calibration Layer — {title}

> **SYNTHETIC HISTORICAL CALIBRATION DATA — FOR DEMONSTRATION ONLY**
>
> This section adds a full APS 113-aligned calibration loop on top of the
> proxy baseline above. Workout data is synthetically generated (2014-2024,
> 10-year window) to demonstrate methodology; real production calibration
> requires an internal workout tape.

### Calibration Pipeline (APS 113 s.32-68)

| Step | Function | APS 113 |
|------|----------|---------|
| 1 | Load/generate historical workout data | s.44, Att A |
| 2 | `compute_realised_lgd()` — LIP costs, cure leg | s.32-34 |
| 3 | `classify_economic_regime()` + `assign_regime_to_workouts()` | s.43, s.46 |
| 4 | `segment_lgd()` — product-specific segment keys | s.52 |
| 5 | `compute_long_run_lgd()` — vintage-EWA method | s.43-44 |
| 6 | `compare_model_vs_actual()` — proxy vs realised | s.60-62 |
| 7 | `apply_downturn_overlay()` + `apply_correlation_adjustment()` | s.46-50, s.55-57 |
| 8 | `MoCRegister` + `apply_moc()` — 5 APS 113 s.65 sources | s.63-65 |
| 9 | `apply_regulatory_floor()` — {floor_note} | s.58 |
| 10 | Export 9 CSV outputs | — |
| 11 | `run_full_validation_suite()` — Gini, HL, PSI, OOT | s.66-68 |

**Correct APS 113 order:** LR-LGD → downturn overlay → correlation adj →
MoC → floor (MoC applied to downturn LGD, not long-run LGD, per s.63).
"""

    # ── Cell 2: Imports and config ───────────────────────────────────────────
    code_imports = f"""\
# APS 113 Calibration Layer — imports and configuration
import os, sys
from pathlib import Path
sys.path.insert(0, os.path.abspath('..'))

from src.calibration_utils import (
    compute_realised_lgd,
    segment_lgd,
    compute_long_run_lgd,
    compare_model_vs_actual,
    classify_economic_regime,
    assign_regime_to_workouts,
    apply_downturn_overlay,
    apply_correlation_adjustment,
    build_lgd_pd_annual_series,
    estimate_lgd_pd_correlation,
    MoCRegister,
    apply_moc,
    apply_regulatory_floor,
    run_full_validation_suite,
    generate_compliance_map,
    export_compliance_map,
    CALIBRATION_STEP_ORDER,
)
from src.generators import GENERATOR_MAP, generate_all_historical_workouts

PRODUCT = "{product}"
SEGMENT_KEYS = {seg_keys_repr}
MODEL_LGD_COL = "{model_lgd_col}"

HISTORY_DIR = Path('..') / 'data' / 'generated' / 'historical'
OUTPUTS_DIR = Path('..') / 'outputs' / 'tables'
UPSTREAM_PARQUET = Path('..') / 'data' / 'exports' / 'macro_regime_flags.parquet'

HISTORY_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Product: {{PRODUCT}}")
print(f"Segment keys: {{SEGMENT_KEYS}}")
print(f"APS 113 calibration pipeline — step order:")
for step, fn, ref in CALIBRATION_STEP_ORDER:
    print(f"  Step {{step:>2}}: {{fn:<35}} {{ref}}")
"""

    # ── Cell 3: Steps 1-11 ──────────────────────────────────────────────────
    code_pipeline = f"""\
# ── STEP 1: Load or generate historical workout data (APS 113 s.44, Att A) ─
parquet_path = HISTORY_DIR / f"{{PRODUCT}}_workouts.parquet"

if parquet_path.exists():
    cal_loans = pd.read_parquet(parquet_path)
    # cashflows stored alongside or re-generated
    try:
        cal_cashflows = pd.read_parquet(
            HISTORY_DIR / f"{{PRODUCT}}_cashflows.parquet"
        )
    except FileNotFoundError:
        cal_cashflows = None
    print(f"Loaded {{len(cal_loans):,}} historical workout loans from Parquet")
else:
    print(f"Parquet not found — generating synthetic workout data for {{PRODUCT}}")
    results = generate_all_historical_workouts(
        seed=42, output_dir=HISTORY_DIR, write_parquet=True, products=[PRODUCT]
    )
    cal_loans = results[PRODUCT]["loans"]
    cal_cashflows = results[PRODUCT].get("cashflows")
    print(f"Generated {{len(cal_loans):,}} synthetic workout loans (2014-2024)")

print(f"Date range: {{cal_loans['default_date'].min()}} to {{cal_loans['default_date'].max()}}")
print(f"Columns: {{list(cal_loans.columns)}}")

# ── STEP 2: Compute realised LGD (APS 113 s.32-34) ─────────────────────────
# LIP costs (Loss Identification Period, ~90 days) auto-detected if cashflows available
lgd_df = compute_realised_lgd(
    loans=cal_loans,
    cashflows=cal_cashflows,
    ead_col="ead_at_default",
    recovery_col="recovery_amount",
    cost_col="direct_costs",
    lip_window_days=90,
)
print(f"\\nStep 2: Realised LGD computed")
print(f"  EAD-weighted realised LGD: {{(lgd_df['realised_lgd'] * lgd_df['ead_at_default']).sum() / lgd_df['ead_at_default'].sum():.2%}}")
display(lgd_df[['realised_lgd', 'ead_at_default']].describe().round(4))

# ── STEP 3: Economic regime classification (APS 113 s.43, s.46-50) ─────────
upstream_path = str(UPSTREAM_PARQUET) if UPSTREAM_PARQUET.exists() else None
regime_df = classify_economic_regime(
    upstream_parquet_path=upstream_path,
    method="upstream_first",
)
print(f"\\nStep 3: Economic regimes classified")
print(f"  Data source: {{regime_df['data_source'].iloc[0]}}")
display(regime_df[['year', 'regime', 'is_downturn_period', 'data_source']].head(12))

lgd_df = assign_regime_to_workouts(lgd_df, regime_df)
downturn_pct = lgd_df.get('is_downturn_period', pd.Series([False])).mean()
print(f"  Downturn observations: {{downturn_pct:.1%}} of portfolio")

# ── STEP 4: Segment by product-specific keys (APS 113 s.52) ────────────────
segmented_df = segment_lgd(lgd_df, SEGMENT_KEYS)
low_count = segmented_df[segmented_df.get('segment_flag', '') == 'low_count'] if 'segment_flag' in segmented_df.columns else pd.DataFrame()
print(f"\\nStep 4: Segmentation complete")
print(f"  Segments: {{segmented_df.groupby(SEGMENT_KEYS, observed=True).ngroups}}")
if not low_count.empty:
    print(f"  Low-count segments flagged (<20 obs): {{len(low_count)}}")

# ── STEP 5: Long-run LGD — vintage-EWA (APS 113 s.43-44) ─────────────────
lr_lgd_df = compute_long_run_lgd(
    segmented_df,
    segment_keys=SEGMENT_KEYS,
    method="vintage_ewa",
)
print(f"\\nStep 5: Long-run LGD by segment (vintage-EWA)")
display(lr_lgd_df.round(4))
lr_lgd_df.to_csv(OUTPUTS_DIR / f"{{PRODUCT}}_long_run_lgd_by_segment.csv", index=False)

# ── STEP 6: Compare model vs actual (APS 113 s.60-62) ──────────────────────
# 'model_lgd' here = proxy model LGD from the section above (lgd_final if present)
if MODEL_LGD_COL in cal_loans.columns:
    compare_input = lgd_df.merge(
        cal_loans[['loan_id', MODEL_LGD_COL] if 'loan_id' in cal_loans.columns else [MODEL_LGD_COL]],
        left_index=True, right_index=True, how='left',
    ) if MODEL_LGD_COL not in lgd_df.columns else lgd_df
else:
    compare_input = lgd_df.copy()
    compare_input['model_lgd'] = lr_lgd_df['long_run_lgd'].mean() if 'long_run_lgd' in lr_lgd_df.columns else 0.25

model_col_to_use = MODEL_LGD_COL if MODEL_LGD_COL in compare_input.columns else 'model_lgd'
mva_df = compare_model_vs_actual(
    compare_input,
    model_lgd_col=model_col_to_use,
    segment_keys=SEGMENT_KEYS,
)
print(f"\\nStep 6: Model vs actual comparison")
display(mva_df.round(4))
mva_df.to_csv(OUTPUTS_DIR / f"{{PRODUCT}}_model_vs_actual_comparison.csv", index=False)

# ── STEP 7: Downturn overlay + Frye-Jacobs correlation adj (s.46-50, s.55-57)
# Reuses apply_downturn_overlay from existing lgd_calculation.py
dt_lgd = apply_downturn_overlay(segmented_df, product=PRODUCT)
print(f"\\nStep 7: Downturn overlay applied")
if 'lgd_downturn' in dt_lgd.columns:
    ewa_dt = (dt_lgd['lgd_downturn'] * dt_lgd['ead_at_default']).sum() / dt_lgd['ead_at_default'].sum()
    ewa_lr = (dt_lgd['realised_lgd'] * dt_lgd['ead_at_default']).sum() / dt_lgd['ead_at_default'].sum()
    print(f"  EWA Long-run LGD:  {{ewa_lr:.2%}}")
    print(f"  EWA Downturn LGD:  {{ewa_dt:.2%}}")
    downturn_col = 'lgd_downturn'
else:
    dt_lgd['lgd_downturn'] = dt_lgd['realised_lgd'] * 1.15  # fallback scalar
    downturn_col = 'lgd_downturn'

# Frye-Jacobs correlation adjustment (APS 113 s.55-57)
try:
    lgd_ts, pd_ts = build_lgd_pd_annual_series(dt_lgd)
    macro_for_corr = regime_df.rename(columns={{'gdp_growth': 'gdp_growth_yoy'}})
    corr_result = estimate_lgd_pd_correlation(lgd_ts, pd_ts, macro_for_corr)
    dt_lgd['lgd_downturn_corr_adj'] = apply_correlation_adjustment(
        dt_lgd[downturn_col], corr_result['rho'], corr_result['macro_shock_std']
    )
    downturn_col = 'lgd_downturn_corr_adj'
    print(f"  Frye-Jacobs rho={{corr_result['rho']:.3f}}, adj factor={{corr_result['lgd_dt_adjustment_factor']:.3f}}")
except Exception as exc:
    print(f"  Frye-Jacobs skipped: {{exc}}")

dt_lgd.to_csv(OUTPUTS_DIR / f"{{PRODUCT}}_downturn_lgd_by_segment.csv", index=False)

# ── STEP 8: MoC register + apply MoC (AFTER downturn — APS 113 s.63-65) ───
# Determine regime data source for MoC model_approximation component
data_src = regime_df['data_source'].iloc[0] if 'data_source' in regime_df.columns else 'synthetic'
n_downturn_yrs = int(regime_df['is_downturn_period'].sum()) if 'is_downturn_period' in regime_df.columns else 0

psi_approx = 0.05  # placeholder — full PSI computed in Step 11
bias_approx = float(mva_df['bias'].abs().mean()) if 'bias' in mva_df.columns else 0.02

moc_register = MoCRegister(product=PRODUCT, regime_data_source=data_src)
moc_df = moc_register.build_moc_register(
    segment_df=segmented_df,
    segment_keys=SEGMENT_KEYS,
    n_downturn_vintages=n_downturn_yrs,
    psi_value=psi_approx,
    backtesting_bias=bias_approx,
)
print(f"\\nStep 8: MoC register built")
display(moc_df.round(4))
moc_df.to_csv(OUTPUTS_DIR / f"{{PRODUCT}}_moc_register.csv", index=False)

lgd_with_moc = apply_moc(dt_lgd[downturn_col], moc_df, segment_col=SEGMENT_KEYS[0] if SEGMENT_KEYS else None)
dt_lgd['lgd_with_moc'] = lgd_with_moc
moc_ewa = (lgd_with_moc * dt_lgd['ead_at_default']).sum() / dt_lgd['ead_at_default'].sum()
print(f"  EWA LGD after MoC: {{moc_ewa:.2%}}")

# ── STEP 9: Regulatory floors (APS 113 s.58) ──────────────────────────────
dt_lgd['lgd_final_calibrated'] = apply_regulatory_floor(dt_lgd['lgd_with_moc'], product=PRODUCT)
final_ewa = (dt_lgd['lgd_final_calibrated'] * dt_lgd['ead_at_default']).sum() / dt_lgd['ead_at_default'].sum()
floor_binding_pct = (dt_lgd['lgd_final_calibrated'] > dt_lgd['lgd_with_moc']).mean()
print(f"\\nStep 9: Regulatory floor applied")
print(f"  EWA Final Calibrated LGD: {{final_ewa:.2%}}")
print(f"  Floor binding for: {{floor_binding_pct:.1%}} of loans")

dt_lgd.to_csv(OUTPUTS_DIR / f"{{PRODUCT}}_final_calibrated_lgd.csv", index=False)

# ── STEP 10: Export all outputs ────────────────────────────────────────────
# Already exported: long_run_lgd_by_segment, model_vs_actual, downturn_lgd, moc_register, final_calibrated_lgd
# Export remaining:
lgd_df[['realised_lgd', 'ead_at_default']].assign(product=PRODUCT).to_csv(
    OUTPUTS_DIR / f"{{PRODUCT}}_historical_workouts.csv", index=False
)
regime_df.to_csv(OUTPUTS_DIR / f"{{PRODUCT}}_regime_classification.csv", index=False)

# Calibration adjustments summary
cal_adj_summary = pd.DataFrame({{
    'product': [PRODUCT],
    'ewa_realised_lgd': [(lgd_df['realised_lgd'] * lgd_df['ead_at_default']).sum() / lgd_df['ead_at_default'].sum()],
    'ewa_long_run_lgd': [lr_lgd_df['long_run_lgd'].mean() if 'long_run_lgd' in lr_lgd_df.columns else None],
    'ewa_downturn_lgd': [(dt_lgd.get('lgd_downturn', dt_lgd['realised_lgd']) * dt_lgd['ead_at_default']).sum() / dt_lgd['ead_at_default'].sum()],
    'ewa_lgd_with_moc': [(dt_lgd['lgd_with_moc'] * dt_lgd['ead_at_default']).sum() / dt_lgd['ead_at_default'].sum()],
    'ewa_lgd_final': [final_ewa],
    'floor_binding_pct': [floor_binding_pct],
    'regime_data_source': [data_src],
    'n_loans': [len(lgd_df)],
    'calibration_date': [pd.Timestamp.today().date()],
}})
cal_adj_summary.to_csv(OUTPUTS_DIR / f"{{PRODUCT}}_calibration_adjustments.csv", index=False)
print(f"\\nStep 10: All outputs exported to {{OUTPUTS_DIR}}")

# ── STEP 11: Full validation suite (APS 113 s.66-68) ──────────────────────
print(f"\\nStep 11: Running full validation suite...")
try:
    val_results = run_full_validation_suite(
        loans=dt_lgd,
        predicted_col='lgd_final_calibrated',
        actual_col='realised_lgd',
        segment_col=SEGMENT_KEYS[0] if SEGMENT_KEYS else None,
        date_col='default_date' if 'default_date' in dt_lgd.columns else None,
        product=PRODUCT,
    )
    print(f"  Gini: {{val_results.get('gini', 'n/a')}}")
    print(f"  Calibration ratio: {{val_results.get('calibration_ratio', 'n/a')}}")
    print(f"  PSI: {{val_results.get('psi', 'n/a')}}")
    if 'summary_table' in val_results:
        val_results['summary_table'].to_csv(
            OUTPUTS_DIR / f"{{PRODUCT}}_validation_report.csv", index=False
        )
    if 'backtest_results' in val_results:
        val_results['backtest_results'].to_csv(
            OUTPUTS_DIR / f"{{PRODUCT}}_backtest_results.csv", index=False
        )
except Exception as exc:
    print(f"  Validation suite error (non-fatal): {{exc}}")
"""

    # ── Cell 4: Summary waterfall + APS 113 compliance ───────────────────────
    code_summary = f"""\
# ── Calibration summary waterfall ──────────────────────────────────────────
import matplotlib.pyplot as plt

try:
    stages = {{
        'Realised LGD\\n(2014-2024)': (lgd_df['realised_lgd'] * lgd_df['ead_at_default']).sum() / lgd_df['ead_at_default'].sum(),
        'Long-run LGD\\n(vintage-EWA)': lr_lgd_df['long_run_lgd'].mean() if 'long_run_lgd' in lr_lgd_df.columns else None,
        'Downturn LGD': (dt_lgd.get(downturn_col, dt_lgd['realised_lgd']) * dt_lgd['ead_at_default']).sum() / dt_lgd['ead_at_default'].sum(),
        '+ MoC': (dt_lgd['lgd_with_moc'] * dt_lgd['ead_at_default']).sum() / dt_lgd['ead_at_default'].sum(),
        'Final\\n(Floor Applied)': final_ewa,
    }}
    labels = [k for k, v in stages.items() if v is not None]
    values = [v for v in stages.values() if v is not None]
    colors = ['#3498db', '#2ecc71', '#e67e22', '#e74c3c', '#8e44ad'][:len(labels)]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor='white', width=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{{val:.1%}}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax.set_ylabel('EAD-Weighted LGD')
    ax.set_title(f'APS 113 Calibration Waterfall — {title}')
    ax.set_ylim(0, max(values) * 1.35)
    ax.axhline(values[-1], color='black', ls=':', lw=1, label=f'Final: {{values[-1]:.1%}}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        Path('..') / 'outputs' / 'figures' / f'{product}_calibration_waterfall.png',
        dpi=150, bbox_inches='tight',
    )
    plt.show()
except Exception as exc:
    print(f"Waterfall chart error (non-fatal): {{exc}}")

# ── APS 113 compliance snapshot ─────────────────────────────────────────────
compliance_df = generate_compliance_map(
    calibration_results={{PRODUCT: {{'long_run_lgd_by_segment': True, 'calibration_steps': True}}}},
    moc_registers={{PRODUCT: moc_df}},
    regime_data_source=data_src,
    products=[PRODUCT],
)
print("\\n=== APS 113 Compliance Snapshot ===")
display(compliance_df[['section_ref', 'requirement', 'status', 'reviewer_note']].set_index('section_ref'))
export_compliance_map(compliance_df, OUTPUTS_DIR / f"{product}_aps113_compliance.csv")

# Final summary
print("\\n=== Calibration Summary ===")
display(cal_adj_summary.round(4))
print(f"\\nAll calibration outputs in: {{OUTPUTS_DIR}}")
print(f"SYNTHETIC HISTORICAL CALIBRATION DATA — FOR DEMONSTRATION ONLY")
"""

    return [
        {"cell_type": "markdown", "source": md_header},
        {"cell_type": "code",     "source": code_imports},
        {"cell_type": "code",     "source": code_pipeline},
        {"cell_type": "code",     "source": code_summary},
    ]


def _make_cell_obj(cell_type: str, source: str) -> dict:
    """Build a minimal nbformat 4 cell dict."""
    if cell_type == "markdown":
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": source,
        }
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def process_notebook(cfg: dict) -> None:
    nb_path = NB_DIR / cfg["notebook"]
    if not nb_path.exists():
        print(f"  SKIP (not found): {nb_path.name}")
        return

    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    # Idempotency check — skip if calibration section already added
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if CALIBRATION_MARKER in src:
            print(f"  SKIP (already calibrated): {nb_path.name}")
            return

    new_cells = _make_cells(cfg)
    for c in new_cells:
        nb["cells"].append(_make_cell_obj(c["cell_type"], c["source"]))

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")

    print(f"  OK ({len(new_cells)} cells added): {nb_path.name}")


def main() -> None:
    print(f"Adding APS 113 calibration sections to {len(PRODUCT_CONFIG)} notebooks...\n")
    for cfg in PRODUCT_CONFIG:
        process_notebook(cfg)
    print("\nDone.")


if __name__ == "__main__":
    main()

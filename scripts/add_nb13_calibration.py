"""Append the APS 113 calibration comparison section to notebook 13."""
from __future__ import annotations
import json
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
NB_PATH = REPO_ROOT / "notebooks" / "13_cross_product_comparison.ipynb"
MARKER = "## APS 113 Calibration Comparison"

MD_HEADER = """\
---

## APS 113 Calibration Comparison — Cross-Product

> **SYNTHETIC HISTORICAL CALIBRATION DATA — FOR DEMONSTRATION ONLY**
>
> This section overlays the APS 113 calibration results from notebooks 02-12
> onto the proxy-based comparison above. It shows:
> calibrated LGD waterfall, seniority rank-ordering, MoC summary,
> APRA benchmark comparison, and indicative RWA impact.

**Correct APS 113 pipeline order per product:**
Realised LGD → Long-run LGD (vintage-EWA) → Downturn overlay →
Frye-Jacobs correlation adj → MoC (s.63-65) → Regulatory floor (s.58) → Final
"""

CODE_LOAD = """\
# ── Load per-product calibration adjustment summaries (from notebooks 02-12) ─
import os, sys
from pathlib import Path
sys.path.insert(0, os.path.abspath('..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

OUTPUTS = Path('..') / 'outputs' / 'tables'
PRODUCTS = [
    'mortgage', 'commercial_cashflow', 'receivables', 'trade_contingent',
    'asset_equipment', 'development_finance', 'cre_investment',
    'residual_stock', 'land_subdivision', 'bridging', 'mezz_second_mortgage',
]


def _load_cal_adj(product: str) -> dict | None:
    path = OUTPUTS / f'{product}_calibration_adjustments.csv'
    if not path.exists():
        return None
    row = pd.read_csv(path).iloc[0].to_dict()
    row['product'] = product
    return row


def _load_moc(product: str) -> pd.DataFrame | None:
    path = OUTPUTS / f'{product}_moc_register.csv'
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df['product'] = product
    return df


rows = [r for p in PRODUCTS if (r := _load_cal_adj(p)) is not None]
cal_df = pd.DataFrame(rows)
moc_frames = [f for p in PRODUCTS if (f := _load_moc(p)) is not None]
moc_all = pd.concat(moc_frames, ignore_index=True) if moc_frames else pd.DataFrame()

if cal_df.empty:
    print("No calibration outputs found. Run notebooks 02-12 calibration sections first.")
    print("Or: python scripts/run_calibration_pipeline.py --products all")
else:
    print(f"Loaded calibration results for {len(cal_df)} products")
    cols = ['ewa_realised_lgd', 'ewa_long_run_lgd', 'ewa_downturn_lgd',
            'ewa_lgd_with_moc', 'ewa_lgd_final']
    display(cal_df.set_index('product')[[c for c in cols if c in cal_df.columns]].round(4))
"""

CODE_WATERFALL = """\
# ── Calibration waterfall: all products ─────────────────────────────────────
if not cal_df.empty:
    numeric_cols = ['ewa_realised_lgd', 'ewa_long_run_lgd', 'ewa_downturn_lgd',
                    'ewa_lgd_with_moc', 'ewa_lgd_final']
    stage_labels = ['Realised\\n(2014-24)', 'Long-run\\n(EWA)',
                    'Downturn', '+ MoC', 'Final\\n(Floor)']
    available_cols = [c for c in numeric_cols if c in cal_df.columns]
    n_stages = len(available_cols)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Left: grouped bar chart — all pipeline stages per product
    x = np.arange(len(cal_df))
    width = 0.14
    colors = ['#3498db', '#2ecc71', '#e67e22', '#e74c3c', '#8e44ad']
    for i, (col, label) in enumerate(zip(available_cols, stage_labels[:n_stages])):
        axes[0].bar(x + i * width, cal_df[col], width, label=label,
                    color=colors[i % len(colors)], alpha=0.85)
    axes[0].set_xticks(x + width * (n_stages - 1) / 2)
    axes[0].set_xticklabels(cal_df['product'].str.replace('_', '\\n'), fontsize=7)
    axes[0].set_ylabel('EAD-Weighted LGD')
    axes[0].set_title('APS 113 Calibration Waterfall — All Products')
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    axes[0].legend(fontsize=8)

    # Right: final calibrated LGD rank order
    sorted_df = cal_df.sort_values('ewa_lgd_final', ascending=True)
    bars = axes[1].barh(sorted_df['product'], sorted_df['ewa_lgd_final'],
                        color='#8e44ad', alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, sorted_df['ewa_lgd_final']):
        axes[1].text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                     f'{val:.1%}', va='center', fontsize=9)
    axes[1].set_xlabel('Final Calibrated LGD (EWA)')
    axes[1].set_title('Final LGD Rank Order (APS 113 s.58 floor applied)')
    axes[1].xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

    plt.tight_layout()
    plt.savefig(Path('..') / 'outputs' / 'figures' / 'cross_product_calibration_waterfall.png',
                dpi=150, bbox_inches='tight')
    plt.show()
"""

CODE_SENIORITY = """\
# ── Seniority rank-ordering check (APS 113 best practice) ──────────────────
# Expected: more senior exposures should generally show lower LGD.
# This validates rank-ordering reasonableness across the product hierarchy.
SENIORITY_ORDER = {
    'mortgage': 1, 'cre_investment': 2, 'asset_equipment': 3,
    'commercial_cashflow': 4, 'development_finance': 5, 'receivables': 6,
    'trade_contingent': 7, 'land_subdivision': 8, 'residual_stock': 9,
    'bridging': 10, 'mezz_second_mortgage': 11,
}

if not cal_df.empty and 'ewa_lgd_final' in cal_df.columns:
    from scipy.stats import spearmanr

    rank_df = cal_df[['product', 'ewa_lgd_final']].copy()
    rank_df['seniority_rank'] = rank_df['product'].map(SENIORITY_ORDER)
    rank_df = rank_df.dropna(subset=['seniority_rank']).sort_values('seniority_rank')
    rank_df['lgd_rank'] = rank_df['ewa_lgd_final'].rank()
    rank_df['rank_diff'] = rank_df['lgd_rank'] - rank_df['seniority_rank']
    rank_df['rank_ordering_ok'] = rank_df['rank_diff'].abs() <= 2

    rho, pval = spearmanr(rank_df['seniority_rank'], rank_df['lgd_rank'])

    print('=== Seniority Rank-Ordering Check ===')
    print(f'  Spearman rho (seniority vs LGD rank): {rho:.3f}  p={pval:.3f}')
    print(f'  Expected: positive correlation (senior = first-ranking = lower LGD)')
    violations = (~rank_df['rank_ordering_ok']).sum()
    print(f'  Rank violations (|diff| > 2): {violations}')
    if violations == 0:
        print('  PASS — rank ordering consistent with seniority hierarchy')
    else:
        print('  REVIEW — investigate rank violations against model assumptions')
    display(rank_df[['product', 'seniority_rank', 'lgd_rank', 'ewa_lgd_final',
                      'rank_diff', 'rank_ordering_ok']].round(3))
"""

CODE_MOC = """\
# ── MoC summary across all products (APS 113 s.63-65) ──────────────────────
if not moc_all.empty:
    print('=== MoC Summary — All Products ===')
    moc_components = [c for c in moc_all.columns
                      if c.endswith('_moc') and c != 'total_moc']
    if 'total_moc' in moc_all.columns:
        moc_summary = (
            moc_all.groupby('product')[['total_moc'] + moc_components]
            .mean()
            .round(4)
        )
        display(moc_summary)

        if moc_components:
            fig, ax = plt.subplots(figsize=(13, 5))
            moc_pivot = moc_all.groupby('product')[moc_components].mean()
            moc_pivot.plot(kind='bar', stacked=True, ax=ax,
                           colormap='tab10', edgecolor='white', width=0.7)
            ax.set_ylabel('MoC Add-on (absolute LGD)')
            ax.set_title('APS 113 s.65 MoC Components by Product — Five Required Sources')
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
            ax.legend(loc='upper right', fontsize=8)
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()
            plt.savefig(Path('..') / 'outputs' / 'figures' / 'cross_product_moc_summary.png',
                        dpi=150, bbox_inches='tight')
            plt.show()
    moc_all.to_csv(OUTPUTS / 'moc_summary_all_products.csv', index=False)
    print('Saved outputs/tables/moc_summary_all_products.csv')
else:
    print('MoC registers not generated yet — run notebooks 02-12 calibration sections')
"""

CODE_APRA_RWA = """\
# ── APRA ADI benchmark comparison ───────────────────────────────────────────
apra_path = OUTPUTS / 'apra_benchmark_comparison.csv'
if apra_path.exists():
    apra_df = pd.read_csv(apra_path)
    print('=== APRA ADI Benchmark Comparison ===')
    print('NOTE: APRA impairment ratio is a directional proxy only — not a direct LGD benchmark.')
    display(apra_df.round(4))
else:
    print('APRA benchmark not generated — run: python scripts/run_calibration_pipeline.py')

# ── Indicative RWA impact: proxy vs calibrated LGD ─────────────────────────
if not cal_df.empty and 'ewa_lgd_final' in cal_df.columns:

    def _irb_rwa_approx(ead: float, pd_: float, lgd: float) -> float:
        # Simplified IRB corporate RWA. APS 113 Attachment C.
        r = (0.12 * (1 - np.exp(-50 * pd_)) / (1 - np.exp(-50))
             + 0.24 * (1 - (1 - np.exp(-50 * pd_)) / (1 - np.exp(-50))))
        z = (pd_ * 1.06 + (pd_ * (1 - pd_)) ** 0.5 * 1.645
             * (r ** 0.5 / (1 - r) ** 0.5 - pd_ * 0.5))
        k = max(lgd * z, 0.0)
        return k * ead * 12.5

    rwa_df = cal_df[['product']].copy()
    rwa_df['proxy_lgd'] = cal_df.get('ewa_realised_lgd',
                                      cal_df['ewa_lgd_final'] * 0.85)
    rwa_df['calibrated_lgd'] = cal_df['ewa_lgd_final']
    PD = 0.02
    EAD = 100_000_000  # $100m illustrative

    rwa_df['rwa_proxy'] = rwa_df['proxy_lgd'].apply(
        lambda lgd: _irb_rwa_approx(EAD, PD, lgd))
    rwa_df['rwa_calibrated'] = rwa_df['calibrated_lgd'].apply(
        lambda lgd: _irb_rwa_approx(EAD, PD, lgd))
    rwa_df['rwa_delta_pct'] = (
        (rwa_df['rwa_calibrated'] - rwa_df['rwa_proxy']) / rwa_df['rwa_proxy']
    )

    print('\\n=== Indicative RWA Impact: Proxy vs Calibrated LGD ===')
    print('(Illustrative: $100m EAD per product, flat PD=2%, simplified IRB formula)')
    display(rwa_df.set_index('product')[
        ['proxy_lgd', 'calibrated_lgd', 'rwa_proxy', 'rwa_calibrated', 'rwa_delta_pct']
    ].round(4))
    rwa_df.to_csv(OUTPUTS / 'cross_product_rwa_impact.csv', index=False)
    print('Saved outputs/tables/cross_product_rwa_impact.csv')
"""


def make_cell(cell_type: str, source: str) -> dict:
    if cell_type == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": source}
    return {
        "cell_type": "code", "execution_count": None,
        "metadata": {}, "outputs": [], "source": source,
    }


def main() -> None:
    with open(NB_PATH, encoding="utf-8") as f:
        nb = json.load(f)

    # Idempotency check
    for c in nb["cells"]:
        if MARKER in "".join(c.get("source", [])):
            print(f"SKIP (already present): {NB_PATH.name}")
            return

    new_cells = [
        make_cell("markdown", MD_HEADER),
        make_cell("code", CODE_LOAD),
        make_cell("code", CODE_WATERFALL),
        make_cell("code", CODE_SENIORITY),
        make_cell("code", CODE_MOC),
        make_cell("code", CODE_APRA_RWA),
    ]
    for c in new_cells:
        nb["cells"].append(c)

    with open(NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")

    print(f"Added {len(new_cells)} cells to {NB_PATH.name}")


if __name__ == "__main__":
    main()

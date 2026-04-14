"""
APS 113 Compliance Map Generator.

Produces outputs/tables/aps113_compliance_map.csv — a section-by-section
assessment of this repo's implementation against APRA APS 113 requirements.

For each product module and major methodology step, records:
    - Which APS 113 section governs it
    - Whether implementation is 'met', 'partial', or 'not_met'
    - Evidence file and detail
    - Reviewer note

APS 113 Reference: All sections.
APRA Prudential Standard APS 113 — Capital Adequacy: Internal Ratings-based
Approach to Credit Risk.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# APS 113 requirement registry
# ---------------------------------------------------------------------------

APS113_REQUIREMENTS = {
    "s32_lip_costs": {
        "section": "s.32",
        "requirement": "LGD computation includes Loss Identification Period costs",
        "description": (
            "Costs incurred between actual default and formal identification "
            "(typically 60-90 days) must be included in LGD numerator."
        ),
    },
    "s37_ead_definition": {
        "section": "s.37",
        "requirement": "EAD correctly defined at time of default",
        "description": "EAD includes drawn balance, accrued interest, fees.",
    },
    "s43_long_run_lgd": {
        "section": "s.43",
        "requirement": "Long-run LGD estimated as exposure-weighted average through cycle",
        "description": "Must include full economic cycle data. Vintage-EWA method used.",
    },
    "s44_observation_period": {
        "section": "s.44 / Att A",
        "requirement": "Minimum observation period: 7y mortgage, 5y others",
        "description": "Observation window: 2014-2024 (10 years) exceeds minimum.",
    },
    "s46_downturn_lgd": {
        "section": "s.46-50",
        "requirement": "Downturn LGD >= long-run LGD; reflects downturn conditions",
        "description": "Downturn overlay applied using product-specific scalars.",
    },
    "s50_discount_rate": {
        "section": "s.50",
        "requirement": "Contractual interest rate used for discounting workout cashflows",
        "description": (
            "Primary: RBA B6 indicator lending rates by product type and vintage year. "
            "Fallback: RBA cash rate + 300bps."
        ),
    },
    "s52_segmentation": {
        "section": "s.52",
        "requirement": "LGD segmented by risk-homogeneous groups",
        "description": "Product-specific segment keys; segments with < 20 obs flagged.",
    },
    "s58_lgd_floors": {
        "section": "s.58",
        "requirement": "LGD subject to prescribed minimums (floors)",
        "description": (
            "Mortgage: 10% (LMI) / 15% (no LMI). "
            "Commercial: 15-40% depending on security. "
            "Applied after MoC per run_calibration_pipeline()."
        ),
    },
    "s60_model_vs_actual": {
        "section": "s.60-62",
        "requirement": "Model LGD compared to historical realised LGD",
        "description": "compare_model_vs_actual() produces model-vs-actual table per segment.",
    },
    "s63_moc": {
        "section": "s.63",
        "requirement": "Margin of Conservatism added to final parameter estimate",
        "description": (
            "MoC applied AFTER downturn overlay (correct order: LR → downturn → MoC → floor). "
            "Five APS 113 s.65 sources evaluated per segment."
        ),
    },
    "s65_moc_sources": {
        "section": "s.65",
        "requirement": "MoC covers: data quality, model error, incomplete workout, cyclicality, instability",
        "description": "All five sources implemented in MoCRegister with individual basis-point addons.",
    },
    "s66_validation": {
        "section": "s.66-68",
        "requirement": "Model validated against independent datasets",
        "description": (
            "Gini/AUROC, Hosmer-Lemeshow, PSI, OOT backtest, conservatism test "
            "implemented in validation_suite.run_full_validation_suite()."
        ),
    },
}

# Products in scope
ALL_PRODUCTS = [
    "mortgage", "commercial_cashflow", "receivables", "trade_contingent",
    "asset_equipment", "development_finance", "cre_investment",
    "residual_stock", "land_subdivision", "bridging", "mezz_second_mortgage",
]


# ---------------------------------------------------------------------------
# Main compliance map generator
# ---------------------------------------------------------------------------

def generate_compliance_map(
    calibration_results: dict[str, dict] | None = None,
    moc_registers: dict[str, pd.DataFrame] | None = None,
    regime_data_source: str = "synthetic",
    products: list[str] | None = None,
) -> pd.DataFrame:
    """
    Generate the APS 113 compliance map.

    Parameters
    ----------
    calibration_results : dict of product → calibration output dicts
        (from run_calibration_pipeline())
    moc_registers : dict of product → MoC register DataFrames
    regime_data_source : 'rba_abs_real' or 'synthetic'
    products : list of products to include. Defaults to all 11.

    Returns
    -------
    DataFrame with columns:
        product, section_ref, requirement_description, status,
        evidence_file, evidence_detail, reviewer_note, aps113_section

    Status values: 'met' | 'partial' | 'not_met' | 'not_applicable'
    """
    products = products or ALL_PRODUCTS
    rows = []

    for product in products:
        for req_key, req in APS113_REQUIREMENTS.items():
            status, evidence, reviewer_note = _assess_requirement(
                req_key=req_key,
                product=product,
                calibration_results=calibration_results,
                moc_registers=moc_registers,
                regime_data_source=regime_data_source,
            )
            rows.append({
                "product": product,
                "requirement_key": req_key,
                "section_ref": req["section"],
                "requirement": req["requirement"],
                "description": req["description"],
                "status": status,
                "evidence_file": evidence,
                "reviewer_note": reviewer_note,
            })

    result = pd.DataFrame(rows)

    # Summary counts
    for product in products:
        prod_df = result[result["product"] == product]
        met = (prod_df["status"] == "met").sum()
        partial = (prod_df["status"] == "partial").sum()
        not_met = (prod_df["status"] == "not_met").sum()
        logger.info(
            "APS 113 compliance (%s): met=%d | partial=%d | not_met=%d",
            product, met, partial, not_met,
        )

    return result


def _assess_requirement(
    req_key: str,
    product: str,
    calibration_results: dict | None,
    moc_registers: dict | None,
    regime_data_source: str,
) -> tuple[str, str, str]:
    """Return (status, evidence_file, reviewer_note) for a requirement."""

    cal = calibration_results.get(product, {}) if calibration_results else {}

    if req_key == "s32_lip_costs":
        return (
            "met",
            f"outputs/tables/{product}_historical_workouts.csv",
            "lip_costs column present in all generators. Auto-detection in compute_realised_lgd().",
        )

    if req_key == "s37_ead_definition":
        return (
            "met",
            f"outputs/tables/{product}_historical_workouts.csv",
            "ead_at_default computed at default date in all generators.",
        )

    if req_key == "s43_long_run_lgd":
        has_lr = "long_run_lgd_by_segment" in cal
        return (
            "met" if has_lr else "partial",
            f"outputs/tables/{product}_long_run_lgd_by_segment.csv",
            "vintage_ewa method averages through cycle. 10-year window used."
            if has_lr else "Calibration pipeline not yet run for this product.",
        )

    if req_key == "s44_observation_period":
        return (
            "met",
            f"data/generated/historical/{product}_workouts.parquet",
            "2014-2024 observation window (10 years) exceeds APS 113 Att A minimums.",
        )

    if req_key == "s46_downturn_lgd":
        has_dt = "calibration_steps" in cal
        return (
            "met" if has_dt else "partial",
            f"outputs/tables/{product}_downturn_lgd_by_segment.csv",
            "Downturn overlay applied via product-specific scalar from overlay_parameters.csv.",
        )

    if req_key == "s50_discount_rate":
        status = "met" if regime_data_source == "rba_abs_real" else "partial"
        return (
            status,
            "outputs/tables/rba_discount_rate_register.csv",
            "RBA B6 indicator lending rates used as primary discount rate proxy. "
            "Fallback: RBA cash rate + 300bps. "
            + ("Real RBA data in use." if status == "met" else "Synthetic fallback in use."),
        )

    if req_key == "s52_segmentation":
        return (
            "met",
            f"outputs/tables/{product}_long_run_lgd_by_segment.csv",
            "Product-specific segment keys defined. Segments < 20 obs flagged as low_count.",
        )

    if req_key == "s58_lgd_floors":
        has_floors = "calibration_steps" in cal
        return (
            "met" if has_floors else "partial",
            f"outputs/tables/{product}_final_calibrated_lgd.csv",
            "Policy floors applied in run_calibration_pipeline() step 4. "
            "Floors sourced from overlay_parameters.csv.",
        )

    if req_key == "s60_model_vs_actual":
        return (
            "partial",
            f"outputs/tables/{product}_model_vs_actual_comparison.csv",
            "Comparison is synthetic-data-vs-synthetic-model (no real workout tape). "
            "Directional only. Partial compliance for demonstration repo.",
        )

    if req_key == "s63_moc":
        has_moc = moc_registers and product in moc_registers and not moc_registers[product].empty
        return (
            "met" if has_moc else "partial",
            f"outputs/tables/{product}_moc_register.csv",
            "Correct order: downturn → MoC → floor. MoC applied to downturn LGD per s.63.",
        )

    if req_key == "s65_moc_sources":
        has_moc = moc_registers and product in moc_registers and not moc_registers[product].empty
        return (
            "met" if has_moc else "partial",
            f"outputs/tables/{product}_moc_register.csv",
            "All 5 APS 113 s.65 sources evaluated with individual bps add-ons.",
        )

    if req_key == "s66_validation":
        return (
            "partial",
            f"outputs/tables/{product}_backtest_results.csv",
            "Gini, HL, PSI, OOT, conservatism implemented in validation_suite.py. "
            "Partial (not 'met') because workout data is synthetic — not independent.",
        )

    return "not_applicable", "", ""


def validate_observation_periods(
    loans: pd.DataFrame,
    product: str,
    date_col: str = "default_date",
) -> dict:
    """
    Assert observation period meets APS 113 Attachment A minimums.

    Returns
    -------
    dict with:
        product, n_years, minimum_required, compliant,
        oldest_default, newest_default, aps113_section
    """
    from src.generators.base_generator import OBSERVATION_PERIODS_BY_PRODUCT

    dates = pd.to_datetime(loans[date_col], errors="coerce").dropna()
    if dates.empty:
        return {
            "product": product, "n_years": 0, "compliant": False,
            "aps113_section": "s.44 / Att A",
        }

    oldest = dates.min()
    newest = dates.max()
    n_years = (newest - oldest).days / 365.25

    min_req = OBSERVATION_PERIODS_BY_PRODUCT.get(product, 5)
    compliant = n_years >= min_req

    if not compliant:
        logger.error(
            "APS 113 Att A: %s observation period %.1f years < minimum %d years.",
            product, n_years, min_req,
        )

    return {
        "product": product,
        "n_years": round(float(n_years), 1),
        "minimum_required": min_req,
        "compliant": compliant,
        "oldest_default": str(oldest.date()),
        "newest_default": str(newest.date()),
        "aps113_section": "s.44 / Att A",
    }


def export_compliance_map(
    compliance_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Write compliance map to CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    compliance_df.to_csv(output_path, index=False)
    logger.info(
        "Exported APS 113 compliance map (%d rows) to %s",
        len(compliance_df), output_path,
    )

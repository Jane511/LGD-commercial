"""
RBA Indicator Lending Rates Loader — per-vintage discount rate proxy.

Replaces the 'RBA cash rate + 300bps' fallback with real product-specific
lending rates from RBA Table B6 (Lending Rates), per APS 113 s.50 which
requires using the contractual interest rate on the defaulted exposure.

Source: Reserve Bank of Australia, Table B6
URL: https://www.rba.gov.au/statistics/tables/

Usage:
    from src.rba_rates_loader import load_rba_lending_rates, get_discount_rate_for_loan

    rates_df = load_rba_lending_rates()
    rate, source = get_discount_rate_for_loan("mortgage_owner_occupier", 2019, rates_df)
    # rate = 0.048, source = 'rba_b6'

APS 113 Reference: s.50 — discount rate is contractual rate on defaulted exposure.
"""
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Product → RBA B6 column mapping
# ---------------------------------------------------------------------------

# Maps internal product_type strings to RBA B6 rate column names.
# Rate column names correspond to columns in data/external/rba_b6_rates.csv.
RBA_RATE_MAP: dict[str, str] = {
    # Residential mortgage
    "mortgage":                          "mortgage_owner_occupier_variable",
    "mortgage_owner_occupier":           "mortgage_owner_occupier_variable",
    "mortgage_investor":                 "mortgage_investor_variable",
    "mortgage_fixed":                    "mortgage_fixed_3yr",
    "bridging":                          "mortgage_owner_occupier_variable",  # secured on residential
    "mezz_second_mortgage":              "mortgage_investor_variable",         # typically investor-style
    # Commercial
    "commercial_cashflow":               "small_business_variable",
    "receivables":                       "small_business_variable",
    "trade_contingent":                  "small_business_variable",
    "asset_equipment":                   "small_business_variable",
    # Property-backed commercial
    "development_finance":               "large_business_variable",
    "cre_investment":                    "large_business_variable",
    "residual_stock":                    "large_business_variable",
    "land_subdivision":                  "large_business_variable",
    # Generic fallback
    "personal":                          "personal_variable",
}

DEFAULT_RATES_PATH = Path(__file__).parent.parent / "data" / "external" / "rba_b6_rates.csv"
FALLBACK_SPREAD_BPS = 300   # RBA cash rate + 300bps when no B6 rate available


def load_rba_lending_rates(
    rates_path: str | Path | None = None,
    vintage_year_range: tuple[int, int] = (2014, 2024),
) -> pd.DataFrame:
    """
    Load RBA B6 indicator lending rates from CSV.

    Parameters
    ----------
    rates_path : path to rba_b6_rates.csv. Defaults to data/external/rba_b6_rates.csv.
    vintage_year_range : (start_year, end_year) inclusive. Records outside this range
        are dropped.

    Returns
    -------
    DataFrame with columns:
        year (int), product_type (str), rate_decimal (float), rate_source (str)

    The returned frame has one row per (year, product_type) combination.
    rate_source is 'rba_b6' for real data and 'rba_cash_plus_300bps' for fallbacks.

    APS 113 s.50: Use contractual rate on defaulted exposure. This function provides
    the best available proxy when the actual contractual rate is not in the workout record.
    """
    path = Path(rates_path) if rates_path else DEFAULT_RATES_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"RBA B6 rates file not found at {path}. "
            "Expected: data/external/rba_b6_rates.csv. "
            "Download from https://www.rba.gov.au/statistics/tables/ (Table B6)."
        )

    raw = pd.read_csv(path, comment="#")
    raw["year"] = pd.to_numeric(raw["year"], errors="coerce").astype("Int64")

    start_yr, end_yr = vintage_year_range
    raw = raw[(raw["year"] >= start_yr) & (raw["year"] <= end_yr)].copy()

    if raw.empty:
        raise ValueError(
            f"No RBA rate data found for year range {vintage_year_range}. "
            f"Check {path}."
        )

    # Melt from wide to long: one row per (year, product_type)
    rate_columns = [
        "mortgage_owner_occupier_variable",
        "mortgage_investor_variable",
        "mortgage_fixed_3yr",
        "small_business_variable",
        "large_business_variable",
        "personal_variable",
        "cash_plus_300bps",
    ]
    available_cols = [c for c in rate_columns if c in raw.columns]
    melted = raw.melt(
        id_vars=["year"],
        value_vars=available_cols,
        var_name="rate_column",
        value_name="rate_pct",
    )
    melted["rate_decimal"] = pd.to_numeric(melted["rate_pct"], errors="coerce") / 100.0
    melted["rate_source"] = melted["rate_column"].apply(
        lambda c: "rba_b6" if c != "cash_plus_300bps" else "rba_cash_plus_300bps"
    )

    logger.info(
        "Loaded RBA B6 lending rates: %d years × %d rate series from %s",
        raw["year"].nunique(),
        len(available_cols),
        path.name,
    )
    return melted[["year", "rate_column", "rate_decimal", "rate_source"]].reset_index(drop=True)


def get_discount_rate_for_loan(
    product_type: str,
    vintage_year: int,
    rates_df: pd.DataFrame,
    fallback_spread_bps: int = FALLBACK_SPREAD_BPS,
) -> tuple[float, str]:
    """
    Return (rate_decimal, source_label) for a given product type and vintage year.

    Priority:
    1. RBA B6 rate matching product_type → vintage_year
    2. Cash rate + fallback_spread_bps for the same year
    3. Global default: 0.07 (7.0%) with source 'hardcoded_fallback'

    Parameters
    ----------
    product_type : one of the keys in RBA_RATE_MAP (e.g., 'mortgage', 'commercial_cashflow')
    vintage_year : origination year (int)
    rates_df : output of load_rba_lending_rates()
    fallback_spread_bps : basis points to add to cash rate when B6 rate unavailable

    Returns
    -------
    (rate_decimal, source_label)

    APS 113 s.50: Fallback to RBA cash rate + 300bps is explicitly mentioned as an
    acceptable fallback when the contractual rate is not available.
    """
    rate_col = RBA_RATE_MAP.get(product_type)

    if rate_col is not None:
        mask = (rates_df["rate_column"] == rate_col) & (rates_df["year"] == vintage_year)
        matched = rates_df[mask]
        if not matched.empty:
            rate = float(matched.iloc[0]["rate_decimal"])
            if np.isfinite(rate) and rate > 0:
                return rate, "rba_b6"

    # Try cash+spread fallback for that year
    cash_mask = (rates_df["rate_column"] == "cash_plus_300bps") & (rates_df["year"] == vintage_year)
    cash_matched = rates_df[cash_mask]
    if not cash_matched.empty:
        rate = float(cash_matched.iloc[0]["rate_decimal"])
        if np.isfinite(rate) and rate > 0:
            logger.debug(
                "Using RBA cash+%dbps fallback for product=%s year=%d: rate=%.4f",
                fallback_spread_bps, product_type, vintage_year, rate,
            )
            return rate, "rba_cash_plus_300bps"

    # Hard fallback
    logger.warning(
        "No RBA rate found for product=%s year=%d. Using hardcoded 7.0%% fallback.",
        product_type, vintage_year,
    )
    return 0.07, "hardcoded_fallback"


def build_discount_rate_register(
    loans: pd.DataFrame,
    rates_df: pd.DataFrame,
    product_type: str,
    origination_year_col: str = "origination_year",
    occupancy_col: str | None = "occupancy_type",
) -> pd.DataFrame:
    """
    Build a per-loan discount rate register for audit/export.

    Applies get_discount_rate_for_loan() to each row, resolving mortgage
    sub-type (owner-occupier vs investor) when occupancy_col is available.

    Returns
    -------
    DataFrame with original index plus columns:
        discount_rate, discount_rate_source

    Export to outputs/tables/rba_discount_rate_register.csv.
    """
    rates = []
    sources = []

    for _, row in loans.iterrows():
        year = int(row.get(origination_year_col, 2019))

        # Resolve mortgage sub-type for more accurate rate
        effective_product = product_type
        if product_type == "mortgage" and occupancy_col and occupancy_col in row:
            occ = str(row[occupancy_col]).lower()
            if "invest" in occ:
                effective_product = "mortgage_investor"
            else:
                effective_product = "mortgage_owner_occupier"

        rate, source = get_discount_rate_for_loan(effective_product, year, rates_df)
        rates.append(rate)
        sources.append(source)

    result = loans.copy()
    result["discount_rate"] = rates
    result["discount_rate_source"] = sources
    return result


def export_discount_rate_register(
    loans: pd.DataFrame,
    rates_df: pd.DataFrame,
    product_type: str,
    output_path: str | Path,
    origination_year_col: str = "origination_year",
) -> pd.DataFrame:
    """
    Build and write the discount rate register to CSV.

    Per-loan audit trail required for APS 113 s.50 documentation.
    """
    register = build_discount_rate_register(loans, rates_df, product_type, origination_year_col)
    cols = ["loan_id", "discount_rate", "discount_rate_source"] if "loan_id" in register.columns else ["discount_rate", "discount_rate_source"]
    out = register[cols].copy()
    out["product_type"] = product_type
    out["aps113_section"] = "s.50"
    out["note"] = "Discount rate for workout LGD — contractual rate proxy per APS 113 s.50"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    logger.info("Exported discount rate register (%d rows) to %s", len(out), output_path)
    return out

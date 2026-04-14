"""
APRA ADI Performance Statistics — Peer Benchmarking.

Uses publicly available APRA ADI Performance Statistics as a cross-check
benchmark for calibrated LGD estimates.

IMPORTANT LIMITATION:
    APRA publishes system-wide impairment ratios (specific provisions / gross
    impaired assets), which are NOT the same as realised LGD. They represent
    expected provisioning behaviour, not confirmed loss rates.

    These are used as DIRECTIONAL benchmarks only. A calibrated LGD of 30%
    being compared to an APRA impairment ratio of 28% is evidence of broad
    consistency, not a precise validation.

Source:
    Australian Prudential Regulation Authority
    Quarterly ADI Performance Statistics
    https://www.apra.gov.au/quarterly-authorised-deposit-taking-institution-statistics

APS 113 Reference:
    Not directly cited — this is a governance/peer benchmarking addition.
    Consistent with APRA's model risk guidance on benchmarking.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_APRA_PATH = Path(__file__).parent.parent / "data" / "external" / "apra_adi_statistics.csv"

# Map from internal product names to APRA data columns
APRA_PRODUCT_MAP: dict[str, str] = {
    "mortgage":              "housing_loans_impairment_ratio",
    "commercial_cashflow":   "business_loans_impairment_ratio",
    "receivables":           "business_loans_impairment_ratio",
    "trade_contingent":      "business_loans_impairment_ratio",
    "asset_equipment":       "business_loans_impairment_ratio",
    "development_finance":   "construction_loans_impairment_ratio",
    "cre_investment":        "construction_loans_impairment_ratio",
    "residual_stock":        "construction_loans_impairment_ratio",
    "land_subdivision":      "construction_loans_impairment_ratio",
    "bridging":              "construction_loans_impairment_ratio",
    "mezz_second_mortgage":  "business_loans_impairment_ratio",
}


def load_apra_adi_benchmarks(
    apra_data_path: str | Path | None = None,
    product_types: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load and process APRA ADI Performance Statistics.

    Parameters
    ----------
    apra_data_path : path to apra_adi_statistics.csv. Defaults to
        data/external/apra_adi_statistics.csv.
    product_types : list of product names to include. None = all.

    Returns
    -------
    DataFrame with columns:
        year, product_type, apra_system_impairment_ratio,
        apra_implied_lgd_proxy, percentile_25, percentile_75,
        data_vintage, data_source, benchmark_note
    """
    path = Path(apra_data_path) if apra_data_path else DEFAULT_APRA_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"APRA ADI statistics file not found at {path}. "
            "Expected: data/external/apra_adi_statistics.csv. "
            "Download from: https://www.apra.gov.au/quarterly-authorised-deposit-taking-institution-statistics"
        )

    raw = pd.read_csv(path, comment="#")
    raw["year"] = pd.to_numeric(raw["year"], errors="coerce").astype("Int64")

    products = product_types or list(APRA_PRODUCT_MAP.keys())
    rows = []
    for product in products:
        col = APRA_PRODUCT_MAP.get(product)
        if col not in raw.columns:
            logger.warning("APRA column '%s' not found for product '%s'. Skipping.", col, product)
            continue

        # Annual average from quarterly data
        annual = (
            raw.groupby("year")[col]
            .agg(
                apra_system_impairment_ratio="mean",
                percentile_25=lambda x: np.nanpercentile(x, 25),
                percentile_75=lambda x: np.nanpercentile(x, 75),
            )
            .reset_index()
        )
        annual["product_type"] = product
        # LGD proxy: impairment ratio is a rough LGD proxy but typically
        # understates actual LGD (provisions are forward-looking estimates)
        annual["apra_implied_lgd_proxy"] = annual["apra_system_impairment_ratio"]
        annual["data_source"] = "apra_adi_statistics"
        annual["data_vintage"] = "2014-2024"
        annual["benchmark_note"] = (
            "APRA system impairment ratio = specific provisions / gross impaired assets. "
            "Directional benchmark only — NOT a direct LGD equivalent. "
            "Calibrated LGD should generally exceed this proxy (provisioning ≠ realised loss)."
        )
        rows.append(annual)

    if not rows:
        return pd.DataFrame()

    result = pd.concat(rows, ignore_index=True)
    logger.info(
        "Loaded APRA ADI benchmarks: %d product-years across %d products.",
        len(result), result["product_type"].nunique(),
    )
    return result


def generate_benchmark_comparison(
    calibrated_lgd: pd.DataFrame,
    apra_benchmarks: pd.DataFrame,
    product_col: str = "product",
    lgd_col: str = "final_lgd",
    year_col: str = "year",
) -> pd.DataFrame:
    """
    Compare calibrated long-run LGD against APRA system benchmark.

    Parameters
    ----------
    calibrated_lgd : DataFrame with calibrated LGD per product (+ year if available)
    apra_benchmarks : output of load_apra_adi_benchmarks()
    lgd_col : column with calibrated LGD values

    Returns
    -------
    DataFrame with columns:
        product_type, calibrated_lgd_ewa, apra_benchmark_proxy,
        vs_benchmark_diff, vs_benchmark_pct,
        benchmark_adequacy_flag, benchmark_note, aps113_note

    benchmark_adequacy_flag values:
        'consistent'  — calibrated LGD within +/- 30% of APRA proxy
        'conservative' — calibrated LGD > 130% of APRA proxy (possibly over-cautious)
        'below_benchmark' — calibrated LGD < 70% of APRA proxy (requires investigation)
    """
    # Portfolio-level average if no year dimension
    if lgd_col not in calibrated_lgd.columns:
        raise ValueError(f"calibrated_lgd missing column '{lgd_col}'.")

    if product_col not in calibrated_lgd.columns:
        raise ValueError(f"calibrated_lgd missing column '{product_col}'.")

    apra_avg = apra_benchmarks.groupby("product_type")["apra_system_impairment_ratio"].mean()
    cal_avg = calibrated_lgd.groupby(product_col)[lgd_col].mean()

    rows = []
    for product in cal_avg.index:
        cal = float(cal_avg[product])
        benchmark = float(apra_avg.get(product, np.nan))

        if np.isnan(benchmark):
            flag = "no_benchmark_available"
            diff = np.nan
            diff_pct = np.nan
        else:
            diff = cal - benchmark
            diff_pct = diff / benchmark if benchmark > 0 else np.nan
            if diff_pct < -0.30:
                flag = "below_benchmark"
            elif diff_pct > 0.30:
                flag = "conservative"
            else:
                flag = "consistent"

        rows.append({
            "product_type": product,
            "calibrated_lgd_ewa": round(cal, 6),
            "apra_benchmark_proxy": round(benchmark, 6) if np.isfinite(benchmark) else np.nan,
            "vs_benchmark_diff": round(diff, 6) if np.isfinite(diff) else np.nan,
            "vs_benchmark_pct": round(diff_pct, 4) if np.isfinite(diff_pct) else np.nan,
            "benchmark_adequacy_flag": flag,
            "benchmark_note": (
                "APRA impairment ratio is a provisioning proxy, not realised LGD. "
                "Calibrated LGD should exceed APRA proxy in most cases."
            ),
            "aps113_note": "Peer benchmarking — governance/model risk guidance.",
        })

    result = pd.DataFrame(rows)
    n_below = (result["benchmark_adequacy_flag"] == "below_benchmark").sum()
    if n_below > 0:
        logger.warning(
            "generate_benchmark_comparison: %d product(s) below APRA benchmark. "
            "Consider reviewing calibration.",
            n_below,
        )
    return result


def export_benchmark_comparison(
    comparison_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Write benchmark comparison to CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_path, index=False)
    logger.info("Exported APRA benchmark comparison (%d rows) to %s", len(comparison_df), output_path)

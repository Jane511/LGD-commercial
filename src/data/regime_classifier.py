"""
Economic Cycle Regime Classifier.

Identifies downturn periods for use in APS 113 s.46–50 downturn LGD estimation.

Priority (upstream_first mode):
  1. Load from industry-analysis repo's macro_regime_flags.parquet (real RBA/ABS data)
  2. Compute from a supplied macro DataFrame
  3. Use built-in synthetic macro series (2014-2024 proxy based on known history)

The classifier tags workout records with macro_regime and downturn_flag, which
feed into:
  - compute_long_run_lgd() (identifies downturn vs non-downturn vintages)
  - MoCRegister.cyclicality_moc (checks whether sample includes a downturn)
  - apply_downturn_overlay() (calibrates the uplift magnitude)

APS 113 References:
  - s.43: Long-run average across full economic cycle
  - s.46-50: Downturn LGD >= long-run LGD; must reflect downturn conditions
  - s.55-57: Downturn definition and calibration basis

SYNTHETIC DATA DISCLAIMER:
When operating in synthetic mode (no upstream parquet), regime data is proxy-based.
All outputs include a data_source column to identify the provenance.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in synthetic macro series 2014-2024
# Calibrated to known Australian economic history (RBA/ABS public data)
# ---------------------------------------------------------------------------

_SYNTHETIC_MACRO_SERIES = pd.DataFrame({
    "year": list(range(2014, 2025)),
    "gdp_growth_yoy": [2.6, 2.3, 2.8, 2.4, 2.7, 1.9, -2.2, 5.0, 3.6, 2.0, 1.8],
    "unemployment_rate": [0.061, 0.062, 0.057, 0.056, 0.053, 0.052, 0.069, 0.052, 0.036, 0.038, 0.040],
    "credit_spread_bps": [130, 145, 140, 125, 118, 130, 290, 175, 225, 240, 200],
    "rba_cash_rate": [0.025, 0.020, 0.017, 0.015, 0.015, 0.010, 0.003, 0.001, 0.019, 0.041, 0.043],
    "house_price_growth_yoy": [0.085, 0.072, 0.068, 0.110, 0.085, 0.035, -0.023, 0.220, -0.048, -0.019, 0.042],
    "regime": [
        "expansion", "mild_stress", "mild_stress", "expansion", "expansion",
        "expansion", "severe_stress", "severe_stress", "mild_stress", "mild_stress", "expansion",
    ],
    "is_downturn_period": [False, False, False, False, False, False, True, True, False, False, False],
    "data_source": ["synthetic"] * 11,
})

# Default path for upstream macro_regime_flags.parquet from industry-analysis repo
_DEFAULT_UPSTREAM_PATH = Path(__file__).parent.parent / "data" / "exports" / "macro_regime_flags.parquet"


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

def classify_economic_regime(
    macro_df: pd.DataFrame | None = None,
    upstream_parquet_path: str | Path | None = None,
    date_col: str = "year",
    method: str = "upstream_first",
) -> pd.DataFrame:
    """
    Classify each year as expansion / mild_stress / severe_stress.

    Parameters
    ----------
    macro_df : optional DataFrame with macro time series. Used when
        method='scoring' or as fallback when upstream not found.
    upstream_parquet_path : path to macro_regime_flags.parquet from the
        industry-analysis repo. When present, takes highest priority.
        Defaults to data/exports/macro_regime_flags.parquet.
    date_col : column name for year/date in macro_df (default: 'year')
    method : 'upstream_first' | 'scoring' | 'synthetic'
        - upstream_first: try upstream parquet, then macro_df, then synthetic
        - scoring: use macro_df with multi-indicator scoring (requires macro_df)
        - synthetic: always use built-in synthetic series

    Returns
    -------
    DataFrame with columns:
        year (int), regime (str), is_downturn_period (bool),
        gdp_growth_yoy, unemployment_rate, credit_spread_bps,
        data_source ('rba_abs_real' | 'synthetic')

    APS 113 s.43: classifier output is used to identify through-the-cycle
    vintages and ensure at least one downturn period is represented.
    """
    if method == "upstream_first":
        return _classify_upstream_first(upstream_parquet_path, macro_df)
    elif method == "scoring":
        if macro_df is None:
            raise ValueError("method='scoring' requires macro_df to be provided.")
        return _classify_from_macro_df(macro_df, date_col)
    elif method == "synthetic":
        return _get_synthetic_regimes()
    else:
        raise ValueError(f"Unknown method '{method}'. Choose: upstream_first, scoring, synthetic.")


def _classify_upstream_first(
    upstream_path: str | Path | None,
    macro_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Try upstream parquet → macro_df → synthetic, in order."""
    # 1. Try upstream parquet
    path = Path(upstream_path) if upstream_path else _DEFAULT_UPSTREAM_PATH
    if path.exists():
        try:
            regimes = _load_upstream_parquet(path)
            logger.info(
                "Loaded economic regimes from upstream parquet (real RBA/ABS data): %s",
                path.name,
            )
            return regimes
        except Exception as exc:
            logger.warning("Failed to load upstream regime parquet (%s): %s. Falling back.", path, exc)

    if not path.exists():
        logger.info(
            "Upstream macro_regime_flags.parquet not found at %s. "
            "Run the industry-analysis repo first to produce real RBA/ABS regimes. "
            "Falling back to synthetic macro series.",
            path,
        )

    # 2. Try macro_df
    if macro_df is not None:
        try:
            return _classify_from_macro_df(macro_df, date_col="year")
        except Exception as exc:
            logger.warning("Failed to classify from macro_df: %s. Falling back to synthetic.", exc)

    # 3. Synthetic fallback
    logger.info("Using built-in synthetic macro regime series (2014-2024).")
    return _get_synthetic_regimes()


def _load_upstream_parquet(path: Path) -> pd.DataFrame:
    """
    Load and standardise macro_regime_flags.parquet from the industry-analysis repo.

    The upstream parquet may have various column names depending on the
    industry-analysis repo version. This function normalises to the standard
    output contract.
    """
    raw = pd.read_parquet(path)

    # Find the year/date column
    year_col = _find_column(raw, ["year", "date", "period", "year_end"])
    if year_col is None:
        raise ValueError(f"Cannot find year column in {path}. Columns: {list(raw.columns)}")

    # Extract year from date if needed
    if raw[year_col].dtype == "object" or pd.api.types.is_datetime64_any_dtype(raw[year_col]):
        raw["year"] = pd.to_datetime(raw[year_col]).dt.year
    else:
        raw["year"] = pd.to_numeric(raw[year_col], errors="coerce").astype("Int64")

    # Find regime column
    regime_col = _find_column(raw, ["macro_regime", "regime", "scenario_name", "economic_regime"])
    downturn_col = _find_column(raw, ["is_downturn", "downturn_flag", "regime_flag", "stress_regime_flag"])

    # Build standard output
    result = pd.DataFrame()
    result["year"] = raw["year"]

    if regime_col:
        result["regime"] = raw[regime_col].astype(str)
    else:
        # Infer from downturn flag
        if downturn_col:
            result["regime"] = np.where(raw[downturn_col].astype(bool), "severe_stress", "expansion")
        else:
            raise ValueError(f"Cannot find regime column in {path}. Columns: {list(raw.columns)}")

    if downturn_col:
        result["is_downturn_period"] = raw[downturn_col].astype(bool)
    else:
        result["is_downturn_period"] = result["regime"] == "severe_stress"

    # Optional macro factor columns
    for target, candidates in [
        ("gdp_growth_yoy", ["gdp_growth_yoy", "gdp_growth", "real_gdp_growth"]),
        ("unemployment_rate", ["unemployment_rate", "unemployment", "unemp_rate"]),
        ("credit_spread_bps", ["credit_spread_bps", "credit_spread", "spread_bps"]),
    ]:
        src = _find_column(raw, candidates)
        result[target] = pd.to_numeric(raw[src], errors="coerce") if src else np.nan

    result["data_source"] = "rba_abs_real"
    return result.sort_values("year").reset_index(drop=True)


def _classify_from_macro_df(macro_df: pd.DataFrame, date_col: str = "year") -> pd.DataFrame:
    """
    Classify regimes from a supplied macro DataFrame using multi-indicator scoring.

    Scoring method:
      - GDP < 0.5%: severe stress signal
      - Unemployment > 7%: severe stress signal
      - Credit spread > 300bps: severe stress signal
      If 2+ severe signals: severe_stress
      If 1 severe signal or GDP < 1.5%: mild_stress
      Else: expansion
    """
    df = macro_df.copy()
    if date_col != "year":
        df["year"] = pd.to_datetime(df[date_col]).dt.year

    gdp = pd.to_numeric(df.get("gdp_growth_yoy", np.nan), errors="coerce")
    unemp = pd.to_numeric(df.get("unemployment_rate", np.nan), errors="coerce")
    spread = pd.to_numeric(df.get("credit_spread_bps", np.nan), errors="coerce")

    severe_signals = (
        (gdp < 0.005).astype(int).fillna(0)
        + (unemp > 0.07).astype(int).fillna(0)
        + (spread > 300).astype(int).fillna(0)
    )
    mild_signals = (
        ((gdp >= 0.005) & (gdp < 0.015)).astype(int).fillna(0)
        + ((unemp >= 0.055) & (unemp <= 0.07)).astype(int).fillna(0)
    )

    regimes = []
    for s, m in zip(severe_signals, mild_signals):
        if s >= 2:
            regimes.append("severe_stress")
        elif s >= 1 or m >= 1:
            regimes.append("mild_stress")
        else:
            regimes.append("expansion")

    result = df[["year"]].copy()
    result["regime"] = regimes
    result["is_downturn_period"] = [r == "severe_stress" for r in regimes]
    result["gdp_growth_yoy"] = gdp.values if hasattr(gdp, "values") else gdp
    result["unemployment_rate"] = unemp.values if hasattr(unemp, "values") else unemp
    result["credit_spread_bps"] = spread.values if hasattr(spread, "values") else spread
    result["data_source"] = "supplied_macro_df"
    return result.sort_values("year").reset_index(drop=True)


def _get_synthetic_regimes() -> pd.DataFrame:
    """Return the built-in synthetic macro series."""
    return _SYNTHETIC_MACRO_SERIES.copy()


# ---------------------------------------------------------------------------
# Helper: assign regimes to workout records
# ---------------------------------------------------------------------------

def assign_regime_to_workouts(
    loans: pd.DataFrame,
    regimes: pd.DataFrame,
    default_date_col: str = "default_date",
    default_year_col: str = "default_year",
) -> pd.DataFrame:
    """
    Join economic regime classification to loan-level workout data.

    Adds or overwrites:
        macro_regime (str): expansion / mild_stress / severe_stress
        downturn_flag (int): 1 if severe_stress

    Parameters
    ----------
    loans : loan-level workout DataFrame (from any product generator)
    regimes : output of classify_economic_regime()
    default_date_col : column in loans containing the default date
    default_year_col : column in loans containing the default year (int)

    Returns
    -------
    loans with macro_regime and downturn_flag columns updated.
    """
    df = loans.copy()

    # Ensure default_year column exists
    if default_year_col not in df.columns:
        if default_date_col in df.columns:
            df[default_year_col] = pd.to_datetime(df[default_date_col]).dt.year
        else:
            logger.warning(
                "Neither %s nor %s found in loans. Regime assignment skipped.",
                default_year_col, default_date_col,
            )
            return df

    regime_map = regimes.set_index("year")["regime"].to_dict()
    downturn_map = regimes.set_index("year")["is_downturn_period"].to_dict()

    df["macro_regime"] = df[default_year_col].map(regime_map).fillna("expansion")
    df["downturn_flag"] = df[default_year_col].map(downturn_map).fillna(False).astype(int)
    df["regime_data_source"] = regimes["data_source"].iloc[0] if len(regimes) > 0 else "unknown"

    n_downturn = df["downturn_flag"].sum()
    logger.info(
        "Assigned regimes to %d loans: %d in downturn periods (%.1f%%)",
        len(df), n_downturn, 100 * n_downturn / len(df) if len(df) > 0 else 0,
    )
    return df


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def export_regime_classification(
    regimes: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Write the regime classification table to CSV.

    Output: outputs/tables/economic_regime_classification.csv
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out = regimes.copy()
    out["aps113_section"] = "s.43, s.46-50"
    out["note"] = (
        "Economic cycle regime classification used for downturn LGD calibration. "
        "severe_stress years are treated as downturn periods per APS 113 s.46."
    )
    out.to_csv(output_path, index=False)
    logger.info("Exported regime classification to %s", output_path)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return first candidate column name found in df.columns, or None."""
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    return None

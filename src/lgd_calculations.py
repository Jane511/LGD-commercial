"""
Canonical workout-based LGD calculation engine.

This module implements the workout LGD method required under APRA APS 113
for non-traded exposures. It supplements (does not replace) src/lgd_calculation.py
which contains the existing proxy LGD engines.

Key function: compute_realised_lgd()

APS 113 References:
  - s.32:  Loss Identification Period (LIP) — costs in window between actual default
           and formal identification are included in the LGD numerator
  - s.43:  Long-run average LGD through a complete economic cycle
  - s.44:  Minimum observation periods (7y mortgage, 5y others)
  - s.46:  Downturn LGD must be >= long-run LGD
  - s.49-51: Workout method — discount cashflows at contractual rate
  - s.52:  Segmentation requirements
  - s.54:  Exposure-weighted aggregation

NAMING NOTE:
  This file is lgd_calculations.py (with 's').
  The existing proxy engine is lgd_calculation.py (without 's').
  Both can coexist: import lgd_calculation for proxy engines,
  import lgd_calculations for calibration functions.

REUSE:
  exposure_weighted_average() is imported from lgd_calculation.py (not re-implemented).
  apply_downturn_overlay() is imported from lgd_calculation.py (not re-implemented).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse existing utilities from the proxy engine — do not re-implement
from src.lgd_calculation import (
    apply_downturn_overlay,
    exposure_weighted_average,
    build_weighted_lgd_output,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LGD_FLOOR_ABSOLUTE = 0.0      # cures floor at 0 per APS 113 s.32
LGD_CAP_ABSOLUTE = 1.50       # cap for extreme excess-cost cases
DEFAULT_LIP_WINDOW_DAYS = 90   # typical AU bank LIP identification window


# ---------------------------------------------------------------------------
# Core function: compute_realised_lgd
# ---------------------------------------------------------------------------

def compute_realised_lgd(
    loans: pd.DataFrame,
    cashflows: pd.DataFrame,
    *,
    ead_col: str = "ead_at_default",
    recovery_col: str = "recovery_amount",
    cost_col: str = "direct_costs",
    indirect_cost_col: str = "indirect_costs",
    lip_costs: float | pd.Series | None = None,
    discount_rate_col: str = "discount_rate",
    cashflow_date_col: str = "cashflow_date",
    default_date_col: str = "default_date",
    is_cured_col: str = "is_cured",
    cure_recovery_col: str | None = "cure_recovery_amount",
    lip_window_days: int = DEFAULT_LIP_WINDOW_DAYS,
    include_accrued_interest: bool = False,
    accrued_interest_col: str = "accrued_interest_at_default",
) -> pd.DataFrame:
    """
    Compute workout-based realised LGD per APS 113 s.32-34 and s.49-51.

    Two workout paths are handled:

    Non-cured path (is_cured=False):
        LGD = 1 - (PV_recoveries - PV_direct_costs - PV_indirect_costs - LIP_costs) / EAD

    Cured path (is_cured=True):
        LGD = 1 - PV(cure_recovery) / EAD
        (near-zero; adjusted for arrears/admin costs in the cure period)

    PV formula (per APS 113 s.50):
        PV(CF at day t) = CF / (1 + discount_rate)^(t/365)

    Loss Identification Period (APS 113 s.32):
        Costs in [default_date, default_date + lip_window_days] are classified
        as LIP costs and treated as a first charge against recovery (not discounted
        separately — they reduce net recovery at time 0 effectively).

    Parameters
    ----------
    loans : DataFrame, one row per defaulted facility.
        Required columns: loan_id, ead_col, default_date_col, discount_rate_col, is_cured_col.

    cashflows : DataFrame, one row per cashflow event.
        Required columns: loan_id, cashflow_date_col, recovery_col, cost_col, indirect_cost_col.

    lip_costs : scalar, Series aligned to loans.index, or None.
        - None: auto-detected from cashflows within lip_window_days of default_date.
        - scalar: applied uniformly.
        - Series: must align to loans.index.

    lip_window_days : number of days after default_date in which costs are LIP.
        Typical Australian bank practice is 60-90 days. Default: 90.

    include_accrued_interest : if True, add accrued_interest_at_default to EAD denominator.
        Some APRA institutions capitalise accrued interest into EAD per APS 113 s.37.

    Returns
    -------
    DataFrame (same index as loans) with original columns plus:
        realised_lgd          (float, clipped to [0.0, 1.5])
        pv_recoveries         (float)
        pv_direct_costs       (float)
        pv_indirect_costs     (float)
        lip_cost_applied      (float)
        ead_denominator       (float)
        net_recovery_pv       (float)
        workout_lgd_source    (str: 'cured' | 'workout' | 'partial_workout')
        lgd_computation_note  (str: audit trail)
    """
    _validate_inputs(loans, cashflows, ead_col, default_date_col, discount_rate_col, is_cured_col)

    # Build cashflow index: cashflows keyed by loan_id
    cf = cashflows.copy()
    cf[cashflow_date_col] = pd.to_datetime(cf[cashflow_date_col])
    cf_by_loan = cf.groupby("loan_id")

    results = []
    for _, loan in loans.iterrows():
        row = _compute_single_loan_lgd(
            loan=loan,
            cf_by_loan=cf_by_loan,
            ead_col=ead_col,
            recovery_col=recovery_col,
            cost_col=cost_col,
            indirect_cost_col=indirect_cost_col,
            lip_costs=lip_costs,
            discount_rate_col=discount_rate_col,
            cashflow_date_col=cashflow_date_col,
            default_date_col=default_date_col,
            is_cured_col=is_cured_col,
            cure_recovery_col=cure_recovery_col,
            lip_window_days=lip_window_days,
            include_accrued_interest=include_accrued_interest,
            accrued_interest_col=accrued_interest_col,
        )
        results.append(row)

    result_df = pd.DataFrame(results, index=loans.index)
    # Merge calibration columns back onto loans
    output = loans.copy()
    for col in ["realised_lgd", "pv_recoveries", "pv_direct_costs", "pv_indirect_costs",
                "lip_cost_applied", "ead_denominator", "net_recovery_pv",
                "workout_lgd_source", "lgd_computation_note"]:
        output[col] = result_df[col].values

    n_cured = (output["workout_lgd_source"] == "cured").sum()
    n_workout = (output["workout_lgd_source"] == "workout").sum()
    logger.info(
        "compute_realised_lgd: %d loans | cured=%d (%.0f%%) | workout=%d (%.0f%%) | "
        "mean LGD=%.3f",
        len(output), n_cured, 100 * n_cured / len(output),
        n_workout, 100 * n_workout / len(output),
        output["realised_lgd"].mean(),
    )
    return output


def _compute_single_loan_lgd(
    loan: pd.Series,
    cf_by_loan,
    ead_col, recovery_col, cost_col, indirect_cost_col,
    lip_costs, discount_rate_col, cashflow_date_col, default_date_col,
    is_cured_col, cure_recovery_col, lip_window_days,
    include_accrued_interest, accrued_interest_col,
) -> dict:
    """Compute realised LGD for a single loan row."""
    loan_id = loan["loan_id"]
    ead = float(loan[ead_col])
    if ead <= 0:
        raise ValueError(f"loan_id={loan_id}: EAD must be > 0, got {ead}")

    # EAD denominator
    ead_denom = ead
    if include_accrued_interest and accrued_interest_col in loan.index:
        acc = float(loan.get(accrued_interest_col, 0) or 0)
        ead_denom = ead + acc

    default_dt = pd.Timestamp(loan[default_date_col])
    discount_rate = float(loan.get(discount_rate_col, 0.07))
    if discount_rate <= 0:
        discount_rate = 0.07
        logger.debug("loan_id=%s: invalid discount_rate, using 7%% fallback.", loan_id)

    # Check cure flag
    is_cured = bool(loan.get(is_cured_col, False))

    if is_cured and cure_recovery_col and cure_recovery_col in loan.index:
        cure_recovery = float(loan.get(cure_recovery_col, 0) or 0)
        pv_cure = _pv_cashflow(cure_recovery, 0, discount_rate)   # at default date
        net_recovery = pv_cure
        lgd = max(LGD_FLOOR_ABSOLUTE, 1.0 - net_recovery / ead_denom)
        lgd = min(lgd, LGD_CAP_ABSOLUTE)
        return {
            "realised_lgd": round(lgd, 6),
            "pv_recoveries": round(pv_cure, 2),
            "pv_direct_costs": 0.0,
            "pv_indirect_costs": 0.0,
            "lip_cost_applied": 0.0,
            "ead_denominator": round(ead_denom, 2),
            "net_recovery_pv": round(net_recovery, 2),
            "workout_lgd_source": "cured",
            "lgd_computation_note": "Cure path: LGD from cure recovery PV. APS 113 s.32.",
        }

    # Workout (non-cured) path
    # Get cashflows for this loan
    try:
        loan_cf = cf_by_loan.get_group(loan_id).copy()
    except KeyError:
        loan_cf = pd.DataFrame(columns=[cashflow_date_col, recovery_col, cost_col, indirect_cost_col])

    loan_cf[cashflow_date_col] = pd.to_datetime(loan_cf[cashflow_date_col])
    loan_cf["days_from_default"] = (loan_cf[cashflow_date_col] - default_dt).dt.days.clip(lower=0)

    lip_window_end = default_dt + pd.Timedelta(days=lip_window_days)

    # Compute LIP costs
    if lip_costs is None:
        # Auto-detect: costs within LIP window
        lip_mask = loan_cf[cashflow_date_col] <= lip_window_end
        lip_cost_val = float(loan_cf.loc[lip_mask, cost_col].sum()) if not loan_cf.empty else 0.0
        # Also include lip_costs column from loan if present
        if "lip_costs" in loan.index:
            lip_cost_val = max(lip_cost_val, float(loan.get("lip_costs", 0) or 0))
    elif isinstance(lip_costs, pd.Series):
        lip_cost_val = float(lip_costs.get(loan.name, 0) or 0)
    else:
        lip_cost_val = float(lip_costs)

    # PV of recoveries (all cashflows, not just post-LIP)
    pv_recoveries = 0.0
    if not loan_cf.empty and recovery_col in loan_cf.columns:
        for _, cf_row in loan_cf.iterrows():
            amt = float(cf_row.get(recovery_col, 0) or 0)
            if amt > 0:
                pv_recoveries += _pv_cashflow(amt, int(cf_row["days_from_default"]), discount_rate)

    # PV of direct costs (post-LIP window only; LIP costs already captured above)
    pv_direct_costs = 0.0
    if not loan_cf.empty and cost_col in loan_cf.columns:
        post_lip_cf = loan_cf[loan_cf[cashflow_date_col] > lip_window_end]
        for _, cf_row in post_lip_cf.iterrows():
            amt = float(cf_row.get(cost_col, 0) or 0)
            if amt > 0:
                pv_direct_costs += _pv_cashflow(amt, int(cf_row["days_from_default"]), discount_rate)

    # PV of indirect costs
    pv_indirect_costs = 0.0
    if not loan_cf.empty and indirect_cost_col in loan_cf.columns:
        for _, cf_row in loan_cf.iterrows():
            amt = float(cf_row.get(indirect_cost_col, 0) or 0)
            if amt > 0:
                pv_indirect_costs += _pv_cashflow(amt, int(cf_row["days_from_default"]), discount_rate)

    # Workout LGD formula (APS 113 s.49-51):
    # LGD = (EAD - (PV_recoveries - PV_direct_costs - PV_indirect_costs - LIP_costs)) / EAD
    net_recovery = pv_recoveries - pv_direct_costs - pv_indirect_costs - lip_cost_val
    lgd = 1.0 - net_recovery / ead_denom
    lgd = max(LGD_FLOOR_ABSOLUTE, min(lgd, LGD_CAP_ABSOLUTE))

    # Determine if workout is complete
    source = "workout"
    note = "Non-cured workout path. PV cashflows discounted at contractual rate. APS 113 s.49-51."
    if loan_cf.empty:
        source = "partial_workout"
        note = "No cashflow records found — LGD computed from loan-level fields only."

    return {
        "realised_lgd": round(lgd, 6),
        "pv_recoveries": round(pv_recoveries, 2),
        "pv_direct_costs": round(pv_direct_costs, 2),
        "pv_indirect_costs": round(pv_indirect_costs, 2),
        "lip_cost_applied": round(lip_cost_val, 2),
        "ead_denominator": round(ead_denom, 2),
        "net_recovery_pv": round(net_recovery, 2),
        "workout_lgd_source": source,
        "lgd_computation_note": note,
    }


def _pv_cashflow(amount: float, days_from_default: int, annual_rate: float) -> float:
    """PV = CF / (1 + r)^(t/365). APS 113 s.50."""
    if days_from_default <= 0:
        return amount
    return amount / (1.0 + annual_rate) ** (days_from_default / 365.0)


# ---------------------------------------------------------------------------
# Segmentation and long-run LGD
# ---------------------------------------------------------------------------

def segment_lgd(
    df: pd.DataFrame,
    segment_keys: list[str],
    lgd_col: str = "realised_lgd",
    ead_col: str = "ead_at_default",
    min_segment_count: int = 20,
) -> pd.DataFrame:
    """
    Compute exposure-weighted LGD by segment with credibility flagging.

    Segments with < min_segment_count observations are flagged for potential
    collapse or pooling per APRA model risk guidance (APS 113 s.52).

    Parameters
    ----------
    df : DataFrame with realised_lgd and ead_at_default columns
    segment_keys : list of column names to group by (product-specific)
    lgd_col : column with realised LGD values
    ead_col : column with EAD values
    min_segment_count : minimum defaults per segment before credibility flag

    Returns
    -------
    DataFrame with columns:
        [segment_keys..., n_obs, total_ead, ew_lgd, ew_lgd_std,
         credibility_flag, segment_key_concat]

    APS 113 s.52: Segments must be homogeneous and contain sufficient observations.
    """
    df = df.copy()
    df[ead_col] = pd.to_numeric(df[ead_col], errors="coerce").fillna(0)
    df[lgd_col] = pd.to_numeric(df[lgd_col], errors="coerce")

    groups = df.groupby(segment_keys, observed=True)

    rows = []
    for keys, grp in groups:
        if not isinstance(keys, tuple):
            keys = (keys,)
        n_obs = len(grp)
        total_ead = grp[ead_col].sum()
        valid = grp[[lgd_col, ead_col]].dropna()
        ew_lgd = float(exposure_weighted_average(valid, lgd_col, ead_col)) if len(valid) > 0 else np.nan
        ew_lgd_std = float(valid[lgd_col].std()) if len(valid) > 1 else np.nan
        cred_flag = "low_count" if n_obs < min_segment_count else "adequate"

        row = {k: v for k, v in zip(segment_keys, keys)}
        row.update({
            "n_obs": n_obs,
            "total_ead": round(total_ead, 2),
            "ew_lgd": round(ew_lgd, 6) if np.isfinite(ew_lgd) else np.nan,
            "ew_lgd_std": round(ew_lgd_std, 6) if np.isfinite(ew_lgd_std) else np.nan,
            "credibility_flag": cred_flag,
            "segment_key_concat": "|".join(str(v) for v in keys),
            "aps113_section": "s.52",
        })
        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        logger.warning("segment_lgd: no segments produced. Check segment_keys=%s.", segment_keys)
    else:
        logger.info(
            "segment_lgd: %d segments | %d with low_count flag",
            len(result), (result["credibility_flag"] == "low_count").sum(),
        )
    return result


def compute_long_run_lgd(
    df: pd.DataFrame,
    segment_keys: list[str],
    lgd_col: str = "realised_lgd",
    ead_col: str = "ead_at_default",
    vintage_col: str = "default_year",
    min_years: int = 5,
    method: str = "vintage_ewa",
) -> pd.DataFrame:
    """
    Compute long-run average LGD per APS 113 s.43.

    Methods
    -------
    'vintage_ewa':
        1. Compute EWA LGD per vintage year per segment.
        2. Average across vintage years (equal weight per year).
        This method averages through economic cycles as APS 113 s.43 requires.
        It prevents COVID-year over-weighting that occurs with simple EWA.

    'ewa':
        Simple exposure-weighted average across all observations.
        WARNING: This over-weights high-default-count years (typically downturns).

    Parameters
    ----------
    df : DataFrame with realised LGD, EAD, and default_year
    segment_keys : grouping columns
    min_years : minimum number of distinct vintage years required
    method : 'vintage_ewa' (recommended) | 'ewa'

    Returns
    -------
    DataFrame with columns:
        [segment_keys..., long_run_lgd, n_vintages, vintage_range,
         min_vintage_lgd, max_vintage_lgd, cycle_coverage_flag,
         n_downturn_vintages, aps113_section]

    Raises
    ------
    ValueError if fewer than min_years distinct vintages are present across all segments.

    APS 113 s.43: Long-run average must reflect full economic cycle.
    APS 113 s.44: Minimum 7 years for mortgage, 5 years for other.
    """
    df = df.copy()
    df[ead_col] = pd.to_numeric(df[ead_col], errors="coerce").fillna(0)
    df[lgd_col] = pd.to_numeric(df[lgd_col], errors="coerce")
    df[vintage_col] = pd.to_numeric(df[vintage_col], errors="coerce")

    # Validate minimum observation period
    n_vintages_total = df[vintage_col].nunique()
    if n_vintages_total < min_years:
        raise ValueError(
            f"compute_long_run_lgd: only {n_vintages_total} distinct vintage years in dataset. "
            f"Minimum required: {min_years} (APS 113 Attachment A). "
            "Extend observation period or check vintage_col name."
        )

    groups = df.groupby(segment_keys, observed=True)
    rows = []

    for keys, grp in groups:
        if not isinstance(keys, tuple):
            keys = (keys,)
        valid = grp[[lgd_col, ead_col, vintage_col]].dropna()

        if method == "vintage_ewa":
            # Step 1: EWA LGD per vintage year
            vintage_lgds = []
            for year, yr_grp in valid.groupby(vintage_col):
                ewa = float(exposure_weighted_average(yr_grp, lgd_col, ead_col))
                if np.isfinite(ewa):
                    vintage_lgds.append({"year": year, "lgd": ewa})
            if not vintage_lgds:
                long_run_lgd = np.nan
            else:
                # Step 2: Equal-weight average across vintages
                long_run_lgd = float(np.mean([v["lgd"] for v in vintage_lgds]))
        else:  # simple ewa
            long_run_lgd = float(exposure_weighted_average(valid, lgd_col, ead_col))

        # Cycle coverage
        n_vintages = valid[vintage_col].nunique()
        vintage_range = (
            f"{int(valid[vintage_col].min())}-{int(valid[vintage_col].max())}"
            if not valid.empty else "N/A"
        )
        # Count downturn vintages in segment
        from src.generators.base_generator import DOWNTURN_YEARS
        n_downturn_vintages = len(set(valid[vintage_col].dropna().astype(int)) & DOWNTURN_YEARS)
        cycle_flag = "adequate" if n_downturn_vintages >= 1 else "no_downturn_in_sample"

        row = {k: v for k, v in zip(segment_keys, keys)}
        row.update({
            "long_run_lgd": round(long_run_lgd, 6) if np.isfinite(long_run_lgd) else np.nan,
            "n_vintages": n_vintages,
            "vintage_range": vintage_range,
            "min_vintage_lgd": round(float(min(v["lgd"] for v in vintage_lgds)), 6) if method == "vintage_ewa" and vintage_lgds else np.nan,
            "max_vintage_lgd": round(float(max(v["lgd"] for v in vintage_lgds)), 6) if method == "vintage_ewa" and vintage_lgds else np.nan,
            "cycle_coverage_flag": cycle_flag,
            "n_downturn_vintages": n_downturn_vintages,
            "method": method,
            "aps113_section": "s.43-44",
        })
        rows.append(row)

    result = pd.DataFrame(rows)
    n_no_downturn = (result["cycle_coverage_flag"] == "no_downturn_in_sample").sum()
    if n_no_downturn > 0:
        logger.warning(
            "compute_long_run_lgd: %d segments have no downturn vintage in sample. "
            "MoC cyclicality component will be triggered. (APS 113 s.43)",
            n_no_downturn,
        )
    return result


def compare_model_vs_actual(
    df: pd.DataFrame,
    model_lgd_col: str,
    actual_lgd_col: str = "realised_lgd",
    segment_keys: list[str] | None = None,
    ead_col: str = "ead_at_default",
) -> pd.DataFrame:
    """
    Compare model LGD (from proxy engine) vs historical realised LGD.

    This comparison directly feeds into:
      - MoCRegister.model_error_moc (when bias > 5%)
      - calibration_adjustments.csv
      - Cross-product model adequacy review

    Returns
    -------
    DataFrame with per-segment columns:
        model_lgd_ewa, actual_lgd_ewa, bias (model - actual),
        bias_pct, is_conservative (model >= actual),
        ead_coverage_pct, n_obs, aps113_section

    APS 113 s.60-62: Models must be compared to actual loss experience.
    When model LGD < actual LGD, a calibration adjustment is required.
    """
    df = df.copy()
    df[ead_col] = pd.to_numeric(df[ead_col], errors="coerce").fillna(0)
    df[model_lgd_col] = pd.to_numeric(df[model_lgd_col], errors="coerce")
    df[actual_lgd_col] = pd.to_numeric(df[actual_lgd_col], errors="coerce")

    total_ead = df[ead_col].sum()
    rows = []

    groups = [(None, df)] if segment_keys is None else list(df.groupby(segment_keys, observed=True))

    for keys, grp in groups:
        valid = grp[[model_lgd_col, actual_lgd_col, ead_col]].dropna()
        if valid.empty:
            continue

        model_ewa = float(exposure_weighted_average(valid, model_lgd_col, ead_col))
        actual_ewa = float(exposure_weighted_average(valid, actual_lgd_col, ead_col))
        bias = model_ewa - actual_ewa
        bias_pct = bias / actual_ewa if actual_ewa > 0 else np.nan
        is_conservative = model_ewa >= actual_ewa
        ead_coverage = valid[ead_col].sum() / total_ead if total_ead > 0 else np.nan

        row = {}
        if keys is not None and segment_keys:
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {k: v for k, v in zip(segment_keys, keys)}

        row.update({
            "model_lgd_ewa": round(model_ewa, 6),
            "actual_lgd_ewa": round(actual_ewa, 6),
            "bias": round(bias, 6),
            "bias_pct": round(bias_pct, 6) if np.isfinite(bias_pct) else np.nan,
            "is_conservative": is_conservative,
            "ead_coverage_pct": round(ead_coverage, 4) if np.isfinite(ead_coverage) else np.nan,
            "n_obs": len(valid),
            "aps113_section": "s.60-62",
            "note": (
                "Model underestimates — calibration adjustment required."
                if not is_conservative else "Model conservative — no mandatory adjustment."
            ),
        })
        rows.append(row)

    result = pd.DataFrame(rows)
    n_under = (~result["is_conservative"]).sum() if len(result) > 0 else 0
    if n_under > 0:
        logger.warning(
            "compare_model_vs_actual: %d segments where model LGD < actual LGD "
            "(calibration adjustment required per APS 113 s.60-62).",
            n_under,
        )
    return result


def compute_calibration_adjustment(
    model_lgd: float,
    actual_lgd: float,
    method: str = "additive",
) -> dict:
    """
    Compute calibration adjustment to close the model-vs-actual gap.

    APS 113 s.60-62: When model LGD < actual, an upward adjustment is mandatory.

    Parameters
    ----------
    method : 'additive' | 'multiplicative'
        - additive: adjustment = actual_lgd - model_lgd
        - multiplicative: adjustment = actual_lgd / model_lgd

    Returns dict with:
        adjustment_type, model_lgd, actual_lgd, raw_adjustment,
        adjusted_lgd, calibration_note
    """
    if method == "additive":
        adj = actual_lgd - model_lgd
        adjusted = model_lgd + adj
    elif method == "multiplicative":
        adj = actual_lgd / model_lgd if model_lgd > 0 else 1.0
        adjusted = model_lgd * adj
    else:
        raise ValueError(f"Unknown calibration method '{method}'. Use: additive, multiplicative.")

    return {
        "adjustment_type": method,
        "model_lgd": round(model_lgd, 6),
        "actual_lgd": round(actual_lgd, 6),
        "raw_adjustment": round(adj, 6),
        "adjusted_lgd": round(max(adjusted, actual_lgd), 6),   # never go below actual
        "is_conservative_after_adjustment": adjusted >= actual_lgd,
        "aps113_section": "s.60-62",
        "calibration_note": (
            "Mandatory upward adjustment applied (model < actual)."
            if actual_lgd > model_lgd else "No adjustment required (model >= actual)."
        ),
    }


def apply_regulatory_floor(
    lgd_series: pd.Series,
    floor: float,
    product: str = "",
) -> pd.Series:
    """
    Apply regulatory/policy floor to LGD estimates.

    APS 113 s.58: LGD must not be set below prescribed minimums.
    The floor is applied AFTER MoC (see moc_framework.py for application order).

    Parameters
    ----------
    floor : decimal floor (e.g., 0.10 for 10%)

    Returns
    -------
    Series with LGD values floored at min(lgd, floor).
    """
    floored = lgd_series.clip(lower=floor)
    n_floored = (lgd_series < floor).sum()
    if n_floored > 0:
        logger.info(
            "apply_regulatory_floor(%s): floor=%.0f%% applied to %d/%d loans (%.1f%%).",
            product, 100 * floor, n_floored, len(lgd_series),
            100 * n_floored / len(lgd_series),
        )
    return floored


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_inputs(loans, cashflows, ead_col, default_date_col, discount_rate_col, is_cured_col):
    for col in ["loan_id", ead_col, default_date_col]:
        if col not in loans.columns:
            raise ValueError(f"loans DataFrame missing required column '{col}'.")
    if "loan_id" not in cashflows.columns:
        raise ValueError("cashflows DataFrame missing required column 'loan_id'.")
    if (pd.to_numeric(loans[ead_col], errors="coerce") <= 0).any():
        n_bad = (pd.to_numeric(loans[ead_col], errors="coerce") <= 0).sum()
        raise ValueError(f"{n_bad} rows have EAD <= 0 in column '{ead_col}'.")
    if discount_rate_col not in loans.columns:
        logger.warning(
            "Discount rate column '%s' not in loans. "
            "Will use 7%% hardcoded fallback per loan. "
            "Run rba_rates_loader.build_discount_rate_register() to set real rates.",
            discount_rate_col,
        )
    if is_cured_col not in loans.columns:
        logger.warning(
            "'%s' column not found. Treating all loans as non-cured (workout path).",
            is_cured_col,
        )

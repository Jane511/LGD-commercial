"""
LGD calculation engines for Australian bank lending products.

Provides product-specific LGD computation, segmentation, regulatory overlays,
and the full pipeline from realised LGD through to final APRA-compliant estimates.

Products:
  1. Residential Mortgage  (two-stage cure model + APRA overlays)
  2. Commercial Cash Flow   (security-type segmentation + APS 112)
  3. Development Finance    (completion-stage scenario model)

Common pipeline:
  Realised LGD -> Exposure-weighted long-run -> Downturn adjustment -> MoC -> Regulatory overlays
"""
import logging
import numpy as np
import pandas as pd
from scipy.special import expit  # logistic function

logger = logging.getLogger(__name__)

from .industry_risk_integration import (
    compute_industry_downturn_scalar,
    compute_industry_moc_adjustment,
    compute_industry_recovery_haircut,
    compute_working_capital_lgd_adjustment,
)
from .overlay_parameters import OverlayParameterManager
from .segmentation import apply_standard_segments, build_segmentation_consistency_report


# ==========================================================================
# COMMON UTILITIES
# ==========================================================================

def exposure_weighted_average(df, lgd_col, ead_col="ead", group_col=None):
    """
    Compute exposure-weighted average LGD.

    Formula: LR_LGD_s = Sum(LGD_i * EAD_i) / Sum(EAD_i)

    Parameters
    ----------
    df : DataFrame with LGD and EAD columns
    lgd_col : column name for LGD values
    ead_col : column name for EAD values
    group_col : optional grouping column(s) for segment-level averages

    Returns
    -------
    DataFrame with exposure-weighted LGD per segment, or scalar if no grouping
    """
    if group_col is None:
        total_ead = df[ead_col].sum()
        if total_ead == 0:
            return 0.0
        return (df[lgd_col] * df[ead_col]).sum() / total_ead

    def _ewa(g):
        total = g[ead_col].sum()
        if total == 0:
            return 0.0
        return (g[lgd_col] * g[ead_col]).sum() / total

    if isinstance(group_col, str):
        group_col = [group_col]

    result = (
        df.groupby(group_col, observed=True)
        .apply(_ewa, include_groups=False)
        .reset_index()
    )
    result.columns = list(group_col) + ["lgd_long_run"]
    counts = df.groupby(group_col, observed=True).size().reset_index(name="count")
    eads = df.groupby(group_col, observed=True)[ead_col].sum().reset_index(name="total_ead")
    result = result.merge(counts, on=group_col).merge(eads, on=group_col)
    return result


def build_weighted_lgd_output(
    df,
    group_cols=None,
    ead_col="ead",
    base_col="lgd_base",
    downturn_col="lgd_downturn",
    final_col="lgd_final",
):
    """
    Build exposure-weighted LGD outputs with explicit base/downturn/final fields.
    """
    if df is None or len(df) == 0:
        cols = [
            "facility_count",
            "total_ead",
            "ead_weighted_lgd_base",
            "ead_weighted_lgd_downturn",
            "ead_weighted_lgd_final",
        ]
        if group_cols is None:
            return pd.DataFrame(columns=cols)
        group_cols = [group_cols] if isinstance(group_cols, str) else list(group_cols)
        return pd.DataFrame(columns=group_cols + cols)

    work = df.copy()
    if ead_col not in work.columns:
        work[ead_col] = 1.0

    resolved_base_col = base_col
    if resolved_base_col not in work.columns and "realised_lgd" in work.columns:
        resolved_base_col = "realised_lgd"

    def _summarise(g):
        return pd.Series(
            {
                "facility_count": len(g),
                "total_ead": pd.to_numeric(g[ead_col], errors="coerce").fillna(0.0).sum(),
                "ead_weighted_lgd_base": (
                    exposure_weighted_average(g, resolved_base_col, ead_col)
                    if resolved_base_col in g.columns
                    else np.nan
                ),
                "ead_weighted_lgd_downturn": (
                    exposure_weighted_average(g, downturn_col, ead_col)
                    if downturn_col in g.columns
                    else np.nan
                ),
                "ead_weighted_lgd_final": (
                    exposure_weighted_average(g, final_col, ead_col)
                    if final_col in g.columns
                    else np.nan
                ),
            }
        )

    if group_cols is None:
        return _summarise(work).to_frame().T

    if isinstance(group_cols, str):
        group_cols = [group_cols]
    valid_groups = [c for c in group_cols if c in work.columns]
    if not valid_groups:
        return _summarise(work).to_frame().T

    out = (
        work.groupby(valid_groups, observed=True)
        .apply(lambda g: _summarise(g), include_groups=False)
        .reset_index()
    )
    return out


def apply_downturn_overlay(lgd_series, method="scalar", scalar=1.10, addon=0.0):
    """
    Apply downturn adjustment to long-run LGD.

    Methods:
      - 'scalar':  Downturn_LGD = LR_LGD * scalar
      - 'additive': Downturn_LGD = LR_LGD + addon
      - 'combined': Downturn_LGD = LR_LGD * scalar + addon
    """
    if method == "scalar":
        return lgd_series * scalar
    elif method == "additive":
        return lgd_series + addon
    elif method == "combined":
        return lgd_series * scalar + addon
    raise ValueError(f"Unknown downturn method: {method}")


def add_margin_of_conservatism(lgd_series, margin):
    """Add margin of conservatism (additive, in decimal form e.g. 0.02 = 2pp)."""
    return lgd_series + margin


def _apply_downturn_scalar(
    base_scalar,
    macro_multiplier,
    lower: float = 1.00,
    upper: float = 1.90,
):
    """Compute clip(base_scalar * macro_multiplier, lower, upper).

    Centralises clip-bounds logic shared across all product downturn scalar functions.
    """
    return np.clip(base_scalar * macro_multiplier, lower, upper)


def _validate_final_lgd(df: pd.DataFrame, product: str) -> None:
    """
    Sanity-check lgd_final after all overlays are applied.

    Hard failures (raise ValueError):
      - any lgd_final outside [0, 1]

    Soft warnings (log WARNING):
      - portfolio mean outside [5%, 70%]
      - lgd_final < lgd_base where a downturn scalar > 1 was applied
    """
    lgd = df["lgd_final"]

    out_of_range = ((lgd < 0) | (lgd > 1)).sum()
    if out_of_range:
        raise ValueError(
            f"{product}: {out_of_range} lgd_final value(s) are outside [0, 1] after clipping — "
            f"check overlay logic."
        )

    mean_lgd = lgd.mean()
    if mean_lgd < 0.05 or mean_lgd > 0.70:
        logger.warning(
            "%s: portfolio mean lgd_final = %.1f%% is outside the plausible range [5%%, 70%%]. "
            "Review overlay parameters.",
            product,
            mean_lgd * 100,
        )

    if "lgd_base" in df.columns and "combined_downturn_scalar" in df.columns:
        scalar = _coerce_numeric_series(df, "combined_downturn_scalar", default=1.0).fillna(1.0)
        uplift_mask = scalar > 1.0
        if uplift_mask.any():
            violation = (df.loc[uplift_mask, "lgd_final"] < df.loc[uplift_mask, "lgd_base"]).sum()
            if violation:
                logger.warning(
                    "%s: %d row(s) have lgd_final < lgd_base despite combined_downturn_scalar > 1. "
                    "Check floor/clip interactions.",
                    product,
                    violation,
                )


def _coerce_numeric_series(df, column, default=np.nan):
    """Return a numeric Series for `column`, or a default-filled Series if missing."""
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    original_na_mask = df[column].isna()
    numeric = pd.to_numeric(df[column], errors="coerce")
    new_na_count = int(numeric.isna().sum()) - int(original_na_mask.sum())
    if new_na_count > 0:
        logger.warning(
            "Column '%s': %d value(s) could not be converted to numeric and were set to NaN",
            column,
            new_na_count,
        )
    return numeric


def _year_bucket_for_unemployment(default_dates):
    """
    Build a year-bucket label for fallback unemployment reporting.
    """
    if default_dates is None:
        return pd.Series(dtype=object)
    dates = pd.to_datetime(default_dates, errors="coerce")
    if not isinstance(dates, pd.Series):
        dates = pd.Series(dates)
    years = dates.dt.year
    bucket = np.where(
        years.isin([2020, 2021]),
        "covid_2020_2021",
        np.where(
            years.isin([2022, 2023]),
            "normalising_2022_2023",
            np.where(years.notna(), "baseline_other_year", "missing_default_date"),
        ),
    )
    return pd.Series(bucket, index=years.index)


def _year_to_unemployment_shock(default_dates):
    """
    Convert default year into a simple unemployment shock proxy.

    This is intentionally transparent and interview-friendly:
      - 2020-2021: COVID stress period
      - 2022-2023: normalisation
      - other years: neutral baseline
    """
    if default_dates is None:
        return pd.Series(dtype=float)
    dates = pd.to_datetime(default_dates, errors="coerce")
    if not isinstance(dates, pd.Series):
        dates = pd.Series(dates)
    years = dates.dt.year
    shocks = np.where(
        years.isin([2020, 2021]),
        0.03,
        np.where(years.isin([2022, 2023]), 0.015, np.where(years.notna(), 0.01, 0.02)),
    )
    return pd.Series(shocks, index=years.index, dtype=float)


def _resolve_discount_rate_proxy(df, product_baseline=0.05):
    """
    Resolve discount-rate proxy and source labels using fallback hierarchy:
      1) max(contract_rate_proxy, cost_of_funds_proxy)
      2) contract-only fallback
      3) cost-of-funds-only fallback
      4) provided discount_rate fallback
      5) product baseline fallback
    """
    index = df.index
    contract = _coerce_numeric_series(df, "contract_rate_proxy")
    cof = _coerce_numeric_series(df, "cost_of_funds_proxy")
    provided = _coerce_numeric_series(df, "discount_rate")

    has_contract = contract.notna()
    has_cof = cof.notna()
    has_provided = provided.notna()

    rate = pd.Series(float(product_baseline), index=index)
    source = pd.Series("product_baseline_fallback", index=index, dtype=object)

    both = has_contract & has_cof
    rate.loc[both] = np.maximum(contract.loc[both], cof.loc[both])
    source.loc[both] = "max_contract_vs_cost_of_funds"

    contract_only = has_contract & ~has_cof
    rate.loc[contract_only] = contract.loc[contract_only]
    source.loc[contract_only] = "contract_only_fallback"

    cof_only = ~has_contract & has_cof
    rate.loc[cof_only] = cof.loc[cof_only]
    source.loc[cof_only] = "cost_of_funds_only_fallback"

    provided_only = ~(has_contract | has_cof) & has_provided
    rate.loc[provided_only] = provided.loc[provided_only]
    source.loc[provided_only] = "provided_discount_rate_fallback"

    # Warn when low-priority fallbacks (tiers 4 & 5) are used
    n_provided_fallback = int(provided_only.sum())
    n_baseline_fallback = int((source == "product_baseline_fallback").sum())
    if n_provided_fallback:
        logger.warning(
            "_resolve_discount_rate_proxy: %d row(s) are using tier-4 provided_discount_rate_fallback "
            "— no contract_rate_proxy or cost_of_funds_proxy available.",
            n_provided_fallback,
        )
    if n_baseline_fallback:
        logger.warning(
            "_resolve_discount_rate_proxy: %d row(s) are using tier-5 product_baseline_fallback (%.1f%%) "
            "— no rate data available at all. Review input data quality.",
            n_baseline_fallback,
            product_baseline * 100,
        )

    clipped = rate.clip(lower=0.0)
    n_clipped = (rate < 0).sum()
    if n_clipped:
        logger.warning(
            "_resolve_discount_rate_proxy: %d row(s) had negative rates and were clipped to 0.",
            n_clipped,
        )
    return clipped, source


def _resolve_mortgage_arrears_proxy(df):
    """
    Resolve arrears stage using observed value when present, else:
      1) a fitted/modelled estimate if provided (`arrears_stage_model`), else
      2) transparent proxy rules.
    """
    index = df.index
    observed = df.get("arrears_stage")
    if observed is None:
        observed = pd.Series(np.nan, index=index)
    observed = observed.astype("string")
    use_proxy = observed.isna()

    # Optional hook: allow an upstream fitted model to provide an estimate.
    # This keeps the engine deterministic while supporting bank-style calibration workflows.
    modelled = df.get("arrears_stage_model")
    if modelled is not None:
        modelled = modelled.astype("string")
        use_modelled = use_proxy & modelled.notna()
        stage = observed.where(~use_modelled, modelled)
        # Only treat remaining missing as proxy-required.
        use_proxy = stage.isna()
        observed = stage

    ltv = _coerce_numeric_series(df, "ltv_at_default", default=0.85).fillna(0.85)
    dti = _coerce_numeric_series(df, "dti", default=0.35).fillna(0.35)
    score = _coerce_numeric_series(df, "credit_score", default=680).fillna(680)
    proxy = np.where(
        (ltv >= 1.00) | (dti >= 0.50) | (score < 580),
        "90+",
        np.where(
            (ltv >= 0.90) | (dti >= 0.42) | (score < 640),
            "60-89",
            "30-59",
        ),
    )
    stage = observed.where(~use_proxy, pd.Series(proxy, index=index))
    return stage.fillna("60-89"), use_proxy.astype(bool)


def _resolve_mortgage_behaviour_proxy(df):
    """
    Resolve repayment behaviour using observed value when present, else:
      1) a fitted/modelled estimate if provided (`repayment_behaviour_model`), else
      2) proxy scorecard rules.
    """
    index = df.index
    observed = df.get("repayment_behaviour")
    if observed is None:
        observed = pd.Series(np.nan, index=index)
    observed = observed.astype("string")
    use_proxy = observed.isna()

    # Optional hook: allow an upstream fitted model to provide an estimate.
    modelled = df.get("repayment_behaviour_model")
    if modelled is not None:
        modelled = modelled.astype("string")
        use_modelled = use_proxy & modelled.notna()
        behaviour = observed.where(~use_modelled, modelled)
        use_proxy = behaviour.isna()
        observed = behaviour

    loan_type = df.get("loan_type", pd.Series("P&I", index=index)).astype(str)
    seasoning = _coerce_numeric_series(df, "seasoning_months", default=24).fillna(24)
    dti = _coerce_numeric_series(df, "dti", default=0.35).fillna(0.35)
    ltv = _coerce_numeric_series(df, "ltv_at_default", default=0.85).fillna(0.85)
    score = _coerce_numeric_series(df, "credit_score", default=680).fillna(680)

    behaviour_score = (
        np.where(loan_type.str.upper().str.contains("P&I"), 1, 0)
        + np.where(seasoning >= 24, 1, 0)
        + np.where(dti >= 0.45, -1, 0)
        + np.where(ltv >= 0.90, -1, 0)
        + np.where(score >= 700, 1, 0)
    )
    proxy = np.where(
        behaviour_score >= 2,
        "strong",
        np.where(behaviour_score <= -1, "weak", "stable"),
    )
    behaviour = observed.where(~use_proxy, pd.Series(proxy, index=index))
    return behaviour.fillna("stable"), use_proxy.astype(bool)


def _resolve_mortgage_liquidation_loss(df, discount_rate_series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Estimate liquidation loss (LGD conditional on property sale) from simple cashflow proxies.

    This is closer to bank practice than back-solving from realised_lgd, but still designed to
    run with sparse demo inputs. If required fields are missing, returns NaNs and a source label.

    Required (at least one collateral value):
      - `property_value_forced_sale` OR `property_value_at_default`
    Also uses:
      - `ead`
      - optional cost/timing inputs: `selling_cost_rate`, `legal_cost_rate`, `holding_cost_rate_pa`,
        `time_to_sell_months`, `sale_price_haircut`

    Returns:
      - liquidation_loss (0..1)
      - liquidation_loss_source (string label per row)
    """
    index = df.index
    ead = _coerce_numeric_series(df, "ead", default=np.nan)
    ead = ead.replace(0, np.nan)

    forced_sale_value = _coerce_numeric_series(df, "property_value_forced_sale", default=np.nan)
    at_default_value = _coerce_numeric_series(df, "property_value_at_default", default=np.nan)

    has_forced = forced_sale_value.notna()
    has_default = at_default_value.notna()
    collateral = forced_sale_value.where(has_forced, at_default_value)
    source = pd.Series(
        np.where(has_forced, "property_value_forced_sale", np.where(has_default, "property_value_at_default", "missing_collateral_value")),
        index=index,
        dtype="string",
    )

    selling_cost_rate = _coerce_numeric_series(df, "selling_cost_rate", default=0.08).fillna(0.08).clip(0, 0.20)
    legal_cost_rate = _coerce_numeric_series(df, "legal_cost_rate", default=0.01).fillna(0.01).clip(0, 0.10)
    holding_cost_rate_pa = _coerce_numeric_series(df, "holding_cost_rate_pa", default=0.02).fillna(0.02).clip(0, 0.20)
    time_to_sell_months = _coerce_numeric_series(df, "time_to_sell_months", default=9.0).fillna(9.0).clip(0, 60)

    haircut = _coerce_numeric_series(df, "sale_price_haircut", default=0.10).fillna(0.10).clip(0, 0.50)

    # Approximate cashflows:
    # - Sale proceeds realised at time_to_sell_months
    # - Holding cost accrues over the period on collateral (simple approximation)
    gross_sale_proceeds = collateral * (1 - haircut)
    selling_cost = gross_sale_proceeds * selling_cost_rate
    legal_cost = gross_sale_proceeds * legal_cost_rate
    holding_cost = gross_sale_proceeds * holding_cost_rate_pa * (time_to_sell_months / 12.0)
    net_proceeds = (gross_sale_proceeds - selling_cost - legal_cost - holding_cost).clip(lower=0.0)

    discount_rate = pd.to_numeric(discount_rate_series, errors="coerce").fillna(0.05).clip(0, 0.30)
    discount_factor = 1 / (1 + discount_rate) ** (time_to_sell_months / 12.0)
    pv_recovery = net_proceeds * discount_factor

    recovery_rate = (pv_recovery / ead).clip(0, 2.0)
    liquidation_loss = (1 - recovery_rate).clip(0, 1)

    # If we have no collateral value, mark output NaN to allow fallback to other methods.
    liquidation_loss = liquidation_loss.where(source != "missing_collateral_value", np.nan)
    return liquidation_loss, source


def mortgage_macro_downturn_scalar(
    df,
    base_scalar=1.08,
    return_detail=False,
    house_price_coeff=0.40,
    unemployment_coeff=1.20,
    rate_shock_coeff=1.60,
):
    """
    Mortgage downturn scalar linked to macro drivers.

    Drivers:
      - House price decline
      - Unemployment shock
      - Rate shock
    """
    if "house_price_decline" in df.columns:
        hpd_raw = pd.to_numeric(df["house_price_decline"], errors="coerce")
        house_price_source = np.where(
            hpd_raw.notna(),
            "observed_house_price_decline",
            "house_price_fallback",
        )
        hpd = hpd_raw.fillna(0.0)
    elif {"property_value_orig", "property_value_at_default"}.issubset(df.columns):
        hpd = (
            (df["property_value_orig"] - df["property_value_at_default"])
            / df["property_value_orig"].replace(0, np.nan)
        ).clip(lower=0).fillna(0.0)
        house_price_source = pd.Series(
            "derived_from_property_values", index=df.index
        )
    else:
        # Fallback: stressed LTV above 80% often proxies collateral value decline.
        hpd = _coerce_numeric_series(df, "ltv_at_default", default=0.0).fillna(0.0)
        hpd = ((hpd - 0.80) / 0.40).clip(lower=0, upper=0.30)
        house_price_source = pd.Series("ltv_proxy_fallback", index=df.index)

    if "unemployment_shock" in df.columns:
        unemp_obs = pd.to_numeric(df["unemployment_shock"], errors="coerce")
        year_fallback = _year_to_unemployment_shock(df.get("default_date"))
        if year_fallback.empty:
            year_fallback = pd.Series(0.02, index=df.index, dtype=float)
        unemp = unemp_obs.fillna(year_fallback).fillna(0.02)
        unemployment_source = pd.Series(
            np.where(
                unemp_obs.notna(),
                "observed_unemployment_shock",
                "year_bucket_fallback",
            ),
            index=df.index,
        )
    else:
        year_fallback = _year_to_unemployment_shock(df.get("default_date"))
        if year_fallback.empty:
            unemp = pd.Series(0.02, index=df.index, dtype=float)
            unemployment_source = pd.Series(
                "neutral_constant_fallback", index=df.index
            )
        else:
            unemp = year_fallback.fillna(0.02)
            dates = pd.to_datetime(df.get("default_date"), errors="coerce")
            unemployment_source = pd.Series(
                np.where(
                    dates.notna(),
                    "year_bucket_fallback",
                    "neutral_constant_fallback",
                ),
                index=df.index,
            )
    unemp = np.clip(unemp, 0, 0.06)
    unemployment_year_bucket = _year_bucket_for_unemployment(df.get("default_date"))

    if "rate_shock" in df.columns:
        rate_obs = pd.to_numeric(df["rate_shock"], errors="coerce")
        if {"discount_rate", "contract_rate_proxy"}.issubset(df.columns):
            fallback_rate = (
                pd.to_numeric(df["discount_rate"], errors="coerce")
                - pd.to_numeric(df["contract_rate_proxy"], errors="coerce")
            ).clip(lower=0).fillna(0.0)
        else:
            fallback_rate = pd.Series(0.0, index=df.index)
        rate = rate_obs.fillna(fallback_rate).fillna(0.0)
        rate_source = pd.Series(
            np.where(rate_obs.notna(), "observed_rate_shock", "rate_shock_fallback"),
            index=df.index,
        )
    elif {"discount_rate", "contract_rate_proxy"}.issubset(df.columns):
        rate = (df["discount_rate"] - df["contract_rate_proxy"]).clip(lower=0)
        rate_source = pd.Series("derived_discount_minus_contract", index=df.index)
    elif "discount_rate" in df.columns:
        rate = (pd.to_numeric(df["discount_rate"], errors="coerce") - 0.04).clip(lower=0)
        rate_source = pd.Series("derived_discount_minus_baseline", index=df.index)
    else:
        rate = pd.Series(0.0, index=df.index)
        rate_source = pd.Series("neutral_zero_rate_shock", index=df.index)
    rate = np.clip(rate, 0, 0.05)

    macro_multiplier = 1 + house_price_coeff * hpd + unemployment_coeff * unemp + rate_shock_coeff * rate
    scalar = _apply_downturn_scalar(base_scalar, macro_multiplier, lower=1.00, upper=1.60)

    if not return_detail:
        return scalar

    detail = pd.DataFrame(
        {
            "house_price_driver": hpd,
            "house_price_source": house_price_source,
            "unemployment_driver": unemp,
            "unemployment_source": unemployment_source,
            "unemployment_year_bucket": unemployment_year_bucket,
            "rate_shock_driver": rate,
            "rate_shock_source": rate_source,
        },
        index=df.index,
    )
    return scalar, detail


def commercial_macro_downturn_scalar(
    df,
    base_scalar,
    value_decline_coeff=0.30,
    cashflow_weakness_coeff=0.10,
    recovery_delay_coeff=0.08,
):
    """
    Commercial downturn scalar linked to macro drivers.

    Drivers:
      - Collateral value decline proxy
      - Weaker cashflow (ICR stress)
      - Longer time to recovery
    """
    if "value_decline" in df.columns:
        value_decline = pd.to_numeric(df["value_decline"], errors="coerce").fillna(0.0)
    elif "security_coverage_ratio" in df.columns:
        value_decline = (1 - pd.to_numeric(df["security_coverage_ratio"], errors="coerce")).clip(lower=0)
    else:
        value_decline = pd.Series(0.10, index=df.index)
    value_decline = np.clip(value_decline, 0, 0.50)

    if "icr" in df.columns:
        icr_vals = pd.to_numeric(df["icr"], errors="coerce").fillna(1.5)
    else:
        icr_vals = pd.Series(1.5, index=df.index)
    cashflow_weakness = ((1.5 - icr_vals) / 1.5).clip(lower=0, upper=1)

    if "workout_months" in df.columns:
        workout = pd.to_numeric(df["workout_months"], errors="coerce").fillna(18)
    else:
        workout = pd.Series(18, index=df.index)
    recovery_delay = ((workout - 18) / 24).clip(lower=0, upper=1)

    macro_multiplier = (
        1
        + value_decline_coeff * value_decline
        + cashflow_weakness_coeff * cashflow_weakness
        + recovery_delay_coeff * recovery_delay
    )
    return _apply_downturn_scalar(base_scalar, macro_multiplier, lower=1.00, upper=1.70)


def development_macro_downturn_scalar(
    df,
    base_scalar,
    grv_decline_coeff=0.35,
    cost_overrun_coeff=0.25,
    sell_through_delay_coeff=0.12,
):
    """
    Development downturn scalar linked to macro drivers.

    Drivers:
      - GRV decline
      - Cost overrun
      - Slower sell-through (longer workout period)
    """
    if "grv_decline" in df.columns:
        grv_decline = pd.to_numeric(df["grv_decline"], errors="coerce").fillna(0.0)
    elif {"grv", "as_is_value"}.issubset(df.columns):
        grv_decline = (
            (df["grv"] - df["as_is_value"]) / df["grv"].replace(0, np.nan)
        ).clip(lower=0).fillna(0.0)
    else:
        grv_decline = pd.Series(0.10, index=df.index)
    grv_decline = np.clip(grv_decline, 0, 0.60)

    if {"cost_to_complete", "ead"}.issubset(df.columns):
        cost_overrun = (df["cost_to_complete"] / df["ead"].replace(0, np.nan)).clip(lower=0).fillna(0.0)
    else:
        cost_overrun = pd.Series(0.10, index=df.index)
    cost_overrun = np.clip(cost_overrun, 0, 0.40)

    workout = _coerce_numeric_series(df, "workout_months", default=18).fillna(18)
    sell_through_delay = ((workout - 18) / 18).clip(lower=0, upper=1)

    macro_multiplier = (
        1
        + grv_decline_coeff * grv_decline
        + cost_overrun_coeff * cost_overrun
        + sell_through_delay_coeff * sell_through_delay
    )
    return _apply_downturn_scalar(base_scalar, macro_multiplier, lower=1.00, upper=1.90)


def cashflow_macro_downturn_scalar(
    df,
    base_scalar,
    dscr_weakness_coeff=0.12,
    utilisation_coeff=0.06,
    recovery_delay_coeff=0.08,
):
    """
    Cashflow-lending downturn scalar linked to liquidity/utilisation proxies.
    """
    dscr = _coerce_numeric_series(df, "dscr", default=1.3).fillna(1.3)
    utilisation = _coerce_numeric_series(df, "utilisation", default=0.65).fillna(0.65)
    workout = _coerce_numeric_series(df, "workout_months", default=18).fillna(18)

    dscr_weakness = ((1.3 - dscr) / 1.3).clip(lower=0, upper=1)
    util_pressure = (utilisation - 0.65).clip(lower=0, upper=0.60)
    recovery_delay = ((workout - 18) / 24).clip(lower=0, upper=1)

    macro_multiplier = (
        1
        + dscr_weakness_coeff * dscr_weakness
        + utilisation_coeff * util_pressure
        + recovery_delay_coeff * recovery_delay
    )
    return _apply_downturn_scalar(base_scalar, macro_multiplier, lower=1.00, upper=1.90)


def _resolve_base_scalar_from_parameters(out, product, params):
    if product == "mortgage":
        scalar = params.get_value("mortgage", "base_downturn_scalar", default=1.08)
        return pd.Series(float(scalar), index=out.index)
    if product == "commercial":
        scalar_map = params.get_map("commercial", "base_downturn_scalar", "security_type:")
        return out["security_type"].map(scalar_map).fillna(
            params.get_value("commercial", "base_downturn_scalar", default=1.15)
        )
    if product == "development":
        scalar_map = params.get_map("development", "base_downturn_scalar", "completion_stage:")
        return out["completion_stage"].map(scalar_map).fillna(
            params.get_value("development", "base_downturn_scalar", default=1.25)
        )
    if product == "cashflow_lending":
        scalar_map = params.get_map("cashflow_lending", "base_downturn_scalar", "cashflow_product:")
        return out["cashflow_product"].map(scalar_map).fillna(
            params.get_value("cashflow_lending", "base_downturn_scalar", default=1.15)
        )
    return pd.Series(1.10, index=out.index)


def resolve_overlay_contract(
    df,
    product,
    parameter_manager,
    scenario_id="baseline",
    base_scalar=None,
    pd_band_addon=None,
):
    """
    Canonical overlay resolver with deterministic precedence:
    base scalar -> macro overlay -> industry adjustment.
    """
    out = df.copy()
    params = parameter_manager
    if base_scalar is None:
        base_scalar = _resolve_base_scalar_from_parameters(out, product, params)
    else:
        base_scalar = pd.Series(base_scalar, index=out.index)

    if pd_band_addon is not None:
        base_scalar = base_scalar + pd_band_addon

    if product == "mortgage":
        macro_scalar, _ = mortgage_macro_downturn_scalar(
            out,
            base_scalar=base_scalar,
            return_detail=True,
            house_price_coeff=params.get_value("mortgage", "macro_house_price_coeff", default=0.40),
            unemployment_coeff=params.get_value("mortgage", "macro_unemployment_coeff", default=1.20),
            rate_shock_coeff=params.get_value("mortgage", "macro_rate_shock_coeff", default=1.60),
        )
    elif product == "commercial":
        macro_scalar = commercial_macro_downturn_scalar(
            out,
            base_scalar=base_scalar,
            value_decline_coeff=params.get_value("commercial", "macro_value_decline_coeff", default=0.30),
            cashflow_weakness_coeff=params.get_value("commercial", "macro_cashflow_weakness_coeff", default=0.10),
            recovery_delay_coeff=params.get_value("commercial", "macro_recovery_delay_coeff", default=0.08),
        )
    elif product == "development":
        macro_scalar = development_macro_downturn_scalar(
            out,
            base_scalar=base_scalar,
            grv_decline_coeff=params.get_value("development", "macro_grv_decline_coeff", default=0.35),
            cost_overrun_coeff=params.get_value("development", "macro_cost_overrun_coeff", default=0.25),
            sell_through_delay_coeff=params.get_value("development", "macro_sell_through_delay_coeff", default=0.12),
        )
    else:
        macro_scalar = cashflow_macro_downturn_scalar(
            out,
            base_scalar=base_scalar,
            dscr_weakness_coeff=params.get_value("cashflow_lending", "macro_dscr_weakness_coeff", default=0.12),
            utilisation_coeff=params.get_value("cashflow_lending", "macro_utilisation_coeff", default=0.06),
            recovery_delay_coeff=params.get_value("cashflow_lending", "macro_recovery_delay_coeff", default=0.08),
        )

    alpha = params.get_value("all", "industry_downturn_alpha", default=0.15)
    if "industry_risk_score" in out.columns:
        combined = compute_industry_downturn_scalar(
            out["industry_risk_score"],
            macro_scalar,
            alpha=alpha,
        )
        industry_adj = combined / pd.Series(macro_scalar, index=out.index).replace(0, np.nan)
        overlay_source = "macro_plus_industry"
    else:
        combined = macro_scalar
        industry_adj = pd.Series(1.0, index=out.index)
        overlay_source = "macro_only"

    detail = pd.DataFrame(
        {
            "macro_downturn_scalar": pd.Series(macro_scalar, index=out.index),
            "industry_downturn_adjustment": industry_adj.fillna(1.0),
            "combined_downturn_scalar": pd.Series(combined, index=out.index),
            "overlay_source": overlay_source,
            "parameter_version": params.meta.version,
            "scenario_id": str(scenario_id),
        },
        index=out.index,
    )
    return detail


# ==========================================================================
# 1. RESIDENTIAL MORTGAGE LGD ENGINE
# ==========================================================================

class MortgageLGDEngine:
    """
    Australian residential mortgage LGD engine.

    Implements:
    - Two-stage model: P(Cure) then E[LGD|Loss]
    - Exposure-weighted long-run LGD by segment
    - Downturn overlay (scalar or macro-linked)
    - Margin of conservatism
    - APRA overlays: LMI recognition, 10%/15% floor, standard/non-standard
    """

    # APRA regulatory parameters
    LGD_FLOOR_STANDARD = 0.10
    LGD_FLOOR_NON_STANDARD = 0.15
    LMI_BENEFIT_FACTOR = 0.20  # 20% LGD reduction for eligible LMI
    APRA_SCALAR = 1.10  # applied to RWA, not LGD

    # Default model parameters
    DEFAULT_DOWNTURN_SCALAR = 1.08
    DEFAULT_MOC = 0.02  # 2 percentage points

    def __init__(self, downturn_scalar=None, moc=None, parameter_manager=None, scenario_id="baseline"):
        self.parameter_manager = parameter_manager or OverlayParameterManager()
        self.scenario_id = scenario_id
        self.downturn_scalar = downturn_scalar or self.parameter_manager.get_value(
            "mortgage", "base_downturn_scalar", default=self.DEFAULT_DOWNTURN_SCALAR
        )
        self.moc = moc or self.parameter_manager.get_value(
            "mortgage", "base_moc", default=self.DEFAULT_MOC
        )

    @staticmethod
    def segment_loans(df):
        """
        Create segmentation columns for mortgage portfolio.

        Segments:
          Level 1: mortgage_class (Standard / Non-Standard)
          Level 2: ltv_band (LTV at default)
          Level 3: lmi_eligible
        """
        out = df.copy()

        # LTV bands at default
        bins = [0, 0.60, 0.70, 0.80, 0.90, float("inf")]
        labels = ["<60%", "60-70%", "70-80%", "80-90%", "90%+"]
        out["ltv_band"] = pd.cut(
            out["ltv_at_default"], bins=bins, labels=labels, right=True
        )

        # Credit score bands
        score_bins = [0, 550, 620, 680, 740, 900]
        score_labels = ["<550", "550-620", "620-680", "680-740", "740+"]
        out["score_band"] = pd.cut(
            out["credit_score"], bins=score_bins, labels=score_labels, right=True
        )

        return apply_standard_segments(out, product="mortgage")

    def compute_long_run_lgd(self, df, segments=None):
        """
        Compute exposure-weighted long-run LGD by segment.

        Default segments: mortgage_class x ltv_band
        """
        if segments is None:
            segments = ["mortgage_class", "ltv_band"]
        segmented = self.segment_loans(df)
        return exposure_weighted_average(
            segmented, lgd_col="realised_lgd", ead_col="ead", group_col=segments
        )

    def compute_weighted_outputs(self, df, group_cols=None):
        """
        Compute weighted base/downturn/final LGD outputs for mortgage.
        """
        if group_cols is None:
            group_cols = ["mortgage_class", "ltv_band", "lmi_eligible"]
        segmented = self.segment_loans(df)
        return build_weighted_lgd_output(
            segmented,
            group_cols=group_cols,
            base_col="lgd_base",
            downturn_col="lgd_downturn",
            final_col="lgd_final",
        )

    def apply_apra_overlays(self, df):
        """
        Apply full APRA overlay chain to loan-level LGD estimates.

        Pipeline:
          1. Build cure-aware base LGD (`lgd_base`)
          2. Apply macro-linked downturn scalar
          3. Add margin of conservatism
          4. Apply LMI benefit (for eligible loans)
          5. Apply LGD floor (10% standard, 15% non-standard)

        Returns DataFrame with intermediate and final columns.
        """
        out = df.copy()

        # Governance hooks: discount-rate fallback source.
        discount_rate_proxy_used, discount_rate_source = _resolve_discount_rate_proxy(
            out, product_baseline=0.05
        )
        out["discount_rate_proxy_used"] = discount_rate_proxy_used
        out["discount_rate_source"] = discount_rate_source

        # Governance hooks: arrears and repayment-behaviour proxy flags.
        arrears_stage_proxy, arrears_proxy_flag = _resolve_mortgage_arrears_proxy(out)
        behaviour_proxy, behaviour_proxy_flag = _resolve_mortgage_behaviour_proxy(out)
        out["arrears_stage_proxy"] = arrears_stage_proxy
        out["proxy_arrears_flag"] = arrears_proxy_flag
        out["repayment_behaviour_proxy"] = behaviour_proxy
        out["proxy_behaviour_flag"] = behaviour_proxy_flag
        out["borrower_type_proxy"] = (
            out.get("occupancy", pd.Series("Unknown", index=out.index))
            .astype(str)
            .str.lower()
            .str.replace("-", "_", regex=False)
        )

        # Mortgage cure proxy for reporting.
        arrears_num = out["arrears_stage_proxy"].map(
            {"30-59": 0, "60-89": 1, "90+": 2}
        ).fillna(1)
        behaviour_num = out["repayment_behaviour_proxy"].map(
            {"weak": -1, "stable": 0, "strong": 1}
        ).fillna(0)
        owner_bonus = np.where(
            out["borrower_type_proxy"].str.contains("owner"), 0.04, 0.0
        )
        ltv = _coerce_numeric_series(out, "ltv_at_default", default=0.85).fillna(0.85)
        dti = _coerce_numeric_series(out, "dti", default=0.35).fillna(0.35)

        # Optional hook: allow an upstream fitted model to provide P(cure).
        # If present, this becomes the primary estimate; the proxy scorecard remains a fallback.
        cure_rate_model = out.get("cure_rate_model")
        if cure_rate_model is not None:
            cure_rate_model = pd.to_numeric(cure_rate_model, errors="coerce")
        cure_rate_proxy = (
            0.36
            - 0.20 * (ltv - 0.80).clip(lower=0, upper=0.50)
            - 0.08 * arrears_num
            - 0.12 * (dti - 0.35).clip(lower=0, upper=0.25)
            + 0.06 * behaviour_num
            + owner_bonus
        )
        cure_rate = pd.Series(cure_rate_proxy, index=out.index, dtype=float)
        if cure_rate_model is not None:
            cure_rate = cure_rate.where(cure_rate_model.isna(), cure_rate_model)
        cure_rate = cure_rate.clip(0.02, 0.75)
        out["cure_rate_proxy"] = cure_rate

        # Prefer a cashflow-style liquidation estimate if collateral value inputs are present;
        # otherwise fall back to the original demo back-solve using realised_lgd.
        liquidation_loss_cf, liquidation_source = _resolve_mortgage_liquidation_loss(
            out, discount_rate_series=out["discount_rate_proxy_used"]
        )
        liquidation_backsolve = (
            out["realised_lgd"] / (1 - out["cure_rate_proxy"]).replace(0, np.nan)
        ).fillna(out["realised_lgd"]).clip(0, 1)
        out["liquidation_loss_proxy"] = liquidation_backsolve.where(
            liquidation_loss_cf.isna(), liquidation_loss_cf
        ).clip(0, 1)
        out["liquidation_loss_source"] = liquidation_source.where(
            liquidation_loss_cf.notna(), "backsolved_from_realised_lgd"
        )
        out["lgd_cure_proxy"] = (
            (1 - out["cure_rate_proxy"]) * out["liquidation_loss_proxy"]
        ).clip(0, 1)
        out["foreclosure_channel_flag"] = np.where(
            out.get("resolution_type", pd.Series("Property Sale", index=out.index))
            .astype(str)
            .str.lower()
            .eq("cure"),
            0,
            1,
        )
        out["cure_overlay_applied_flag"] = True
        out["cure_overlay_source"] = "mortgage_proxy_two_stage"
        out["lgd_base"] = out["lgd_cure_proxy"].clip(0, 1)

        # Step 1: Macro-linked downturn (house prices, unemployment, rate shock)
        out["downturn_scalar"], macro_detail = mortgage_macro_downturn_scalar(
            out,
            base_scalar=self.downturn_scalar,
            return_detail=True,
            house_price_coeff=self.parameter_manager.get_value(
                "mortgage", "macro_house_price_coeff", default=0.40
            ),
            unemployment_coeff=self.parameter_manager.get_value(
                "mortgage", "macro_unemployment_coeff", default=1.20
            ),
            rate_shock_coeff=self.parameter_manager.get_value(
                "mortgage", "macro_rate_shock_coeff", default=1.60
            ),
        )
        out = pd.concat([out, macro_detail], axis=1)
        overlay_detail = resolve_overlay_contract(
            out,
            product="mortgage",
            parameter_manager=self.parameter_manager,
            scenario_id=self.scenario_id,
            base_scalar=self.downturn_scalar,
        )
        out["macro_downturn_scalar"] = overlay_detail["macro_downturn_scalar"]
        out["industry_downturn_adjustment"] = overlay_detail["industry_downturn_adjustment"]
        out["downturn_scalar"] = overlay_detail["combined_downturn_scalar"]
        out["combined_downturn_scalar"] = overlay_detail["combined_downturn_scalar"]
        out["overlay_source"] = overlay_detail["overlay_source"]
        out["parameter_version"] = overlay_detail["parameter_version"]
        out["scenario_id"] = overlay_detail["scenario_id"]
        out["lgd_downturn"] = out["lgd_base"] * out["downturn_scalar"]

        # Step 2: MoC
        out["lgd_with_moc"] = add_margin_of_conservatism(
            out["lgd_downturn"], self.moc
        )

        # Step 3: LMI adjustment
        out["lgd_after_lmi"] = np.where(
            out["lmi_eligible"] == 1,
            out["lgd_with_moc"] * (1 - self.LMI_BENEFIT_FACTOR),
            out["lgd_with_moc"],
        )

        # Step 4: Floor
        floor = np.where(
            out["mortgage_class"] == "Standard",
            self.LGD_FLOOR_STANDARD,
            self.LGD_FLOOR_NON_STANDARD,
        )
        out["lgd_final"] = np.maximum(out["lgd_after_lmi"], floor)

        # Cap at 100%
        out["lgd_final"] = out["lgd_final"].clip(0, 1)

        _validate_final_lgd(out, product="mortgage")
        return apply_standard_segments(out, product="mortgage")

    def compute_illustrative_rwa(self, df, pd_estimate=0.02, maturity=5):
        """
        Compute illustrative RWA using simplified IRB risk-weight function.

        K = LGD * [N(sqrt(1/(1-R)) * G(PD) + sqrt(R/(1-R)) * G(0.999)) - PD]
            * (1 + (M - 2.5) * b) / (1 - 1.5 * b)
        RWA = K * 12.5 * EAD
        RWA_APRA = RWA * 1.10
        """
        from scipy.stats import norm

        lgd = df["lgd_final"].values
        ead = df["ead"].values
        pd_val = pd_estimate

        # Residential mortgage correlation (Basel)
        R = 0.15

        # Maturity adjustment
        b = (0.11852 - 0.05478 * np.log(pd_val)) ** 2
        maturity_adj = (1 + (maturity - 2.5) * b) / (1 - 1.5 * b)

        # Capital requirement
        K = lgd * (
            norm.cdf(
                np.sqrt(1 / (1 - R)) * norm.ppf(pd_val)
                + np.sqrt(R / (1 - R)) * norm.ppf(0.999)
            )
            - pd_val
        ) * maturity_adj

        rwa = K * 12.5 * ead
        rwa_apra = rwa * self.APRA_SCALAR

        out = df.copy()
        out["capital_K"] = np.round(K, 6)
        out["rwa"] = np.round(rwa, 2)
        out["rwa_after_apra_scalar"] = np.round(rwa_apra, 2)
        return out


# ==========================================================================
# 2. COMMERCIAL CASH FLOW LGD ENGINE
# ==========================================================================

class CommercialLGDEngine:
    """
    Australian commercial cash-flow lending LGD engine.

    Implements:
    - Security-type segmentation (Property, PPSR, GSR)
    - Coverage-ratio-based analysis
    - Industry segmentation
    - Exposure-weighted long-run LGD
    - Downturn overlay by security type
    - Margin of conservatism
    - APS 112 supervisory LGD fallback
    """

    # Supervisory LGD (APS 112 / standardised approach fallback)
    SUPERVISORY_LGD = {
        "Senior Secured": 0.35,
        "Senior Unsecured": 0.45,
        "Subordinated": 0.75,
    }

    # Downturn scalars by security type
    DOWNTURN_SCALARS = {
        "Property": 1.15,
        "PPSR - P&E": 1.20,
        "PPSR - Receivables": 1.20,
        "PPSR - Mixed": 1.20,
        "GSR Only": 1.15,
    }

    DEFAULT_MOC = 0.03  # 3 percentage points

    def __init__(self, moc=None, parameter_manager=None, scenario_id="baseline"):
        self.parameter_manager = parameter_manager or OverlayParameterManager()
        self.scenario_id = scenario_id
        self.moc = moc or self.parameter_manager.get_value(
            "commercial", "base_moc", default=self.DEFAULT_MOC
        )

    @staticmethod
    def segment_loans(df):
        """
        Create segmentation columns for commercial portfolio.

        Segments:
          Level 1: security_type
          Level 2: coverage_band
          Level 3: industry_risk_band (if industry_risk_score present)
          Level 4: borrower_size
        """
        out = df.copy()

        # Security coverage bands
        bins = [0, 0.50, 0.80, 1.00, 1.20, float("inf")]
        labels = ["<50%", "50-80%", "80-100%", "100-120%", "120%+"]
        out["coverage_band"] = pd.cut(
            out["security_coverage_ratio"], bins=bins, labels=labels, right=True
        )

        # Borrower size
        rev_bins = [0, 5e6, 20e6, 75e6, float("inf")]
        rev_labels = ["Micro (<$5M)", "Small ($5-20M)", "Mid ($20-75M)", "Large (>$75M)"]
        out["borrower_size"] = pd.cut(
            out["annual_revenue"], bins=rev_bins, labels=rev_labels, right=True
        )

        # Industry risk band (from industry analysis integration)
        if "industry_risk_score" in out.columns:
            risk_bins = [0, 2.5, 3.0, 5.0]
            risk_labels = ["Low", "Medium", "Elevated"]
            out["industry_risk_band"] = pd.cut(
                out["industry_risk_score"], bins=risk_bins,
                labels=risk_labels, right=True
            )

        return apply_standard_segments(out, product="commercial")

    def compute_long_run_lgd(self, df, segments=None):
        """Compute exposure-weighted long-run LGD by segment."""
        if segments is None:
            segments = ["security_type", "coverage_band"]
        segmented = self.segment_loans(df)
        return exposure_weighted_average(
            segmented, lgd_col="realised_lgd", ead_col="ead", group_col=segments
        )

    def compute_weighted_outputs(self, df, group_cols=None):
        """
        Compute weighted base/downturn/final LGD outputs for commercial.
        """
        if group_cols is None:
            group_cols = ["security_type", "industry"]
        segmented = self.segment_loans(df)
        return build_weighted_lgd_output(
            segmented,
            group_cols=group_cols,
            base_col="lgd_base",
            downturn_col="lgd_downturn",
            final_col="lgd_final",
        )

    def apply_overlays(self, df):
        """
        Apply downturn, MoC, and regulatory overlays with industry risk adjustments.

        Enhanced pipeline:
          1. Apply industry recovery haircut to realised LGD (if available)
             and set base LGD (`lgd_base`)
          2. Map base downturn scalar by security type, then adjust for industry risk
          3. Apply industry-adjusted MoC
          4. Add working capital LGD adjustment (if available)
          5. Floor to supervisory LGD by seniority
          6. Cap at 100%
        """
        out = df.copy()
        has_industry = "industry_risk_score" in out.columns

        # Governance hook: discount-rate fallback source tracking.
        discount_rate_proxy_used, discount_rate_source = _resolve_discount_rate_proxy(
            out, product_baseline=0.06
        )
        out["discount_rate_proxy_used"] = discount_rate_proxy_used
        out["discount_rate_source"] = discount_rate_source
        out["icr_driver"] = _coerce_numeric_series(out, "icr")
        out["dscr_driver"] = _coerce_numeric_series(out, "dscr")
        out["industry_driver"] = out.get(
            "industry", pd.Series("Unknown", index=out.index)
        ).astype(str)

        # Step 1: Industry recovery haircut (raises effective LGD pre-overlays)
        if has_industry:
            out["industry_recovery_haircut"] = compute_industry_recovery_haircut(
                out["industry_risk_score"]
            )
            out["lgd_industry_adjusted"] = (
                out["realised_lgd"] + out["industry_recovery_haircut"]
            )
        else:
            out["lgd_industry_adjusted"] = out["realised_lgd"]
        out["lgd_base"] = out["lgd_industry_adjusted"].clip(0, 1)

        # Step 2: Canonical downturn overlay resolver
        base_scalar = _resolve_base_scalar_from_parameters(out, "commercial", self.parameter_manager)
        overlay_detail = resolve_overlay_contract(
            out,
            product="commercial",
            parameter_manager=self.parameter_manager,
            scenario_id=self.scenario_id,
            base_scalar=base_scalar,
        )
        out["macro_downturn_scalar"] = overlay_detail["macro_downturn_scalar"]
        out["industry_downturn_adjustment"] = overlay_detail["industry_downturn_adjustment"]
        out["downturn_scalar"] = overlay_detail["combined_downturn_scalar"]
        out["combined_downturn_scalar"] = overlay_detail["combined_downturn_scalar"]
        out["overlay_source"] = overlay_detail["overlay_source"]
        out["parameter_version"] = overlay_detail["parameter_version"]
        out["scenario_id"] = overlay_detail["scenario_id"]
        out["lgd_downturn"] = out["lgd_base"] * out["downturn_scalar"]

        # Step 3: MoC (adjusted by industry risk)
        beta = self.parameter_manager.get_value("all", "industry_moc_beta", default=0.20)
        if has_industry:
            out["industry_moc"] = compute_industry_moc_adjustment(
                out["industry_risk_score"], self.moc, beta=beta
            )
        else:
            out["industry_moc"] = self.moc
        out["lgd_with_moc"] = out["lgd_downturn"] + out["industry_moc"]

        # Step 4: Working capital overlay (additive)
        if "wc_lgd_overlay_score" in out.columns:
            out["wc_lgd_adjustment"] = compute_working_capital_lgd_adjustment(
                out["wc_lgd_overlay_score"]
            )
            out["lgd_with_moc"] = out["lgd_with_moc"] + out["wc_lgd_adjustment"]

        # Step 4b: Simplified cure overlay proxy for reporting.
        secured_mask = out["security_type"].isin(
            ["Property", "PPSR - P&E", "PPSR - Receivables", "PPSR - Mixed"]
        )
        coverage = _coerce_numeric_series(
            out, "security_coverage_ratio", default=1.0
        ).fillna(1.0)
        icr = _coerce_numeric_series(out, "icr", default=1.5).fillna(1.5)
        workout = _coerce_numeric_series(out, "workout_months", default=18).fillna(18)
        cure_proxy = (
            0.08
            + 0.20 * (coverage - 0.80).clip(lower=0, upper=0.60)
            + 0.10 * ((icr - 1.20) / 1.80).clip(lower=0, upper=1.0)
            - 0.10 * ((workout - 18) / 24).clip(lower=0, upper=1.0)
        )
        cure_proxy = np.where(secured_mask, cure_proxy, cure_proxy * 0.35)
        out["cure_overlay_rate_proxy"] = pd.Series(cure_proxy, index=out.index).clip(0.0, 0.30)
        out["liquidation_loss_proxy"] = out["lgd_with_moc"].clip(0, 1)
        out["lgd_after_cure_overlay_proxy"] = (
            (1 - out["cure_overlay_rate_proxy"]) * out["liquidation_loss_proxy"]
        ).clip(0, 1)
        out["cure_overlay_applied_flag"] = True
        out["cure_overlay_source"] = np.where(
            secured_mask,
            "secured_proxy",
            "dampened_unsecured_proxy",
        )

        # Step 5: Supervisory LGD floor by seniority
        out["supervisory_lgd"] = out["seniority"].map(self.SUPERVISORY_LGD).fillna(0.45)
        out["lgd_with_moc"] = np.maximum(out["lgd_with_moc"], out["supervisory_lgd"])

        # Step 6: Final -- cap at 100%
        out["lgd_final"] = out["lgd_with_moc"].clip(0, 1)

        _validate_final_lgd(out, product="commercial")
        return apply_standard_segments(out, product="commercial")

    @staticmethod
    def sme_firm_size_adjustment(revenue_millions):
        """
        Basel/APRA SME firm-size adjustment to correlation parameter.

        Adjustment = -0.04 * (1 - (S - 5) / 45)
        where S = annual revenue in AUD millions, capped at 5-50.
        """
        s = np.clip(revenue_millions, 5, 50)
        return -0.04 * (1 - (s - 5) / 45)


# ==========================================================================
# 3. DEVELOPMENT FINANCE LGD ENGINE
# ==========================================================================

class DevelopmentLGDEngine:
    """
    Australian development finance LGD engine.

    Implements:
    - Completion-stage segmentation (primary LGD driver)
    - Fund-to-complete scenario analysis
    - Pre-sale rescission risk
    - Downturn overlay by completion stage
    - Margin of conservatism (highest of all products)
    - Specialised lending / slotting framework
    """

    # Downturn scalars by completion stage
    DOWNTURN_SCALARS = {
        "Pre-Construction": 1.20,
        "Early Construction": 1.25,
        "Mid-Construction": 1.30,
        "Near-Complete": 1.15,
        "Complete Unsold": 1.20,
    }

    DEFAULT_MOC = 0.05  # 5 percentage points (highest due to data limitations)

    # APRA slotting categories (supervisory risk weights)
    SLOTTING_RW = {
        "Strong": 0.70,
        "Good": 0.90,
        "Satisfactory": 1.15,
        "Weak": 2.50,
        "Default": 0.0,  # deducted from capital
    }

    # HVCRE higher correlation multiplier
    HVCRE_CORRELATION_MULTIPLIER = 1.25

    def __init__(self, moc=None, parameter_manager=None, scenario_id="baseline"):
        self.parameter_manager = parameter_manager or OverlayParameterManager()
        self.scenario_id = scenario_id
        self.moc = moc or self.parameter_manager.get_value(
            "development", "base_moc", default=self.DEFAULT_MOC
        )

    @staticmethod
    def segment_loans(df):
        """
        Create segmentation columns for development portfolio.

        Segments:
          Level 1: completion_stage (primary driver)
          Level 2: development_type
          Level 3: presale_band
          Level 4: lvr_band
          Level 5: industry_risk_band (if available)
        """
        out = df.copy()

        # Pre-sale coverage bands
        ps_bins = [0, 0.30, 0.60, 0.80, 1.0, float("inf")]
        ps_labels = ["<30%", "30-60%", "60-80%", "80-100%", "100%+"]
        out["presale_band"] = pd.cut(
            out["presale_coverage"], bins=ps_bins, labels=ps_labels, right=True
        )

        # LVR bands (as-if-complete)
        lvr_bins = [0, 0.55, 0.65, 0.75, float("inf")]
        lvr_labels = ["<55%", "55-65%", "65-75%", "75%+"]
        out["lvr_band"] = pd.cut(
            out["lvr_as_if_complete"], bins=lvr_bins, labels=lvr_labels, right=True
        )

        # Industry risk band (from development type -> industry mapping)
        if "industry_risk_score" in out.columns:
            risk_bins = [0, 2.5, 3.0, 5.0]
            risk_labels = ["Low", "Medium", "Elevated"]
            out["industry_risk_band"] = pd.cut(
                out["industry_risk_score"], bins=risk_bins,
                labels=risk_labels, right=True
            )

        return apply_standard_segments(out, product="development")

    def compute_long_run_lgd(self, df, segments=None):
        """Compute exposure-weighted long-run LGD by segment."""
        if segments is None:
            segments = ["completion_stage"]
        segmented = self.segment_loans(df)
        return exposure_weighted_average(
            segmented, lgd_col="realised_lgd", ead_col="ead", group_col=segments
        )

    def compute_weighted_outputs(self, df, group_cols=None):
        """
        Compute weighted base/downturn/final LGD outputs for development.
        """
        if group_cols is None:
            group_cols = ["completion_stage", "development_type"]
        segmented = self.segment_loans(df)
        return build_weighted_lgd_output(
            segmented,
            group_cols=group_cols,
            base_col="lgd_base",
            downturn_col="lgd_downturn",
            final_col="lgd_final",
        )

    def apply_overlays(self, df):
        """
        Apply downturn, MoC, and regulatory overlays with industry risk adjustments.

        Enhanced pipeline:
          1. Apply industry recovery haircut (if available)
             and set base LGD (`lgd_base`)
          2. Map base downturn scalar by completion stage, adjust for industry risk
          3. Apply industry-adjusted MoC
          4. Cap at 100%
        """
        out = df.copy()
        has_industry = "industry_risk_score" in out.columns
        discount_rate_proxy_used, discount_rate_source = _resolve_discount_rate_proxy(
            out, product_baseline=0.07
        )
        out["discount_rate_proxy_used"] = discount_rate_proxy_used
        out["discount_rate_source"] = discount_rate_source
        out["grv_driver"] = _coerce_numeric_series(out, "grv")
        out["completion_pct_driver"] = _coerce_numeric_series(out, "completion_pct")
        out["cost_to_complete_driver"] = _coerce_numeric_series(out, "cost_to_complete")

        # Step 1: Industry recovery haircut
        if has_industry:
            out["industry_recovery_haircut"] = compute_industry_recovery_haircut(
                out["industry_risk_score"]
            )
            out["lgd_industry_adjusted"] = (
                out["realised_lgd"] + out["industry_recovery_haircut"]
            )
        else:
            out["lgd_industry_adjusted"] = out["realised_lgd"]
        out["lgd_base"] = out["lgd_industry_adjusted"].clip(0, 1)

        # Step 2: Canonical downturn overlay resolver
        base_scalar = _resolve_base_scalar_from_parameters(out, "development", self.parameter_manager)
        overlay_detail = resolve_overlay_contract(
            out,
            product="development",
            parameter_manager=self.parameter_manager,
            scenario_id=self.scenario_id,
            base_scalar=base_scalar,
        )
        out["macro_downturn_scalar"] = overlay_detail["macro_downturn_scalar"]
        out["industry_downturn_adjustment"] = overlay_detail["industry_downturn_adjustment"]
        out["downturn_scalar"] = overlay_detail["combined_downturn_scalar"]
        out["combined_downturn_scalar"] = overlay_detail["combined_downturn_scalar"]
        out["overlay_source"] = overlay_detail["overlay_source"]
        out["parameter_version"] = overlay_detail["parameter_version"]
        out["scenario_id"] = overlay_detail["scenario_id"]
        out["lgd_downturn"] = out["lgd_base"] * out["downturn_scalar"]

        # Step 3: MoC (adjusted by industry risk)
        beta = self.parameter_manager.get_value("all", "industry_moc_beta", default=0.20)
        if has_industry:
            out["industry_moc"] = compute_industry_moc_adjustment(
                out["industry_risk_score"], self.moc, beta=beta
            )
        else:
            out["industry_moc"] = self.moc
        out["lgd_with_moc"] = out["lgd_downturn"] + out["industry_moc"]

        # Step 3b: Simplified cure overlay proxy for reporting.
        stage_base = {
            "Pre-Construction": 0.03,
            "Early Construction": 0.05,
            "Mid-Construction": 0.08,
            "Near-Complete": 0.14,
            "Complete Unsold": 0.10,
        }
        stage_component = out["completion_stage"].map(stage_base).fillna(0.05)
        presale = _coerce_numeric_series(out, "presale_coverage", default=0.50).fillna(0.50)
        lvr = _coerce_numeric_series(out, "lvr_as_if_complete", default=0.70).fillna(0.70)
        cure_proxy = (
            stage_component
            + 0.12 * (presale - 0.50).clip(lower=0, upper=0.50)
            - 0.15 * (lvr - 0.70).clip(lower=0, upper=0.40)
        ).clip(0.01, 0.25)
        out["cure_overlay_rate_proxy"] = cure_proxy
        out["liquidation_loss_proxy"] = out["lgd_with_moc"].clip(0, 1)
        out["lgd_after_cure_overlay_proxy"] = (
            (1 - out["cure_overlay_rate_proxy"]) * out["liquidation_loss_proxy"]
        ).clip(0, 1)
        out["cure_overlay_applied_flag"] = True
        out["cure_overlay_source"] = "development_stage_presale_lvr_proxy"

        # Step 4: Cap at 100%
        out["lgd_final"] = out["lgd_with_moc"].clip(0, 1)

        _validate_final_lgd(out, product="development")
        return apply_standard_segments(out, product="development")

    def assign_slotting(self, df):
        """
        Assign APRA slotting categories based on project characteristics.

        Heuristic scoring:
          - Completion stage
          - Pre-sale coverage
          - LVR
          - Sponsor track record proxy (years_in_business not available, use unit_count as scale proxy)

        Returns DataFrame with 'slotting_category' and 'slotting_rw' columns.
        """
        out = df.copy()
        score = np.zeros(len(df))

        # Completion: later is better
        stage_scores = {
            "Pre-Construction": 0,
            "Early Construction": 1,
            "Mid-Construction": 2,
            "Near-Complete": 4,
            "Complete Unsold": 3,
        }
        score += out["completion_stage"].map(stage_scores).fillna(1).values

        # Pre-sales: higher is better
        score += np.where(out["presale_coverage"] > 0.80, 3,
                 np.where(out["presale_coverage"] > 0.60, 2,
                 np.where(out["presale_coverage"] > 0.30, 1, 0)))

        # LVR: lower is better
        score += np.where(out["lvr_as_if_complete"] < 0.55, 3,
                 np.where(out["lvr_as_if_complete"] < 0.65, 2,
                 np.where(out["lvr_as_if_complete"] < 0.75, 1, 0)))

        # Industry risk: lower risk is better (0-2 points)
        if "industry_risk_score" in out.columns:
            score += np.where(out["industry_risk_score"] < 2.5, 2,
                     np.where(out["industry_risk_score"] < 3.0, 1, 0))

        # Map to categories
        out["slotting_score"] = score
        out["slotting_category"] = np.where(
            score >= 9, "Strong",
            np.where(score >= 7, "Good",
            np.where(score >= 4, "Satisfactory",
                     "Weak"))
        )
        out["slotting_rw"] = out["slotting_category"].map(self.SLOTTING_RW)

        return out

    def scenario_analysis(self, df, grv_decline=0.0, cost_overrun=0.0,
                          rescission_rate=0.0, sales_extension_months=0):
        """
        Run scenario analysis on development LGD.

        Adjusts key drivers and recomputes approximate LGD impact:
          - grv_decline: fractional decline in GRV (e.g., -0.20 for 20% drop)
          - cost_overrun: fractional increase in cost-to-complete (e.g., 0.10 for 10%)
          - rescission_rate: fraction of pre-sales that rescind
          - sales_extension_months: additional months to sell (increases holding/discount cost)

        Returns DataFrame with scenario LGD estimates.
        """
        out = df.copy()

        stressed_grv = out["grv"] * (1 + grv_decline)
        stressed_ctc = out["cost_to_complete"] * (1 + cost_overrun)
        stressed_presale = out["presale_value"] * (1 - rescission_rate)

        # Approximate stressed recovery
        recovery_if_ftc = np.where(
            out["fund_to_complete"] == 1,
            stressed_grv * 0.90 - stressed_ctc,
            out["as_is_value"] * (1 + grv_decline) * 0.75,
        )
        recovery_if_ftc = np.maximum(recovery_if_ftc, 0)

        # Additional discount for extended sales period
        extra_discount = (1 + out["discount_rate"]) ** (sales_extension_months / 12)
        recovery_pv = recovery_if_ftc / extra_discount

        # Approximate stressed costs (holding + receiver + legal)
        stressed_costs = out["ead"] * 0.08  # ~8% total costs

        econ_loss = out["ead"] + stressed_costs - recovery_pv
        out["scenario_lgd"] = (econ_loss / out["ead"]).clip(0, 1)

        out["scenario_description"] = (
            f"GRV {grv_decline*100:+.0f}%, CTC {cost_overrun*100:+.0f}%, "
            f"Rescission {rescission_rate*100:.0f}%, +{sales_extension_months}mo sales"
        )

        return out


# ==========================================================================
# 4. CASH FLOW LENDING LGD ENGINE (PD-aligned)
# ==========================================================================

class CashFlowLendingLGDEngine:
    """
    Australian cash flow lending LGD engine aligned with PD Scorecard.

    Implements:
    - PD score band segmentation (A-E from WoE logistic regression)
    - Product-type segmentation (8 cash flow lending products)
    - PD-band-adjusted downturn add-ons
    - DSCR stress overlay
    - Conduct overlay (Green / Amber / Red)
    - Industry risk adjustments (shared with CommercialLGDEngine)
    - Supervisory LGD floors by product and seniority

    Key principle: PD and LGD must be internally consistent -- higher PD
    borrowers face systematically worse recovery outcomes due to weaker
    cash flow capacity during workout.
    """

    # Supervisory LGD floors (APS 112 fallback)
    SUPERVISORY_LGD = {
        "Senior Secured": 0.35,
        "Senior Unsecured": 0.45,
        "Subordinated": 0.75,
    }

    # Product-specific supervisory floors (unsecured cash flow products
    # warrant higher floors than collateralised commercial)
    PRODUCT_LGD_FLOOR = {
        "Business Term Loan": 0.45,
        "Working Capital Facility": 0.50,
        "Trade Finance": 0.40,
        "Equipment Finance": 0.35,
        "Invoice Finance": 0.35,
        "Merchant Cash Advance": 0.55,
        "Business Line of Credit": 0.50,
        "Professional Practice Loan": 0.40,
    }

    # Base downturn scalars by product type
    BASE_DOWNTURN_SCALARS = {
        "Business Term Loan": 1.15,
        "Working Capital Facility": 1.20,
        "Trade Finance": 1.10,
        "Equipment Finance": 1.10,
        "Invoice Finance": 1.10,
        "Merchant Cash Advance": 1.25,
        "Business Line of Credit": 1.20,
        "Professional Practice Loan": 1.10,
    }

    # PD-band downturn add-ons: worse PD -> more severe downturn impact
    PD_BAND_DOWNTURN_ADDON = {
        "A": -0.03,
        "B": -0.01,
        "C":  0.00,
        "D":  0.03,
        "E":  0.05,
    }

    # MoC multiplier by PD band (higher PD -> more model uncertainty)
    MOC_PD_MULTIPLIER = {
        "A": 0.85,
        "B": 0.95,
        "C": 1.00,
        "D": 1.10,
        "E": 1.25,
    }

    # Conduct overlay (additive, in decimal)
    CONDUCT_OVERLAY = {
        "Green": 0.000,
        "Amber": 0.005,
        "Red":   0.015,
    }

    DEFAULT_MOC = 0.04  # 4pp base (higher than secured commercial)

    def __init__(self, moc=None, parameter_manager=None, scenario_id="baseline"):
        self.parameter_manager = parameter_manager or OverlayParameterManager()
        self.scenario_id = scenario_id
        self.moc = moc or self.parameter_manager.get_value(
            "cashflow_lending", "base_moc", default=self.DEFAULT_MOC
        )

    @staticmethod
    def segment_loans(df):
        """
        Create segmentation columns for cash flow lending portfolio.

        Segments:
          Level 1: pd_score_band (A-E)
          Level 2: cashflow_product
          Level 3: industry_risk_band
          Level 4: conduct_classification
        """
        out = df.copy()

        # DSCR bands
        dscr_bins = [0, 1.0, 1.3, 1.8, 2.5, float("inf")]
        dscr_labels = ["<1.0x", "1.0-1.3x", "1.3-1.8x", "1.8-2.5x", "2.5x+"]
        if "dscr" in out.columns:
            out["dscr_band"] = pd.cut(
                out["dscr"], bins=dscr_bins, labels=dscr_labels, right=True
            )

        # Bureau score bands
        if "bureau_score" in out.columns:
            bscore_bins = [0, 500, 580, 650, 720, 900]
            bscore_labels = ["<500", "500-580", "580-650", "650-720", "720+"]
            out["bureau_band"] = pd.cut(
                out["bureau_score"], bins=bscore_bins,
                labels=bscore_labels, right=True
            )

        # Industry risk band
        if "industry_risk_score" in out.columns:
            risk_bins = [0, 2.5, 3.0, 5.0]
            risk_labels = ["Low", "Medium", "Elevated"]
            out["industry_risk_band"] = pd.cut(
                out["industry_risk_score"], bins=risk_bins,
                labels=risk_labels, right=True
            )

        return apply_standard_segments(out, product="cashflow_lending")

    def compute_long_run_lgd(self, df, segments=None):
        """Compute exposure-weighted long-run LGD by segment."""
        if segments is None:
            segments = ["pd_score_band", "cashflow_product"]
        segmented = self.segment_loans(df)
        return exposure_weighted_average(
            segmented, lgd_col="realised_lgd", ead_col="ead", group_col=segments
        )

    def apply_overlays(self, df):
        """
        Apply PD-aligned downturn, MoC, and regulatory overlays.

        Enhanced pipeline:
          1. Industry recovery haircut (if available)
          2. PD-band-adjusted downturn scalar
          3. PD-band-adjusted MoC + industry adjustment
          4. DSCR stress overlay
          5. Conduct overlay
          6. Working capital adjustment (if available)
          7. Product-specific supervisory LGD floor
          8. Cap at 100%
        """
        out = df.copy()
        has_industry = "industry_risk_score" in out.columns

        # Step 1: Industry recovery haircut
        if has_industry:
            out["industry_recovery_haircut"] = compute_industry_recovery_haircut(
                out["industry_risk_score"]
            )
            out["lgd_industry_adjusted"] = (
                out["realised_lgd"] + out["industry_recovery_haircut"]
            )
        else:
            out["lgd_industry_adjusted"] = out["realised_lgd"]

        # Step 2: PD-band-adjusted base scalar + canonical overlay resolver
        base_scalar = _resolve_base_scalar_from_parameters(out, "cashflow_lending", self.parameter_manager)
        pd_addon_map = self.parameter_manager.get_map("cashflow_lending", "pd_band_downturn_addon", "pd_band:")
        pd_addon = out["pd_score_band"].map(pd_addon_map).fillna(0.0)
        effective_scalar = base_scalar + pd_addon

        overlay_detail = resolve_overlay_contract(
            out,
            product="cashflow_lending",
            parameter_manager=self.parameter_manager,
            scenario_id=self.scenario_id,
            base_scalar=effective_scalar,
        )
        out["macro_downturn_scalar"] = overlay_detail["macro_downturn_scalar"]
        out["industry_downturn_adjustment"] = overlay_detail["industry_downturn_adjustment"]
        out["downturn_scalar"] = overlay_detail["combined_downturn_scalar"]
        out["combined_downturn_scalar"] = overlay_detail["combined_downturn_scalar"]
        out["overlay_source"] = overlay_detail["overlay_source"]
        out["parameter_version"] = overlay_detail["parameter_version"]
        out["scenario_id"] = overlay_detail["scenario_id"]
        out["lgd_downturn"] = out["lgd_industry_adjusted"] * out["downturn_scalar"]

        # Step 3: PD-band-adjusted MoC + industry
        pd_moc_map = self.parameter_manager.get_map("cashflow_lending", "moc_pd_multiplier", "pd_band:")
        pd_moc_mult = out["pd_score_band"].map(pd_moc_map).fillna(1.0)
        base_moc = self.moc * pd_moc_mult
        beta = self.parameter_manager.get_value("all", "industry_moc_beta", default=0.20)
        if has_industry:
            out["industry_moc"] = compute_industry_moc_adjustment(
                out["industry_risk_score"], base_moc, beta=beta
            )
        else:
            out["industry_moc"] = base_moc
        out["lgd_with_moc"] = out["lgd_downturn"] + out["industry_moc"]

        # Step 4: DSCR stress overlay
        if "dscr" in out.columns:
            out["dscr_stress_overlay"] = np.maximum(
                (1.3 - out["dscr"]) * 0.02, 0
            )
            out["lgd_with_moc"] = out["lgd_with_moc"] + out["dscr_stress_overlay"]

        # Step 5: Conduct overlay
        if "conduct_classification" in out.columns:
            conduct_map = self.parameter_manager.get_map(
                "cashflow_lending", "conduct_overlay", "conduct:"
            )
            out["conduct_overlay"] = out["conduct_classification"].map(conduct_map).fillna(0.0)
            out["lgd_with_moc"] = out["lgd_with_moc"] + out["conduct_overlay"]

        # Step 6: Working capital adjustment
        if "wc_lgd_overlay_score" in out.columns:
            out["wc_lgd_adjustment"] = compute_working_capital_lgd_adjustment(
                out["wc_lgd_overlay_score"]
            )
            out["lgd_with_moc"] = out["lgd_with_moc"] + out["wc_lgd_adjustment"]

        # Step 7: Product-specific supervisory floor
        product_floor = out["cashflow_product"].map(
            self.PRODUCT_LGD_FLOOR
        ).fillna(0.45)
        seniority_floor = out["seniority"].map(
            self.SUPERVISORY_LGD
        ).fillna(0.45)
        out["supervisory_lgd"] = np.maximum(product_floor, seniority_floor)
        out["lgd_with_moc"] = np.maximum(out["lgd_with_moc"], out["supervisory_lgd"])

        # Step 8: Cap at 100%
        out["lgd_final"] = out["lgd_with_moc"].clip(0, 1)

        _validate_final_lgd(out, product="cashflow_lending")
        return apply_standard_segments(out, product="cashflow_lending")


# ==========================================================================
# GOVERNANCE REPORTING HOOKS
# ==========================================================================

def _summarise_usage_counts(df, product, topic, column):
    """Summarise usage counts and exposure by category for a reporting column."""
    if column not in df.columns:
        return pd.DataFrame(columns=[
            "product", "topic", "usage_value", "loan_count", "total_ead", "ead_share"
        ])
    work = df.copy()
    work["usage_value"] = work[column].astype(str).fillna("missing")
    if "ead" not in work.columns:
        work["ead"] = 1.0
    grouped = (
        work.groupby("usage_value", observed=True)
        .agg(loan_count=("usage_value", "size"), total_ead=("ead", "sum"))
        .reset_index()
    )
    total_ead = grouped["total_ead"].sum()
    grouped["ead_share"] = grouped["total_ead"] / total_ead if total_ead > 0 else 0.0
    grouped.insert(0, "topic", topic)
    grouped.insert(0, "product", product)
    return grouped


def build_fallback_usage_report(results):
    """
    Build fallback usage counts across products from overlay outputs.
    """
    rows = []
    topic_map = {
        "discount_rate_source": "discount_rate_fallback",
        "house_price_source": "house_price_fallback",
        "unemployment_source": "unemployment_fallback",
        "rate_shock_source": "rate_shock_fallback",
    }
    for product, payload in results.items():
        if not isinstance(payload, dict) or "loans_with_overlays" not in payload:
            continue
        df = payload["loans_with_overlays"]
        for col, topic in topic_map.items():
            part = _summarise_usage_counts(df, product, topic, col)
            if not part.empty:
                rows.append(part)
    if not rows:
        return pd.DataFrame(
            columns=["product", "topic", "usage_value", "loan_count", "total_ead", "ead_share"]
        )
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["product", "topic", "usage_value"]).reset_index(drop=True)


def build_unemployment_fallback_report(results):
    """
    Build mortgage year-bucket unemployment fallback report.
    """
    mtg = results.get("mortgage", {}).get("loans_with_overlays")
    if mtg is None or len(mtg) == 0:
        return pd.DataFrame(columns=[
            "year_bucket", "unemployment_source", "loan_count", "total_ead",
            "mean_unemployment_shock", "ead_weighted_unemployment_shock",
        ])

    df = mtg.copy()
    if "unemployment_year_bucket" not in df.columns:
        df["unemployment_year_bucket"] = _year_bucket_for_unemployment(df.get("default_date"))
    if "unemployment_source" not in df.columns:
        df["unemployment_source"] = "unreported"
    if "ead" not in df.columns:
        df["ead"] = 1.0
    if "unemployment_driver" not in df.columns:
        df["unemployment_driver"] = _year_to_unemployment_shock(df.get("default_date"))

    grouped = (
        df.groupby(["unemployment_year_bucket", "unemployment_source"], observed=True)
        .apply(
            lambda g: pd.Series(
                {
                    "loan_count": len(g),
                    "total_ead": pd.to_numeric(g["ead"], errors="coerce").fillna(0.0).sum(),
                    "mean_unemployment_shock": pd.to_numeric(
                        g["unemployment_driver"], errors="coerce"
                    ).fillna(0.0).mean(),
                    "ead_weighted_unemployment_shock": exposure_weighted_average(
                        g.assign(
                            unemployment_driver=pd.to_numeric(
                                g["unemployment_driver"], errors="coerce"
                            ).fillna(0.0)
                        ),
                        "unemployment_driver",
                        "ead",
                    ),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    grouped = grouped.rename(
        columns={"unemployment_year_bucket": "year_bucket"}
    )
    return grouped.sort_values(["year_bucket", "unemployment_source"]).reset_index(drop=True)


def build_proxy_flag_report(results):
    """
    Build proxy arrears and behavioural flag report for mortgage.
    """
    mtg = results.get("mortgage", {}).get("loans_with_overlays")
    if mtg is None or len(mtg) == 0:
        return pd.DataFrame(columns=[
            "flag_type", "flag_value", "loan_count", "total_ead", "ead_share"
        ])
    rows = [
        _summarise_usage_counts(
            mtg, "mortgage", "proxy_arrears_flag", "proxy_arrears_flag"
        ),
        _summarise_usage_counts(
            mtg, "mortgage", "proxy_behaviour_flag", "proxy_behaviour_flag"
        ),
    ]
    out = pd.concat(rows, ignore_index=True).rename(
        columns={"topic": "flag_type", "usage_value": "flag_value"}
    )
    return out[["flag_type", "flag_value", "loan_count", "total_ead", "ead_share"]]


def build_cure_overlay_flag_report(results):
    """
    Build cure overlay reporting flag summary across products.
    """
    rows = []
    for product in ["mortgage", "commercial", "development", "cashflow_lending"]:
        payload = results.get(product)
        if not isinstance(payload, dict):
            continue
        df = payload.get("loans_with_overlays")
        if df is None or len(df) == 0:
            continue
        part_flag = _summarise_usage_counts(
            df, product, "cure_overlay_applied_flag", "cure_overlay_applied_flag"
        )
        if not part_flag.empty:
            rows.append(part_flag)
        part_source = _summarise_usage_counts(
            df, product, "cure_overlay_source", "cure_overlay_source"
        )
        if not part_source.empty:
            rows.append(part_source)
    if not rows:
        return pd.DataFrame(columns=[
            "product", "topic", "usage_value", "loan_count", "total_ead", "ead_share"
        ])
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["product", "topic", "usage_value"]).reset_index(drop=True)


def build_overlay_trace_report(results):
    """
    Build compact trace report for canonical overlay outputs.
    """
    rows = []
    for product, payload in results.items():
        if not isinstance(payload, dict) or "loans_with_overlays" not in payload:
            continue
        df = payload["loans_with_overlays"]
        required = [
            "macro_downturn_scalar",
            "industry_downturn_adjustment",
            "combined_downturn_scalar",
            "overlay_source",
            "parameter_version",
            "scenario_id",
        ]
        if df is None or len(df) == 0 or not set(required).issubset(df.columns):
            continue
        ead = pd.to_numeric(df.get("ead", 1.0), errors="coerce").fillna(0.0)
        part = pd.DataFrame(
            {
                "product": product,
                "loan_count": [len(df)],
                "total_ead": [ead.sum()],
                "avg_macro_downturn_scalar": [pd.to_numeric(df["macro_downturn_scalar"], errors="coerce").mean()],
                "avg_industry_downturn_adjustment": [pd.to_numeric(df["industry_downturn_adjustment"], errors="coerce").mean()],
                "avg_combined_downturn_scalar": [pd.to_numeric(df["combined_downturn_scalar"], errors="coerce").mean()],
                "overlay_source_mode": [df["overlay_source"].astype(str).mode().iloc[0]],
                "parameter_version": [df["parameter_version"].astype(str).mode().iloc[0]],
                "scenario_id": [df["scenario_id"].astype(str).mode().iloc[0]],
            }
        )
        rows.append(part)

    if not rows:
        return pd.DataFrame(
            columns=[
                "product",
                "loan_count",
                "total_ead",
                "avg_macro_downturn_scalar",
                "avg_industry_downturn_adjustment",
                "avg_combined_downturn_scalar",
                "overlay_source_mode",
                "parameter_version",
                "scenario_id",
            ]
        )
    return pd.concat(rows, ignore_index=True).sort_values("product").reset_index(drop=True)


def build_run_metadata_report(results):
    rows = []
    meta = results.get("run_metadata", {})
    if not isinstance(meta, dict) or not meta:
        return pd.DataFrame(
            columns=[
                "seed",
                "parameter_version",
                "parameter_hash",
                "scenario_id",
                "generated_at_utc",
                "input_contract_check",
            ]
        )
    rows.append(
        {
            "seed": meta.get("seed"),
            "parameter_version": meta.get("parameter_version"),
            "parameter_hash": meta.get("parameter_hash"),
            "scenario_id": meta.get("scenario_id"),
            "generated_at_utc": meta.get("generated_at_utc"),
            "input_contract_check": meta.get("input_contract_check"),
        }
    )
    return pd.DataFrame(rows)


def build_governance_reporting_tables(results, parameter_manager=None):
    """
    Build all governance reporting tables used for remediation reporting.
    """
    parameter_report = (
        parameter_manager.build_parameter_version_report()
        if parameter_manager is not None
        else pd.DataFrame(
            columns=[
                "parameter_version",
                "parameter_hash",
                "parameter_file",
                "check",
                "status",
                "detail",
            ]
        )
    )
    return {
        "fallback_usage_report.csv": build_fallback_usage_report(results),
        "unemployment_year_bucket_report.csv": build_unemployment_fallback_report(results),
        "proxy_flags_report.csv": build_proxy_flag_report(results),
        "cure_overlay_report.csv": build_cure_overlay_flag_report(results),
        "overlay_trace_report.csv": build_overlay_trace_report(results),
        "parameter_version_report.csv": parameter_report,
        "segmentation_consistency_report.csv": build_segmentation_consistency_report(results),
        "run_metadata_report.csv": build_run_metadata_report(results),
    }


# ==========================================================================
# FULL PIPELINE: Run all products
# ==========================================================================

def run_full_pipeline(
    datasets,
    include_reporting=False,
    parameter_manager=None,
    scenario_id="baseline",
    seed=42,
):
    """
    Run the complete LGD pipeline for all products.

    Parameters
    ----------
    datasets : dict from generate_all_datasets()

    Returns
    -------
    dict with keys 'mortgage', 'commercial', 'development', 'cashflow_lending',
    each containing 'loans_with_overlays', 'segment_summary', and (for core
    products) 'weighted_output' DataFrames.
    When include_reporting=True, adds key 'reporting_tables' with governance
    remediation outputs.
    """
    params = parameter_manager or OverlayParameterManager()
    results = {}
    results["run_metadata"] = {
        "seed": int(seed),
        "parameter_version": params.meta.version,
        "parameter_hash": params.meta.parameter_hash,
        "scenario_id": str(scenario_id),
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "input_contract_check": "passed",
    }

    # --- Mortgage ---
    mtg = MortgageLGDEngine(parameter_manager=params, scenario_id=scenario_id)
    mtg_loans = datasets["mortgage"]["loans"]
    mtg_with_overlays = mtg.apply_apra_overlays(mtg_loans)
    mtg_with_overlays = mtg.compute_illustrative_rwa(mtg_with_overlays)
    mtg_segments = mtg.compute_long_run_lgd(mtg_loans)
    mtg_weighted = mtg.compute_weighted_outputs(mtg_with_overlays)
    results["mortgage"] = {
        "loans_with_overlays": mtg_with_overlays,
        "segment_summary": mtg_segments,
        "weighted_output": mtg_weighted,
    }

    # --- Commercial ---
    com = CommercialLGDEngine(parameter_manager=params, scenario_id=scenario_id)
    com_loans = datasets["commercial"]["loans"]
    com_with_overlays = com.apply_overlays(com_loans)
    com_segments = com.compute_long_run_lgd(com_loans)
    com_weighted = com.compute_weighted_outputs(com_with_overlays)
    results["commercial"] = {
        "loans_with_overlays": com_with_overlays,
        "segment_summary": com_segments,
        "weighted_output": com_weighted,
    }

    # --- Development ---
    dev = DevelopmentLGDEngine(parameter_manager=params, scenario_id=scenario_id)
    dev_loans = datasets["development"]["loans"]
    dev_with_overlays = dev.apply_overlays(dev_loans)
    dev_with_overlays = dev.assign_slotting(dev_with_overlays)
    dev_segments = dev.compute_long_run_lgd(dev_loans)
    dev_weighted = dev.compute_weighted_outputs(dev_with_overlays)
    results["development"] = {
        "loans_with_overlays": dev_with_overlays,
        "segment_summary": dev_segments,
        "weighted_output": dev_weighted,
    }

    # --- Cash Flow Lending ---
    if "cashflow_lending" in datasets:
        cfl = CashFlowLendingLGDEngine(parameter_manager=params, scenario_id=scenario_id)
        cfl_loans = datasets["cashflow_lending"]["loans"]
        cfl_with_overlays = cfl.apply_overlays(cfl_loans)
        cfl_segments = cfl.compute_long_run_lgd(cfl_loans)
        results["cashflow_lending"] = {
            "loans_with_overlays": cfl_with_overlays,
            "segment_summary": cfl_segments,
        }

    if include_reporting:
        results["reporting_tables"] = build_governance_reporting_tables(
            results, parameter_manager=params
        )

    return results

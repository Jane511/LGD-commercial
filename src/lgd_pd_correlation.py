"""
LGD-PD Correlation — Frye-Jacobs Systematic Factor Model.

Models the dependence between Loss Given Default (LGD) and Probability of
Default (PD) through common macro-economic factors. Both LGD and PD are
driven by the same systematic risk factors (GDP, unemployment, credit spreads),
causing LGD to be elevated precisely when defaults are highest — a double
adverse effect that must be captured in downturn LGD calibration.

Method: Frye-Jacobs (2001), adapted for APS 113 s.55-57
=========================================================
    1. Regress annual realised LGD on macro factors Z → residuals ε_L
    2. Regress annual default rate on macro factors Z → residuals ε_D
    3. ρ = Corr(ε_L, ε_D)
    4. Correlation-adjusted downturn LGD:
         LGD_dt_adj = LGD_LR × (1 + ρ × macro_shock_std)

The correlation adjustment is applied BEFORE MoC and after the base
downturn overlay, as an additional sensitivity test rather than a
mandatory uplift (Australian APRA practice treats this as a stress scenario
rather than a base parameter in most frameworks).

Data dependency:
    - Annual realised LGD: from compute_realised_lgd() aggregated by year
    - Annual default rates: from generators or industry-analysis repo
    - Macro factors: from macro_regime_flags.parquet (real RBA/ABS data)

APS 113 References:
    s.55-57: Downturn LGD estimation methodology
    s.43: Through-the-cycle consideration

References:
    Frye, J., Jacobs, M. (2012). "Credit Loss and Systematic LGD."
    Journal of Credit Risk 8(1):109-140.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:
    from sklearn.linear_model import LinearRegression
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "scikit-learn not available. LGD-PD correlation will use numpy OLS fallback."
    )

logger = logging.getLogger(__name__)


def estimate_lgd_pd_correlation(
    lgd_time_series: pd.DataFrame,
    pd_time_series: pd.DataFrame,
    macro_factors: pd.DataFrame,
    lgd_col: str = "realised_lgd_ewa",
    pd_col: str = "default_rate",
    year_col: str = "year",
    macro_factor_cols: list[str] | None = None,
    min_years: int = 5,
) -> dict:
    """
    Estimate the LGD-PD systematic correlation (rho) using common macro factors.

    Parameters
    ----------
    lgd_time_series : annual EWA realised LGD per year
        Required columns: year, lgd_col
    pd_time_series : annual observed default rate per year
        Required columns: year, pd_col
    macro_factors : annual macro time series (from regime_classifier or upstream)
        Required columns: year + at least one of macro_factor_cols
    lgd_col : column name for LGD in lgd_time_series
    pd_col : column name for default rate in pd_time_series
    year_col : year column name in all DataFrames
    macro_factor_cols : list of columns to use as regressors.
        Defaults to: ['gdp_growth_yoy', 'unemployment_rate', 'credit_spread_bps']
    min_years : minimum years of data required for regression

    Returns
    -------
    dict with:
        rho (float) : Pearson correlation between LGD and PD residuals
        rho_ci (tuple) : 95% confidence interval for rho
        lgd_factor_r2 (float) : R² of LGD regression on macro factors
        pd_factor_r2 (float) : R² of PD regression on macro factors
        macro_shock_std (float) : standard deviation of composite macro shock
        lgd_dt_adjustment_factor (float) : 1 + rho * macro_shock_std
        regression_summary (DataFrame) : factor loadings for both regressions
        n_years (int) : years used in regression
        interpretation (str) : human-readable interpretation
        aps113_note (str) : APS 113 reference
        data_source (str) : source of macro data

    Raises
    ------
    ValueError if fewer than min_years of matching data available.
    """
    if macro_factor_cols is None:
        macro_factor_cols = [
            col for col in ["gdp_growth_yoy", "unemployment_rate", "credit_spread_bps"]
            if col in macro_factors.columns
        ]
    if not macro_factor_cols:
        raise ValueError(
            "No macro factor columns found in macro_factors DataFrame. "
            f"Available: {list(macro_factors.columns)}"
        )

    # Merge all three series on year
    merged = (
        lgd_time_series[[year_col, lgd_col]]
        .merge(pd_time_series[[year_col, pd_col]], on=year_col, how="inner")
        .merge(macro_factors[[year_col] + macro_factor_cols], on=year_col, how="inner")
        .dropna()
    )

    if len(merged) < min_years:
        raise ValueError(
            f"estimate_lgd_pd_correlation: only {len(merged)} years of matching data. "
            f"Minimum required: {min_years}. "
            "Check that lgd_time_series, pd_time_series, and macro_factors share common years."
        )

    logger.info(
        "LGD-PD correlation: %d years | macro factors: %s",
        len(merged), macro_factor_cols,
    )

    X = merged[macro_factor_cols].values
    y_lgd = merged[lgd_col].values
    y_pd = merged[pd_col].values

    # Normalise macro factors
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # Regress LGD and PD on macro factors
    resid_lgd, r2_lgd, coef_lgd = _ols_regression(X_std, y_lgd)
    resid_pd, r2_pd, coef_pd = _ols_regression(X_std, y_pd)

    # Correlation between residuals
    rho, p_value = stats.pearsonr(resid_lgd, resid_pd)
    n = len(merged)
    # 95% CI for rho using Fisher z-transformation
    rho_ci = _pearson_ci(rho, n, alpha=0.05)

    # Macro shock standard deviation (composite)
    macro_shock_std = float(np.std(np.dot(X_std, coef_lgd)))

    # Frye-Jacobs adjustment factor
    adj_factor = 1.0 + rho * macro_shock_std

    # Regression summary
    reg_summary = pd.DataFrame({
        "macro_factor": macro_factor_cols,
        "lgd_beta": coef_lgd.flatten(),
        "pd_beta": coef_pd.flatten(),
    })

    interpretation = _interpret_rho(rho, r2_lgd, r2_pd)
    data_source = str(macro_factors.get("data_source", pd.Series(["unknown"])).iloc[0]) if "data_source" in macro_factors.columns else "unknown"

    result = {
        "rho": round(float(rho), 6),
        "rho_p_value": round(float(p_value), 6),
        "rho_ci": (round(rho_ci[0], 6), round(rho_ci[1], 6)),
        "lgd_factor_r2": round(float(r2_lgd), 6),
        "pd_factor_r2": round(float(r2_pd), 6),
        "macro_shock_std": round(float(macro_shock_std), 6),
        "lgd_dt_adjustment_factor": round(float(adj_factor), 6),
        "regression_summary": reg_summary,
        "n_years": int(n),
        "macro_factor_cols": macro_factor_cols,
        "interpretation": interpretation,
        "aps113_note": (
            "LGD-PD correlation estimated per Frye-Jacobs (2001) approach. "
            "Informs downturn LGD sensitivity. APS 113 s.55-57."
        ),
        "data_source": data_source,
    }

    logger.info(
        "LGD-PD correlation: rho=%.3f (p=%.3f) | adj_factor=%.3f | "
        "LGD R²=%.2f | PD R²=%.2f",
        rho, p_value, adj_factor, r2_lgd, r2_pd,
    )
    return result


def apply_correlation_adjustment(
    downturn_lgd: pd.Series,
    rho: float,
    macro_shock_std: float,
    cap_adjustment: float = 0.10,
) -> pd.Series:
    """
    Apply Frye-Jacobs correlation adjustment to downturn LGD.

    Larger rho → higher downturn LGD (LGD rises when defaults also rise).
    This is applied after the base downturn overlay and before MoC.

    Parameters
    ----------
    downturn_lgd : Series of base downturn LGD values
    rho : LGD-PD correlation coefficient from estimate_lgd_pd_correlation()
    macro_shock_std : standard deviation of composite macro shock
    cap_adjustment : maximum uplift as absolute add-on (default: 10pp)

    Returns
    -------
    Series: downturn_lgd × max(1.0, adj_factor), clipped by cap.

    APS 113 s.55-57: Downturn estimation must account for relationships
    between default rates and loss rates.
    """
    adj_factor = 1.0 + rho * macro_shock_std
    adj_factor = max(1.0, adj_factor)   # never reduce below base downturn LGD

    adjusted = downturn_lgd * adj_factor
    # Apply cap: don't increase by more than cap_adjustment in absolute terms
    adjusted = adjusted.clip(upper=downturn_lgd + cap_adjustment)

    n_affected = (adjusted > downturn_lgd).sum()
    logger.info(
        "apply_correlation_adjustment: rho=%.3f | adj_factor=%.3f | "
        "%d/%d loans affected | mean uplift=%.2f%%",
        rho, adj_factor, n_affected, len(downturn_lgd),
        100 * (adjusted - downturn_lgd).mean(),
    )
    return adjusted


def build_lgd_pd_annual_series(
    loans: pd.DataFrame,
    lgd_col: str = "realised_lgd",
    ead_col: str = "ead_at_default",
    year_col: str = "default_year",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build annual LGD and default rate time series from loan-level data.

    Returns
    -------
    (lgd_ts, pd_ts): tuple of annual DataFrames
        lgd_ts: year, realised_lgd_ewa, n_defaults, total_ead
        pd_ts: year, default_rate (defaults / prior_year_exposure — proxy)
    """
    loans = loans.copy()
    loans[ead_col] = pd.to_numeric(loans[ead_col], errors="coerce").fillna(0)
    loans[lgd_col] = pd.to_numeric(loans[lgd_col], errors="coerce")
    loans[year_col] = pd.to_numeric(loans[year_col], errors="coerce")

    lgd_ts_rows = []
    for year, grp in loans.groupby(year_col):
        valid = grp[[lgd_col, ead_col]].dropna()
        if valid.empty:
            continue
        total_ead = valid[ead_col].sum()
        ewa_lgd = float((valid[lgd_col] * valid[ead_col]).sum() / total_ead) if total_ead > 0 else float(valid[lgd_col].mean())
        lgd_ts_rows.append({
            "year": int(year),
            "realised_lgd_ewa": round(ewa_lgd, 6),
            "n_defaults": len(valid),
            "total_ead": round(total_ead, 2),
        })

    lgd_ts = pd.DataFrame(lgd_ts_rows).sort_values("year").reset_index(drop=True)

    # Default rate proxy: n_defaults / cumulative_loans (rough)
    total_n = len(loans)
    pd_ts = lgd_ts[["year", "n_defaults"]].copy()
    pd_ts["default_rate"] = (pd_ts["n_defaults"] / (total_n / pd_ts["year"].nunique())).clip(0, 1)
    pd_ts = pd_ts[["year", "default_rate"]].copy()

    return lgd_ts, pd_ts


def export_correlation_report(
    correlation_result: dict,
    output_path: str | Path,
    product: str = "",
) -> pd.DataFrame:
    """Export LGD-PD correlation results as a CSV report."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    row = {
        "product": product,
        "rho": correlation_result["rho"],
        "rho_lower_95ci": correlation_result["rho_ci"][0],
        "rho_upper_95ci": correlation_result["rho_ci"][1],
        "rho_p_value": correlation_result["rho_p_value"],
        "lgd_factor_r2": correlation_result["lgd_factor_r2"],
        "pd_factor_r2": correlation_result["pd_factor_r2"],
        "macro_shock_std": correlation_result["macro_shock_std"],
        "lgd_dt_adjustment_factor": correlation_result["lgd_dt_adjustment_factor"],
        "n_years": correlation_result["n_years"],
        "macro_factors_used": "|".join(correlation_result["macro_factor_cols"]),
        "data_source": correlation_result["data_source"],
        "interpretation": correlation_result["interpretation"],
        "aps113_note": correlation_result["aps113_note"],
    }
    report = pd.DataFrame([row])
    report.to_csv(output_path, index=False)
    logger.info("Exported LGD-PD correlation report to %s", output_path)
    return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ols_regression(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """OLS regression. Returns (residuals, R², coefficients)."""
    if _SKLEARN_AVAILABLE:
        model = LinearRegression(fit_intercept=True)
        model.fit(X, y)
        y_hat = model.predict(X)
        coef = model.coef_.reshape(-1, 1)
    else:
        # Numpy fallback
        X_int = np.column_stack([np.ones(len(X)), X])
        coef_full, _, _, _ = np.linalg.lstsq(X_int, y, rcond=None)
        y_hat = X_int @ coef_full
        coef = coef_full[1:].reshape(-1, 1)

    resid = y - y_hat
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return resid, float(r2), coef


def _pearson_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """95% confidence interval for Pearson correlation via Fisher z-transformation."""
    if n <= 3:
        return (-1.0, 1.0)
    z = np.arctanh(np.clip(r, -0.9999, 0.9999))
    se = 1.0 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    lower = np.tanh(z - z_crit * se)
    upper = np.tanh(z + z_crit * se)
    return float(lower), float(upper)


def _interpret_rho(rho: float, r2_lgd: float, r2_pd: float) -> str:
    """Generate human-readable interpretation of correlation results."""
    strength = (
        "strong positive" if rho > 0.5 else
        "moderate positive" if rho > 0.3 else
        "weak positive" if rho > 0.1 else
        "negligible" if abs(rho) <= 0.1 else
        "negative"
    )
    return (
        f"LGD-PD correlation is {strength} (rho={rho:.3f}). "
        f"Macro factors explain {100*r2_lgd:.0f}% of LGD variance and "
        f"{100*r2_pd:.0f}% of PD variance. "
        + (
            "Positive rho means LGD is elevated in high-default periods — "
            "the correlation adjustment increases downturn LGD. "
            if rho > 0 else
            "Negative rho is unexpected and may indicate data artefacts. "
            "Review calibration data before applying adjustment."
        )
    )

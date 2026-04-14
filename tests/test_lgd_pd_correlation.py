"""
Tests for src/lgd_pd_correlation.py — Frye-Jacobs LGD-PD correlation.

Covers:
- estimate_lgd_pd_correlation: rho in [-1,1], adj_factor >= 1
- apply_correlation_adjustment: never reduces LGD below base
- build_lgd_pd_annual_series: correct shape and columns
- Raises ValueError on insufficient data
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_annual_series(n=10, seed=0):
    rng = np.random.default_rng(seed)
    years = list(range(2014, 2014 + n))
    lgd_ts = pd.DataFrame({
        "year": years,
        "realised_lgd_ewa": rng.uniform(0.15, 0.55, n),
        "n_defaults": rng.integers(20, 200, n),
        "total_ead": rng.uniform(1e6, 10e6, n),
    })
    pd_ts = pd.DataFrame({
        "year": years,
        "default_rate": rng.uniform(0.01, 0.08, n),
    })
    macro = pd.DataFrame({
        "year": years,
        "gdp_growth_yoy": rng.uniform(-0.03, 0.04, n),
        "unemployment_rate": rng.uniform(0.04, 0.10, n),
        "credit_spread_bps": rng.uniform(50, 300, n),
    })
    return lgd_ts, pd_ts, macro


# ── estimate_lgd_pd_correlation ───────────────────────────────────────────────

def test_estimate_returns_dict():
    from src.lgd_pd_correlation import estimate_lgd_pd_correlation
    lgd_ts, pd_ts, macro = _make_annual_series(10)
    result = estimate_lgd_pd_correlation(lgd_ts, pd_ts, macro)
    assert isinstance(result, dict)


def test_rho_in_valid_range():
    from src.lgd_pd_correlation import estimate_lgd_pd_correlation
    lgd_ts, pd_ts, macro = _make_annual_series(10)
    result = estimate_lgd_pd_correlation(lgd_ts, pd_ts, macro)
    assert "rho" in result
    assert -1.0 <= result["rho"] <= 1.0


def test_adj_factor_at_least_one():
    """Correlation adjustment factor must be >= 1 (never reduce downturn LGD)."""
    from src.lgd_pd_correlation import estimate_lgd_pd_correlation
    lgd_ts, pd_ts, macro = _make_annual_series(10)
    result = estimate_lgd_pd_correlation(lgd_ts, pd_ts, macro)
    assert result["lgd_dt_adjustment_factor"] >= 1.0, \
        "Adjustment factor < 1 would reduce downturn LGD — must be floored at 1"


def test_confidence_interval_present():
    from src.lgd_pd_correlation import estimate_lgd_pd_correlation
    lgd_ts, pd_ts, macro = _make_annual_series(10)
    result = estimate_lgd_pd_correlation(lgd_ts, pd_ts, macro)
    assert "rho_ci" in result
    lo, hi = result["rho_ci"]
    assert lo <= result["rho"] <= hi


def test_regression_summary_dataframe():
    from src.lgd_pd_correlation import estimate_lgd_pd_correlation
    lgd_ts, pd_ts, macro = _make_annual_series(10)
    result = estimate_lgd_pd_correlation(lgd_ts, pd_ts, macro)
    assert "regression_summary" in result
    assert isinstance(result["regression_summary"], pd.DataFrame)


def test_interpretation_string():
    from src.lgd_pd_correlation import estimate_lgd_pd_correlation
    lgd_ts, pd_ts, macro = _make_annual_series(10)
    result = estimate_lgd_pd_correlation(lgd_ts, pd_ts, macro)
    assert "interpretation" in result
    assert isinstance(result["interpretation"], str)
    assert len(result["interpretation"]) > 10


def test_raises_on_insufficient_years():
    from src.lgd_pd_correlation import estimate_lgd_pd_correlation
    lgd_ts, pd_ts, macro = _make_annual_series(3)  # only 3 years, min_years=5
    with pytest.raises(ValueError, match="years"):
        estimate_lgd_pd_correlation(lgd_ts, pd_ts, macro, min_years=5)


def test_raises_on_no_macro_factors():
    from src.lgd_pd_correlation import estimate_lgd_pd_correlation
    lgd_ts, pd_ts, _ = _make_annual_series(10)
    empty_macro = pd.DataFrame({"year": lgd_ts["year"]})  # no factor columns
    with pytest.raises(ValueError, match="macro"):
        estimate_lgd_pd_correlation(lgd_ts, pd_ts, empty_macro)


# ── apply_correlation_adjustment ─────────────────────────────────────────────

def test_apply_correlation_adj_non_negative_uplift():
    """Adjustment should not reduce LGD below base downturn LGD."""
    from src.lgd_pd_correlation import apply_correlation_adjustment
    rng = np.random.default_rng(42)
    base = pd.Series(rng.uniform(0.2, 0.5, 50))
    adjusted = apply_correlation_adjustment(base, rho=0.4, macro_shock_std=0.5)
    assert (adjusted >= base - 1e-9).all(), \
        "apply_correlation_adjustment must not reduce LGD below base"


def test_apply_correlation_adj_with_negative_rho():
    """Negative rho: adj_factor floored at 1.0, so LGD unchanged."""
    from src.lgd_pd_correlation import apply_correlation_adjustment
    base = pd.Series([0.25, 0.30, 0.35])
    adjusted = apply_correlation_adjustment(base, rho=-0.5, macro_shock_std=1.0)
    pd.testing.assert_series_equal(adjusted, base, check_names=False)


def test_apply_correlation_adj_cap():
    """Absolute cap: adjustment cannot exceed cap_adjustment above base."""
    from src.lgd_pd_correlation import apply_correlation_adjustment
    base = pd.Series([0.30])
    cap = 0.05
    adjusted = apply_correlation_adjustment(base, rho=0.99, macro_shock_std=10.0,
                                             cap_adjustment=cap)
    assert float(adjusted.iloc[0]) <= float(base.iloc[0]) + cap + 1e-9


# ── build_lgd_pd_annual_series ────────────────────────────────────────────────

def test_build_annual_series_shape():
    from src.lgd_pd_correlation import build_lgd_pd_annual_series
    import numpy as np
    rng = np.random.default_rng(0)
    n = 80
    years = np.tile(np.arange(2016, 2024), int(np.ceil(n / 8)))[:n]
    loans = pd.DataFrame({
        "realised_lgd": rng.uniform(0.1, 0.6, n),
        "ead_at_default": rng.uniform(50_000, 500_000, n),
        "default_year": years,
    })
    lgd_ts, pd_ts = build_lgd_pd_annual_series(loans)
    assert isinstance(lgd_ts, pd.DataFrame)
    assert isinstance(pd_ts, pd.DataFrame)
    assert "realised_lgd_ewa" in lgd_ts.columns
    assert "default_rate" in pd_ts.columns
    assert len(lgd_ts) == len(pd_ts)
